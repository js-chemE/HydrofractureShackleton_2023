import re
import glob
from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import Affine
from rasterio.warp import reproject, calculate_default_transform, aligned_target #, Resampling

import numpy as np
from skimage import measure, morphology
import os
from typing import Dict, List
import shapely.geometry
import geopandas as gpd
import pandas as pd





import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TILES = [181, 182, 183]
MIN_LAKE_AREA = 0.0018 * 1e6

def merge_split_tiles(base_path: str):
    """
    Given the base path (without the trailing offset codes and extension),
    finds all four split tiles named like:
      base_path-0000000000-0000000000.tif
      base_path-0000000000-0000016384.tif
      base_path-0000016384-0000000000.tif
      base_path-0000016384-0000016384.tif
    Merges them into a single GeoTIFF at base_path.tif (keeps originals).
    """

    # Build glob pattern to find all split parts
    pattern = f"{base_path}-*-*.tif"
    tiles = sorted(glob.glob(pattern))
    if not tiles:
        logger.info(f"No split tiles found for base '{base_path}'.")
        return

    # Open all parts
    src_files = [rasterio.open(fp) for fp in tiles]
    # Merge
    mosaic, out_transform = merge(src_files)

    # Prepare metadata
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw"
    })

    # Write merged file beside the splits
    out_fp = f"{base_path}.tif"
    with rasterio.open(out_fp, "w", **out_meta) as dst:
        dst.write(mosaic)
    logger.info(f"Merged {len(tiles)} tiles → '{out_fp}'.")

    # Close sources
    for src in src_files:
        src.close()

def merge_all_split(folder: str):
    """
    Scans `folder` for:
      1) All split tiles matching pattern *_<10digit>-<10digit>.tif
         Groups them by base path, merges each group (if not already merged).
      2) Leaves any standalone .tif alone.
    The merged outputs are written as base_path.tif alongside originals.
    """
    folder = Path(folder)
    # regex to detect split-tile suffix: -1234567890-0987654321.tif
    split_re = re.compile(r"^(.+)-\d{10}-\d{10}\.tif$")

    # Map base_path → list of split file paths
    groups = {}
    for tif in folder.glob("*.tif"):
        m = split_re.match(tif.name)
        if m:
            base = folder / m.group(1)
            groups.setdefault(str(base), []).append(str(tif))

    # Merge each group
    for base_path, tiles in groups.items():
        out_fp = f"{base_path}.tif"
        if Path(out_fp).exists():
            logger.info(f"Skipped '{out_fp}' (already exists).")
            continue
        # Use our function: it will re-glob the same tiles anyway
        merge_split_tiles(base_path)

    logger.info("Done scanning folder.")

"""===================================================================================================
    Post-processing lakes
==================================================================================================="""

def delete_split_tiles(folder: str):
    """
    Deletes all split tiles matching pattern *_<10digit>-<10digit>.tif recursively in all subfolders
    """
    folder = Path(folder)
    # regex to detect split-tile suffix: -1234567890-0987654321.tif
    split_re = re.compile(r"^(.+)-\d{10}-\d{10}\.tif$")

    # Use rglob to recursively find all .tif files
    for tif in folder.rglob("*.tif"):
        m = split_re.match(tif.name)
        if m:
            logger.info(f"Deleting '{tif}'...")
            try:
                os.remove(tif)
            except Exception as e:
                logger.error(f"Failed to delete {tif}:) {e}") 

"""===================================================================================================
    Post-processing lakes
==================================================================================================="""

def get_min_pixels(min_lake_area: float, res: int) -> int:
    return int(min_lake_area / res**2)


def reduce_lakes_all_summer(folder: str, masks: Dict[int, str], out_folder: str | None = None, skip_dir: List[str] | None = None):
    if out_folder is None:
        out_folder = folder
    os.makedirs(out_folder, exist_ok=True)

    logger.info(f"Post-processing lakes in '{folder}'...")
    for dir in os.listdir(folder):
        if skip_dir and dir in skip_dir:
            logger.info(f"Skipping directory '{dir}'...")
            continue
        dir_path = os.path.join(folder, dir)
        out_path = os.path.join(out_folder, dir)
        if os.path.isdir(dir_path):
            logger.info(f"Processing directory '{dir}'...")
            reduce_lakes_summer(dir_path, masks, out_path)

def reduce_lakes_summer(folder: str, masks: np.ndarray, out_folder: str | None = None):
    if out_folder is None:
        out_folder = folder
    os.makedirs(out_folder, exist_ok=True)

    logger.info(f"Reducing lakes in folder: {folder}")
    files = os.listdir(folder)
    tifffiles = [f for f in files if f.endswith('m.tif')]
    for it, tile in enumerate(TILES):
        logger.info(f"Processing tile {tile} in {os.path.split(folder)[-1]}...")
        tile_files = [f for f in tifffiles if f.startswith(f"tile-{tile}")]
        if not tile_files:
            logger.warning(f"No files found for tile {tile} in {folder}.")
            continue
        logger.info(f"Found {len(tile_files)} files for tile {tile}.")
        for file in tile_files:
            file_path = os.path.join(folder, file)
            out_path = os.path.join(out_folder, file)
            reduce_lakes(file_path, masks[tile], out_path)

def reduce_lakes(file_path: str, mask_path: np.ndarray, out_path: str | None = None) -> tuple:
    if out_path is None:
        out_path = file_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with rasterio.open(file_path) as tiff:
        meta = tiff.meta
        transform = tiff.transform
        clouds = tiff.read(1)
        lakes = tiff.read(2)
        depth = tiff.read(3)
    
    with rasterio.open(mask_path) as tiff:
            vx = tiff.read(
                out_shape=(tiff.count, int(meta["height"]), int(meta["width"])),
                resampling=Resampling.nearest,
            )[0]
            mask = vx * 0 + 1

    """Removing Lakes that don't have any depth or are on the ocean."""
    depth_mask = np.nan_to_num(depth, nan=-9999)
    depth_mask[depth_mask == 0] = -9999
    lakes[depth_mask == -9999] = np.nan
    lakes = lakes * mask

    """Minimum Lake Area"""
    segmented_raw, num = measure.label(label_image=np.nan_to_num(lakes, 0), background=0, return_num=True, connectivity=2)  # type: ignore
    segmented = morphology.remove_small_objects(
        segmented_raw, get_min_pixels(MIN_LAKE_AREA, abs(transform[0]))
    )
    lakes_reduced = segmented.copy()
    lakes_reduced[lakes_reduced >= 1] = 1
    lakes_reduced[lakes_reduced < 1] = 0
    lakes_reduced = lakes_reduced.astype(np.float16)
    lakes_reduced[lakes_reduced < 1] = np.nan
    depth_reduced = depth * lakes_reduced

    stats = {}
    stats_reduced = {}
    stats["plakes"] = segmented_raw.max()
    stats_reduced["plakes"] = segmented.max()
    stats["lakes"] = len(np.unique(segmented_raw)) - 1
    stats_reduced["lakes"] = len(np.unique(segmented)) - 1
    stats["lakes_area"] = stats["plakes"] * transform[0]**2
    stats_reduced["lakes_area"] = stats_reduced["plakes"] * transform[0]**2
    stats["lakes_volume"] = np.nan_to_num(depth.copy(), nan=0).sum() * transform[0]**2
    stats_reduced["lakes_volume"] = np.nan_to_num(depth_reduced.copy(), nan=0).sum() * transform[0]**2

    meta["compress"] = "ZSTD"
    meta["predictor"] = 2
    with rasterio.open(
                out_path, "w", **meta
            ) as src:
        src.write_band(1, clouds.astype(np.uint8))
        src.write_band(2, lakes_reduced.astype(np.uint8))
        src.write_band(3, depth_reduced.astype(np.float16))

    logger.info(f"Processed file {file_path} with stats: {stats["plakes"]} vs {stats_reduced["plakes"]}")

"""===================================================================================================
    Datasets
==================================================================================================="""

def resample_dataset(
    raster, transform, target_width, target_height, mode: str = "bilinear"
):
    if raster.ndim == 2:
        count = 1
    elif raster.ndim > 3:
        raise ValueError("To High dimension of raster")
    else:
        count = raster.shape[0]
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=raster.shape[-2],
            width=raster.shape[-1],
            count=count,
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            if count == 1 and raster.ndim == 2:
                dataset.write(raster, 1)
            else:
                for i in range(count):
                    dataset.write(raster[i], i + 1)

        if mode == "bilinear":
            resamp = Resampling.bilinear
        elif mode == "nearest":
            resamp = Resampling.nearest
        elif mode == "average":
            resamp = Resampling.average
        else:
            raise Exception

        with memfile.open() as src:
            resampled = src.read(
                out_shape=(count, int(target_height), int(target_width)),
                resampling=resamp,
            )
    return resampled



def clip_dataset(raster, transform, coords):
    if raster.ndim == 2:
        count = 1
    else:
        count = raster.shape[0]
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=raster.shape[-2],
            width=raster.shape[-1],
            count=count,
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            if raster.ndim == 2:
                dataset.write(raster, 1)
            else:
                for i in range(count):
                    # print(raster[i])
                    dataset.write(raster[i], i + 1)
                    # print(dataset.meta)

        with memfile.open() as src:
            out_img, out_transform = mask(dataset=src, shapes=coords, crop=True)

            # win_transform = rw.transform(win, src.transform)
    return out_img, out_transform


def vectorize(
    da, transform, crs, attribute_col="attribute", dtype="float32", **rasterio_kwargs
):
    vectors = rasterio.features.shapes(
        source=da.astype(dtype), transform=transform, **rasterio_kwargs
    )

    vectors = list(vectors)
    coords = [p for p, v in vectors]
    values = [v for p, v in vectors]

    polygons = [shapely.geometry.shape(p) for p in coords]

    gdf = gpd.GeoDataFrame(
        data={attribute_col: values}, geometry=polygons, crs=str(crs)
    )  # type: ignore
    return gdf

def get_value(raster: np.ndarray, transform: Affine, x: float, y: float):
    if raster.ndim == 2:
        count = 1
    elif raster.ndim > 3:
        raise ValueError("To High dimension of raster")
    else:
        count = raster.shape[0]
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=raster.shape[-2],
            width=raster.shape[-1],
            count=count,
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            if count == 1 and raster.ndim == 2:
                dataset.write(raster, 1)
            else:
                for i in range(count):
                    dataset.write(raster[i], i + 1)
            row, col = dataset.index(x, y)
            return dataset.read(1)[row, col]



def parse_filename(filename):
    if 'LC08' in filename:
        # Landsat 8: time between LC08 and date
        match = re.search(r'LC08_(\d{6})_(\d{8})\.tif$', filename)
        if match:
            time_str = match.group(1)   # e.g., '113106'
            date_str = match.group(2)   # e.g., '20161208'
            try:    
                dt = pd.to_datetime(f'{date_str}{time_str}', format='%Y%m%d%H%M%S')
            except ValueError:
                try:
                    logger.warning(f"Failed to parse date from 'date_str' : '{date_str}' and 'time_str' : '{time_str}' in filename: {filename}")
                    logger.warning("Trying to parse without time...")
                    time_str = '000000'  # Default time if not provided
                    dt = pd.to_datetime(f'{date_str}{time_str}', format='%Y%m%d%H%M%S')
                except ValueError:
                    logger.error(f"Failed to parse date from 'date_str' : '{date_str}' and 'time_str' : '{time_str}' in filename: {filename}")
                    return None, None
            logger.debug(f"Parsed Landsat 8 date: {dt} from filename: {filename}")
            return 'L8', dt

    elif 'S1A' in filename or 'S1B' in filename:
        # Sentinel-1: first timestamp in format YYYYMMDDTHHMMSS
        match = re.search(r'_(\d{8}T\d{6})_', filename)
        if match:
            try:
                dt = pd.to_datetime(match.group(1), format='%Y%m%dT%H%M%S')
            except ValueError:
                logger.error(f"Failed to parse date from {match.group(1)} in filename: {filename}")
                return None, None
            logger.debug(f"Parsed Sentinel-1 date: {dt} from filename: {filename}")
            return 'S1', dt

    elif re.search(r'\d{8}T\d{6}_\d{8}T\d{6}_T\d{2}[A-Z]{3}', filename):
        # Sentinel-2: first timestamp in format YYYYMMDDTHHMMSS
        match = re.search(r'_(\d{8}T\d{6})_', filename)
        if match:
            try:
                dt = pd.to_datetime(match.group(1), format='%Y%m%dT%H%M%S')
            except ValueError:
                logger.error(f"Failed to parse date from {match.group(1)} in filename: {filename}")
                return None, None
            logger.debug(f"Parsed Sentinel-2 date: {dt} from filename: {filename}")
            return 'S2', dt

    logger.warning(f"Could not parse date from filename: {filename}")
    return None, None

def get_nearby_images(filenames, target_date, before_after=(5, 11), s1_before_after=(4, 6)):
    s2_l8_entries = []
    s1_entries = []

    for fname in filenames:
        try:
            sat_type, dt = parse_filename(fname)
            if dt is None:
                continue
            if sat_type in {'L8', 'S2'}:
                s2_l8_entries.append((dt, fname))
            elif sat_type == 'S1':
                s1_entries.append((dt, fname))
        except Exception as e:
            logger.error(f"Error parsing filename '{fname}': {e}")
            continue

    def get_nearest(entries, n_before, n_after, total, label):
        if not entries:
            return []
        timestamps = [dt for dt, _ in entries]
        target_date_dt = pd.to_datetime(target_date)
        idx = pd.Series(timestamps).searchsorted(target_date_dt)
        start = max(0, idx - n_before) # Include the target date itself
        end = min(len(entries), idx + n_after)
        sliced = entries[start:end]
        # Return filenames sorted by datetime
        return [fname for _, fname in sorted(sliced)]

    # Get 12 for L8/S2: e.g., 5 before, 7 after
    l8_s2_list = get_nearest(s2_l8_entries, before_after[0], before_after[1], before_after[0] + before_after[1], "L8/S2")
    
    # Get 10 for S1: e.g., 4 before, 6 after
    s1_list = get_nearest(s1_entries, s1_before_after[0], s1_before_after[1], s1_before_after[0] + s1_before_after[1], "S1")

    return l8_s2_list, s1_list


def reproject_in_memory(src, target_crs):
    # Compute the transform for the destination CRS
    transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds
    )

    # Allocate destination array
    dst_array = np.empty((src.count, height, width), dtype=src.dtypes[0])

    for i in range(src.count):
        reproject(
            source=src.read(i + 1),
            destination=dst_array[i],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

    # Update metadata
    meta = src.meta.copy()
    meta.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    return dst_array, meta

def rotate_transform_90deg_clockwise(src):
    # Get original transform and image dimensions
    old_transform = src.transform
    width, height = src.width, src.height
    res = old_transform.a  # pixel size (assumes square pixels)

    # Step 1: Define new transform manually (90° clockwise rotation)
    # After rotation: new width = original height, new height = original width
    new_width = height
    new_height = width

    # Step 2: Build new affine transform
    # Clockwise rotation: x' = y, y' = -x
    # So top-left of new image needs to correspond to bottom-left of original
    # This is a pivot around the origin + shift
    new_transform = Affine(
        0, res, 0,
        -res, 0, height * res
    )

    return new_transform, new_width, new_height

def reproject_with_custom_transform(src, new_transform, new_width, new_height, target_crs):
    dst_array = np.empty((src.count, new_height, new_width), dtype=src.dtypes[0])

    for i in range(src.count):
        reproject(
            source=src.read(i + 1),
            destination=dst_array[i],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=new_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

    meta = src.meta.copy()
    meta.update({
        'crs': target_crs,
        'transform': new_transform,
        'width': new_width,
        'height': new_height
    })

    return dst_array, meta
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.io import MemoryFile
from affine import Affine
import rioxarray as rioxr
import geopandas as gpd
import os
from typing import List
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def merge_tiles(folder: str, files: List[str], new_folder: str | None = None):
    if new_folder is None:
        new_folder = folder

    # Open all parts
    src_files = [rasterio.open(os.path.join(folder, f)) for f in files]
    # Merge
    mosaic, out_transform = merge(src_files, nodata=np.nan)

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
    new_name = "_".join(files[0].split("_")[1:])
    out_fp = os.path.join(new_folder, new_name)
    with rasterio.open(out_fp, "w", **out_meta) as dst:
        dst.write(mosaic)
    logger.info(f"Merged {len(files)} tiles → '{new_name}'.")

    # Close sources
    for src in src_files:
        src.close()

def merge_tiles_from_folder(folder: str, keywords: str | List[str]= "dmg", new_folder: str | None = None):
    if new_folder is None:
        new_folder = folder

    if isinstance(keywords, str):
        keywords = [keywords]

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        logger.info(f"Creating folder '{new_folder}'.")

    for keyword in keywords:
        files = [f for f in os.listdir(folder) if f.endswith(f"_{keyword}.tif")]
        if not files:
            logger.info(f"No split tiles found for base '{folder}'.")
            return
        merge_tiles(folder, files, new_folder)


def resample_tiles(
    folder: str,
    files: List[str],
    new_folder: str | None = None,
    scale: float = 10,
    resampling_method: str = "average"
):
    method_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3
    }

    resample_method = method_map.get(resampling_method.lower())
    if resample_method is None:
        raise ValueError(f"Invalid resampling method: '{resampling_method}'")

    for f in files:
        with rasterio.open(os.path.join(folder, f)) as src:
            raster = src.read(1)
            #raster = np.nan_to_num(raster, nan=0)
            #if src.nodata is not None:
            #    raster[raster == src.nodata] = 0
            # Compute new dimensions
            new_width = int(src.width // scale)
            new_height = int(src.height // scale)
            old_transform = src.transform

            # Calculate new transform and metadata
            transform, width, height = calculate_default_transform(
                src.crs, src.crs, src.width, src.height, *src.bounds, dst_width=new_width, dst_height=new_height
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'height': height,
                'width': width,
                'transform': transform,
                'compress': 'lzw',
                'nodata': 0,
            })

            new_f = f.replace("30m", f"{int(old_transform[0]*scale)}m").replace("_output_10px", "")
            with rasterio.open(os.path.join(new_folder, new_f), 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    raster = src.read(i)
                    raster = np.nan_to_num(raster, nan=0)
                    if src.nodata is not None:
                        raster[raster == src.nodata] = 0
                    reproject(
                        source=raster,
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        dst_nodata= 0,
                        resampling=resample_method,
                    )
            
        logger.info(f"Resampled '{f}' to '{os.path.join(new_folder, new_f)}' with scale {scale} and method '{resampling_method}'.")


def resample_tiles_from_folder(
        folder: str,
        keywords: str | List[str]= "dmg",
        new_folder: str | None = None, scale: float = 100,
        resampling_method: str = "average",
        mask_act_dmg: bool = False
):
    if new_folder is None:
        new_folder = folder

    if isinstance(keywords, str):
        keywords = [keywords]
        
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        logger.info(f"Creating folder '{new_folder}'.")

    for keyword in keywords:
        files = [f for f in os.listdir(folder) if f.endswith(f"_{keyword}.tif")]
        if not files:
            logger.info(f"No split tiles found for base '{folder}'.")
            return
        resample_tiles(folder, files, new_folder, scale, resampling_method)
    
    if mask_act_dmg:
        logger.info("Masking active crevasses with damage tiles.")
        dmg_files = [f for f in os.listdir(new_folder) if f.endswith("_dmg.tif")]
        act_files = [f for f in os.listdir(new_folder) if f.endswith("_act.tif")]
        dmg_files.sort()
        act_files.sort()

        for dmg_file, act_file in zip(dmg_files, act_files):
            logger.info(f"Masking '{act_file}'")
            logger.info(f"with '{dmg_file}'")
            with rasterio.open(os.path.join(new_folder, dmg_file)) as dmg_src:
                dmg_raster = dmg_src.read(1)
                dmg_transform = dmg_src.transform
            with rasterio.open(os.path.join(new_folder, act_file)) as act_src:
                act_raster = act_src.read(1)
                act_transform = act_src.transform
                act_raster[dmg_raster == 0] = 0  # Mask out zero damage areas
            with rasterio.open(os.path.join(new_folder, act_file), 'w', **act_src.meta) as dst:
                dst.write(act_raster, 1)

def resample_dataset(raster, transform, target_res, mode: str = "bilinear"):
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
            meta={
                "driver": "GTiff",
                "height": raster.shape[-2],
                "width": raster.shape[-1],
                "count": count,
                "dtype": raster.dtype,
                "transform": transform,
                "crs": rasterio.crs.CRS.from_epsg(4326),
                "nodata": np.nan,
            }
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

        target_height = abs(raster.shape[-2] * transform.a // target_res)
        target_width = abs(raster.shape[-1] * transform.e // target_res)
        with memfile.open() as src:
            resampled = src.read(
                out_shape=(count, int(target_height), int(target_width)),
                resampling=resamp,
            )
            meta = src.meta
        resampled_transform = Affine(
            target_res,
            0,
            transform[2],
            0,
            -target_res,
            transform[5],
        )
        resampled_meta = meta.copy()
        resampled_meta.update(
            {
                "driver": "GTiff",
                "height": int(target_height),
                "width": int(target_width),
                "transform": resampled_transform,
            }
        )
    return resampled, resampled_transform, resampled_meta


def mask_dataset(
    raster, transform, masking_object, mode: str = "shape", filled: bool = True
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
            bounds = dataset.bounds
        if mode == "shape":
            masking_object = masking_object.cx[
                bounds.left : bounds.right, bounds.bottom : bounds.top
            ]
        else:
            raise Exception

        with memfile.open() as src:
            masked, masked_transform = rasterio.mask.mask(
                src, masking_object, crop=True, filled=filled
            )
    return masked, masked_transform


def mask_tile(
    folder: str,
    files: List[str],
    mask_object: gpd.GeoSeries | gpd.GeoDataFrame | str,
    new_folder: str | None = None,
    normalize: bool = True
):
    
    for f in files:
        input_path = os.path.join(folder, f)
        if new_folder is None:
            new_folder = folder
        else:
            os.makedirs(new_folder, exist_ok=True)

        output_path = os.path.join(new_folder, f.replace(".tif", "_masked.tif"))

        with rasterio.open(input_path) as src:
            meta = src.meta.copy()
            meta.update({"compress": "lzw"})
            bounds = src.bounds

            # Determine if mask is vector (shapefile) or raster
            if isinstance(mask_object, gpd.GeoSeries) or isinstance(mask_object, gpd.GeoDataFrame):
                mask_object = mask_object.cx[
                bounds.left : bounds.right, bounds.bottom : bounds.top
                ]
                out_image, out_transform = mask(src, mask_object, crop=True)
                meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": src.nodata if src.nodata is not None else np.nan
                })
            else:
                raise NotImplementedError("Raster mask not implemented yet.")

            with rasterio.open(output_path, "w", **meta) as dst:
                out_image[out_image == 0] = np.nan # dst.nodata
                if normalize:
                    out_image = out_image / np.nanmax(out_image)
                dst.write(out_image)
        logger.info(f"Masked '{f}' → '{output_path}'.")

def mask_tiles_from_folder(
    folder: str,
    mask_object: str,
    keywords: str | List[str] | None = None,
    new_folder: str | None = None
):
    if isinstance(keywords, str):
        keywords = [keywords]

    if new_folder is None:
        new_folder = folder
    else:
        os.makedirs(new_folder, exist_ok=True)
        logger.info(f"Creating folder '{new_folder}'.")

    if keywords is None:
        files = [f for f in os.listdir(folder) if f.endswith(f".tif")]
        if not files:
                logger.warning(f"No tiles found in '{folder}'.")
                return
        mask_tile(folder, files, mask_object, new_folder)
    else:
        for keyword in keywords:
            files = [f for f in os.listdir(folder) if f.endswith(f"_{keyword}.tif")]
            if not files:
                logger.warning(f"No tiles found for keyword '{keyword}' in '{folder}'.")
                continue
            mask_tile(folder, files, mask_object, new_folder)

def normalize_array(arr):
    arr = np.array(arr, dtype=float)
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return np.zeros_like(arr)  # Avoid division by zero
    return (arr - min_val) / (max_val - min_val)

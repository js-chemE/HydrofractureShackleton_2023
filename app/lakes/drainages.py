import logging
import numpy as np
from skimage import measure
import rasterio
import rasterio.windows as rw
from rasterio.transform import Affine
from rasterio.crs import CRS
import geopandas as gpd
import shapely
import pandas as pd
import os
from app.lakes.tiffiles import resample_dataset
from app.lakes.tiffiles import clip_dataset
from app.lakes.tiffiles import vectorize

from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TILES = [181, 182, 183]

def get_dates(filename: str):
    start = pd.Timestamp(filename.split("_")[2])
    end = pd.Timestamp(filename.split("_")[3])
    mean = start + (end - start) / 2
    return start, end, mean

def run_drainage_detection(
        folder_path: str,
        output_path: str = None,
        rgb_path: str = None,
        fname: str = "drainages",
        dominant_satellite: str = "L8",
        min_shrink: float = 0.8,
        min_lake_area: float = 0.054 * 1e6,
):
    """
    Run the drainage detection process.
    Folder path includes all tiles for one summer.
    """
    if output_path is None:
        output_path = folder_path
    else:
        os.makedirs(output_path, exist_ok=True)

    all_files = os.listdir(folder_path)

    drains = []
    for tile in TILES:
        logger.info(f"Processing tile {tile} in {os.path.split(folder_path)[-1]}.")
        tile_files = [f for f in all_files if f.startswith(f"tile-{tile}")]

        

        if not tile_files:
            logger.warning(f"No files found for tile {tile} in {folder_path}.")
            continue

        

        sorted_files = sorted(tile_files, key=lambda x: get_dates(x)[2])

        """dominant window"""
        file_dome = [w for w in sorted_files if w.split("_")[1] == dominant_satellite][0]
        with rasterio.open(
            os.path.join(folder_path, file_dome)
        ) as src:
            dom_raster = src.read()
            dom_transform = src.transform
            dom_meta = src.meta
        dom_res = dom_meta["transform"][0]

        """start window 0"""
        ifile_0 = 0 

        """Going over all windows and collect drains"""
        while True:
            file_0 = sorted_files[ifile_0]
            logger.info(f"Processing file '{file_0}' for tile {tile}.")
            start0, end0, mean0 = get_dates(file_0)
            sat0 = file_0.split("_")[1]

            remaining_files = [f for f in sorted_files if get_dates(f)[0] >= start0 and get_dates(f)[1] <= end0 + pd.Timedelta(days=20) and f != file_0]
            
            logger.debug(f"Remaining files:")
            for rfile in remaining_files:
                logger.debug(f"{rfile}")

            with rasterio.open(
                os.path.join(folder_path, file_0)
            ) as src:
                tif0 = src.read()
                tif0_transform = src.transform
                tif0_meta = src.meta
            
            if sat0 != dominant_satellite:
                logger.info(f"Resampling file '{file_0}' to match the dominant satellite {dominant_satellite}.")
                tif0 = resample_dataset(
                    tif0,
                    tif0_transform,
                    dom_meta["width"],
                    dom_meta["height"],
                    mode="nearest",
                )

            logger.info(f"Initial window established for tile {tile} with file '{file_0}'.")
            logger.info(f"Number of files to process for this window: {len(remaining_files)}.")

            
            for irfile, file_1 in enumerate(remaining_files):
                start1, end1, mean1 = get_dates(file_1)
                sat1 = file_1.split("_")[1]

                """Comparison to Window 1"""
                logger.info(f"{irfile:02d}: Comparing with file '{file_1}' for tile {tile}.")

                with rasterio.open(
                    os.path.join(folder_path, file_1)
                ) as src:
                    tif1 = src.read()
                    tif1_transform = src.transform
                    tif1_meta = src.meta
                
                if sat1 != dominant_satellite:
                    logger.info(f"Resampling file '{file_1}' to match the dominant satellite {dominant_satellite}.")
                    tif1 = resample_dataset(
                        tif1,
                        tif1_transform,
                        dom_meta["width"],
                        dom_meta["height"],
                        mode="nearest",
                    )
                
                if rgb_path is not None:
                    rgb_file_1 = file_1.replace(".tif", "_rgb.tif")
                    if not os.path.exists(os.path.join(rgb_path, rgb_file_1)):
                        logger.warning(f"RGB file '{rgb_file_1}' not found for '{file_1}'. Using default image.")
                        rgb1 = np.ones_like(tif0[0])
                        rgb1_transform = tif0_transform
                        rgb1_meta = tif0_meta
                    else:
                        logger.info(f"Found RGB data for '{file_1}' at '{rgb_file_1}'.")
                        with rasterio.open(
                            os.path.join(rgb_path, rgb_file_1)
                        ) as src:
                            rgb1 = src.read(1)
                            rgb1_transform = src.transform
                            rgb1_meta = src.meta

                        if sat1 != dominant_satellite:
                            rgb1 = resample_dataset(
                                rgb1,
                                rgb1_transform,
                                dom_meta["width"],
                                dom_meta["height"],
                                mode="bilinear",
                            )

                    img1 = rgb1.copy() * 0 + 1
                    logger.info(f"Using RGB data from '{rgb_file_1}' for drainage detection.")
                else:
                    img1 = np.ones_like(tif0[0])
                    logger.warning(f"No RGB data available for '{file_1}', using default image.")

                clouds1 = tif1[0]
                noclouds1 = np.ones_like(clouds1)
                noclouds1[clouds1 > 0] = np.nan

                
                drainage_rpt1 = find_drainages(
                    tif0[1] * noclouds1 * img1,
                    tif1[1],
                    min_shrink=min_shrink,
                    min_lake_pixel_size=int(min_lake_area / dom_res**2),
                )

                drain1 = drainage2vector(
                    drainage_rpt1, dom_transform, crs=CRS.from_string("EPSG:3031")
                )
                drain1["tile"] = tile
                drain1["ifile_0"] = ifile_0
                drain1["area"] = drain1["geometry"].area
                drain1["file_0"] = file_0
                drain1["date-0"] = mean0
                drain1["sat-0"] = sat0
                drain1["start-0"] = start0.strftime("%Y-%m-%d")
                drain1["end-0"] = end0.strftime("%Y-%m-%d")
                drain1["ifile_1"] = irfile
                drain1["file_1"] = file_1
                drain1["date-1"] = mean1.strftime("%Y-%m-%d")
                drain1["sat-1"] = sat1
                drain1["start-1"] = start1.strftime("%Y-%m-%d")
                drain1["end-1"] = end1.strftime("%Y-%m-%d")
                drains.append(drain1)
                logger.info(f"-> {drain1.shape[0]:04d} drainages detected.")
            
            logger.info(f"Processed {len(remaining_files)} files for '{file_0}'.")

            ifile_0 += 1
            # if ifile_0 == 2:
            #     break
            if ifile_0 >= len(sorted_files):
                logger.info(f"Finished processing {len(remaining_files)} files.")
                break
        #2020break
    gdf = gpd.GeoDataFrame(pd.concat(drains, ignore_index=True))
    gdf.set_crs(CRS.from_string("EPSG:3031"), inplace=True)
    gdf.to_file(os.path.join(output_path, f"{fname}.shp"))
        
def find_drainages(
    w0: np.ndarray, w1: np.ndarray, min_shrink=0.8, min_lake_pixel_size: int = 0
):
    drain_rpt = {}
    drain_rpt["seg0"] = np.array(measure.label(label_image=np.nan_to_num(w0, nan=0), background=0, connectivity=2))  # type: ignore
    counts0 = np.unique(drain_rpt["seg0"], return_counts=True)  # type: ignore
    drain_rpt["counted_ids0"] = counts0[0][counts0[1] >= min_lake_pixel_size]
    drain_rpt["pixel_sizes0"] = counts0[1][counts0[1] >= min_lake_pixel_size]
    dropped_ids = counts0[0][counts0[1] < min_lake_pixel_size]

    drain_rpt["seg1"] = drain_rpt["seg0"] * np.nan_to_num(w1, nan=0).astype(int)
    drain_rpt["seg1"][np.isin(drain_rpt["seg1"], dropped_ids)] = 0
    counts1 = np.unique(drain_rpt["seg1"], return_counts=True)  # type: ignore

    drain_rpt["counted_ids1"] = counts1[0]
    drain_rpt["pixel_sizes1"] = counts1[1]

    """Missing"""
    missing_lake_bools = np.isin(
        drain_rpt["counted_ids0"], drain_rpt["counted_ids1"], invert=True
    )
    drain_rpt["missing_lake_ids"] = drain_rpt["counted_ids0"][missing_lake_bools]
    drain_rpt["missing_lake_sizes"] = drain_rpt["pixel_sizes0"][missing_lake_bools]
    raster_drain_bool = np.isin(drain_rpt["seg0"], drain_rpt["missing_lake_ids"])  # type: ignore
    drain_rpt["drain"] = np.zeros(drain_rpt["seg0"].shape)  # type: ignore
    drain_rpt["drain"][raster_drain_bool] = 1
    drain_rpt["drain"][np.invert(raster_drain_bool)] = 0

    """Shrinking"""
    remaining_lake_ids = drain_rpt["counted_ids0"][np.invert(missing_lake_bools)]
    # print(remaining_lake_ids)
    shrink_perc = (
        -1
        * (
            drain_rpt["pixel_sizes1"]
            - drain_rpt["pixel_sizes0"][np.invert(missing_lake_bools)]
        )
        / drain_rpt["pixel_sizes0"][np.invert(missing_lake_bools)]
    )
    # shrink_perc = perc[perc < 0]
    drain_rpt["shrinking_lake_ids"] = remaining_lake_ids[shrink_perc > min_shrink]
    drain_rpt["shrinking_lake_sizes"] = drain_rpt["pixel_sizes0"][
        np.invert(missing_lake_bools)
    ][shrink_perc > min_shrink]
    drain_rpt["shrinking_percs"] = shrink_perc[shrink_perc > min_shrink]

    raster_shrink_bool = np.isin(drain_rpt["seg0"], drain_rpt["shrinking_lake_ids"])  # type: ignore
    drain_rpt["shrink0"] = np.zeros(drain_rpt["seg0"].shape)  # type: ignore
    drain_rpt["shrink0"][raster_shrink_bool] = 1
    drain_rpt["shrink0"][np.invert(raster_shrink_bool)] = 0
    drain_rpt["shrink1"] = np.zeros(drain_rpt["seg1"].shape)  # type: ignore
    drain_rpt["shrink1"][raster_shrink_bool] = 1
    drain_rpt["shrink1"][np.invert(raster_shrink_bool)] = 0

    drain_rpt["fracture"] = drain_rpt["drain"] + drain_rpt["shrink0"]
    drain_rpt["fracture"][drain_rpt["drain"] > 0] = 1
    return drain_rpt

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

def drainage2vector(drainage_rpt, transform, crs):
    drain = {}
    mask_drain = drainage_rpt["drain"].copy()
    mask_drain[mask_drain == 0] = np.nan
    mask_shrink0 = drainage_rpt["shrink0"].copy()
    mask_shrink0[mask_shrink0 == 0] = np.nan
    mask_shrink1 = drainage_rpt["shrink1"].copy()
    mask_shrink1[mask_shrink1 == 0] = np.nan
    drain["drain"] = vectorize(
        drainage_rpt["seg0"],
        transform=transform,
        crs=crs,
        mask=mask_drain.astype(np.uint8),
    )
    drain["shrink0"] = vectorize(
        drainage_rpt["seg0"],
        transform=transform,
        crs=crs,
        mask=mask_shrink0.astype(np.uint8),
    )
    drain["shrink1"] = vectorize(
        drainage_rpt["seg1"],
        transform=transform,
        crs=crs,
        mask=mask_shrink1.astype(np.uint8),
    )

    """Setting Window"""
    drain["drain"].rename(columns={"attribute": "lake id"}, inplace=True)
    drain["shrink0"].rename(columns={"attribute": "lake id"}, inplace=True)
    drain["shrink1"].rename(columns={"attribute": "lake id"}, inplace=True)

    """Setting Window"""
    drain["drain"]["window"] = 0
    drain["shrink0"]["window"] = 0
    drain["shrink1"]["window"] = 1

    """Setting Type"""
    drain["drain"]["type"] = "drain"
    drain["shrink0"]["type"] = "shrink"
    drain["shrink1"]["type"] = "shrink"

    gdfs = [drain["drain"], drain["shrink0"], drain["shrink1"]]
    drain["drain"].crs
    # pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    gdf.set_crs(drain["drain"].crs, inplace=True)
    gdf = gdf[gdf["lake id"] != 0.0]
    return gdf


def assert_rgb_depth(
    drainages_0: pd.DataFrame,
    drainages_1: pd.DataFrame,
    # rgb_path: str,
    tucket_path: str,
    max_fraction_depth_nan: float = 0.6,
    min_median_depth: float = 0.65,
    min_std: float = 0.25,
    buffer: int = 500,
    offset: int = 1000
):
    infos = {}
    for ilake, lake_criteria in enumerate(drainages_0["criteria"].unique()):
        infos[lake_criteria] = {
            "status": "unknown",
            "reason": "unknown",
            "median-0": None,
            "std-0": None,
            "fraction-0": None
        }
        lake0_df = drainages_0[drainages_0["criteria"] == lake_criteria].reset_index(drop=True)
        lake0 = lake0_df.explode(ignore_index=False, index_parts=False)  # type: ignore
        try:
            lake1_df = drainages_1[drainages_1["criteria"] == lake_criteria].reset_index(drop=True)
            lake1 = lake1_df.explode(ignore_index=False, index_parts=False)  # type: ignore
        except KeyError:
            logger.warning(f"Lake '{lake_criteria}' not found in drainages_1. Skipping.")
            lake1 = None
        minx = min(lake0.bounds.minx) - offset
        miny = min(lake0.bounds.miny) - offset
        maxx = max(lake0.bounds.maxx) + offset
        maxy = max(lake0.bounds.maxy) + offset
        file_0 = lake0["file_0"].iloc[0]
        file_1 = lake0["file_1"].iloc[0]

        with rasterio.open(
            os.path.join(tucket_path, file_0)
        ) as src:
            tif0 = src.read(
                window=rw.from_bounds(minx, miny, maxx, maxy, src.transform)
            )
            tif0_transform = Affine(
                src.transform[0], 0, minx, 0, src.transform[4], maxy
            )
        
        
        """Check if depth not nan"""
        try: 
            tif0_clipped, tif0_clipped_transform = clip_dataset(
                tif0, tif0_transform, coords=[geom for geom in lake0["geometry"]]
            )
        except rasterio.errors.RasterioIOError as e:
            logger.error(f"Error clipping dataset for lake '{lake_criteria}': {e}")
            infos[lake_criteria]["status"] = "invalid"
            infos[lake_criteria]["reason"] = "clipping"
            continue

        number_notclipped = tif0_clipped[2][tif0_clipped[2] == 0].size
        number_depth = tif0_clipped[2][tif0_clipped[2] > 0].size
        number_nodepth = tif0_clipped[2].size - number_notclipped - number_depth
        fraction_depth = 1 - number_nodepth / (tif0_clipped[2].size - number_notclipped)
        infos[lake_criteria]["fraction_depth"] = fraction_depth
        if fraction_depth < max_fraction_depth_nan:
            infos[lake_criteria]["status"] = "invalid"
            infos[lake_criteria]["reason"] = "nodepth"
            logger.debug(f"Lake '{lake_criteria}' has {fraction_depth:.2f} fraction of nodepth data.")
            continue

        with rasterio.open(
            os.path.join(tucket_path, file_1)
        ) as src:
            tif1 = src.read(
                window=rw.from_bounds(minx, miny, maxx, maxy, src.transform)
            )
            tif1_transform = Affine(
                src.transform[0], 0, minx, 0, src.transform[4], maxy
            )

        """Check if median depth is bigger"""
        depth0 = tif0_clipped[2]
        depth0[depth0 == 0] = np.nan
        median0 = np.nanmedian(depth0)
        mean0 = np.nanmean(depth0)
        infos[lake_criteria]["median-0"] = median0
        infos[lake_criteria]["mean-0"] = mean0
        infos[lake_criteria]["volume-0"] = mean0 * lake0["area-0"].iloc[0]

        if lake1 is not None:
            try: 
                tif1_clipped, tif1_clipped_transform = clip_dataset(
                    tif1, tif1_transform, coords=[geom for geom in lake1["geometry"]]
                )
                depth1 = tif1_clipped[2]
                depth1[depth1 == 0] = np.nan
                median1 = np.nanmedian(depth1)
                mean1 = np.nanmean(depth1)
                infos[lake_criteria]["median-1"] = median1
                infos[lake_criteria]["mean-1"] = mean1
                infos[lake_criteria]["volume-1"] = mean1 * lake1["area-1"].iloc[0]

            except rasterio.errors.RasterioIOError as e:
                logger.error(f"Error clipping dataset for lake '{lake_criteria}': {e}")
                infos[lake_criteria]["median-1"] = 0
                infos[lake_criteria]["mean-1"] = 0
                infos[lake_criteria]["volume-1"] = 0
                #continue

            except ValueError as e:
                logger.error(f"Value error for lake '{lake_criteria}': {e}")
                infos[lake_criteria]["median-1"] = 0
                infos[lake_criteria]["mean-1"] = 0
                infos[lake_criteria]["volume-1"] = 0
                #continue

            

        if median0 <= min_median_depth:
            infos[lake_criteria]["status"] = "invalid"
            infos[lake_criteria]["reason"] = "median_depth"
            logger.debug(f"Lake '{lake_criteria}' has median depth {median0:.2f} m < {min_median_depth:.2f} m.")
            continue

        """Check if std depth is bigger"""
        std = np.nanstd(depth0)
        infos[lake_criteria]["std_depth"] = std
        if std <= min_std:
            infos[lake_criteria]["status"] = "invalid"
            infos[lake_criteria]["reason"] = "std_depth"
            logger.debug(f"Lake '{lake_criteria}' has std depth {std:.2f} m < {min_std:.2f} m.")
            continue
        
        """Check clouds"""
        clouds0 = vectorize(
            tif0[0],
            transform=tif0_transform,
            crs=lake0.crs,
            mask=tif0[0].astype(np.uint8),
        )
        clouds1 = vectorize(
            tif1[0],
            transform=tif1_transform,
            crs=lake0.crs,
            mask=tif1[0].astype(np.uint8),
        )

        if clouds0.shape[0] != 0:
            clouds1_buffered = gpd.GeoSeries(
                [clouds1.geometry.buffer(buffer).unary_union]
            )
            clouds1_buffered_gdf = gpd.GeoDataFrame(geometry=clouds1_buffered, crs=clouds1.crs)  # type: ignore
            bool0 = lake0.dissolve().intersects(clouds1_buffered_gdf)
            if any(bool0):
                infos[lake_criteria]["status"] = "invalid"
                infos[lake_criteria]["reason"] = "clouds-0"
                continue

        if clouds1.shape[0] != 0:
            clouds1_buffered = gpd.GeoSeries(
                [clouds1.geometry.buffer(buffer).unary_union]
            )
            clouds1_buffered_gdf = gpd.GeoDataFrame(geometry=clouds1_buffered, crs=clouds1.crs)  # type: ignore
            bool1 = lake0.dissolve().intersects(clouds1_buffered_gdf)
            if any(bool1):
                infos[lake_criteria]["status"] = "invalid"
                infos[lake_criteria]["reason"] = "clouds-1"
                continue
        infos[lake_criteria]["status"] = "valid"
        infos[lake_criteria]["reason"] = "passed all checks"

    reasons = [infos[lake]["reason"] for lake in drainages_0["criteria"].unique()]
    invalid_lakes = [lake for lake, info in infos.items() if info["status"] == "invalid"]
    logger.info(f"-> {len(infos.keys()):04d} lakes.")
    logger.info(f"-  {sum(r == 'clipping' for r in reasons):04d} based on clipping.")
    logger.info(f"-  {sum(r == 'nodepth' for r in reasons):04d} based on nodepth.")
    logger.info(f"-  {sum(r == 'median_depth' for r in reasons):04d} based on median-depth.")
    logger.info(f"-  {sum(r == 'std_depth' for r in reasons):04d} based on std-depth.")
    logger.info(f"-  {sum(r == 'clouds-0' for r in reasons):04d} based on clouds in window 0.")
    logger.info(f"-  {sum(r == 'clouds-1' for r in reasons):04d} based on clouds in window 1.")
    logger.info(f"=  {len(drainages_0["criteria"].unique()) - len(invalid_lakes):04d} remain.")
    return infos

def combine_filter_drainages(
    drainage_path: str,
    output_path: str,
    # rgb_path: str = None,
    tucket_path: str = None,
    max_days: int = 10,
    min_lake_area: float = 0.054 * 1e6,
    freezing_months: List[int] = [3, 4, 5, 6, 7, 8, 9, 10, 11],
    max_fraction_depth_nan: float = 0.6,
    min_median_depth: float = 0.65,
    min_std: float = 0.25,
    buffer: int = 500,
    offset: int = 1000,
):

    os.makedirs(output_path, exist_ok=True)

    logger.info("Combining and filtering drainages.")
    logger.info(f"Loading drainages from {drainage_path}.")
    drainages = gpd.read_file(drainage_path)
    drainages_4326 = drainages.to_crs(epsg=4326, inplace=False)
    drainages_4326["centroid"] = drainages_4326.geometry.centroid
    drainages["lon"] = drainages_4326.centroid.x
    drainages["lat"] = drainages_4326.centroid.y

    drainages["criteria"] = (
        drainages["tile"].astype(str) + "_" +
        drainages["ifile_0"].astype(int).astype(str) + "_" +
        drainages["sat-0"].astype(str) + "_" +
        drainages["ifile_1"].astype(int).astype(str) + "_" +
        drainages["sat-1"].astype(str) + "_" +
        drainages["lake id"].astype(int).astype(str) + "_" +
        drainages["date-0"].astype(str) + "_" +
        drainages["date-1"].astype(str) + "_" +
        #(drainages["lon"] * 1e5).round().astype(int).astype(str) + "_" +
        #(drainages["lat"] * 1e5).round().astype(int).astype(str) + "_" +
        drainages["type"].astype(str)
    )

    drainages["date-1"] = drainages["date-1"].astype("datetime64[ns]")
    drainages["date-0"] = drainages["date-0"].astype("datetime64[ns]")
    drainages["start-1"] = pd.to_datetime(drainages["start-1"], format="%Y-%m-%d")
    drainages["start-0"] = pd.to_datetime(drainages["start-0"], format="%Y-%m-%d")
    drainages["end-1"] = pd.to_datetime(drainages["end-1"], format="%Y-%m-%d")
    drainages["end-0"] = pd.to_datetime(drainages["end-0"], format="%Y-%m-%d")

    drainages["days"] = (
        drainages["date-1"] - drainages["date-0"]
    ).dt.days
    drainages["days_long"] = (
        drainages["end-1"] - drainages["start-0"]
    ).dt.days

    logger.info(f"Loaded drainages of shape {drainages.shape}.")

    """Dissolve"""
    drainages_dissolved = drainages.dissolve(by=["criteria", "window"], as_index=False)
    logger.info(f"New shape {drainages_dissolved.shape} after dissolving by criteria and window.")

    """Seperating"""
    drainages_0 = drainages_dissolved[drainages_dissolved["window"] == 0].copy()
    drainages_1 = drainages_dissolved[drainages_dissolved["window"] == 1].copy()
    drainages_0["area-0"] = drainages_0.geometry.area
    drainages_1["area-1"] = drainages_1.geometry.area
    logger.info(f"Separated into ({drainages_0.shape[0]}) drainages in window 0 and ({drainages_1.shape[0]}) in window 1.")

    """Filter days"""
    drainages_filtered_days = drainages_0[drainages_0["days"] <= max_days].copy()
    logger.info(f"New shape {drainages_filtered_days.shape} after filtering by days <= {max_days}.")

    """Within Melting Period"""
    drainages_melting = drainages_filtered_days[
        ~(drainages_filtered_days["start-0"].dt.month.isin(freezing_months)) &
        ~(drainages_filtered_days["end-1"].dt.month.isin(freezing_months))
    ]
    logger.info(f"New shape {drainages_melting.shape} after filtering out freezing months {freezing_months}.")

    """Area"""
    drainages_0_area = drainages_melting[drainages_melting["area-0"] >= min_lake_area]
    logger.info(f"New shape {drainages_0_area.shape} after filtering by area >= {min_lake_area} km².")
    drainages_0_area_criteria = drainages_0_area["criteria"].unique()
    drainages_1_area = drainages_1[
        np.isin(drainages_1["criteria"], drainages_0_area_criteria)
    ].copy()

    """Assert RGB Depth"""
    asserted_infos = assert_rgb_depth(
        drainages_0 = drainages_0_area,
        drainages_1 = drainages_1_area,
        # rgb_path = rgb_path,
        tucket_path = tucket_path,
        max_fraction_depth_nan=max_fraction_depth_nan,
        min_median_depth=min_median_depth,
        min_std=min_std,
        buffer=buffer,
        offset=offset
    )
    drainages_asserted = drainages_0_area.copy()
    new_columns = drainages_asserted.apply(
        lambda row: asserted_infos[row["criteria"]], axis=1, result_type="expand"
    )
    drainages_asserted = pd.concat([drainages_asserted, new_columns], axis=1)

    lakes_to_remove = [asserted_criteria for asserted_criteria, info in asserted_infos.items() if info["status"] == "invalid"]
    drainages_passed = drainages_asserted[~drainages_asserted["criteria"].isin(lakes_to_remove)].reset_index(drop=True)
    drainages_failed = drainages_asserted[drainages_asserted["criteria"].isin(lakes_to_remove)].reset_index(drop=True)

    logger.info(f"New shape {drainages_passed.shape} after asserting RGB, depth, and clouds.")


    """Merge"""
    drainages_0failed = drainages_failed.reset_index(drop=True)
    drainages_1failed = drainages_1[
        np.isin(drainages_1["criteria"], drainages_failed["criteria"].unique())
    ].reset_index(drop=True)

    drainages_0 = drainages_passed.reset_index(drop=True)
    drainages_1 = drainages_1[
        np.isin(drainages_1["criteria"], drainages_passed["criteria"].unique())
    ].reset_index(drop=True)
    logger.info(f"Filtered window 1 drainages to ({drainages_1.shape[0]}) matching criteria from window 0.")

    """Export"""
    fname = os.path.split(drainage_path)[-1]
    drainages_0.to_file(os.path.join(output_path, f"{fname.replace('.shp', '_0.shp')}"))
    drainages_1.to_file(os.path.join(output_path, f"{fname.replace('.shp', '_1.shp')}"))
    drainages_0failed.to_file(os.path.join(output_path, f"{fname.replace('.shp', '_0_failed.shp')}"))
    drainages_1failed.to_file(os.path.join(output_path, f"{fname.replace('.shp', '_1_failed.shp')}"))
    logger.info(f"Exported drainages to {output_path}.")


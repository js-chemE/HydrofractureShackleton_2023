from typing import List

import geopandas as gpd
import pandas as pd

from .config_handling import read_config


def combine_shape_files(paths: List[str]) -> gpd.GeoDataFrame:
    drains: List[gpd.GeoDataFrame] = []

    for f in paths:
        drains.append(gpd.read_file(f))

    all_drains = gpd.GeoDataFrame(pd.concat(drains, ignore_index=True))
    return all_drains


def load_drain_file(
    file: str, crs: str = read_config().get("Geospatial", "crs_wgs84_str")
):
    # Read the excel file containing drain data
    raw_drains: pd.DataFrame = pd.read_excel(file)

    # Create a GeoDataFrame from the raw drain data
    drains: gpd.GeoDataFrame = gpd.GeoDataFrame(
        data=raw_drains,  # type: ignore
        geometry=gpd.points_from_xy(raw_drains.lon, raw_drains.lat),
        crs=crs,
    )

    # Convert date columns to datetime format
    date_columns = [
        c for c in drains.columns if any(s in c for s in ["start", "end", "date"])
    ]
    for col in date_columns:
        drains[col] = pd.to_datetime(drains[col], format="%Y-%m-%d")

    # Convert satellite columns to categorical format
    sat_columns = [c for c in drains.columns if any(s in c for s in ["sat"])]
    for col in sat_columns:
        drains[col] = drains[col].astype("category")

    return drains

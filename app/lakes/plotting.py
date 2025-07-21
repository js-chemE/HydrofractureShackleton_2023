import logging
import numpy as np
import rasterio
import rasterio.windows as rw
from rasterio.transform import Affine
from rasterio.crs import CRS
import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from typing import List


def get_geodataframes(path: str, filter_in: List[str] | None = None) -> List[gpd.GeoDataFrame]:
    """
    Get all GeoDataFrames from the specified path.
    
    Args:
        path (str): Path to the directory containing GeoDataFrames.
        
    Returns:
        List[gpd.GeoDataFrame]: List of GeoDataFrames.
    """
    if filter_in is None:
        filter_in = []

    geodataframes = []
    for file in os.listdir(path):
        if file.endswith('.shp') and (any(f in file for f in filter_in) or len(filter_in) == 0):
            file_path = os.path.join(path, file)
            gdf = gpd.read_file(file_path)
            geodataframes.append(gdf)
    return geodataframes

def plot_all_drainages(path:str, include_failed: bool = False) -> None:
    try:
        gdf_0 = get_geodataframes(path, ["_0."])[0]
        gdf_1 = get_geodataframes(path, ["_1."])[0]
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        gdf_0 = None
        gdf_1 = None
    except IndexError as e:
        logging.error(f"Index error: {e}. Ensure that the shapefiles are present in the directory.")
        gdf_0 = None
        gdf_1 = None

    try:
        gdf_0_failed = get_geodataframes(path, ["_0_failed."])[0] if include_failed else None
        gdf_1_failed = get_geodataframes(path, ["_1_failed."])[0] if include_failed else None
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}. Failed layers will not be plotted.")
        gdf_0_failed = None
        gdf_1_failed = None
        return
    except IndexError as e:
        logging.error(f"Index error: {e}. Ensure that the failed shapefiles are present in the directory.")
        gdf_0_failed = None
        gdf_1_failed = None

    if gdf_0 is not None:
        fig, ax = plt.subplots()
        gdf_0.centroid.plot(ax = ax, color='blue', edgecolor='black', alpha=0.5, label='Drainage 0')
        plt.show()



def truncate_colormap(cmap, minval=0.0, maxval=0.8, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap
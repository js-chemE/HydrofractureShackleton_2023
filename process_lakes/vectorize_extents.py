import app
import pandas as pd
import os

import rasterio
from rasterio.merge import merge

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app.setup_logger(use_console_handler=True, use_file_handler=False)

path_cleaned = r"D:\PhD\21_Experiments\TidesDamageDriver\03_cleaned\L8S2-tucket_extents"
path_cleaned_vector = r"D:\PhD\21_Experiments\TidesDamageDriver\03_cleaned\L8S2-tucket_extents_vector"

YEARS = [2016, 2018, 2019, 2020] # , 2018, 2019, 2020

for year in YEARS: #os.listdir(path_reduced):
    file = os.path.join(path_cleaned, f"{year}_extent.tif")

    gdf = app.lakes.tiffiles.raster_to_vector(file, value=1, out_folder=path_cleaned_vector)
    logger.info(f"Converted raster >>{file}<< to vector with >>{len(gdf)}<< geometries and total area {gdf.area.sum()}.")
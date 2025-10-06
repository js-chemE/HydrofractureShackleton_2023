import app
import pandas as pd
import os

import rasterio
from rasterio.merge import merge

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app.setup_logger(use_console_handler=True, use_file_handler=False)

path_combined = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket_extents_combined"
path_cleaned = r"D:\PhD\21_Experiments\TidesDamageDriver\03_cleaned\L8S2-tucket_extents"

YEARS = [2016, 2018, 2019, 2020] # , 2018, 2019, 2020

for year in YEARS: #os.listdir(path_reduced):
    files_years = [f for f in os.listdir(path_combined) if str(year) in f]

    logger.info(f"Merging {len(files_years)} files for year >>{year}<< into combined extent")
    for f in files_years:
        logger.info(f" - {f}")


    src_files = [rasterio.open(os.path.join(path_combined, f)) for f in files_years]
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
    out_fp = os.path.join(path_cleaned, f"{year}_extent.tif")
    with rasterio.open(out_fp, "w", **out_meta) as dst:
        dst.write(mosaic)
    logger.info(f"Merged {len(files_years)} tiles → '{out_fp}'.")

    # Close sources
    for src in src_files:
        src.close()

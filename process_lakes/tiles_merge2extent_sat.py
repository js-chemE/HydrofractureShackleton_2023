import app
import pandas as pd
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app.setup_logger(use_console_handler=True, use_file_handler=False)



path_extent = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket_extents"
path_combined = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket_extents_combined"

YEARS = [2016, 2018, 2019, 2020] # , 2018, 2019, 2020
TILES = ["tile-181", "tile-182", "tile-183"] # , "tile-182", "tile-183"
SATS = ["L8", "S2"]
dominant_sat = "S2"  # Choose the dominant satellite for final naming

for year in YEARS: #os.listdir(path_reduced):
    if not os.path.isdir(os.path.join(path_extent, str(year))):
        continue

    for tile in TILES:
        files_year_tile = [f for f in os.listdir(os.path.join(path_extent, str(year))) if tile in f]
        if len(files_year_tile) != len(SATS):
            logger.warning(f"Expected {len(SATS)} files for >>{str(year)}<< of >>{tile}<< but found {len(files_year_tile)}")
            continue
        
        dominant_sat_id = [i for i, s in enumerate(SATS) if s == dominant_sat][0]
        logger.info(f"Processing >>{tile}<< >>{year}<< with dominant sat >>{dominant_sat}<< (id >>{dominant_sat_id}<<) and files: {files_year_tile}")
        app.lakes.tiffiles.merge_satextent2extent(
            folder=os.path.join(path_extent, str(year)),
            files=files_year_tile,
            dominant_sat_id=dominant_sat_id,
            new_folder=path_combined,
            new_name=f"{tile}_{year}_extent.tif",
        )
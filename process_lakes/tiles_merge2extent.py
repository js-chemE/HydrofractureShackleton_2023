import app
import pandas as pd
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app.setup_logger(use_console_handler=True, use_file_handler=False)


path_reduced = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket_reduced"
path_extent = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket_extents"

YEARS = [2016, 2018, 2019, 2020] # , 2018, 2019, 2020
TILES = ["tile-181", "tile-182", "tile-183"] # , "tile-182", "tile-183"
SATS = ["L8", "S2"]
for year in YEARS: #os.listdir(path_reduced):
    if not os.path.isdir(os.path.join(path_reduced, str(year))):
        continue

    for tile in TILES:
        files_year_tile = [f for f in os.listdir(os.path.join(path_reduced, str(year))) if tile in f]
        if len(files_year_tile) < 1:
            logger.warning(f"No files found for >>{str(year)}<< of >>{tile}<<")
            continue

        for sat in SATS:
            files_year_tile_sat = [f for f in files_year_tile if sat in f]
            if len(files_year_tile_sat) < 1:
                logger.warning(f"No files found for >>{str(year)}<< of >>{tile}<< and >>{sat}<<")
                continue
            logger.info(f"Merging {len(files_year_tile_sat)} files for >>{str(year)}<< of >>{tile}<< and >>{sat}<<")

            date_start = pd.Timestamp(f"{year}-12-01")
            date_end = pd.Timestamp(f"{year + 1}-03-01")
            app.lakes.tiffiles.merge_tiles2extent(
                folder=os.path.join(path_reduced, str(year)),
                files=files_year_tile_sat,
                new_folder=os.path.join(path_extent, str(year)),
                new_name=f"{tile}_{year}_{sat}_extent.tif",
                date_start=date_start,
                date_end=date_end,
            )
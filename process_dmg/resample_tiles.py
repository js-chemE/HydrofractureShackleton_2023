import app
import os
import geopandas as gpd

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app.setup_logger(use_console_handler=True, use_file_handler=False)

nerd_path = r"D:\PhD\21_Experiments\TidesDamageDriver\03_cleaned\dmg_merged"
dmg_path = r"D:\PhD\21_Experiments\TidesDamageDriver\03_cleaned\dmg_resampled"
dmg_mask_path = r"D:\PhD\21_Experiments\TidesDamageDriver\03_cleaned\dmg_resampled_masked"
iceshelf_path = r"D:\PhD\21_Experiments\TidesDamageDriver\01_raw\greene2022_iceshelves"

YEARS = [2016, 2018, 2019, 2020]

for year in os.listdir(nerd_path): # YEARS: #
    year = str(year)
    if not os.path.isdir(os.path.join(nerd_path, year)):
        continue
    app.dmg.tiffiles.resample_tiles_from_folder(
        folder=os.path.join(nerd_path, year),   
        keywords=["emax", "act", "dmg", "crevSig"],
        new_folder=os.path.join(dmg_path, year),
        scale = 10,
        resampling_method="average",
        mask_act_dmg = False
    )

all_iceshelves_files = os.listdir(iceshelf_path)
ICE_BUFFER = 150
SHELF_BUFFER = 6e3

XMIN = 2.50e6
XMAX = 2.76e6
YMAX = -0.215e6
YMIN = -0.591e6

for year in YEARS: #os.listdir(nerd_path):
    year = str(year)
    year_ = int(year)
    while True:
        iceshelves = [
            gpd.read_file(os.path.join(iceshelf_path, f)).cx[XMIN:XMAX, YMIN:YMAX]
            for f in all_iceshelves_files
            if f.endswith(".shp") and (f"{year_}" in f)
        ]
        if len(iceshelves) == 0:
            logger.warning(f"No iceshelves found for {year}.")
            year_ -= 1
        else:
            break
    if len(iceshelves) == 1:
        ice = iceshelves[0]
        ice.geometry = ice.buffer(ICE_BUFFER)
        ice = ice.dissolve()
        ice.geometry = ice.buffer(-ICE_BUFFER)
        ice_buffer = ice.buffer(SHELF_BUFFER)
    else:
        raise ValueError(
            f"Multiple iceshelves found for {year}: {', '.join([i.name for i in iceshelves])}"
        )
    
    fs = [f for f in os.listdir(os.path.join(dmg_path, year)) if "masked" in f]
    for f in fs:
        os.remove(os.path.join(dmg_path, year, f))

    app.dmg.tiffiles.mask_tiles_from_folder(
        folder=os.path.join(dmg_path, year),
        mask_object=ice_buffer,
        new_folder=os.path.join(dmg_mask_path, year),
    )
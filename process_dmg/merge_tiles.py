import app
import os
app.setup_logger(use_console_handler=True, use_file_handler=False)

nerd_path = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\NeRD"
dmg_path = r"D:\PhD\21_Experiments\TidesDamageDriver\03_cleaned\dmg_merged"

YEARS = [2016, 2018, 2019, 2020] 
for year in os.listdir(nerd_path): #YEARS:# 
    year = str(year)
    if not os.path.isdir(os.path.join(nerd_path, year)):
        continue
    app.dmg.tiffiles.merge_tiles_from_folder(
        folder=os.path.join(nerd_path, year),
        keywords=["emax", "emin", "alphaC", "delta-alpha", "delta-theta", "act", "crevSig", "dmg"],
        new_folder=os.path.join(dmg_path, year)
    )
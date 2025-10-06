import app
import os

app.setup_logger(use_console_handler=True, use_file_handler=False)

YEARS = ["2016", "2018", "2019", "2020"] #] # Extend to other years as needed

for year in YEARS:

    path = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket_reduced", year)
    rgb_folder = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\01_raw\L8S2-mosaics", year)

    out = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\00_drainages_70", year)
    if not os.path.exists(out):
        os.makedirs(out)

    app.lakes.drainages.run_drainage_detection(path, out, rgb_folder, f"{year}_drainages", min_shrink=0.7)
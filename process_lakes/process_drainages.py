import app
import os

app.setup_logger(use_console_handler=True, use_file_handler=False)

YEARS = ["2016", "2018", "2019", "2020", "2022", "2023"] #] # Extend to other years as needed

for year in YEARS:
    drainage_path = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\00_drainages_70", year)
    out_path = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\01_drainages_70", year)
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    tucket_folder = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket_reduced", year)

    app.lakes.drainages.combine_filter_drainages(
        drainage_path = os.path.join(drainage_path, f"{year}_drainages.shp"),
        output_path = out_path,
        tucket_path = tucket_folder,
        )
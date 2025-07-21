import app
import os

app.setup_logger(use_console_handler=True, use_file_handler=False)

year = "2020"

drainage_path = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\00_drainages", year)
out_path = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\01_drainages", year)


rgb_folder = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\01_raw\L8S2-mosaics", year)
tucket_folder = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket_reduced", year)

app.lakes.drainages.combine_filter_drainages(
    drainage_path = os.path.join(drainage_path, f"{year}_drainages.shp"),
    output_path = out_path,
    rgb_path= rgb_folder,
    tucket_path = tucket_folder,
    )
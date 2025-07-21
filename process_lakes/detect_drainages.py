import app
import os

app.setup_logger(use_console_handler=True, use_file_handler=False)

year = "2016"
path = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket_reduced", year)
out = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\00_drainages", year)
rgb_folder = os.path.join(r"D:\PhD\21_Experiments\TidesDamageDriver\01_raw\L8S2-mosaics", year)

app.lakes.drainages.run_drainage_detection(path, out, rgb_folder, f"{year}_drainages")
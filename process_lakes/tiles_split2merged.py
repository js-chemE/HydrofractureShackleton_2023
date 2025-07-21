import app
import os

app.setup_logger(use_console_handler=True, use_file_handler=False)
path = r"D:\PhD\21_Experiments\TidesDamageDriver\01_raw\L8S2-tucket"

for year in os.listdir(path):
    year_path = os.path.join(path, year)
    app.lakes.tiffiles.merge_all_split(year_path)
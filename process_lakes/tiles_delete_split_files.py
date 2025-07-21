import app
import os

app.setup_logger(use_console_handler=True, use_file_handler=False)
path = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket"

app.lakes.tiffiles.delete_split_tiles(path)
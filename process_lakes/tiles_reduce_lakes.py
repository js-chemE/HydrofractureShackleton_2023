import app
import os

app.setup_logger(use_console_handler=True, use_file_handler=False)
path = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket"
out = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\L8S2-tucket_reduced"
vx_folder = r"D:\PhD\21_Experiments\TidesDamageDriver\01_raw\velocity"

MASKS = {
    181: [os.path.join(vx_folder, f) for f in os.listdir(vx_folder) if f.startswith("tile-181")][0],
    182: [os.path.join(vx_folder, f) for f in os.listdir(vx_folder) if f.startswith("tile-182")][0],
    183: [os.path.join(vx_folder, f) for f in os.listdir(vx_folder) if f.startswith("tile-183")][0],
}
print(MASKS)
app.lakes.tiffiles.reduce_lakes_all_summer(path, MASKS, out)
import app
import os
app.setup_logger(use_console_handler=True, use_file_handler=False)

nerd_path = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\NeRD"
v_path = r"D:\PhD\21_Experiments\TidesDamageDriver\01_raw\velocity"

for year in os.listdir(nerd_path):
    
    fs_ = [f for f in os.listdir(os.path.join(nerd_path, year)) if f.endswith("delta-angle.tif")]
    for f_ in fs_:
        os.remove(os.path.join(nerd_path, year, f_))
    
    fs__ = [f for f in os.listdir(os.path.join(nerd_path, year)) if f.startswith("S1_")]
    for f__ in fs__:
        os.remove(os.path.join(nerd_path, year, f__))
    
    fs = [f for f in os.listdir(os.path.join(nerd_path, year)) if f.endswith("alphaC.tif")]
    for f in fs:
        tile = f.split("_")[0]
        result = app.dmg.calculate_delta_velocity_fracture(
            file_vx=os.path.join(v_path, tile + "_vx.tif"),
            file_vy=os.path.join(v_path, tile + "_vy.tif"),
            file_alpha_c=os.path.join(nerd_path, year, f)
        )
        result.export(
            basename="_".join(f.split("_")[:-1]),
            path=os.path.join(nerd_path, year)
        )
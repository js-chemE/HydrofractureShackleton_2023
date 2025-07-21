import app
import os

app.setup_logger(use_console_handler=True, use_file_handler=False)

nerd_path = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\NeRD"

TILES = [181,182,183,187,188]
YEARS = [2016, 2018, 2019, 2020]

for year in os.listdir(nerd_path): # YEARS: # 
    year = str(year)
    for tile in TILES:
        fs_delta_alpha = [f for f in os.listdir(os.path.join(nerd_path, year)) if f.endswith("delta-alpha.tif") and "tile-" + str(tile) in f]
        fs_dmg = [f for f in os.listdir(os.path.join(nerd_path, year)) if f.endswith("dmg.tif")and "tile-" + str(tile) in f]
        if len(fs_delta_alpha) != len(fs_dmg):
            raise ValueError(f"Number of delta-alpha and dmg files do not match for tile {tile} in year {year}.")
        if len(fs_delta_alpha) != 1:
            raise ValueError(f"Expected exactly one delta-alpha and dmg file for tile {tile} in year {year}, found {len(fs_delta_alpha)} and {len(fs_dmg)} respectively.")
        
        f_delta_alpha = fs_delta_alpha[0]
        f_dmg = fs_dmg[0]
        result = app.dmg.calculate_activeness(
            file_delta_alpha=os.path.join(nerd_path, year, f_delta_alpha),
            file_dmg=os.path.join(nerd_path, year, f_dmg),
            target=45,
            spread=15
        )
        result.export(
            basename="_".join(f_delta_alpha.split("_")[:-1]),
            path=os.path.join(nerd_path, year)
        )
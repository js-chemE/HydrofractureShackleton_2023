import app
import os
import geopandas as gpd
import pandas as pd
import logging

app.setup_logger(use_console_handler=True, use_file_handler=False)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


drainage_plot_path = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\02_drainages_plots"
drainage_path = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\01_drainages"

out_path = r"D:\PhD\21_Experiments\TidesDamageDriver\02_processed\02_drainages"

gdf0_all = []
gdf1_all = []
years = ["2016", "2018", "2019", "2020"]
for year in years: # os.listdir(drainage_path)
    try:
        gdf0 = gpd.read_file(os.path.join(drainage_path, year, f"{year}_drainages_0.shp"))
        gdf0_failed = gpd.read_file(os.path.join(drainage_path, year, f"{year}_drainages_0_failed.shp"))
        gdf1 = gpd.read_file(os.path.join(drainage_path, year, f"{year}_drainages_1.shp"))
        gdf1_failed = gpd.read_file(os.path.join(drainage_path, year, f"{year}_drainages_1_failed.shp"))
        print(gdf0.columns)

        for gdf in [gdf0, gdf0_failed, gdf1, gdf1_failed]:
            gdf['year'] = year


        criteria =  [f.split(".")[0] for f in os.listdir(os.path.join(drainage_plot_path, year))]
        logger.info(f"Year: {year}, Criteria: {len(criteria)}")
        
        gdf0_c = gdf0[gdf0['criteria'].isin(criteria)]
        gdf0_failed_c = gdf0_failed[gdf0_failed['criteria'].isin(criteria)]
        gdf1_c = gdf1[gdf1['criteria'].isin(criteria)]
        gdf1_failed_c = gdf1_failed[gdf1_failed['criteria'].isin(criteria)]

        for gdf in [gdf0_c, gdf0_failed_c]:
            if not gdf.empty:
                gdf0_all.append(gdf)
        for gdf in [gdf1_c, gdf1_failed_c]:
            if not gdf.empty:
                gdf1_all.append(gdf)

    except Exception as e:
        logger.error(f"Year: {year}, {e}")

gdf0_combined = gpd.GeoDataFrame(pd.concat(gdf0_all, ignore_index=True))
gdf1_combined = gpd.GeoDataFrame(pd.concat(gdf1_all, ignore_index=True))

print(gdf0_combined.shape, gdf1_combined.shape)
print(gdf0_combined.columns)
print(gdf0_combined.head())

gdf0_combined.to_file(os.path.join(out_path, "drainages_0.shp"))
gdf1_combined.to_file(os.path.join(out_path, "drainages_1.shp"))

with open(os.path.join(out_path, "drainages.txt"), "w") as f:
    f.write("[\n")
    for idx, row in gdf0_combined.iterrows():
        id_str = f"'{idx}'"  # quote string ID
        date_start = f"'{row['year']}-12-01'"
        date_end = f"'{int(row['year']) + 1}-03-01'"
        lon = f"{row['lon']:.6f}"  # optional: round coords
        lat = f"{row['lat']:.6f}"
        line = f"  [{id_str}, {date_start}, {date_end}, {lon}, {lat}],\n"
        f.write(line)
    f.write("]\n")
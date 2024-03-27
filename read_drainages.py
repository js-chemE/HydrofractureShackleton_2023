import os
from pprint import pprint
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

import app

filepath = r"D:\[Code]\HydrofractureShackleton_2023\data-hydrofracture\shackleton_hydrofractures_JS_vis.xlsx"
shape_folder = r"D:\[Code]\HydrofractureShackleton_2023\data-hydrofracture"

# drains = app.load_drain_file(filepath)


s = app.combine_shape_files(
    [
        os.path.join(shape_folder, f)
        for f in os.listdir(shape_folder)
        if "L8S2" in f and ".shp" in f
    ]
)
pprint(s)

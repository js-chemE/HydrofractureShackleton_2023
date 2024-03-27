import os
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt

import app

path_drains: str = r"D:\[Code]\HydrofractureShackleton_2023\data-hydrofracture"
files: List[str] = [
    os.path.join(path_drains, f)
    for f in os.listdir(path_drains)
    if ".shp" in f and "L8S2" not in f
]

print(files)

drains = app.get_combined_drains(files)

pprint(drains)

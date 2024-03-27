import os
from pprint import pprint
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.crs
import rasterio.plot as rplt
from affine import Affine
from pydantic import BaseModel, Field
from rasterio.merge import merge

data_folder = r"D:\[Code]\HydrofractureShackleton_2023\data"


def merge_geotiffs(file_paths: List[str], output_file: str):
    # Open all the GeoTIFFs
    datasets = [rasterio.open(path, mode="r") for path in file_paths]

    # Merge the GeoTIFFs
    merged, merged_transform = merge(datasets)
    print(merged_transform)

    # Update the metadata of the merged GeoTIFF
    merged_meta = datasets[0].meta.copy()
    merged_meta.update(
        {
            "driver": "GTiff",
            "height": merged.shape[1],
            "width": merged.shape[2],
            "transform": merged_transform,
        }
    )

    # Write the merged GeoTIFF to the output file
    with rasterio.open(output_file, "w", **merged_meta) as dst:
        dst.write(merged)

    # Close all the opened GeoTIFF datasets
    for dataset in datasets:
        dataset.close()


files = [
    os.path.join(data_folder, f)
    for f in os.listdir(data_folder)
    if "20181101_20181110" in f and "dmg" in f
]
print(files)

out = merge_geotiffs(files, "test.tif")
# save_geotiff(out, os.path.join(data_folder, "merged.tif"))
# rplt.show(out.data, transform=out.transform)

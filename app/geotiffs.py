import os
from pprint import pprint
from typing import List

import numpy as np
import rasterio
import rasterio.crs
import rasterio.plot as rplt
from affine import Affine
from pydantic import BaseModel, Field
from rasterio.merge import merge


class GeoTiff(BaseModel):
    data: np.ndarray
    metadata: dict
    transform: Affine
    crs: rasterio.crs.CRS
    file_path: str = Field(default=None, alias="file_path")

    class Config:
        arbitrary_types_allowed = True


def open_geotiff(file_path: str, show: bool = False) -> GeoTiff:
    with rasterio.open(file_path) as dataset:
        # Access the raster data, metadata, and other properties
        data = dataset.read()
        metadata = dataset.meta
        crs = dataset.crs
        transform = dataset.transform
        # Do further processing or analysis with the data
        # Return the necessary information
        if show:
            rplt.show(data, transform=transform)
        return GeoTiff(
            data=data,
            metadata=metadata,
            transform=transform,
            crs=crs,
            file_path=file_path,
        )


def combine_geotiffs(
    file_paths: List[str], axis: int = 0, reverse=False, show: bool = False
) -> GeoTiff:
    tiffs = [open_geotiff(f, show=show) for f in file_paths]

    if not all(t.transform[0] == tiffs[0].transform[0] for t in tiffs):
        raise ValueError("All GeoTiffs must have the same transform")

    if not all(t.crs == tiffs[0].crs for t in tiffs):
        raise ValueError("All GeoTiffs must have the same CRS")

    if axis == 0:
        tiffs.sort(key=lambda t: t.transform[5], reverse=reverse)
        new_transform = tiffs[0].transform
        data_combined = tiffs[0].data
        # print(data_combined.shape)
        for t in tiffs[1:]:
            data_combined = np.concatenate((data_combined, t.data), axis=axis + 1)
            # print(data_combined.shape)
        # rplt.show(data_combined, transform=new_transform)

        out = tiffs[0]
        out.data = data_combined
        out.transform = new_transform
        out.metadata["height"] = data_combined.shape[1]
        print(out)
    elif axis == 1:
        tiffs.sort(key=lambda t: t.transform[2], reverse=reverse)
        new_transform = tiffs[0].transform
        raise NotImplementedError("Merging along the x-axis is not yet supported")

    return out


def save_geotiff(tiff: GeoTiff, file_path: str) -> None:
    with rasterio.open(file_path, "w", **tiff.metadata) as dataset:
        dataset.write(tiff.data)
        dataset.transform = tiff.transform
        dataset.crs = tiff.crs


def merge_geotiffs(file_paths: List[str], output_file: str):
    # Open all the GeoTIFFs
    datasets = [rasterio.open(path, mode="r") for path in file_paths]

    # Merge the GeoTIFFs
    merged, merged_transform = merge(datasets)

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

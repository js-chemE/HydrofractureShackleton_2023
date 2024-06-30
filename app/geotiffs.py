import os
from pprint import pprint
from typing import List, Tuple

import numpy as np
import rasterio
import rasterio.crs
import rasterio.mask
import rasterio.merge
import rasterio.plot as rplt
from affine import Affine
from pydantic import BaseModel, Field
from rasterio.enums import Resampling
from rasterio.io import MemoryFile


def merge_geotiffs(file_paths: List[str], output_file: str) -> None:
    # Open all the GeoTIFFs
    datasets = [rasterio.open(path, mode="r") for path in file_paths]

    # Merge the GeoTIFFs
    merged, merged_transform = rasterio.merge.merge(datasets)

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


def mask_dataset(
    raster, transform, masking_object, mode: str = "shape", filled: bool = True
):
    if raster.ndim == 2:
        count = 1
    elif raster.ndim > 3:
        raise ValueError("To High dimension of raster")
    else:
        count = raster.shape[0]
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=raster.shape[-2],
            width=raster.shape[-1],
            count=count,
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            if count == 1 and raster.ndim == 2:
                dataset.write(raster, 1)
            else:
                for i in range(count):
                    dataset.write(raster[i], i + 1)
            bounds = dataset.bounds
        if mode == "shape":
            masking_object = masking_object.cx[
                bounds.left : bounds.right, bounds.bottom : bounds.top
            ]
        else:
            raise Exception

        with memfile.open() as src:
            masked, masked_transform = rasterio.mask.mask(
                src, masking_object, crop=True, filled=filled
            )
    return masked, masked_transform


def transpose_raster(raster: np.ndarray, transform: Affine):
    return np.flip(raster.T, axis=0), Affine(
        transform.e,
        0,
        transform.f,
        0,
        -transform.a,
        transform.c + raster.shape[-1] * transform.a,
    )


def open(file_path: str):
    with rasterio.open(file_path) as src:
        raster_original = src.read()
        transform_original = src.transform
        meta = src.meta
    return raster_original, transform_original, meta


def open_resampled(
    file_path: str, target_res, nan: int = 0, mode: str = "average"
) -> Tuple[np.ndarray, Affine]:
    with rasterio.open(file_path) as src:
        raster_original = src.read()
        transform_original = src.transform
        meta = src.meta

    resampled, resampled_transform = resample_dataset(
        np.nan_to_num(raster_original[0], nan=nan),
        transform_original,
        target_res,
        mode=mode,
    )
    return resampled, resampled_transform


def resample_dataset(raster, transform, target_res, mode: str = "bilinear"):
    if raster.ndim == 2:
        count = 1
    elif raster.ndim > 3:
        raise ValueError("To High dimension of raster")
    else:
        count = raster.shape[0]
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=raster.shape[-2],
            width=raster.shape[-1],
            count=count,
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            if count == 1 and raster.ndim == 2:
                dataset.write(raster, 1)
            else:
                for i in range(count):
                    dataset.write(raster[i], i + 1)

        if mode == "bilinear":
            resamp = Resampling.bilinear
        elif mode == "nearest":
            resamp = Resampling.nearest
        elif mode == "average":
            resamp = Resampling.average
        else:
            raise Exception

        target_height = abs(raster.shape[-2] * transform.a // target_res)
        target_width = abs(raster.shape[-1] * transform.e // target_res)
        with memfile.open() as src:
            resampled = src.read(
                out_shape=(count, int(target_height), int(target_width)),
                resampling=resamp,
            )
        resampled_transform = Affine(
            target_res,
            0,
            transform[2],
            0,
            -target_res,
            transform[5],
        )
    return resampled, resampled_transform


def create_active_crevasses_mask(
    d_alpha: np.ndarray, target: float = 45, spread: float = 10
):

    mask0 = abs(d_alpha) <= 90  # only consider values between -90 and 90
    mask1 = np.abs(d_alpha - target) <= spread  # target +- spread
    mask2 = np.abs(-d_alpha - target) <= spread  # -target +- spread

    combined_mask1 = mask0 & mask1
    combined_mask2 = mask0 & mask2

    active_mask = np.zeros(d_alpha.shape) * np.nan
    active_mask[combined_mask1] = 1
    active_mask[combined_mask2] = 1

    active = active_mask.copy()
    active[combined_mask1] = (
        d_alpha[combined_mask1] * active_mask[combined_mask1] - target
    )
    active[combined_mask2] = (
        d_alpha[combined_mask2] * active_mask[combined_mask2] + target
    )
    return active


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

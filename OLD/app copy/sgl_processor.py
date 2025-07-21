import os
import sys

# import time
# from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import pyproj
import rasterio
import rasterio.features
import rasterio.plot as rplt
import rasterio.windows as rw
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiLineString, Polygon, mapping, shape

# from shapely.geometry import MultiPoint, MultiPolygon, Polygon, mapping, shape
from skimage import measure, morphology

import app.sgl_utils

"""----------------------------------------------------------------------------------------
    Options
    ----------------------------------------------------------------------------------------"""


@dataclass
class ProcessorSGLSettings:
    """Dataclass containing all relevant options for the PostProcessorSGL"""

    region = "Shackleton"
    parent_folder = r"D:\[NOT SYNC]\ASRP_Data"
    rangestart: str = "???"
    rangeend: str = "???"
    min_lake_area = 0.0018  # km^2
    crs_ant_str = "EPSG:3031"
    crs_wgs84_str = "EPSG:4326"  # "WGS84"

    @property
    def season_folder(self):
        return os.path.join(
            self.parent_folder,
            self.region.lower() + "_sgl_" + self.rangestart + "_" + self.rangeend,
        )

    @property
    def region_folder(self):
        return os.path.join(self.parent_folder, self.region.lower())

    @property
    def dmg_folder(self):
        return os.path.join(self.parent_folder, self.region.lower() + "_S1_30m")

    @property
    def img_folder(self):
        return os.path.join(self.parent_folder, "images")

    @classmethod
    def from_range(cls, rangestart: str, rangeend: str):
        return cls(rangestart=rangestart, rangeend=rangeend)


"""----------------------------------------------------------------------------------------
    Initialization
    ----------------------------------------------------------------------------------------"""


def initialize(settings: ProcessorSGLSettings) -> None:
    sys.path.append(settings.parent_folder)
    sys.path.append(settings.region_folder)
    sys.path.append(settings.dmg_folder)
    print("X=====================================================")
    print(f"SGLProcessor started for {settings.rangestart} to {settings.rangeend}")
    print("X=====================================================")

    if not os.path.isdir(settings.region_folder):
        os.makedirs(settings.region_folder)
        print("created folder : ", settings.region_folder)

    if not os.path.isdir(settings.season_folder):
        os.makedirs(settings.season_folder)
        print("created folder : ", settings.season_folder)


def check_file_name(settings: ProcessorSGLSettings, filename: str):
    infos = filename.split("_")
    start = infos[2]
    end = infos[3]
    if start == settings.rangestart.replace(
        "-", ""
    ) and end == settings.rangeend.replace("-", ""):
        return True
    else:
        return False


def get_min_pixels(min_lake_area: float, res: int) -> int:
    return int(min_lake_area * 1e6 / res**2)


"""----------------------------------------------------------------------------------------
    Reading of files
    ----------------------------------------------------------------------------------------"""


def read_pre_csv_files(
    settings: ProcessorSGLSettings,
    files: List[str],
    relevant_keywords=["IVS", "weighting", "features"],
):
    csv_inputs = []
    print(f"=====> Read .csv files.")
    relevant_files = [f for f in files if any(x in f for x in relevant_keywords)]
    for f in relevant_files:
        csv_input = {}
        if check_file_name(settings, f):
            csv_input["name"] = f.split("_")[0]
            csv_input["satellite"] = f.split("_")[1]
            csv_input["start"] = f.split("_")[2]
            csv_input["end"] = f.split("_")[3]
            csv_input["type"] = f.split("_")[5]
            csv_input["file"] = f
            try:
                if f.split(".")[0].split("_")[-1] == "IVS":
                    csv_input["type"] = "ivs"
                if f.split(".")[0].split("_")[-1] == "weighting":
                    csv_input["type"] = "weighting"
                if f.split(".")[0].split("_")[-1] == "features":
                    csv_input["type"] = "features"
                csv_input["data"] = pd.read_csv(
                    os.path.join(settings.season_folder, f), index_col=False
                )
                csv_input["data"]["criteria"] = (
                    csv_input["data"]["satellite"]
                    + "_"
                    + csv_input["data"]["start"]
                    + "_"
                    + csv_input["data"]["end"]
                )
                csv_inputs.append(csv_input)
            except Exception as e:
                print(e)
    return csv_inputs


def get_metadata_from_csv(folder: str, rois: List[str]) -> List[dict]:
    print(f"=====> Get metadata from .csv files.")
    files = [f for f in os.listdir(folder)]
    files_csv = [
        f for f in files if f.split("_")[-1] == "meta.csv" and f.split("_")[0] in rois
    ]
    csv_metas = []
    for roi in rois:
        f = [f for f in files_csv if f.split("_")[0] == roi][0]
        csv_meta = {}
        csv_meta["name"] = f.split("_")[0]
        csv_meta["satellite"] = f.split("_")[1]
        csv_meta["start"] = f.split("_")[2]
        csv_meta["end"] = f.split("_")[3]
        csv_meta["file"] = f
        csv_meta["data"] = pd.read_csv(os.path.join(folder, f), index_col=False)
        csv_metas.append(csv_meta)
    return csv_metas


"""----------------------------------------------------------------------------------------
    Processing csv files
    ----------------------------------------------------------------------------------------"""


def combine_csv(csv_inputs, rois: List[str]) -> List[dict]:
    print(f"=====> Combine .csv.")
    csv_combined_inputs = []
    for roi_index, roi in enumerate(rois):
        csv_combined_input = {}
        inputs_filtered = [inp for inp in csv_inputs if inp["name"] == roi]
        features = [inp for inp in inputs_filtered if inp["type"] == "features"][0]
        features["data"] = features["data"][
            [
                "IVSs",
                "Images",
                "criteria",
                "end",
                "numImages",
                "res",
                "satellite",
                "start",
                "windowDays",
            ]
        ]
        weighting = [inp for inp in inputs_filtered if inp["type"] == "weighting"][0]
        weighting["data"] = weighting["data"][
            ["Image_ID", "LPCSs", "criteria", "TLakePixel", "date", "id"]
        ]
        ivs = [inp for inp in inputs_filtered if inp["type"] == "ivs"][0]
        ivs["data"] = ivs["data"][["IVS", "criteria", "IceShelfMask"]]
        combined = pd.merge(
            features["data"],
            weighting["data"],
            how="inner",
            left_on="criteria",
            right_on="criteria",
        )
        combined = pd.merge(
            combined, ivs["data"], how="inner", left_on="criteria", right_on="criteria"
        )
        csv_combined_input["name"] = features["name"]
        csv_combined_input["start"] = features["start"]
        csv_combined_input["end"] = features["end"]
        csv_combined_input["satellite"] = features["satellite"]
        csv_combined_input["data"] = combined
        csv_combined_inputs.append(csv_combined_input)
    return csv_combined_inputs


def export_metadata_csvs(folder: str, csv_combined_inputs: list[dict]) -> None:
    print(f"=====> Export combined metadata .csv files.")
    for csv_input in csv_combined_inputs:
        filename = "_".join(
            [
                csv_input["name"],
                csv_input["satellite"],
                csv_input["start"],
                csv_input["end"],
                "meta",
            ]
        )
        csv_input["data"].to_csv(os.path.join(folder, filename + ".csv"), index=False)


def process_and_export_csv(
    settings: ProcessorSGLSettings, files_csv: list[str], names: list[str]
):
    csv_inputs = read_pre_csv_files(settings, files_csv)
    csv_combined_inputs = combine_csv(csv_inputs, names)
    export_metadata_csvs(settings.season_folder, csv_combined_inputs)


def update_metadata_csvs_from_collection(
    folder: str, collection: app.sgl_utils.WindowCollection
):
    print(f"=====> Update .csv for {collection.name}")
    names = collection.names
    for name in names:
        windows = collection.get_filtered_windows(names=[name])
        filename = "_".join(
            [name, "L8S2", collection.rangestart, collection.rangeend, "meta.csv"]
        )
        data = pd.DataFrame([w.metadata for w in windows])
        data.to_csv(os.path.join(folder, filename), index=False)


"""----------------------------------------------------------------------------------------
    Create and Process Windows
    ----------------------------------------------------------------------------------------"""


def create_collection_from_meta(settings: ProcessorSGLSettings, csv_metas: List[dict]):
    print("=====> Create collection from csv meta.")
    ws = []
    for meta in csv_metas:
        for i in range(len(meta["data"])):
            w = app.sgl_utils.Window()
            w.name = meta["name"]
            w.source = "csv"
            w.rangestart = meta["start"]
            w.rangeend = meta["end"]
            w.start = meta["data"]["start"].iloc[i].replace("-", "")
            w.end = meta["data"]["end"].iloc[i].replace("-", "")
            w.satellite = meta["data"]["satellite"].iloc[i]
            w.metadata = meta["data"].iloc[i].to_dict()
            w.res = app.sgl_utils.RES[w.satellite]
            ws.append(w)
    print(f"     | {len(ws)} windows created for collection.")
    return app.sgl_utils.WindowCollection.from_settings_windows(settings, ws)


def read_maxlake_tifs(input_folder: str, max_lake_files: List[str]):
    tiffs = {}
    for f in max_lake_files:
        with rasterio.open(os.path.join(input_folder, f)) as tiff:
            tiffs[f.split("_")[0]] = tiff
    return tiffs


def read_iceshelf_tifs(input_folder: str, iceshelf_files: List[str]):
    tiffs = {}
    for f in iceshelf_files:
        with rasterio.open(os.path.join(input_folder, f)) as tiff:
            tiffs[f.split("_")[0]] = tiff
    return tiffs


def post_process_tifs(
    settings: ProcessorSGLSettings,
    collection: app.sgl_utils.WindowCollection,
    max_windows=1e3,
    update_tif: bool = False,
    update_csv: bool = False,
    print_metadata: bool = False,
):
    print("=====> Post process tifs and export reduced tifs.")
    if print_metadata:
        print(
            f"     | {'name':13s}",
            f"| {'start':10s}",
            f"| {'end':10s}",
            f"| {'lakes':>8s} {'':9s}",
            f"| {'area':>7s} {'km2':<9s}",
            f"| {'volume':>7s} {'Mm3':<9s}",
        )

    for index_w, w in enumerate(collection.windows):
        """Opening Files"""
        with rasterio.open(os.path.join(settings.season_folder, w.file_gee)) as tiff:
            meta = tiff.meta
            clouds = tiff.read(1)
            lakes = tiff.read(2)
            depth = tiff.read(3)

        with rasterio.open(os.path.join(settings.region_folder, w.file_vx)) as tiff:
            vx = tiff.read(
                out_shape=(tiff.count, int(meta["height"]), int(meta["width"])),
                resampling=Resampling.nearest,
            )[0]
            mask = vx * 0 + 1

        """Removing Lakes that don't have any depth or are on the ocean."""
        depth_mask = np.nan_to_num(depth, nan=-9999)
        depth_mask[depth_mask == 0] = -9999
        lakes[depth_mask == -9999] = np.nan
        lakes = lakes * mask

        """Minimum Lake Area"""
        segmented_raw, num = measure.label(label_image=np.nan_to_num(lakes, 0), background=0, return_num=True, connectivity=2)  # type: ignore
        segmented = morphology.remove_small_objects(
            segmented_raw, get_min_pixels(settings.min_lake_area, w.res)
        )
        lakes_reduced = segmented.copy()
        lakes_reduced[lakes_reduced >= 1] = 1
        lakes_reduced[lakes_reduced < 1] = 0
        lakes_reduced = lakes_reduced.astype(np.float16)
        lakes_reduced[lakes_reduced < 1] = np.nan
        depth_reduced = depth * lakes_reduced

        """Saving Statistics"""
        w.metadata["stat:plakes"] = segmented_raw.max()
        w.metadata["stat:plakes_reduced"] = segmented.max()
        w.metadata["stat:#lakes"] = (
            len(np.unique(segmented_raw)) - 1
        )  # 0 is background and should not be counted
        w.metadata["stat:#lakes_reduced"] = len(np.unique(segmented)) - 1
        w.metadata["stat:alakes"] = w.metadata["stat:plakes"] * w.res**2
        w.metadata["stat:alakes_reduced"] = w.metadata["stat:plakes_reduced"] * w.res**2
        w.metadata["stat:vlakes"] = np.nan_to_num(depth.copy(), nan=0).sum() * w.res**2
        w.metadata["stat:vlakes_reduced"] = (
            np.nan_to_num(depth_reduced.copy(), nan=0).sum() * w.res**2
        )

        if update_tif:
            with rasterio.open(
                os.path.join(settings.season_folder, w.file_reduced), "w", **meta
            ) as src:
                src.write_band(1, clouds)
                src.write_band(2, lakes_reduced)
                src.write_band(3, depth_reduced)

        if w.metadata["stat:plakes"] == 0:
            w.metadata["stat:f#lakes"] = 0
            w.metadata["stat:falakes"] = 0
            w.metadata["stat:fplakes"] = 0
            w.metadata["stat:fvlakes"] = 0
        else:
            w.metadata["stat:fplakes"] = (
                w.metadata["stat:plakes_reduced"] / w.metadata["stat:plakes"]
            )
            w.metadata["stat:f#lakes"] = (
                w.metadata["stat:#lakes_reduced"] / w.metadata["stat:#lakes"]
            )
            w.metadata["stat:falakes"] = (
                w.metadata["stat:alakes_reduced"] / w.metadata["stat:alakes"]
            )
            w.metadata["stat:fvlakes"] = (
                w.metadata["stat:vlakes_reduced"] / w.metadata["stat:vlakes"]
            )

        if print_metadata:
            print(
                f"     | {w.name:10s}",
                f"{w.satellite:2s}",
                f"| {w.start:10s}",
                f"| {w.end:10s}",
                f"| {w.metadata['stat:#lakes_reduced']:8d} ({w.metadata['stat:f#lakes']*100:>6.2f}%)",
                f"| {w.metadata['stat:alakes_reduced']*1e-6:7.3f} ({w.metadata['stat:falakes']*100:>6.2f}%)",
                f"| {w.metadata['stat:vlakes_reduced']*1e-6:7.3f} ({w.metadata['stat:fvlakes']*100:>6.2f}%)",
            )
        if index_w - 1 >= max_windows:
            break

    if update_csv:
        update_metadata_csvs_from_collection(settings.season_folder, collection)
    return collection


"""----------------------------------------------------------------------------------------
Create Max Lake Extent
----------------------------------------------------------------------------------------"""


def export_max_lake_extents(
    folder: str,
    collection: app.sgl_utils.WindowCollection,
    print_metadata: bool = False,
) -> None:
    print("=====> Create and export max lake extents from reduced lake masks.")
    for name in collection.names:
        # max_lake_extents = []
        # metas = {}
        for sat in collection.satellites:
            windows = collection.get_filtered_windows(names=[name], satellites=[sat])
            count = 0
            lake_extent = np.array([])
            meta = {}
            for w in windows:
                with rasterio.open(os.path.join(folder, w.file_reduced)) as tiff:
                    meta = tiff.meta
                    lakes = tiff.read(2)
                if count == 0:
                    lake_extent = np.nan_to_num(lakes, nan=0)
                else:
                    lake_extent = lake_extent + np.nan_to_num(lakes, nan=0)
                count += 1
            lake_extent[lake_extent > 0] = 1
            lake_extent[lake_extent == 0] = np.nan

            meta.update(
                count=1,
            )

            filename = "_".join(
                [name, sat, collection.rangestart, collection.rangeend, "maxlake.tif"]
            )
            with rasterio.open(os.path.join(folder, filename), "w", **meta) as src:
                src.write_band(1, lake_extent)
            if print_metadata:
                print(
                    f"     | {filename} | {np.nan_to_num(lake_extent, nan = 0).sum()}"
                )


def combine_max_lake_extents(folder: str, collection, print_metadata=False) -> None:
    print("=====> Combine and create masks for each single tile and resolution.")
    for name in collection.names:
        filenameS2 = "_".join(
            [name, "S2", collection.rangestart, collection.rangeend, "maxlake.tif"]
        )
        with rasterio.open(os.path.join(folder, filenameS2)) as srcS2:
            extentS2 = srcS2.read(1)
            metaS2 = srcS2.meta
            transformS2 = srcS2.transform

        filenameL8 = "_".join(
            [name, "L8", collection.rangestart, collection.rangeend, "maxlake.tif"]
        )
        with rasterio.open(os.path.join(folder, filenameL8)) as srcL8:
            extentL8 = srcL8.read(
                out_shape=(srcL8.count, int(metaS2["height"]), int(metaS2["width"])),
                resampling=Resampling.nearest,
            )[0]
            metaL8 = srcL8.meta
            transformL8 = srcL8.transform

        extent = np.nan_to_num(extentS2, nan=0) + np.nan_to_num(extentL8, nan=0)
        extent[extent > 0] = 1
        extent[extent == 0] = np.nan

        filename10 = "_".join(
            [
                name,
                "L8S2",
                app.sgl_utils.date_with_comma(collection.rangestart),
                app.sgl_utils.date_with_comma(collection.rangeend),
                "10m_maxlake.tif",
            ]
        )
        with rasterio.open(os.path.join(folder, filename10), "w", **metaS2) as src:
            src.write_band(1, extent)

        with rasterio.open(os.path.join(folder, filename10)) as src:
            extent30 = src.read(
                out_shape=(src.count, int(metaL8["height"]), int(metaL8["width"])),
                resampling=Resampling.nearest,
            )[0]
        extent30[extent30 > 0.0] = 1
        extent30[extent30 <= 0.0] = np.nan
        filename30 = "_".join(
            [
                name,
                "L8S2",
                app.sgl_utils.date_with_comma(collection.rangestart),
                app.sgl_utils.date_with_comma(collection.rangeend),
                "30m_maxlake.tif",
            ]
        )
        with rasterio.open(os.path.join(folder, filename30), "w", **metaL8) as src:
            src.write_band(1, extent30)

        if print_metadata:
            print(
                f"     | {name} | {np.nan_to_num(extent, nan = 0).sum() * 10**2} | {np.nan_to_num(extent30, nan = 0).sum()* 30**2}"
            )


def get_lake_extents(folder: str, collection):
    extents = []
    for f in collection.lake_extent_combined_files:
        extent = {}
        extent["name"] = f.split("_")[0]
        extent["start"] = f.split("_")[2]
        extent["end"] = f.split("_")[3]
        extent["res"] = f.split("_")[4][0:2]
        with rasterio.open(os.path.join(folder, f)) as src:
            extent["data"] = src.read(1)
            extent["meta"] = src.meta
            extent["transform"] = src.transform
        extents.append(extent)
    return extents


def add_extent_statistics(
    folder: str, collection, update_bool: bool = False, print_bool: bool = True
) -> app.sgl_utils.WindowCollection:
    print("=====> Add lake extent statistics.")
    lake_extents = get_lake_extents(folder, collection)
    iceshelf_extents = get_iceshelf_extents(folder, collection)
    if print_bool:
        print(
            f"     | {'name':13s}",
            f"| {'start':10s}",
            f"| {'end':10s}",
            f"| {'extent km2':>10s}",
            f"| {'lake':>8s} {'km2':<8s}",
            f"| {'IVS':>6s}",
        )

    for w in collection.windows:
        lake_extent = [
            tif
            for tif in lake_extents
            if tif["name"] == w.name and int(tif["res"]) == w.res
        ][0]
        iceshelf = [
            tif
            for tif in iceshelf_extents
            if tif["name"] == w.name and int(tif["res"]) == w.res
        ][0]

        # print(lake_extent["tif"].read(1).sum(), np.nan_to_num(lake_extent["tif"].read(1), nan = 0).sum())
        # print(iceshelf["tif"].read(1).sum(), np.nan_to_num(iceshelf["tif"].read(1), nan = 0).sum())
        extent_iceshelf = lake_extent["data"] * iceshelf["data"]
        # extent_iceshelf = np.nan_to_num(lake_extent["tif"].read(1), nan = 0) * np.nan_to_num(iceshelf["tif"].read(1), nan = 0)

        extent_iceshelf[extent_iceshelf > 0] = 1
        extent_iceshelf[extent_iceshelf == 0] = np.nan
        # print(extent_iceshelf.sum(), np.nan_to_num(extent_iceshelf, nan = 0).sum())

        # print(extent.read(1).sum(), extent_iceshelf.sum())
        w.metadata["name"] = w.name
        w.metadata["stat:plakes_extent"] = np.nan_to_num(
            lake_extent["data"], nan=0
        ).sum()
        w.metadata["stat:alakes_extent"] = w.metadata["stat:plakes_extent"] * w.res**2
        w.metadata["stat:falakes_extent"] = (
            w.metadata["stat:alakes_reduced"] / w.metadata["stat:alakes_extent"]
        )
        if print_bool:
            print(
                f"     | {w.name:10s}",
                f"{w.satellite:2s}",
                f"| {w.start:10s}",
                f"| {w.end:10s}",
                f"| {int(w.metadata['stat:alakes_extent'] * 1e-6):8.3f}",
                f"| {int(w.metadata['stat:alakes_reduced'] * 1e-6):8.3f} ({w.metadata['stat:falakes_extent']*100:5.2f}%)",
                f"| {w.metadata['IVS']:6.2f}",
            )
    if update_bool:
        update_metadata_csvs_from_collection(folder, collection)

    return collection


"""----------------------------------------------------------------------------------------
Add Ice Shelf Extent
----------------------------------------------------------------------------------------"""


def get_iceshelf_extents(folder: str, collection: app.sgl_utils.WindowCollection):
    extents = []
    for f in collection.iceshelf_files:
        extent = {}
        extent["name"] = f.split("_")[0]
        extent["satellite"] = f.split("_")[1]
        extent["start"] = f.split("_")[2]
        extent["end"] = f.split("_")[3]
        extent["res"] = app.sgl_utils.RES[extent["satellite"]]
        with rasterio.open(os.path.join(folder, f)) as src:
            extent["data"] = src.read(1)
            extent["meta"] = src.meta
            extent["transform"] = src.transform
        extent["stat:pshelf_extent"] = np.nan_to_num(extent["data"], nan=0).sum()
        extents.append(extent)
    return extents


def add_iceshelf_statistics(
    folder, collection, update_bool: bool = False, print_bool: bool = True
) -> None:
    print("=====> Add ice shelf extent statistics.")
    iceshelf_extents = get_iceshelf_extents(folder, collection)
    lake_extents = get_lake_extents(folder, collection)
    if print_bool:
        print(
            f"     | {'name':13s}",
            f"| {'start':10s}",
            f"| {'end':10s}",
            f"| {'shelf':>10s} {'km2':<9s}",
            f"| {'lakes':>10s} {'km2':<9s}",
            f"| {'IVS':>5s}",
        )

    for w in collection.windows:
        iceshelf_extent = [
            tif
            for tif in iceshelf_extents
            if tif["name"] == w.name and int(tif["res"]) == w.res
        ][0]
        lake_extent = [
            tif
            for tif in lake_extents
            if tif["name"] == w.name and int(tif["res"]) == w.res
        ][0]
        with rasterio.open(os.path.join(folder, w.file_reduced)) as src:
            meta = src.meta
            clouds = src.read(1)
            iceshelf = np.ones(clouds.shape)
            iceshelf[clouds == 1] = np.nan
        with rasterio.open(os.path.join(folder, w.file_rgb)) as tif_rgb:
            image_mask = tif_rgb.read(1) * 0 + 1
        # print(iceshelf.sum(), np.nan_to_num(iceshelf, nan = 0).sum())
        iceshelf_clipped = iceshelf * iceshelf_extent["data"] * image_mask
        visible_lake_extent = iceshelf * lake_extent["data"] * image_mask
        # print(iceshelf_clipped.sum(), np.nan_to_num(iceshelf_clipped, nan = 0).sum())
        w.metadata["stat:pimage"] = np.nan_to_num(image_mask, nan=0).sum()
        w.metadata["stat:aimage"] = w.metadata["stat:pimage"] * w.res**2
        w.metadata["stat:pshelf"] = np.nan_to_num(iceshelf_clipped, nan=0).sum()
        w.metadata["stat:ashelf"] = w.metadata["stat:pshelf"] * w.res**2
        w.metadata["stat:plake_extent_visible"] = np.nan_to_num(
            visible_lake_extent, nan=0
        ).sum()
        w.metadata["stat:alake_extent_visible"] = (
            w.metadata["stat:plake_extent_visible"] * w.res**2
        )
        w.metadata["stat:pshelf_extent"] = iceshelf_extent["stat:pshelf_extent"]
        w.metadata["stat:ashelf_extent"] = w.metadata["stat:pshelf_extent"] * w.res**2
        if w.metadata["stat:pshelf"] == 0:
            w.metadata["stat:fashelf"] = 0
            w.metadata["stat:falake_extent"] = 0
        else:
            w.metadata["stat:fashelf"] = (
                w.metadata["stat:ashelf"] / w.metadata["stat:ashelf_extent"]
            )
            w.metadata["stat:falake_extent"] = (
                w.metadata["stat:alake_extent_visible"]
                / w.metadata["stat:alakes_extent"]
            )
        if print_bool:
            print(
                f"     | {w.name:10s}",
                f"{w.satellite:2s}",
                f"| {w.start:10s}",
                f"| {w.end:10s}",
                f"| {w.metadata['stat:ashelf'] * 1e-6:7.3f} ({w.metadata['stat:fashelf']*100:6.2f}%)",
                f"| {w.metadata['stat:alake_extent_visible'] * 1e-6:7.3f} ({w.metadata['stat:falake_extent']*100:6.2f}%)",
            )
    if update_bool:
        update_metadata_csvs_from_collection(folder, collection)


"""----------------------------------------------------------------------------------------
        ROIs
    ----------------------------------------------------------------------------------------"""


def create_roi_csv(region_name: str, folder: str, file_shp: list[str]) -> None:
    print(f"=====> Create ROI csv in {folder}")
    input_shapes = []
    relevant_files = [
        f
        for f in file_shp
        if region_name in f.split("_")[0] and len(f.split("_")[0]) > len(region_name)
    ]
    for f in relevant_files:
        try:
            shape = {}
            shape["region"] = f.split("_")[0].split("-")[0]
            shape["name"] = f.split("_")[0].split("-")[1]
            shape["file"] = f
            input_shapes.append(shape)
        except Exception as e:
            print(e)
    d = pd.DataFrame(input_shapes)
    d.to_csv(os.path.join(folder, "roi.csv"), index=False)
    print(f"     | -> {d.shape[0]} ROIs found")


def create_roicollection_from_csv(
    settings: ProcessorSGLSettings, name: str, filestr: str = "roi.csv"
) -> app.sgl_utils.ROICollection:
    print(f"=====> Get ROICollection from csv {filestr}")
    data = pd.read_csv(os.path.join(settings.region_folder, filestr), index_col=False)
    collection = app.sgl_utils.ROICollection()  # type: ignore
    collection.name = name
    rois = []
    for i in range(len(data)):
        roi = app.sgl_utils.ROI()
        roi.region = data["region"].iloc[i]
        roi.name = data["name"].iloc[i]
        roi.metadata = data.iloc[i].to_dict()
        roi.data = gpd.GeoDataFrame(
            gpd.read_file(
                os.path.join(settings.region_folder, roi.metadata["file"])
            ).to_crs(settings.crs_ant_str)
        )
        rois.append(roi)
    collection.rois = rois
    print(f"     | {len(collection.rois)} regions created.")
    return collection


def add_roi_statistics(
    settings: ProcessorSGLSettings,
    roi_collection: app.sgl_utils.ROICollection,
    window_collection: app.sgl_utils.WindowCollection,
    print_metadata: bool = False,
    update_bool: bool = False,
) -> app.sgl_utils.ROICollection:
    print(f"=====> Add roi statistics for {roi_collection.name}")
    for roi in roi_collection.rois:
        roi.metadata["tiles"] = []

    for name in window_collection.names:
        f = name + "_shape.shp"
        tile = gpd.GeoDataFrame(
            gpd.read_file(os.path.join(settings.region_folder, f)).to_crs(
                settings.crs_ant_str
            )
        )["geometry"]

        for roi in roi_collection.rois:
            reg = roi.data["geometry"]
            if reg.intersects(tile).all():
                roi.metadata["tiles"].append(name)
    if update_bool:
        update_roi_csv_from_collection(settings.region_folder, roi_collection)
    return roi_collection


def update_roi_csv_from_collection(
    folder, roi_collection: app.sgl_utils.ROICollection, filestr: str = "roi.csv"
):
    print(f"=====> Update roi.csv for {roi_collection.name}")
    data = pd.DataFrame([roi.metadata for roi in roi_collection.rois])
    data.to_csv(os.path.join(folder, filestr), index=False)


"""----------------------------------------------------------------------------------------
        Vectors
    ----------------------------------------------------------------------------------------"""


def get_vectorized_dmgs(
    folder: str,
    collection: app.sgl_utils.WindowCollection,
    min_dmg_threshold: float = 0.0,
    max_dmg_threshold: float = 0.5,
    print_bool: bool = False,
    category_bins: int = 0,
) -> List[Dict[str, Union[str, int, gpd.GeoDataFrame]]]:
    print(f"=====> Get vectorized dmg.")
    dmgs = []
    for f in sorted([f for f in collection.dmg_files]):
        dmg_img = {}
        dmg_img["name"] = f.split("_")[0]
        dmg_img["year"] = int(f.split("_")[2][0:4])
        dmg_img["file"] = f
        with rasterio.open(os.path.join(folder, f)) as src:
            dmg = src.read(1)
            dmg_meta = src.meta
            dmg_transform = src.transform
        with rasterio.open(os.path.join(folder, f)) as src:
            dmg = src.read(1)
            dmg_meta = src.meta
            dmg_transform = src.transform

        mask = dmg.copy()
        if category_bins:
            bins = np.linspace(0, 0.5, category_bins + 1)
            dmg = np.digitize(dmg, bins)

        mask[(mask >= min_dmg_threshold) & (mask <= max_dmg_threshold)] = 1
        mask[(mask < min_dmg_threshold) & (mask > max_dmg_threshold)] = np.nan

        dmg_img["data"] = app.sgl_utils.vectorize(
            np.nan_to_num(dmg, nan=0),
            transform=dmg_transform,
            crs=dmg_meta["crs"],
            mask=mask.astype(np.uint8),
        )
        dmg_img["data"].rename(columns={"attribute": "dmg"}, inplace=True)
        dmg_img["data"].reset_index(drop=True, inplace=True)
        dmgs.append(dmg_img)
        if print_bool:
            print(f"     | {len(dmg_img['data']):6d} | {f}")
    return dmgs


def get_vectorized_lakeextents(
    folder: str, collection: app.sgl_utils.WindowCollection, print_bool: bool = False
) -> List[Dict[str, Union[str, int, gpd.GeoDataFrame]]]:
    print(f"=====> Get vectorized max lake extents.")
    lakeextents = []
    for f in sorted(
        [f for f in collection.lake_extent_combined_files if f.split("_")[4] == "10m"]
    ):
        lakeextent = {}
        lakeextent["name"] = f.split("_")[0]
        lakeextent["year"] = int(f.split("_")[2][0:4])
        lakeextent["file"] = f
        with rasterio.open(os.path.join(folder, f)) as src:
            lakes = src.read(1)
            lakes_meta = src.meta
            lakes_transform = src.transform

        lakes_segmented, num = measure.label(label_image=np.nan_to_num(lakes, 0), background=0, return_num=True, connectivity=2)  # type: ignore
        lakes_segmented = lakes_segmented.astype(float)
        lakes_segmented[lakes_segmented == 0] = np.nan
        lakeextent["data"] = app.sgl_utils.vectorize(
            lakes_segmented,
            transform=lakes_transform,
            crs=lakes_meta["crs"],
            mask=lakes.astype(np.uint8),
        )
        lakeextent["data"].rename(columns={"attribute": "lake id"}, inplace=True)
        lakeextent["data"].dissolve(by="lake id")
        lakeextent["data"]["name"] = lakeextent["name"]
        lakeextent["data"]["criteria"] = lakeextent["data"]["name"] + lakeextent[
            "data"
        ]["lake id"].astype(int).astype(str)
        lakeextents.append(lakeextent)
        if print_bool:
            print(f"     | {len(lakeextent['data']):6d} | {f}")
    return lakeextents


def get_vectorized_crevasse(
    settings: ProcessorSGLSettings,
    collection: app.sgl_utils.WindowCollection,
    target: float = 45,
    spread: float = 15,
    print_bool: bool = False,
) -> List[Dict[str, Union[str, int, gpd.GeoDataFrame]]]:
    print(f"=====> Get vectorized active crevasses for {target}° ({spread}°).")
    crevasses = []
    for f in sorted([f for f in collection.delta_alpha_c_files]):
        crevasse = {}
        crevasse["name"] = f.split("_")[0]
        crevasse["file"] = f
        with rasterio.open(os.path.join(settings.region_folder, f)) as src:
            delta_alpha_c = src.read(1)
            delta_alpha_c_meta = src.meta
            delta_alpha_c_transform = src.transform
        f_dmg = [
            f
            for f in collection.dmg_files
            if f.split("_")[0] == crevasse["name"]
            and f.split("_")[2][:4] == collection.rangestart[:4]
        ][0]

        with rasterio.open(os.path.join(settings.dmg_folder, f_dmg)) as src:
            dmg = src.read(1)
        dmg_mask = np.zeros(dmg.shape)
        dmg_mask[dmg > 0] = 1
        dmg_mask[np.nan_to_num(dmg, nan=0) <= 0] = np.nan
        d_alpha_masked = delta_alpha_c * dmg_mask
        active_mask = app.sgl_utils.create_active_crevasses_mask(
            d_alpha_masked, target, spread
        )
        active_mask[active_mask == 0] = np.nan
        inactive_mask = (
            np.ones(active_mask.shape) - np.nan_to_num(active_mask, nan=0)
        ) * dmg_mask
        inactive_mask[inactive_mask == 0] = np.nan
        crevasse["data active"] = app.sgl_utils.vectorize(
            d_alpha_masked,
            transform=delta_alpha_c_transform,
            crs=delta_alpha_c_meta["crs"],
            mask=active_mask.astype(np.uint8),
        )
        crevasse["data active"].rename(columns={"attribute": "angle"}, inplace=True)
        crevasse["data inactive"] = app.sgl_utils.vectorize(
            d_alpha_masked,
            transform=delta_alpha_c_transform,
            crs=delta_alpha_c_meta["crs"],
            mask=inactive_mask.astype(np.uint8),
        )
        crevasse["data inactive"].rename(columns={"attribute": "angle"}, inplace=True)
        crevasses.append(crevasse)
        if print_bool:
            print(
                f"     | {len(crevasse['data active']):6d} | {len(crevasse['data inactive']):6d} | {f}"
            )
    return crevasses


"""----------------------------------------------------------------------------------------
    Drainages
----------------------------------------------------------------------------------------"""


def export_drainages(
    settings: ProcessorSGLSettings,
    collection: app.sgl_utils.WindowCollection,
    dominant_satellite="L8",
    print_metadata: bool = False,
    min_lake_area: float = 0.054 * 1e6,
    min_ivs: float = 0.25,
    max_windows: int = 100,
):
    print(
        f"=====> Export Drainages with minimum lake area of {min_lake_area*1e-6:.3f} km^2."
    )
    for iname, name in enumerate(collection.names):
        ws = [
            w
            for w in collection.windows
            if w.name == name and w.metadata["stat:falake_extent"] >= min_ivs
        ]
        dates = sorted([w.metadata["date"] for w in ws])

        """dominant window"""
        ws_dom = [w for w in ws if w.satellite == dominant_satellite]
        with rasterio.open(
            os.path.join(settings.season_folder, ws_dom[0].file_reduced)
        ) as src:
            dom_raster = src.read()
            dom_transform = src.transform
            dom_meta = src.meta
        dom_res = ws_dom[0].res

        """start window 0"""
        ws0 = [w for w in ws if w.metadata["date"] == dates[0]]
        w0 = ws0[0]
        with rasterio.open(
            os.path.join(settings.season_folder, w0.file_reduced)
        ) as src:
            tif0 = src.read()
            tif0_transform = src.transform

        if w0.satellite != dominant_satellite:
            tif0 = app.sgl_utils.resample_dataset(
                tif0,
                tif0_transform,
                dom_meta["width"],
                dom_meta["height"],
                mode="nearest",
            )

        if print_metadata:
            print(
                f"     | Dominant window established: {ws_dom[0].satellite} with {dom_raster.shape} in {ws_dom[0].res}m resolution"
            )
            print(
                f"     | Initial window established: {w0.satellite} with {tif0.shape} in {ws_dom[0].res}m resolution"
            )

        """Going over all windows and collect drains"""
        drains = []
        for idate, date in enumerate(dates):
            if idate >= len(dates) - 1:
                break

            ws1 = [w for w in ws if w.metadata["date"] == dates[idate + 1]]
            if len(ws1) != 1:
                continue
            w1 = ws1[0]

            if not w1.satellite == w0.satellite:
                ws1same = [
                    w
                    for w in ws
                    if w.metadata["date"] >= dates[idate + 1]
                    and w.satellite == w0.satellite
                ]
                if len(ws1same) > 0:
                    w1same = ws1same[0]
                else:
                    w1same = False
            else:
                w1same = False

            lakes0 = tif0[1].copy()

            """Comparison to Window 1"""
            with rasterio.open(
                os.path.join(settings.season_folder, w1.file_reduced)
            ) as src:
                tif1 = src.read()
                tif1_transform = src.transform
                tif1_meta = src.meta

            if w1.satellite != dominant_satellite:
                tif1 = app.sgl_utils.resample_dataset(
                    tif1,
                    tif1_transform,
                    dom_meta["width"],
                    dom_meta["height"],
                    mode="nearest",
                )

            with rasterio.open(
                os.path.join(settings.season_folder, w1.file_rgb)
            ) as src:
                rgb1 = src.read(1)
                rgb1_transform = src.transform
                rgb1_meta = src.meta

            if w1.satellite != dominant_satellite:
                rgb1 = app.sgl_utils.resample_dataset(
                    rgb1,
                    rgb1_transform,
                    dom_meta["width"],
                    dom_meta["height"],
                    mode="bilinear",
                )

            img1 = rgb1.copy() * 0 + 1
            clouds1 = tif1[0].copy()
            noclouds1 = np.ones(clouds1.shape)
            noclouds1[clouds1 == 1] = np.nan
            lakes1 = tif1[1].copy()
            # rplt.show_hist(lakes0 * noclouds1 * img1)
            # rplt.show(noclouds1 * img1 * ice)
            drainage_rpt1 = app.sgl_utils.find_drainages(
                lakes0 * noclouds1 * img1,
                lakes1,
                min_lake_pixel_size=int(min_lake_area / (dom_res**2)),
            )
            drain1 = app.sgl_utils.drainage2vector(
                drainage_rpt1, dom_transform, crs=CRS.from_string(settings.crs_ant_str)
            )
            drain1["run"] = idate
            drain1["name"] = name
            drain1["area"] = drain1["geometry"].area
            drain1["date-0"] = date
            drain1["sat-0"] = w0.satellite
            drain1["start-0"] = app.sgl_utils.date_with_comma(w0.start)
            drain1["end-0"] = app.sgl_utils.date_with_comma(w0.end)
            drain1["date-1"] = w1.metadata["date"]
            drain1["sat-1"] = w1.satellite
            drain1["start-1"] = app.sgl_utils.date_with_comma(w1.start)
            drain1["end-1"] = app.sgl_utils.date_with_comma(w1.end)
            drains.append(drain1)

            print(
                f"     | {name}",
                f"| {w0.metadata['stat:falake_extent']*100:>6.2f}",
                f"| {w0.satellite:2s} | {date}",
                f"| {w1.satellite:2s} | {w1.metadata['date']}",
                f"| {len(drainage_rpt1['missing_lake_ids']):3d} + {len(drainage_rpt1['shrinking_lake_ids']):3d}",
            )

            """Comparison to the next window of same satellite."""
            if w1same:
                """Comparison to Window same next window"""
                with rasterio.open(
                    os.path.join(settings.season_folder, w1same.file_reduced)
                ) as src:
                    tif1same = src.read()
                    tif1same_transform = src.transform
                    tif1same_meta = src.meta

                if w1same.satellite != dominant_satellite:
                    tif1same = app.sgl_utils.resample_dataset(
                        tif1same,
                        tif1same_transform,
                        dom_meta["width"],
                        dom_meta["height"],
                        mode="nearest",
                    )

                with rasterio.open(
                    os.path.join(settings.season_folder, w1same.file_rgb)
                ) as src:
                    rgb1same = src.read(1)
                    rgb1same_transform = src.transform
                    rgb1same_meta = src.meta

                if w1same.satellite != dominant_satellite:
                    rgb1same = app.sgl_utils.resample_dataset(
                        rgb1same,
                        rgb1same_transform,
                        dom_meta["width"],
                        dom_meta["height"],
                        mode="bilinear",
                    )

                img1same = rgb1same.copy() * 0 + 1
                clouds1same = tif1same[0].copy()
                noclouds1same = np.ones(clouds1same.shape)
                noclouds1same[clouds1same == 1] = np.nan
                lakes1same = tif1same[1].copy()
                # rplt.show_hist(lakes0 * noclouds1same * img1same)
                # rplt.show(noclouds1same * img1same)
                drainage_rpt1same = app.sgl_utils.find_drainages(
                    lakes0 * noclouds1same * img1same,
                    lakes1same,
                    min_lake_pixel_size=int(min_lake_area / (dom_res**2)),
                )
                drain1same = app.sgl_utils.drainage2vector(
                    drainage_rpt1same,
                    dom_transform,
                    crs=CRS.from_string(settings.crs_ant_str),
                )
                drain1same["run"] = idate
                drain1same["name"] = name
                drain1same["area"] = drain1same["geometry"].area
                drain1same["date-0"] = date
                drain1same["sat-0"] = w0.satellite
                drain1same["start-0"] = app.sgl_utils.date_with_comma(w0.start)
                drain1same["end-0"] = app.sgl_utils.date_with_comma(w0.end)
                drain1same["date-1"] = w1same.metadata["date"]
                drain1same["sat-1"] = w1same.satellite
                drain1same["start-1"] = app.sgl_utils.date_with_comma(w1same.start)
                drain1same["end-1"] = app.sgl_utils.date_with_comma(w1same.end)
                drains.append(drain1same)
                print(
                    f"     | {name}",
                    f"| {w0.metadata['stat:falake_extent']*100:>6.2f}",
                    f"| {w0.satellite:2s} | {date}",
                    f"| {w1same.satellite:2s} | {w1same.metadata['date']}",
                    f"| {len(drainage_rpt1same['missing_lake_ids']):3d} + {len(drainage_rpt1same['shrinking_lake_ids']):3d}",
                )

            w0 = w1
            tif0 = tif1.copy()
            tif0_transform = tif1_transform
            tif0_meta = tif1_meta

        fname = "_".join(
            [
                name,
                "L8S2",
                app.sgl_utils.date_with_comma(collection.rangestart),
                app.sgl_utils.date_with_comma(collection.rangeend),
                f"{dom_res:d}m",
                "drain.shp",
            ]
        )
        gdf = gpd.GeoDataFrame(pd.concat(drains, ignore_index=True))
        gdf.set_crs(CRS.from_string(settings.crs_ant_str), inplace=True)
        gdf.to_file(os.path.join(settings.season_folder, fname))


def check_cloud_cover(
    settings: ProcessorSGLSettings,
    collection: app.sgl_utils.WindowCollection,
    drain_gdf: gpd.GeoDataFrame,
    print_metadata: bool = False,
    max_fraction_depth_nan: float = 0.6,
    min_median_depth: float = 0.65,
    min_std: float = 0.0,
    buffer: int = 500,
    offset: int = 1000,
):
    print(f"=====> Check L8 / S2 images for cloud cover.")

    lakes_to_remove = []
    for lake_criteria in drain_gdf["criteria"].unique():
        gdf_lake = drain_gdf[drain_gdf["criteria"] == lake_criteria].reset_index()
        w0 = [
            w
            for w in collection.windows
            if w.name == gdf_lake["name"].iloc[0]
            and w.metadata["date"] == gdf_lake["date-0"].iloc[0]
        ][0]
        w1 = [
            w
            for w in collection.windows
            if w.name == gdf_lake["name"].iloc[0]
            and w.metadata["date"] == gdf_lake["date-1"].iloc[0]
        ][0]

        lake0 = gdf_lake[gdf_lake["window"] == 0].explode(ignore_index=False, index_parts=False)  # type: ignore
        minx = min(lake0.bounds.minx) - offset
        miny = min(lake0.bounds.miny) - offset
        maxx = max(lake0.bounds.maxx) + offset
        maxy = max(lake0.bounds.maxy) + offset

        with rasterio.open(
            os.path.join(settings.season_folder, w0.file_reduced)
        ) as src:
            tif0 = src.read(
                window=rw.from_bounds(minx, miny, maxx, maxy, src.transform)
            )
            tif0_transform = Affine(
                src.transform[0], 0, minx, 0, src.transform[4], maxy
            )
        with rasterio.open(
            os.path.join(settings.season_folder, w1.file_reduced)
        ) as src:
            tif1 = src.read(
                window=rw.from_bounds(minx, miny, maxx, maxy, src.transform)
            )
            tif1_transform = Affine(
                src.transform[0], 0, minx, 0, src.transform[4], maxy
            )

        """Check if depth not nan"""
        tif0_clipped, tif0_clipped_transform = app.sgl_utils.clip_dataset(
            tif0, tif0_transform, coords=[geom for geom in lake0["geometry"]]
        )

        number_notclipped = tif0_clipped[2][tif0_clipped[2] == 0].size
        number_depth = tif0_clipped[2][tif0_clipped[2] > 0].size
        number_nodepth = tif0_clipped[2].size - number_notclipped - number_depth
        fraction_depth = 1 - number_nodepth / (tif0_clipped[2].size - number_notclipped)
        if fraction_depth < max_fraction_depth_nan:
            lakes_to_remove.append({"criteria": lake_criteria, "reason": "nodepth"})
            continue

        """Check if median depth is bigger"""
        depth0 = tif0_clipped[2]
        depth0[depth0 == 0] = np.nan
        if np.nanmedian(depth0) <= min_median_depth:
            lakes_to_remove.append({"criteria": lake_criteria, "reason": "shallow"})
            continue

        """Check if std depth is bigger"""
        depth0 = tif0_clipped[2]
        depth0[depth0 == 0] = np.nan
        if np.nanstd(depth0) <= min_std:
            lakes_to_remove.append({"criteria": lake_criteria, "reason": "std"})
            continue

        clouds0 = app.sgl_utils.vectorize(
            tif0[0],
            transform=tif0_transform,
            crs=lake0.crs,
            mask=tif0[0].astype(np.uint8),
        )
        clouds1 = app.sgl_utils.vectorize(
            tif1[0],
            transform=tif1_transform,
            crs=lake0.crs,
            mask=tif1[0].astype(np.uint8),
        )

        if clouds0.shape[0] != 0:
            clouds1_buffered = gpd.GeoSeries(
                [clouds1.geometry.buffer(buffer).unary_union]
            )
            clouds1_buffered_gdf = gpd.GeoDataFrame(geometry=clouds1_buffered, crs=clouds1.crs)  # type: ignore
            bool0 = lake0.dissolve().intersects(clouds1_buffered_gdf)
            if any(bool0):
                lakes_to_remove.append(
                    {"criteria": lake_criteria, "reason": "clouds-0"}
                )
                continue

        if clouds1.shape[0] != 0:
            clouds1_buffered = gpd.GeoSeries(
                [clouds1.geometry.buffer(buffer).unary_union]
            )
            clouds1_buffered_gdf = gpd.GeoDataFrame(geometry=clouds1_buffered, crs=clouds1.crs)  # type: ignore
            bool1 = lake0.dissolve().intersects(clouds1_buffered_gdf)
            if any(bool1):
                lakes_to_remove.append(
                    {"criteria": lake_criteria, "reason": "clouds-1"}
                )
                continue

    return lakes_to_remove


def combine_filter_drainages(
    settings: ProcessorSGLSettings,
    collection: app.sgl_utils.WindowCollection,
    filter_bool: bool = True,
    print_metadata: bool = False,
    return_bool: bool = False,
    max_days: int = 10,
    attributes: List[str] = ["criteria", "window"],
    min_lake_area: float = 0.054 * 1e6,
    freezing_months: List[int] = [3, 11],
    max_fraction_depth_nan: float = 0.6,
    min_median_depth: float = 0.65,
    min_std: float = 0.25,
):
    print(f"=====> Combine and filter drainages.")
    gdfs = []
    for name in collection.names:
        fs = [f for f in collection.drain_shp_files if name in f]
        if len(fs) != 1:
            print(f"     | {len(fs)} files found!")
        f = fs[0]

        """Load"""
        gdf = gpd.read_file(os.path.join(settings.season_folder, f))
        gdf["criteria"] = pd.Series(
            gdf["name"]
            + gdf["date-0"]
            + gdf["lake id"].astype(int).astype(str)
            + gdf["date-1"]
            + gdf["sat-0"]
            + gdf["sat-1"]
            + gdf["type"]
        )

        """Dissolve"""
        gdf_diss = gdf.dissolve(by=attributes, as_index=False)

        """Filter days"""
        gdf_diss["day diff"] = pd.Series(pd.to_datetime(gdf_diss["date-1"]) - pd.to_datetime(gdf_diss["date-0"])).dt.days  # type: ignore
        gdf_filtered_daydiff = gdf_diss[gdf_diss["day diff"] <= max_days].copy()

        """Within Melting Period"""
        gdf_melting = gdf_filtered_daydiff[
            ~(
                gdf_filtered_daydiff["date-0"]
                .str[5:7]
                .astype(int)
                .between(freezing_months[0], freezing_months[1])
            )
            & ~(
                gdf_filtered_daydiff["date-1"]
                .str[5:7]
                .astype(int)
                .between(freezing_months[0], freezing_months[1])
            )
        ]

        """Satellite"""
        gdf_sat_filtered = gpd.GeoDataFrame(
            gdf_melting[
                (gdf_melting["sat-0"] == gdf_melting["sat-1"])
                | (
                    (gdf_melting["sat-0"] != gdf_melting["sat-1"]) & gdf_melting["type"]
                    == "shrink"
                )
            ]
        )  # = gpd.GeoDataFrame(pd.concat([gdf_same_sat, gdf_notsame_nodrain]))
        gdf_sat_filtered.set_crs(gdf_melting.crs, inplace=True)

        """Area"""
        gdf_diss0 = gdf_sat_filtered[gdf_sat_filtered["window"] == 0].copy()
        gdf_diss1 = gdf_sat_filtered[gdf_sat_filtered["window"] == 1].copy()
        gdf_diss0["area-0"] = gdf_diss0["geometry"].area
        gdf_area0 = gdf_diss0[gdf_diss0["area-0"] >= min_lake_area]
        gdf_area1 = gdf_diss1[
            np.isin(gdf_diss1["criteria"], gdf_area0["criteria"].unique())
        ]
        gdf_area0.drop(columns=["area-0"], inplace=True)

        gdf_area = gpd.GeoDataFrame(
            pd.concat([gdf_area0, gdf_area1], ignore_index=True)
        )
        gdf_area.set_crs(gdf_sat_filtered.crs)

        if print_metadata:
            print(f"     | {name}")
            print(f"     | {gdf.shape[0]:>5d} -> raw")
            print(
                f"     | {gdf_filtered_daydiff.shape[0]:>5d} -> filtered by time difference of max {max_days} days"
            )
            print(
                f"     | {len(gdf_melting['criteria'].unique()):>5d} -> within melting period"
            )
            print(
                f"     | {len(gdf_sat_filtered['criteria'].unique()):>5d} -> valid satellite combination"
            )
            print(
                f"     | {len(gdf_area['criteria'].unique()):>5d} -> filterd by minimum lake area {min_lake_area*1e-6:.3f} km2"
            )

        gdfs.append(gdf_area)

    all_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    all_gdf.set_crs(gdfs[0].crs, inplace=True)
    print("     |")
    print(f"     | {len(all_gdf['criteria'].unique()):>5d} -> combined events")

    if filter_bool:
        """Check Cloudcover"""
        lakes_to_remove = check_cloud_cover(
            settings,
            collection=collection,
            drain_gdf=all_gdf,
            max_fraction_depth_nan=max_fraction_depth_nan,
            min_median_depth=min_median_depth,
            min_std=min_std,
        )
        reason_nodepth = [l for l in lakes_to_remove if l["reason"] == "nodepth"]
        reason_clouds0 = [l for l in lakes_to_remove if l["reason"] == "clouds-0"]
        reason_clouds1 = [l for l in lakes_to_remove if l["reason"] == "clouds-1"]
        reason_shallow = [l for l in lakes_to_remove if l["reason"] == "shallow"]
        reason_std = [l for l in lakes_to_remove if l["reason"] == "std"]
        print(
            f"     | {len(lakes_to_remove):>5d} -> lakes to remove (nodepth : {len(reason_nodepth)} | shallow : {len(reason_shallow)} | std : {len(reason_std)} | clouds-0 : {len(reason_clouds0)} | clouds-1 : {len(reason_clouds1 )})."
        )

        def filter_lakes(row):
            for lake in lakes_to_remove:
                if row["criteria"] == lake["criteria"]:
                    return False
            return True

        gdf_filtered = all_gdf.copy()
        gdf_filtered["valid"] = all_gdf.apply(filter_lakes, axis=1)
        gdf_filtered_checked = gdf_filtered[gdf_filtered["valid"] == True]
        gdf_filtered_checked.reset_index(inplace=True)
        all_gdf = gdf_filtered_checked.copy()
        print(
            f"     | {len(gdf_filtered_checked['criteria'].unique()):>5d} -> Cloud cover checked."
        )

    fname = "_".join(
        ["L8S2", f"{settings.rangestart}", f"{settings.rangeend}", "drain.shp"]
    )
    all_gdf.to_file(os.path.join(settings.season_folder, fname))
    if return_bool:
        return all_gdf


if __name__ == "__main__":
    pass

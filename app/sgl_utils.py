import os
from dataclasses import dataclass, field
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
import shapely
import shapely.geometry
from matplotlib.colors import LinearSegmentedColormap
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from skimage import (color, data, filters, measure, morphology, segmentation,
                     util)

#from dea import xr_vectorize

RES = {"L8": 30, "S2": 10}

"""----------------------------------------------------------------------------------------
    Windows
----------------------------------------------------------------------------------------"""

@dataclass
class Window:
    name = "???"
    source = "???"
    satellite = "???"
    start = "???"
    end = "???"
    rangestart = "???"
    rangeend = "???"
    metadata = {}
    files = {}
    res = 30

    def __str__(self):
        return f"Window(name = {self.name}, source = {self.source}, satellite = {self.satellite})"
    
    @property
    def file_gee(self):
        return f"{self.name}_{self.satellite}_{date_with_comma(self.start)}_{date_with_comma(self.end)}_{self.res}m.tif"
    @property
    def file_reduced(self):
        return f"{self.name}_{self.satellite}_{date_with_comma(self.start)}_{date_with_comma(self.end)}_{self.res}m_reduced.tif"
    @property
    def file_gee_maxlake(self):
        return f"{self.name}_{self.satellite}_{date_with_comma(self.rangestart)}_{date_with_comma(self.rangeend)}_maxlake.tif"
    
    @property
    def file_dmg(self):
        if self.rangestart[0:4] < self.rangeend[0:4] and int(self.rangestart[4:6]) >= 8:
            year = self.rangestart[0:4]
        elif self.rangestart[0:4] < self.rangeend[0:4] and int(self.rangestart[4:6]) < 8:
            year = str(int(self.rangestart[0:4]) - 1)
        elif self.rangestart[0:4] == self.rangeend[0:4] and int(self.rangestart[4:6]) < 8:
            year = str(int(self.rangestart[0:4]) - 1)
        else:
            year = "2019"
        return f"{self.name.replace('_', '-')}_S1_{year}1101_{year}1110_30m_output_10px_dmg.tif"

    @property
    def file_alpha_c(self):
        if self.rangestart[0:4] < self.rangeend[0:4] and int(self.rangestart[4:6]) >= 8:
            year = self.rangestart[0:4]
        elif self.rangestart[0:4] < self.rangeend[0:4] and int(self.rangestart[4:6]) < 8:
            year = str(int(self.rangestart[0:4]) - 1)
        elif self.rangestart[0:4] == self.rangeend[0:4] and int(self.rangestart[4:6]) < 8:
            year = str(int(self.rangestart[0:4]) - 1)
        else:
            year = "2019"
        return f"{self.name.replace('_', '-')}_S1_{year}1101_{year}1110_30m_output_10px_alphaC.tif"

    @property
    def file_lake_extent(self):
        return f"{self.name}_L8S2_{date_with_comma(self.rangestart)}_{date_with_comma(self.rangeend)}_{self.res}m_maxlake.tif"

    @property
    def file_iceshelf_extent(self):
        return f"{self.name}_{self.satellite}_{date_with_comma(self.rangestart)}_{date_with_comma(self.rangeend)}_iceshelf.tif"
    
    @property
    def file_rgb(self):
        return f"{self.name}_{self.satellite}_{date_with_comma(self.start)}_{date_with_comma(self.end)}_{self.res}m_rgb.tif"
    
    @property
    def file_drain_shp(self):
        return f"{self.name}_L8S2_{date_with_comma(self.rangestart)}_{date_with_comma(self.rangeend)}_30m_drain.shp"
    
    @property
    def file_delta_alpha_c(self):
        if self.rangestart[0:4] < self.rangeend[0:4] and int(self.rangestart[4:6]) >= 8:
            year = self.rangestart[0:4]
        elif self.rangestart[0:4] < self.rangeend[0:4] and int(self.rangestart[4:6]) < 8:
            year = str(int(self.rangestart[0:4]) - 1)
        elif self.rangestart[0:4] == self.rangeend[0:4] and int(self.rangestart[4:6]) < 8:
            year = str(int(self.rangestart[0:4]) - 1)
        else:
            year = "2019"
        return f"{self.name.replace('_', '-')}_S1_{year}1101_{year}1110_delta-alpha.tif"

    
    @property
    def file_vx(self):
        return f"{self.name}_vx.tif"
    
    @property
    def file_vy(self):
        return f"{self.name}_vy.tif"
    
    @property
    def file_emax(self):
        return f"{self.name}_emax.tif"
    
    @property
    def file_emin(self):
        return f"{self.name}_emin.tif"

@dataclass
class WindowCollection:
    name: str = "???"
    rangestart: str = "???"
    rangeend: str = "???"
    windows: List[Window] = field(default_factory=list)

    def __str__(self):
        return f"WindowCollection(name = {self.name}, rangestart = {self.rangestart}, rangeend = {self.rangeend}, count = {self.count})"

    def add_windows(self, windows: list[Window]) -> None:
        self.windows += windows
    
    @property
    def count(self) -> int:
        return len(self.windows)
    
    @property
    def names(self) -> list[str]:
        names = list(set([str(w.name) for w in self.windows]))
        return sorted(names)
    
    @property
    def satellites(self) -> list[str]:
        satellites = list(set([str(w.satellite) for w in self.windows]))
        return sorted(satellites)
    
    @property
    def metadata(self) -> pd.DataFrame:
        return pd.DataFrame([w.metadata for w in self.windows])

    @property
    def dates(self) -> list[str]:
        dates = sorted(self.metadata["date"].unique())
        return dates

    @property
    def iceshelf_files(self) -> list[str]:
        iceshelf_files = list(set([str(w.file_iceshelf_extent) for w in self.windows]))
        return iceshelf_files

    @property
    def max_lake_files(self) -> list[str]:
        max_lake_files = list(set([str(w.file_gee_maxlake) for w in self.windows]))
        return max_lake_files
    
    @property
    def drain_shp_files(self) -> list[str]:
        drain_shp_files = list(set([str(w.file_drain_shp) for w in self.windows]))
        return drain_shp_files
    
    @property
    def lake_extent_combined_files(self) -> list[str]:
        extent_files = list(set([str(w.file_lake_extent) for w in self.windows]))
        return extent_files
    
    @property
    def dmg_files(self) -> list[str]:
        dmg_files = list(set([str(w.file_dmg) for w in self.windows]))
        return dmg_files
    
    def get_filtered_windows(self, names: list[str] = [], satellites: list[str] = [], dates: list[str] = []) -> list[Window]:
        if names == []: names = self.names
        if satellites == []: satellites = self.satellites
        if dates == []: dates = self.dates
        return [w for w in self.windows if w.name in names and w.satellite in satellites and w.metadata["date"] in dates]
    
    
    @property
    def delta_alpha_c_files(self) -> List[str]:
        delta_alpha_c_files = list(set([str(w.file_delta_alpha_c) for w in self.windows]))
        return delta_alpha_c_files
    
    @property
    def emax_files(self) -> List[str]:
        emax_files = list(set([str(w.file_emax) for w in self.windows]))
        return emax_files
    
    @property
    def vx_files(self) -> List[str]:
        vx_files = list(set([str(w.file_vx) for w in self.windows]))
        return vx_files
    
    @property
    def vy_files(self) -> List[str]:
        vy_files = list(set([str(w.file_vy) for w in self.windows]))
        return vy_files
        
    @property
    def total_iceshelf_extent(self) -> int:
        total_extent = 0
        for i, name in enumerate(sorted(self.names)):
            tile_df = self.metadata[self.metadata["name"] == name]
            total_extent += tile_df["stat:ashelf_extent"].mean()
        return int(total_extent)

    @classmethod
    def from_settings(cls, settings):
        return cls(name = settings.region.lower() + settings.rangestart[2:4] + settings.rangeend[2:4],
                    rangestart = settings.rangestart,
                    rangeend = settings.rangeend)

    @classmethod
    def from_settings_windows(cls, settings, windows):
        return cls(name = settings.region.lower() + settings.rangestart[2:4] + settings.rangeend[2:4],
                    rangestart = settings.rangestart,
                    rangeend = settings.rangeend,
                    windows = windows)


def date_with_comma(datestr):
    return "-".join([datestr[0:4], datestr[4:6], datestr[6:]])

def get_angle(gdf: gpd.GeoDataFrame) -> float:
    p1 = gdf["geometry"].exterior[0].coords[1]
    p2 = gdf["geometry"].exterior[0].coords[2]
    return np.arctan(abs(p1[0] - p2[0]) / abs(p1[1] - p2[1])) * 180 / np.pi

def resample_dataset(raster, transform, target_width, target_height, mode: str = "bilinear"):
    if raster.ndim == 2:
        count = 1
    elif raster.ndim > 3:
        raise ValueError("To High dimension of raster")
    else:
        count = raster.shape[0]
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
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

        with memfile.open() as src:
            resampled = src.read(
                out_shape=(
                    count,
                    int(target_height),
                    int(target_width)
                ),
                resampling=resamp,
            )
    return resampled

def window_dataset(raster, transform, left, bottom, right, top):
    if raster.ndim == 2:
        count = 1
    elif raster.ndim > 3:
        raise ValueError("To High dimension of raster")
    else:
        count = raster.shape[0]
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
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
                    
        with memfile.open() as src:
            win = src.read(window = rasterio.windows.from_bounds(left, bottom ,right, top, src.transform))
            
            #win_transform = rw.transform(win, src.transform)
    return win#, win_transform

def create_dataset(raster, transform):
    if raster.ndim == 2:
        count = 1
    elif raster.ndim > 3:
        raise ValueError("To High dimension of raster")
    else:
        count = raster.shape[0]
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
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

            src = dataset        
    return src

def clip_dataset(raster, transform, coords):
    if raster.ndim == 2:
        count = 1
    else:
        count = raster.shape[0]
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=raster.shape[-2],
            width=raster.shape[-1],
            count=count,
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            if raster.ndim == 2:
                    dataset.write(raster, 1)
            else:
                for i in range(count):
                    #print(raster[i])
                    dataset.write(raster[i], i+1)
                    #print(dataset.meta)
                    
        with memfile.open() as src:
            out_img, out_transform = mask(dataset = src, shapes=coords, crop=True)
            
            #win_transform = rw.transform(win, src.transform)
    return out_img, out_transform

def vectorize(da, 
            transform, 
            crs,
            attribute_col='attribute',
            dtype='float32',
            **rasterio_kwargs):    


    vectors = rasterio.features.shapes(source=da.astype(dtype),
                                        transform=transform,
                                        **rasterio_kwargs)

    vectors = list(vectors)
    coords = [p for p,v in vectors]
    values = [v for p,v  in vectors]

    polygons = [shapely.geometry.shape(p) for p in coords]
    
    gdf = gpd.GeoDataFrame(data={attribute_col: values},
                           geometry=polygons,
                           crs=str(crs))  # type: ignore  
    return gdf


def find_drainages(w0: np.ndarray, w1: np.ndarray, min_shrink = 0.8, min_lake_pixel_size: int = 0):
    drain_rpt = {}
    drain_rpt["seg0"] = np.array(measure.label(label_image = np.nan_to_num(w0, nan = 0), background = 0, connectivity = 2))  # type: ignore
    counts0 = np.unique(drain_rpt["seg0"], return_counts = True) # type: ignore
    drain_rpt["counted_ids0"] = counts0[0][counts0[1] >= min_lake_pixel_size]
    drain_rpt["pixel_sizes0"] = counts0[1][counts0[1] >= min_lake_pixel_size]
    dropped_ids = counts0[0][counts0[1] < min_lake_pixel_size]
    
    drain_rpt["seg1"] = drain_rpt["seg0"] * np.nan_to_num(w1, nan = 0).astype(int)
    drain_rpt["seg1"][np.isin(drain_rpt["seg1"], dropped_ids)] = 0
    counts1 = np.unique(drain_rpt["seg1"], return_counts = True) # type: ignore

    drain_rpt["counted_ids1"] = counts1[0]
    drain_rpt["pixel_sizes1"] = counts1[1]
    
        
    """Missing"""
    missing_lake_bools = np.isin(drain_rpt["counted_ids0"], drain_rpt["counted_ids1"], invert = True)
    drain_rpt["missing_lake_ids"] = drain_rpt["counted_ids0"][missing_lake_bools]
    drain_rpt["missing_lake_sizes"] = drain_rpt["pixel_sizes0"][missing_lake_bools]
    raster_drain_bool = np.isin(drain_rpt["seg0"], drain_rpt["missing_lake_ids"]) # type: ignore
    drain_rpt["drain"] = np.zeros(drain_rpt["seg0"].shape) # type: ignore
    drain_rpt["drain"][raster_drain_bool] = 1
    drain_rpt["drain"][np.invert(raster_drain_bool)] = 0

    """Shrinking"""
    remaining_lake_ids = drain_rpt["counted_ids0"][np.invert(missing_lake_bools)]
    #print(remaining_lake_ids)
    shrink_perc =  -1 * (drain_rpt["pixel_sizes1"] - drain_rpt["pixel_sizes0"][np.invert(missing_lake_bools)]) / drain_rpt["pixel_sizes0"][np.invert(missing_lake_bools)]
    #shrink_perc = perc[perc < 0]
    drain_rpt["shrinking_lake_ids"] = remaining_lake_ids[shrink_perc > min_shrink]
    drain_rpt["shrinking_lake_sizes"] = drain_rpt["pixel_sizes0"][np.invert(missing_lake_bools)][shrink_perc > min_shrink]
    drain_rpt["shrinking_percs"] = shrink_perc[shrink_perc > min_shrink]

    raster_shrink_bool = np.isin(drain_rpt["seg0"], drain_rpt["shrinking_lake_ids"]) # type: ignore
    drain_rpt["shrink0"] = np.zeros(drain_rpt["seg0"].shape) # type: ignore
    drain_rpt["shrink0"][raster_shrink_bool] = 1
    drain_rpt["shrink0"][np.invert(raster_shrink_bool)] = 0
    drain_rpt["shrink1"] = np.zeros(drain_rpt["seg1"].shape) # type: ignore
    drain_rpt["shrink1"][raster_shrink_bool] = 1
    drain_rpt["shrink1"][np.invert(raster_shrink_bool)] = 0

    drain_rpt["fracture"] = drain_rpt["drain"] + drain_rpt["shrink0"]
    drain_rpt["fracture"][drain_rpt["drain"] > 0] = 1
    return drain_rpt

def drainage2vector(drainage_rpt, transform, crs):
    drain = {}
    mask_drain = drainage_rpt["drain"].copy()
    mask_drain[mask_drain == 0] = np.nan
    mask_shrink0 = drainage_rpt["shrink0"].copy()
    mask_shrink0[mask_shrink0 == 0] = np.nan
    mask_shrink1 = drainage_rpt["shrink1"].copy()
    mask_shrink1[mask_shrink1 == 0] = np.nan
    drain["drain"] = vectorize(drainage_rpt["seg0"], transform = transform, crs = crs, mask = mask_drain.astype(np.uint8))
    drain["shrink0"] = vectorize(drainage_rpt["seg0"], transform = transform, crs = crs, mask = mask_shrink0.astype(np.uint8))
    drain["shrink1"] = vectorize(drainage_rpt["seg1"], transform = transform, crs = crs, mask = mask_shrink1.astype(np.uint8))
    
    """Setting Window"""
    drain["drain"].rename(columns= {"attribute" : "lake id"}, inplace = True)
    drain["shrink0"].rename(columns= {"attribute" : "lake id"}, inplace = True)
    drain["shrink1"].rename(columns= {"attribute" : "lake id"}, inplace = True)

    """Setting Window"""
    drain["drain"]["window"] = 0
    drain["shrink0"]["window"] = 0
    drain["shrink1"]["window"] = 1

    """Setting Type"""
    drain["drain"]["type"] = "drain"
    drain["shrink0"]["type"] = "shrink"
    drain["shrink1"]["type"] = "shrink"

    gdfs = [drain["drain"], drain["shrink0"], drain["shrink1"]]
    drain["drain"].crs
    #pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    gdf.set_crs(drain["drain"].crs, inplace = True)
    gdf = gdf[gdf["lake id"] != 0.0]
    return gdf

def create_active_crevasses_mask(d_alpha, target, spread):
    active_mask = np.zeros(d_alpha.shape)
    active_mask[(d_alpha >= target - spread) & (d_alpha <= target + spread)] = d_alpha[(d_alpha >= target - spread) & (d_alpha <= target + spread)]
    active_mask[(d_alpha <= -target + spread) & (d_alpha >= -target - spread)] = d_alpha[(d_alpha <= -target + spread) & (d_alpha >= -target - spread)]
    active_mask[(d_alpha >= 180 - target - spread) & (d_alpha <= 180 - target + spread)] = d_alpha[(d_alpha >= 180 - target - spread) & (d_alpha <= 180 - target + spread)]
    active_mask[(d_alpha <= - 180 + target + spread) & (d_alpha >=  -180 + target - spread)] = d_alpha[(d_alpha <= - 180 + target + spread) & (d_alpha >=  -180 + target - spread)]
    active_mask[active_mask == 0] = np.nan
    active_mask[(active_mask > 0) | (active_mask < 0)] = 1

    active_crevasses = d_alpha * active_mask
    high = np.zeros(active_crevasses.shape)
    high[abs(active_crevasses) >= 180 - target - spread] = abs(active_crevasses[abs(active_crevasses) >= 180 - target - spread]) - 180 + target
    low = np.zeros(active_crevasses.shape)
    low[abs(active_crevasses) <= target + spread] = abs(active_crevasses[abs(active_crevasses) < target + spread]) - target
    return abs(high[0]) + abs(low[0]) * active_mask

cmap_clouds = LinearSegmentedColormap.from_list('', ['gray', 'gray'])
cmap_ice = LinearSegmentedColormap.from_list('', ['white', 'white'])
cmap_lakes = LinearSegmentedColormap.from_list('', ['deepskyblue', 'deepskyblue'])








"""----------------------------------------------------------------------------------------
    ROI
----------------------------------------------------------------------------------------"""
@dataclass
class ROI():
    name: str = "???"
    region: str = "???"
    metadata = {}
    data: gpd.GeoDataFrame = gpd.GeoDataFrame()
    images = {}

@dataclass
class ROICollection():
    name: str = "???"
    rois = []

    @property
    def shape_files(self):
        return [roi["file"] for roi in self.rois]

    @property
    def metadata(self) -> pd.DataFrame:
        return pd.DataFrame([roi.metadata for roi in self.rois])

    @property
    def names(self) -> list[str]:
        return [roi.name for roi in self.rois]
    
    def get(self, roi_name):
        return [roi for roi in self.rois if roi.name == roi_name][0]






def read_files_from_folder(folder: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    print(f"=====> Read files from folder:")
    files = [f for f in os.listdir(folder)]
    files_tif = [f for f in files if f.split(".")[-1] == "tif"]
    files_csv = [f for f in files if f.split(".")[-1] == "csv"]
    files_shp = [f for f in files if f.split(".")[-1] == "shp"]
    try:
        names = list(set([f.split("_")[0] for f in files_tif]))
    except: names = []
    print(f"     | -> {len(files_tif)} (.tif), {len(files_csv)} (.csv) and {len(files_shp)} (.shp) for {len(names)} names")
    print(f"     | -> {names}")
    return (files_tif, files_csv, files_shp, names)

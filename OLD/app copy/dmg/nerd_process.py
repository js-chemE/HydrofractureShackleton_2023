import app.dmg.nerd as nerd

import os
import time
import numpy as np
import xarray as xr
from pathos.multiprocessing import ProcessingPool as Pool

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

xr.set_options(keep_attrs=True)

def validate_image_resolution(
        img: xr.DataArray,
        expected_res: int):
    dx = np.unique(img['x'].diff(dim='x'))

    if len(dx) == 1:
        if not dx[0] == expected_res:
            logger.error(f"Configured img resolution ({expected_res}) does not match image resolution ({dx[0]})")
            raise ValueError(f"Configured img resolution ({expected_res}) does not match image resolution ({dx[0]})")
    else:
        logger.error(f"Inconsistent grid spacing; dx values are {dx}")
        raise ValueError(f"Inconsistent grid spacing; dx values are {dx}")
    
def setup_output_dirs(
        base_path: str
        ):
    out_path = os.path.join(base_path, 'damage_detection')
    os.makedirs(out_path, exist_ok=True)

    geotiff_path = os.path.join(out_path, 'geotiffs_python')
    os.makedirs(geotiff_path, exist_ok=True)

    return out_path, geotiff_path

def load_and_validate_image(
        image_path: str,
        dbmin: int | None,
        dbmax: int | None,
        img_res: int
        ):
    im_path, im_name = os.path.split(image_path)
    img = nerd.read_img_to_grayscale(im_path, im_name, dbmin, dbmax)
    validate_image_resolution(img, img_res)
    return img, im_path, im_name

def process_windows(
        img: xr.DataArray,
        wsize: int,
        cores: int):
    logger.info(f'Processing img on {wsize}px windows')
    windows_df = nerd.cut_img_to_windows(img, wsize=wsize)
    logger.info(f'--> {wsize}px windows: {windows_df.shape}')
    
    logger.info(f'--> split img windows into {cores} parts for multi-processing')
    windows_split = np.array_split(windows_df, cores, axis=2)

    if cores == 1:
        logger.info("--> using single core for processing")
        df_out = nerd.process_img_windows(windows_df)
    
    elif cores > 1:
        logger.info(f"--> using {cores} cores for processing")
        with Pool(cores) as pool:
            pool_out = pool.map(nerd.process_img_windows, windows_split)
            df_out = np.concatenate(pool_out)
    
    else:
        logger.error(f"Invalid number of cores: {cores}. Must be >= 1.")
        raise ValueError(f"Invalid number of cores: {cores}. Must be >= 1.")

    return df_out, windows_df


def assemble_results(df_out, windows_df, img_attrs, img_res, wsize):
    da_result = xr.DataArray(
        df_out,
        dims=("sample", "out"),
        coords=(windows_df["sample"], range(8)),
        name="output",
        attrs=img_attrs,
    )

    da_result.attrs.update({
        'long_name': 'Output_NeRD',
        'descriptions': '[theta_1,signal_1, theta_2,signal_2, theta_3,signal_3, theta_4,signal_4]',
        'img_res': img_res,
        'window_size(px)': wsize,
        'window_range(m)': wsize * img_res,
        'crs': 'EPSG:3031'
    })
    return da_result.unstack('sample').transpose("y", "x", "out")


def save_netcdf(data_array, out_dir, fname_base):
    output_path = os.path.join(out_dir, f"{fname_base}.nc")
    try:
        data_array.to_netcdf(output_path)
        logger.info(f"--> output data saved to {output_path}")
    except Exception as e:
        logger.error(f"--> failed to save NetCDF: {e}")


def compute_and_save_outputs(da_result, path2threshold, threshold_fname, source, img_res, wsize, geotiff_path, fname_out):
    alpha_c = da_result.isel(out=0) - 90
    crevSig = da_result.isel(out=1)

    dmg, threshold = nerd.crevsig_to_dmg(
        crevSig, os.path.join(path2threshold, threshold_fname),
        source, img_res, wsize
    )

    alpha_c.rio.to_raster(os.path.join(geotiff_path, f"{fname_out}_alphaC.tif"), driver="COG")
    crevSig.rio.to_raster(os.path.join(geotiff_path, f"{fname_out}_crevSig.tif"), driver="COG")

    if threshold is not None:
        dmg.where(dmg > 0).rio.to_raster(os.path.join(geotiff_path, f"{fname_out}_dmg.tif"), driver="COG")

    logger.info(f"geotiffs saved to {geotiff_path}")


def run_nerd(
    image_path : str,
    img_res : str,
    dbmin: int,
    dbmax: int,
    wsize: int,
    cores: int,
    path2threshold: str,
    overwrite: bool = False,
):
    img, im_path, im_name = load_and_validate_image(image_path, dbmin, dbmax, img_res)
    fname_out = f"{im_name[:-4]}_output_{wsize}px"
    out_path, geotiff_path = setup_output_dirs(im_path)

    if os.path.exists(os.path.join(out_path, f"{fname_out}.nc")) and not overwrite:
        logger.info(f"Output already exists for {fname_out} -- skipping.")
        return

    start_time = time.time()
    df_out, windows_df = process_windows(img, wsize, cores)
    
    duration = time.time() - start_time
    if duration > 60*60:
        logger.info(f"--> done with multiprocessing in {(duration / (60*60)):.2f} hours.")
    else:
        logger.info(f"--> done with multiprocessing in {(time.time() - start_time):.2f} seconds.")

    da_result = assemble_results(df_out, windows_df, img.attrs, img_res, wsize)

    save_netcdf(da_result, out_path, fname_out)

    start_time = time.time()
    compute_and_save_outputs(
        da_result, path2threshold, "dmg_threshold_dictionary.json", "S1", img_res,
        wsize, geotiff_path, fname_out
    )
    duration = time.time() - start_time
    if duration > 60*60:
        logger.info(f"--> done with dmg calculation in {(duration / (60*60)):.2f} hours.")
    else:
        logger.info(f"--> done with dmg calculation in {(time.time() - start_time):.2f} seconds.")

def run_nerd_folder(
    folder: str,
    img_res : str,
    dbmin: int,
    dbmax: int,
    wsize: int,
    cores: int,
    path2threshold: str,
    overwrite: bool = False,
):
    files = [f for f in os.listdir(folder) if f.endswith('.tif')]
    for f in files:
        logger.info(f"Processing {f}...")
        image_path = os.path.join(folder, f)
        run_nerd(
            image_path=image_path,
            img_res=img_res,
            dbmin=dbmin,
            dbmax=dbmax,
            wsize=wsize,
            cores=cores,
            path2threshold=path2threshold,
            overwrite=overwrite
        )
    
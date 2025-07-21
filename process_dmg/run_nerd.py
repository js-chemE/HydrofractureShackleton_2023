from app.dmg.nerd_process import run_nerd_folder
import os
import sys
import app
import logging
import argparse


def run_nerd(path, cores=4, overwrite=False):
    """Run NERD on a Linux Server from the command line for the current folder."""
    app.setup_logger(use_console_handler=True, use_file_handler=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f'Running NERD on folder: {path} with {cores} cores and overwrite={overwrite}')
    run_nerd_folder(
        folder = path,
        img_res = 30,
        dbmin = -15,
        dbmax = 5,
        wsize = 10,
        cores = cores,
        path2threshold=".",
        overwrite=overwrite,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NERD on a folder.")
    parser.add_argument("path", type=str, help="Path to the folder")
    parser.add_argument("--cores", type=int, default=4, help="Number of CPU cores to use")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing results")
    
    args = parser.parse_args()
    run_nerd(args.path, cores=args.cores, overwrite=args.overwrite)
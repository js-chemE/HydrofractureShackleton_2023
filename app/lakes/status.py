import os
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def check_downloaded_files(folder: str):
    files = os.listdir(folder)
    names = []
    tiles = []
    satellites = []
    starts = []
    ends = []
    infos = []
    rests = []
    sizes = []
    criterions = []
    for file in files:
        if file.endswith('.tif'):
            name = file.split('.')[0]
            names.append(name)
            splitted = name.split('_')
            tiles.append(splitted[0])
            satellites.append(splitted[1])
            starts.append(splitted[2])
            ends.append(splitted[3])
            info = splitted[4]
            if len(info.split('-')) > 1:
                rest = "-".join(info.split('-')[1:])
                info = info.split('-')[0]
            else:
                rest = None
            infos.append(info)
            rests.append(rest)
            criterions.append("_".join(splitted[:4] + [info]))
            sizes.append(os.path.getsize(os.path.join(folder, file)) / (1024 * 1024))  # size in MB
    res = pd.DataFrame({
        'tile': tiles,
        'satellite': satellites,
        'start': starts,
        'end': ends,
        'info': infos,
        'rest': rests,
        'size': sizes,
        'criterion': criterions,
        'name': names
    })
    return res

def check_all_downloaded_files(folder: str) -> pd.DataFrame:
    ress = []
    for i_year, year in enumerate(os.listdir(folder)):
        res = check_downloaded_files(os.path.join(folder, year))
        res['year'] = year.split('_')[0]
        ress.append(res)
    res_all = pd.concat(ress, ignore_index=True)
    return res_all
            
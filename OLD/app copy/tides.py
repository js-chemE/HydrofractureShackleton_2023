from typing import List

import pandas as pd

from app.dates import determine_antarctic_summer


def load_tide_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, delimiter=" ", names=["date", "tide"])
    df.date = pd.to_datetime(df.date, format="%Y-%m-%d")
    df["summer"] = [determine_antarctic_summer(d) for d in df.date]
    df.tide = df.tide.astype("float")
    return df


def load_tide_files(filepaths: List[str]) -> pd.DataFrame:
    combined_df = pd.DataFrame()

    for filepath in filepaths:
        df = load_tide_file(filepath)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

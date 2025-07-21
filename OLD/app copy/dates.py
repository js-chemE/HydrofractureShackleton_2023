from datetime import datetime
from typing import List, Union

import pandas as pd
from scipy.interpolate import interp1d


# Define a function to determine the Antarctic summer year based on a given date
def determine_antarctic_summer(date: Union[str, datetime]) -> int:
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.month >= 7:
        summer_year = date.year
    else:
        summer_year = date.year - 1

    return summer_year


def interpolate_dates(
    df: pd.DataFrame, date_column: str, value_column: str, interpolated_dates: List[str]
) -> List[float]:
    df[date_column] = pd.to_datetime(df[date_column])
    df[value_column] = df[value_column].interpolate(
        method="linear", limit_direction="both"
    )
    interpolated_values = df.set_index(date_column)[value_column]
    return interpolated_values.loc[interpolated_dates].to_list()

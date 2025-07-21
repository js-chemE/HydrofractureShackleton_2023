import numpy as np


def categorize(raster: np.ndarray, number_bins: int = 9, medium: int = 1, high: int = 4):

    raster_flatten = raster.data.flatten()
    raster_flatten = raster_flatten[~np.isnan(raster_flatten)]
    quantiles = np.linspace(0, 1, number_bins + 1)
    bin_limits = np.quantile(raster_flatten, quantiles)
    hist = np.histogram(raster_flatten, bins=bin_limits[:-1])
    counts = hist[0]
    limits = hist[1]

    res = {
        "quantiles": quantiles,
        "bin_limits": bin_limits,
        "counts": counts,
        "limits": limits,
        "low": {
            "counts": counts[0:medium],
            "limits": (0, float(limits[medium])),
            "quantile": quantiles[medium],
            "fraction": quantiles[medium]
        },
        "medium": {
            "counts": counts[medium:high],
            "limits": (float(limits[medium]), float(limits[high])),
            "quantile": quantiles[high],
            "fraction": quantiles[high] - quantiles[medium]
        },
        "high": {
            "counts": counts[high:],
            "limits": (float(limits[high]), 1),
            "quantile": quantiles[-1],
            "fraction": quantiles[-1] - quantiles[high]
        }
    }

    return res
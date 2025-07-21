import geopandas as gpd
import numpy as np
import rasterio
import rasterio.plot as rplt

import app

dmg_path = r"data-dmg/2020_S1_30m_dmg.tif"
with rasterio.open(dmg_path) as src:
    raster_original = src.read()
    transform_original = src.transform

print(transform_original)
print(np.count_nonzero(raster_original))

target_res = transform_original.a * 10
print(raster_original.shape)
resampled, resampled_transform = app.geotiffs.resample_dataset(
    raster_original,
    transform_original,
    target_res,
    mode="average",
)

print(np.count_nonzero(resampled))
# rplt.show(resampled, transform=resampled_transform)

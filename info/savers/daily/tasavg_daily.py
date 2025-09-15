"""
tasavg_daily.py
Loads hourly tas data, resamples and saves as tasavg, the daily temperature close to the midpoint between tasmin and tasmax.
"""
from Montreal_UHI_toolbox import get_outputs, save_zarr, standard_rechunk
tas_C,tas_T = get_outputs('tas')
tasavg_C = tas_C.resample(time='1D').mean().rename('tasavg')
tasavg_T = tas_T.resample(time='1D').mean().rename('tasavg')

save_zarr(standard_rechunk(tasavg_C),canopy='C')
print('Saved for CLASS')
save_zarr(standard_rechunk(tasavg_T),canopy='T')
print('Saved for TEB+CLASS')
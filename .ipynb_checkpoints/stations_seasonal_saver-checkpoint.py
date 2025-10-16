"""
stations_seasonal_saver.py

Selects PAVICS 3rd gen homogenised station temperature data within the 2000-2022 (inclusive) simulated period.
Saves as zarr - example to open: 
    tasmax_avg = xr.open_zarr('/home/gulley/UHI_HW_MTL/info/station/seasons_avg_tasmax.zarr')
"""
from Montreal_UHI_toolbox import stations
path = '/runoff/gulley/St_Laurent/intermediates/station'

for field in ['tasmax','tasmin','tas']:
    data = stations.sel(time=slice('2000','2022'))['tasmax']
    data_avg = data.groupby('time.season').mean(dim='time')
    data_std = data.groupby('time.season').std(dim='time')
    data_avg.to_zarr(f'{path}/seasons_avg_{field}.zarr')
    data_std.to_zarr(f'{path}/seasons_std_{field}.zarr')
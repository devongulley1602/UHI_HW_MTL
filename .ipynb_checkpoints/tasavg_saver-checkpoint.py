from Montreal_UHI_toolbox import get_outputs
path = '/runoff/gulley/St_Laurent/intermediates/sim'
field = 'tas'
field_C,field_T = get_outputs(field)

# Make daily tas
tasavg_C = field_C.groupby('time.day').mean(dim='time')
tasavg_T = field_T.groupby('time.day').mean(dim='time')

#tasavg_C.to_zarr(f'{path}/tasavg_daily_C.zarr')
#print('saved for tasavg_C')
tasavg_T.to_zarr(f'{path}/tasavg_daily_T.zarr')
print('saved for tasavg_T')


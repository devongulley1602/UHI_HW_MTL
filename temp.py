from Montreal_UHI_toolbox import get_outputs
path = '/runoff/gulley/St_Laurent/intermediates/sim'
field = 'tas'
field_C,field_T = get_outputs(field)
# Calculate seasonal averages for field
seasonal_std_C = field_C.groupby('time.season').std(dim='time')
seasonal_std_T = field_T.groupby('time.season').std(dim='time')
seasonal_std_C.to_zarr(f'{path}/seasons_std_{field}_noTEB.zarr')
print('noTEB done')
# seasonal_std_T.to_zarr(f'{path}/seasons_std_{field}_TEB.zarr')
# print('TEB done')

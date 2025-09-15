from Montreal_UHI_toolbox import get_outputs
"""
seasonal_sim_avgs.py
Saves tas/min/max seasonal averages across the span of the simulation as static fields.
Saved each as .zarr
"""
path = '/runoff/gulley/St_Laurent/intermediates/sim'

field = ['tasmin','tasmax','tasavg']:
    field_C,field_T = get_outputs(field)
    # Calculate seasonal averages for field
    seasonal_field_C = field_C.groupby('time.season').mean(dim='time')
    seasonal_field_T = field_T.groupby('time.season').mean(dim='time')
    seasonal_field_C.to_zarr(f'{path}/seasons_avg_{field}_noTEB.zarr')
    seasonal_field_T.to_zarr(f'{path}/seasons_avg_{field}_TEB.zarr')

    seasonal_field_C = field_C.groupby('time.season').std(dim='time')
    seasonal_field_T = field_T.groupby('time.season').std(dim='time')
    seasonal_field_C.to_zarr(f'{path}/seasons_std_{field}_noTEB.zarr')
    seasonal_field_T.to_zarr(f'{path}/seasons_std_{field}_TEB.zarr')
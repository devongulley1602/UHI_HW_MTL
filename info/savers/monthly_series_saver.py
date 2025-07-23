"""
monthly_series_saver.py

Loads all field data
Calculate average annual cycle smoothed by monthly averages
Save all data as pkl
"""

import Montreal_UHI_toolbox as tools
#import pickle as pkl

#field_dict_C = {} # CLASS 4d fields
#field_dict_T = {} # CLASS+TEB 4d fields
#for field_name in tools.field_keys:
#    field_dict_C[field_name] = tools.get_outputs(field_name,'.zarr','C')
#    field_dict_T[field_name] = tools.get_outputs(field_name,'.zarr','T')


def monthly_series(da,mask):
    """
    Returns the annual cycle based on monthly averages of a particular field.

    Parameters:
        da : xarray.DataArray   - input field rlat x rlon = 280x280 in time
        mask : xarray.DataArray - boolean spatial mask for urban/rural, high/low vegetation, etc 

    Returns : xarray.DataArray shape (12) - monthly mean of the field
    """
    return da.groupby('time.month').mean(dim='time').where(mask).mean(dim=['rlat','rlon'])


field_name = 'hdxmin' # 'tasmax' 'tasmin' 'hrss'
series_C,series_T = tools.get_outputs(field_name)

# Executing this in a loop will kill the kernal:
# For urban areas
urban_C = monthly_series(series_C,mask=tools.is_urban).values
print(f'urban_C: {urban_C}')
urban_T = monthly_series(series_T,mask=tools.is_urban).values
print(f'urban_T: {urban_T}')

# Now non-urban areas
rural_C = monthly_series(series_C,mask=tools.is_rural).values
print(f'rural_C: {rural_C}')
rural_T = monthly_series(series_T,mask=tools.is_rural).values
print(f'rural_T: {rural_T}')

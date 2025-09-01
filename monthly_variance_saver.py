import Montreal_UHI_toolbox as tools

def monthly_series_std(da,mask):
    """
    Returns the standard deviations of annual cycle based on monthly averages of a particular field.

    Parameters:
        da : xarray.DataArray   - input field rlat x rlon = 280x280 in time
        mask : xarray.DataArray - boolean spatial mask for urban/rural, high/low vegetation, etc 

    Returns : xarray.DataArray shape (12) - monthly mean of the field
    """
    return da.where(mask).groupby('time.month').std(dim='time').mean(dim=['rlat','rlon'])

field_name = 'tas' # 'tasmax' 'tasmin' 'hrss'
series_C,series_T = tools.get_outputs(field_name)

urban_C = monthly_series_std(series_C,mask=tools.is_urban).values
print(f'urban_C: {urban_C}')
#urban_T = monthly_series_std(series_T,mask=tools.is_urban).values
#print(f'urban_T: {urban_T}')

#rural_C = monthly_series_std(series_C,mask=tools.is_rural).values
#print(f'rural_C: {rural_C}')
#rural_T = monthly_series_std(series_T,mask=tools.is_rural).values
#print(f'rural_T: {rural_T}')

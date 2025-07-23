from Montreal_UHI_toolbox import save_zarr, standard_rechunk, get_outputs 
import numpy as np
import xarray as xr
tas_T = get_outputs(field='tas',extension='.zarr',canopy='T')
hrss_T = get_outputs(field='hrss',extension='.zarr',canopy='T')

# Aligning hourly tas to be comparable with the relative humidity 
#tas_3h_C = tas_C.resample(time='3h').nearest().rename('tas_3h')
tas_3h_T = tas_T.resample(time='3h').nearest().rename('tas_3h')

def humidex(e,T):
    """
    Index to indicate how hot or humid the weather feels to the average person according to ECCC technical documentation
    
    params
        e : float
            Vapour pressure [Pa]
        T : float
            Temperature [K]
    returns
        hdx : float 
            Humidex [C]
    """
    return (T - 273.15) + (0.5555)*(e/100 - 10.0)

def SVP(T):
    """
    Calculates the saturation vapour pressure values according to Huang (2018)
    
    parameters
        T : float
            Temperature [K]
    returns
        e_s : float
            Saturation vapour pressure for a given T in Pa
    """
    t = T - 273.15
    e_s_ice = np.exp(43.494 - 6545.8/(t + 278)) / np.square(t+868)
    e_s_wat = np.exp(34.494 - 4924.99/(t+ 237.1)) / np.power(t+105,1.57)
    
    return xr.where(t <= 0, e_s_ice, e_s_wat) # Lazy conditional selection

#e_C = SVP(tas_3h_C)*hrss_C
e_T = SVP(tas_3h_T)*hrss_T
#hdx_C = standard_rechunk(humidex(e_C,tas_3h_C)).rename('hdx')
hdx_T = standard_rechunk(humidex(e_T,tas_3h_T)).rename('hdx')

#save_zarr(hdx_C,canopy='C')
#print('Saved for CLASS')

# Partial saving:

time_index = hdx_T['time'].to_index()
start = 8190*8#time_index.get_indexer([np.datetime64("2002-12-16T00:00:00")], method="nearest")[0]
end = 67209#start + 8190
# end = time_index.get_indexer([np.datetime64("2004-01-01T00:00:00")], method="nearest")[0]
print(f'Saving from index {start} to {end}')
ds_trimmed = hdx_T.isel(time=slice(start, end)).drop_vars(['rlat', 'rlon', 'lat', 'lon'])
save_zarr(ds_trimmed, canopy='T', mode='r+', region={"time": slice(start, end)})
# save_zarr(hdx_T.drop_vars(['rlat', 'rlon', 'lat', 'lon']),canopy='T',mode='r+',region={"time": slice(start, end)})
print('Saved for TEB')

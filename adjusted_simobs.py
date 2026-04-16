"""
Container for tasmin_C, tasmax_C, tasmin_T, tasmax_T, tasmin_S, tasmax_S

Corresponding respectively to CLASS, TEB+CLASS, and Observed min/max daily temperatures at station locations adjusted to a common 54.5m altitude. 

When averaging stations from these sets be sure to check for 80% station data availability threshold.
"""
import xarray as xr
import numpy as np
from Montreal_UHI_toolbox import static_fields_C, add_field_to_stations, add_blurred_field_to_stations, obs, obs_urban, obs_rural, adjust_temp, Z_a
from UHI_statistics import load_daily_simobs

# Extract blurred orography (effective model elevation) from simulation data at station points
adjustment_set = add_blurred_field_to_stations(static_fields_C['orog'],station_set = obs)

Z_b_model = adjustment_set['orog_blurred_std1p5'].values 
Z_b_model = Z_b_model + 2.*np.ones(len(Z_b_model))
print(f'orog (m) for each model station:\n{Z_b_model}\n...to be scaled to {Z_a}m\n\n')

Z_b_obs = adjustment_set['elev'].values
print(f'elev (m) for each actual station:\n{Z_b_obs}\n...to be scaled to {Z_a}m')

# Load all station data 
adjusted = obs

# Adjust minimum and maximum daily temperatures from observed elevation
adjusted = adjusted.assign_coords({'tasmin_S': (('station','time'), adjust_temp(adjusted.tasmin.values + 273.15, z_b = Z_b_obs) - 273.15  )})
adjusted = adjusted.assign_coords({'tasmax_S': (('station','time'), adjust_temp(adjusted.tasmax.values + 273.15, z_b = Z_b_obs) - 273.15  )})
adjusted = adjusted.reset_coords(['tasmin_S', 'tasmax_S'])

# Adjust minimum and maximum daily temperatures from model orography
for f in ['tasmin','tasmax']:
    for m in ['C','T']:
        adjusted = adjusted.assign_coords({f'{f}_{m}': (('time','station') , load_daily_simobs(f, m).values )})
        adjusted = adjusted.reset_coords(f'{f}_{m}')


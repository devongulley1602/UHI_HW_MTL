"""
Seasonal UHI Analysis
========================================

UHI_seasonal and temps_seasonal accessed by [field = tasmin/tasmax][model = C/T][season = JJA/SON/DJF/MAM]['some trait' like UHI, values,]

This module computes seasonal and interannual Urban Heat Island (UHI) statistics from observational station data and simulated model output.
UHI is defined as the difference between spatially averaged urban and rural near-surface (2m in model) air temperatures from point-based 
station locations. These temperatures are adjusted to a common elevation of 54.5m using a dry adiabatic lapse rate.

Daily UHI time series are aggregated to seasonal means for each year, and statistical inference is performed across years to account for
temporal autocorrelation in daily temperature data.

Analysis produces:
- Daily UHI/temperature time series for observations and simulated observations
- Seasonal-mean UHI/temperatures averaged across years
- Interannual standard error, t-statistics, and 95% confidence intervals for each season, temperature variable, and data source for 
  UHI/temperatures
- p-values for UHI
- Calculation of 10-90th inter-percentile ranges

All hypothesis tests are two-sided:
    H0: mean seasonal UHI = 0
    Ha: mean seasonal UHI != 0
    alpha = 0.05

- Daily data are not treated as independent samples.
- Statistical inference is based on interannual variability of seasonal means, treating each year as an independent realisation.
- Missing/masked data are handled implicitly via xarray groupby/count logic.
- Point-based model extractions are subject to a 1.5 cell standard deviation Gaussian blur as is convention for this project.
"""

from Montreal_UHI_toolbox import obs, obs_rural, obs_urban, adjust_temp, add_blurred_field_to_stations, static_fields_C
import xarray as xr
import numpy as np
from scipy import stats

# Adjust based on elevations for actual observations, adjust based on model orography for simulated observations
obs = add_blurred_field_to_stations(static_fields_C['orog'],obs)
obs_rural = add_blurred_field_to_stations(static_fields_C['orog'],obs_rural)
obs_urban = add_blurred_field_to_stations(static_fields_C['orog'],obs_urban)

orog_urban = np.mean(obs_urban.orog_blurred_std1p5).values
orog_rural = np.mean(obs_rural.orog_blurred_std1p5).values
elev_urban = np.mean(obs_urban.elev).values
elev_rural = np.mean(obs_rural.elev).values


def load_daily_simobs(field,model):
    """
    Loads daily simulated observations from /runoff/gulley/St_Laurent/intermediates/sim/series_at_obs_locations
    
    Parameters
    field : string - 'tas', 'tasmin', or 'tasmax'
    model : string - 'C' for CLASS or 'T' for TEB+CLASS

    Returns
    xarray.dataset(station,time) - temperature (corresponding to field) in degrees Celsius
    """
    return xr.open_dataset(f'/runoff/gulley/St_Laurent/intermediates/sim/series_at_obs_locations/{field}_{model}.nc').swap_dims({'points':'station'}).assign_coords({'station': obs.station}).sel(time=slice('2000','2022'))[field] - 273.15

avail_thresh = 0.8 # 80% data availability threshold for observation data
alpha = 0.05

# Loading obs (station data) S for Station ensuring station availability is >= 80% avail_thresh for a given daily sample
urban_S = obs_urban.where(obs_urban.count(dim='station') >= len(obs_urban.station)*avail_thresh,drop=True).mean(dim='station')
rural_S = obs_rural.where(obs_rural.count(dim='station') >= len(obs_rural.station)*avail_thresh,drop=True).mean(dim='station')

# Adjust the rural stations to match the urban station elevations (urban stations don't need to be temperature adjusted to their own elevations of course)
rural_S['tasmax'] = rural_S.tasmax.copy(data=adjust_temp(rural_S.tasmax.values + 273.15, elev_rural)[0] - 273.15)
rural_S['tasmin'] = rural_S.tasmin.copy(data=adjust_temp(rural_S.tasmin.values + 273.15, elev_rural)[0]- 273.15)
rural_S['tas'] = rural_S.tas.copy(data=adjust_temp(rural_S.tas.values + 273.15 , elev_rural)[0]- 273.15)

# Load daily point data from simobs CLASS, simobs TEB+CLASS, and obs
UHI_daily = {}
obsdiff_UHI_daily = {}
tebdiff_UHI_daily = {}
temps_daily = {}

# UHI_seasonal and temps_seasonal accessed by [field = tasmin/tasmax][model = C/T][season = JJA/SON/DJF/MAM]['some trait' like UHI, values,]
UHI_seasonal = {}
UHI_Y = {}
temps_seasonal = {}
temps_Y = {}

# Load simobs and obs into daily dictionaries
for f in ['tasmin','tasmax','tas']:
        # Put the obs UHI into the daily UHI dictionary
        UHI_daily[f'{f}_S'] = (urban_S[f] - rural_S[f]).dropna(dim='time')

        # Load daily observed daily temperatures as well
        # temps_daily[f'{f}_S'] = obs[f] # temperatures unadjusted for elevation
        temps_daily[f'{f}_S'] = obs[f].copy(data=adjust_temp(obs[f].values + 273.15, obs.elev.values) - 273.15)  # temperatures adjusted for elevation
        
        # Loadng simobs (model data) C for CLASS T for TEB+CLASS
        for m in ['C','T']:

            # Before elevation adjustments were made:
            # temps_daily[f'{f}_{m}'] = load_daily_simobs(field=f,model=m)
            # daily = temps_daily[f'{f}_{m}']
            # urban = daily.where(daily.station.isin(obs_urban.station),drop=True).mean(dim='station')
            # rural = daily.where(daily.station.isin(obs_rural.station),drop=True).mean(dim='station')

            # Adjust temperatures based on model orog to observed urban elevation average:
            simobs = load_daily_simobs(field=f,model=m).T # transposed to match adjust_temp function - it was reversed when saved
            adj_simobs = (simobs.copy(data=adjust_temp(simobs.values + 273.15, obs.orog_blurred_std1p5.values + 2) - 273.15)) 
            # Set daily temperatures based on adjusted values
            temps_daily[f'{f}_{m}'] = adj_simobs
            urban = adj_simobs.where(adj_simobs.station.isin(obs_urban.station),drop=True).mean(dim='station')
            rural = adj_simobs.where(adj_simobs.station.isin(obs_rural.station),drop=True).mean(dim='station')
            # Put the UHI simobs UHI for CLASS and TEB+CLASS into the daily UHI dictionary
            UHI_daily[f'{f}_{m}'] = urban - rural

            # Difference of UHI in each model from observation 
            obsdiff_UHI_daily[m] = UHI_daily[f'{f}_{m}'] - UHI_daily[f'{f}_S']
        tebdiff_UHI_daily[f] =  UHI_daily[f'{f}_T'] - UHI_daily[f'{f}_C']

# All temperatures are adjusted based on elevation, moved towards the elevation of the smallest dataset (observed urban elevation) based on constant lapse rate assumptions

# Calculate statistics using interannual variability as a basis for defining seasonal sampling and error
for f in ['tasmin','tasmax','tas']:
    # Seasonal groupings indexed by model/obs
    UHI_seasonal[f] = {}
    temps_seasonal[f] = {}
    
    # Yearly groupings indexed by model/obs
    UHI_Y[f] = {}
    temps_Y[f] = {}

    for m in ['C','T','S']:
        # Groups by season
        UHI_seasonal[f][m] = {'JJA':{'UHI':np.nan,'error':np.nan},'SON': {'UHI':np.nan,'error':np.nan},'DJF':{'UHI':np.nan,'error':np.nan},'MAM':{'UHI':np.nan,'error':np.nan}}
        UHI_Y[f][m] = {'JJA':{'UHI':np.nan,'error':np.nan},'SON': {'UHI':np.nan,'error':np.nan},'DJF':{'UHI':np.nan,'error':np.nan},'MAM':{'UHI':np.nan,'error':np.nan}}
        temps_seasonal[f][m] = {'JJA':{'vals':np.nan,'error':np.nan},'SON': {'vals':np.nan,'error':np.nan},'DJF':{'vals':np.nan,'error':np.nan},'MAM':{'vals':np.nan,'error':np.nan}}
        temps_Y[f][m] = {'JJA':{'vals':np.nan,'error':np.nan},'SON': {'vals':np.nan,'error':np.nan},'DJF':{'vals':np.nan,'error':np.nan},'MAM':{'vals':np.nan,'error':np.nan}}
        
        for s in ['JJA','SON','DJF','MAM']:
            UHI_seasonal[f][m][s] = {}
            temps_seasonal[f][m][s] = {}
            
            # Subannual UHI averages
            UHI_Y[f][m][s] = UHI_daily[f'{f}_{m}'].groupby('time.season')[s].groupby('time.year').mean(dim='time')
            temps_Y[f][m][s] = temps_daily[f'{f}_{m}'].groupby('time.season')[s].groupby('time.year').mean(dim='time')
            n = UHI_Y[f][m][s].count(dim='year')

            # UHI statistics
            UHI_seasonal[f][m][s]['UHI'] = UHI_Y[f][m][s].mean(dim='year').values                               # UHI
            UHI_seasonal[f][m][s]['STD'] = UHI_Y[f][m][s].std(dim='year').values                                # Standard deviation
            UHI_seasonal[f][m][s]['SE'] = (UHI_Y[f][m][s].std(dim='year',ddof=1)/np.sqrt(n)).values             # Standard error
            UHI_seasonal[f][m][s]['T'] = UHI_seasonal[f][m][s]['UHI']/UHI_seasonal[f][m][s]['SE']               # t-statistic
            UHI_seasonal[f][m][s]['PVAL'] = 2*(1 - stats.t.cdf(abs(UHI_seasonal[f][m][s]['T']), n-1))           # 2-sided p-value
            UHI_seasonal[f][m][s]['ERR'] = stats.t.ppf(1 - alpha/2, n-1)*UHI_seasonal[f][m][s]['SE']            # error bars at 1-alpha = 95% confidence
            UHI_seasonal[f][m][s]['PERC'] = np.percentile(UHI_seasonal[f][m][s]['UHI'],[10,90])                 # 10th-90th IPR

            # temp statistics
            temps_seasonal[f][m][s]['vals'] = temps_Y[f][m][s].mean(dim='year').values                          # tempperature at each station
            temps_seasonal[f][m][s]['SE'] = (temps_Y[f][m][s].std(dim='year',ddof=1)/np.sqrt(n)).values         # Standard error
            temps_seasonal[f][m][s]['ERR'] = stats.t.ppf(1 - alpha/2, n-1)*temps_seasonal[f][m][s]['SE']        # error bars at 1-alpha = 95% confidence
            UHI_seasonal[f][m][s]['PERC'] = np.percentile(temps_seasonal[f][m][s]['vals'],[10,90])              # 10th-90th IPR


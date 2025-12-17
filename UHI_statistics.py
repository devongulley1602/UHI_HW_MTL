"""
Seasonal UHI Analysis
========================================

UHI_seasonal and temps_seasonal accessed by [field = tasmin/tasmax][model = C/T][season = JJA/SON/DJF/MAM]['some trait' like UHI, values,]

This module computes seasonal and interannual Urban Heat Island (UHI) statistics from observational station data and simulated model output.
UHI is defined as the difference between spatially averaged urban and rural near-surface (2m in model) air temperatures from point-based 
station locations.

Daily UHI time series are aggregated to seasonal means for each year, and statistical inference is performed across years to account for
temporal autocorrelation in daily temperature data.

Analysis produces:
- Daily UHI/temperature time series for observations and simulated observations
- Seasonal-mean UHI/temperatures averaged across years
- Interannual standard error, t-statistics, and 95% confidence intervals for each season, temperature variable, and data source for 
  UHI/temperatures
- p-values for UHI

All hypothesis tests are two-sided:
    H0: mean seasonal UHI = 0
    Ha: mean seasonal UHI != 0
    alpha = 0.05

- Daily data are not treated as independent samples.
- Statistical inference is based on interannual variability of seasonal means, treating each year as an independent realisation.
- Missing/masked data are handled implicitly via xarray groupby/count logic.
"""

from Montreal_UHI_toolbox import obs, obs_rural, obs_urban
import xarray as xr
import numpy as np
from scipy import stats

avail_thresh = 0.8 # 80% data availability threshold for observation data
alpha = 0.05


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

# Loading obs (station data) S for Station ensuring station availability is >= 80% avail_thresh for a given daily sample
urban_S = obs_urban.where(obs_urban.count(dim='station') >= len(obs_urban.station)*avail_thresh,drop=True).mean(dim='station')
rural_S = obs_rural.where(obs_rural.count(dim='station') >= len(obs_rural.station)*avail_thresh,drop=True).mean(dim='station')

# Load simobs and obs into daily dictionaries
for f in ['tasmin','tasmax','tas']:
        # Put the obs UHI into the daily UHI dictionary
        UHI_daily[f'{f}_S'] = (urban_S[f] - rural_S[f]).dropna(dim='time')

        # Load daily observed daily temperatures as well
        temps_daily[f'{f}_S'] = obs[f]
        
        # Loadng simobs (model data) C for CLASS T for TEB+CLASS
        for m in ['C','T']:
            temps_daily[f'{f}_{m}'] = load_daily_simobs(field=f,model=m)

            daily = temps_daily[f'{f}_{m}']
            urban = daily.where(daily.station.isin(obs_urban.station),drop=True).mean(dim='station')
            rural = daily.where(daily.station.isin(obs_rural.station),drop=True).mean(dim='station')

            # Put the UHI simobs UHI for CLASS and TEB+CLASS into the daily UHI dictionary
            UHI_daily[f'{f}_{m}'] = urban - rural

            # Difference of UHI in each model from observation 
            obsdiff_UHI_daily[m] = UHI_daily[f'{f}_{m}'] - UHI_daily[f'{f}_S']
        tebdiff_UHI_daily[f] =  UHI_daily[f'{f}_T'] - UHI_daily[f'{f}_C']

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
            UHI_seasonal[f][m][s]['SE'] = (UHI_Y[f][m][s].std(dim='year',ddof=1)/np.sqrt(n)).values             # Standard error
            UHI_seasonal[f][m][s]['T'] = UHI_seasonal[f][m][s]['UHI']/UHI_seasonal[f][m][s]['SE']               # t-statistic
            UHI_seasonal[f][m][s]['PVAL'] = 2*(1 - stats.t.cdf(abs(UHI_seasonal[f][m][s]['T']), n-1))           # 2-sided p-value
            UHI_seasonal[f][m][s]['ERR'] = stats.t.ppf(1 - alpha/2, n-1)*UHI_seasonal[f][m][s]['SE']            # error bars at 1-alpha = 95% confidence

            # temp statistics
            temps_seasonal[f][m][s]['vals'] = temps_Y[f][m][s].mean(dim='year').values                          # tempperature at each station
            temps_seasonal[f][m][s]['SE'] = (temps_Y[f][m][s].std(dim='year',ddof=1)/np.sqrt(n)).values         # Standard error
            temps_seasonal[f][m][s]['ERR'] = stats.t.ppf(1 - alpha/2, n-1)*temps_seasonal[f][m][s]['SE']        # error bars at 1-alpha = 95% confidence
"""
Retrieves from .nc and plots monthly data. 
Now obsolete, used for reference (2025-07-22)
"""

from Montreal_UHI_toolbox import *
import plotly.graph_objects as go

# Datasets ending in T are TEB+CLASS, ending in C are CLASS alone
print('Retrieving datasets')
# Set i: 
# 0 hourly temperature
# 1 max daily temperature 
# 2 min daily temperature 
# 3 sensible heat flux 
# 4 latent heat flux
i = 3
field = field_keys[i] # field_keys imported from common_fields

ds_C = xr.open_mfdataset(f'/runoff/gulley/St_Laurent/StLaurent_1km_SL2.5_ERA5_advHU/*{field}.nc')[field]
ds_T = xr.open_mfdataset(f'/runoff/gulley/St_Laurent/StLaurent_1km_SL2.5_ERA5_advHU_TEB/*{field}.nc')[field]

ds_T_annual = ds_T.groupby('time.month').mean(dim='time')
ds_C_annual = ds_C.groupby('time.month').mean(dim='time')

# Urban fraction properties
print('Retrieving urban properties')
static_fields = xr.open_mfdataset('/runoff/gulley/St_Laurent/StLaurent_1km_SL2.5_ERA5_advHU_step0.nc')
urban_fraction_2d = static_fields['furban'].sel(lev=21)
is_rural = urban_fraction_2d < 0.01 # Based on metric used by Roberge and Sushama (2018)
is_urban = urban_fraction_2d > 0.5

# Plot dictionaries for making multiple charts
print('Setting plot variables')
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
color_map = {'TEB+CLASS': 'red', 'CLASS': 'blue','Urban':'red','Rural':'blue'}
dash_map = {
    'Urban TEB+CLASS': 'solid', 'Urban CLASS': 'solid',
    'Rural TEB+CLASS': 'dot', 'Rural CLASS': 'dot',
    'Urban-Rural': 'solid','(TEB+CLASS)-(TEB)':'solid',
    'Urban':'solid', 'Rural':'dot'
}

# Monthly averages over the domain grouped by urban fraction
urban_monthly_series_T =  ds_T_annual.where(is_urban).mean(dim=['rlat','rlon'])
urban_monthly_series_C =  ds_C_annual.where(is_urban).mean(dim=['rlat','rlon'])
rural_monthly_series_T = ds_T_annual.where(is_rural).mean(dim=['rlat','rlon'])
rural_monthly_series_C = ds_C_annual.where(is_rural).mean(dim=['rlat','rlon'])


print('Plotting...')
print(f'Creating {field} plot...') 
fig = go.Figure()

for label, data, model in [
    ('Urban TEB+CLASS', urban_monthly_series_T[field].values, 'TEB+CLASS'),
    ('Urban CLASS',     urban_monthly_series_C[field].values, 'CLASS'),
    ('Rural TEB+CLASS', rural_monthly_series_T[field].values, 'TEB+CLASS'),
    ('Rural CLASS',     rural_monthly_series_C[field].values, 'CLASS')
]:
    fig.add_trace(go.Scatter(
        x=months,
        y=data,
        name=f'{label} {field}',
        line=dict(color=color_map[model], dash=dash_map[label])
    ))

fig.update_layout(
    title=f'Annual Monthly Averaged {field}',
    template='plotly_white',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.35,
        xanchor='center',
        x=0.5
    )
)
print(f'Saving {field}...') 
fig.write_html(f'{field}_monthly_average_annual_series.html')

# Urban-Rural
fig = go.Figure()
print(f'Calculating UHI {field} differences...')
for label, data, model in [
    ('Urban-Rural', urban_monthly_series_T[field] - rural_monthly_series_T[field], 'TEB+CLASS'),
    ('Urban-Rural', urban_monthly_series_C[field] - rural_monthly_series_C[field], 'CLASS'),
]:
    fig.add_trace(go.Scatter(
        x=months,
        y=data.values,
        name=f'{label} {field}',
        line=dict(color=color_map[model], dash=dash_map[label])
    ))
fig.update_layout(
    title=f'Annual Monthly Averaged UHI for {field}',
    template='plotly_white',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.35,
        xanchor='center',
        x=0.5
    )
)
print(f'Saving {field} UHI...') 
fig.write_html(f'{field}_UHI_monthly_average_annual_series.html')

# (TEB+CLASS)-(TEB)
fig = go.Figure()
print(f'Calculating UHI {field} differences...')
for label, data, model in [
    ('Urban', urban_monthly_series_T[field] - urban_monthly_series_C[field],'(TEB+CLASS)-(TEB)'),
    ('Rural', rural_monthly_series_T[field] - rural_monthly_series_C[field],'(TEB+CLASS)-(TEB)' ),
]:
    fig.add_trace(go.Scatter(
        x=months,
        y=data.values,
        name=f'{label} {field}',
        line=dict(color=color_map[label], dash=dash_map[model])
    ))
fig.update_layout(
    title=f'Annual Monthly Averaged (TEB+CLASS)-(TEB) for {field}',
    template='plotly_white',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.35,
        xanchor='center',
        x=0.5
    )
)
print(f'Saving {field} TEBdiff...') 
fig.write_html(f'{field}_TEBdiff_monthly_average_annual_series.html')
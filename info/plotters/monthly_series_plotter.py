"""
Given monthly-grouped averaged data of various fields produced from monthly_series_saver.py
Create graphs for each field
Create monthly UHI and TEBdiff series
"""
import Montreal_UHI_toolbox as tools
import plotly.graph_objects as go
import numpy as np

#monthly stores the monthly average of a particular field keyed by
# {urban or rural}_{C (CLASS) or T (TEB+CLASS)}

monthly = {} # Taken from output given by monthly_series_saver.py (see info/savers/monthly_series_saver.py)
monthly['hdx'] = {'urban_C': [-13.69604752, -12.9559597 ,  -6.75828627 ,  3.23042308 , 13.60251688,
      21.57444827 , 26.28472259 , 25.39775718 , 19.08434232 ,  9.69145241,
       0.72211275,  -7.88446107],
    'urban_T': [-13.22350649 ,-12.53591337 , -6.21991785 ,  3.90518138 , 14.67719893,
      22.60670493 , 27.34531891 , 26.52951434 , 20.16626377 , 10.49278481,
       1.19208307 , -7.23890473],
    'rural_C': [-14.07668567 ,-13.33325275 , -7.48184085  , 2.00567902 , 12.51234693,
      20.3797059 ,  24.91800327  ,24.13080913 , 17.98145977 ,  8.77251834,
      -0.18924984 , -8.46084426],
    'rural_T': [-14.10303794, -13.53502079,  -7.60989267 ,  2.05613978,  12.74289271,
      20.50576976 , 25.06903004 , 24.33306236,  18.1938747  ,  8.89155327,
      -0.13568962,  -8.19731816]}

monthly['hdxmax'] = {'urban_C': [-8.90789456, -7.97284655, -1.30195741,  9.0517855,  19.35949778, 27.4268979,
     32.11271835, 31.06006574, 24.8958237,  14.24843515,  5.06464394, -3.46464055],
    'urban_T': [-8.7328528,  -7.79777543, -1.04686054,  9.50383899, 20.17832146, 28.13148946,
     32.71530753, 31.76511978, 25.67686269, 14.83073834,  5.35651699, -3.0364089 ],
     'rural_C': [-9.19337735, -8.25560994, -2.15345167,  7.50707751, 17.9503039,  25.86050387,
     30.37220983, 29.46095478, 23.61348575, 13.24635893,  4.02722181, -4.04550521],
    'rural_T': [-9.29369342, -8.55314014, -2.34516573,  7.54104915, 18.1549123,  25.95199081,
     30.46976987, 29.62534648, 23.78412834, 13.28960367,  4.12351738, -3.81546042]
}

monthly['hdxmin'] = {'urban_C': [-17.85359478, -17.20958517, -11.29359331,  -1.77422208,   8.34569055,
      16.0869375,   20.47367895,  20.11917647,  14.29401067,   6.11155128,
      -2.90189154, -11.71047188],
     'urban_T': [-16.95523192, -16.3915619,  -10.35401291,  -0.59000808,   9.88073024,
      17.63529584,  22.27415645,  21.84137229,  15.76817974,   7.04235823,
      -2.21156418, -10.72014846],
    'rural_C': [-18.12838113, -17.50831375, -11.77037173,  -2.4934379,    7.69761838,
      15.21656758,  19.49400787,  19.14590864,  13.41932009,   5.26082558,
      -3.68096412, -12.18485923],
    'rural_T':[-18.10370756, -17.64979134, -11.85994158,  -2.41304064,   7.90342587,
      15.34666398,  19.67610793,  19.38848236,  13.64099883,   5.37590087,
      -3.66607344, -11.89563227]}

monthly['hrss'] = {'urban_C': [0.76354022 ,0.74322615 ,0.68418464, 0.62176403, 0.68460278, 0.68656593,
     0.61499082, 0.67803208, 0.7447965,  0.84970325 ,0.78571498, 0.77796491],
    'urban_T': [0.71964182, 0.68223875 ,0.62542459, 0.57337721 ,0.59666939 ,0.60581429,
     0.55980486, 0.61153762 ,0.66246337 ,0.75287738, 0.73602341 ,0.74433182],
    'rural_C': [0.75140444, 0.72249098, 0.67428109, 0.62881956 ,0.66637603, 0.67010321,
     0.62593352 ,0.67353596 ,0.72436832, 0.80223633, 0.77320698, 0.77754175],
    'rural_T': [0.75166426, 0.71736264, 0.67081831, 0.62655535 ,0.65688738, 0.66615648,
     0.62449691, 0.67296972, 0.71864175 ,0.79875898 ,0.77051632, 0.77383574]}

monthly['tas'] = {'urban_C': [263.50931054, 264.30477644 ,269.95456298, 278.63709462, 286.45129633,
     292.08394047 ,295.88015231, 294.659086,   289.8500432,  282.56964414,
     275.93423325, 268.69864337],
    'urban_T': [264.0676357 , 265.00309792, 270.74193504, 279.40435938, 287.63644282,
     293.4207313,  297.08781754 ,295.95154733, 291.09016021, 283.59044459,
     276.45545099, 269.11464915],
    'rural_C': [263.17818213, 263.98173614, 269.32574024, 277.59728936, 285.71218789,
     291.39042058, 294.85478603, 293.86647506, 289.26539,    282.13820098,
     275.22768381, 268.19459355],
    'rural_T': [263.22446144, 264.04042847, 269.39051958, 277.65061147, 285.78315381,
     291.47465687, 294.93783712, 293.9502011,  289.33984283, 282.18225061,
     275.25442633, 268.22121115]}

monthly['tasmax'] = {'urban_C': [268.27168985, 269.0058894 , 274.68248548, 284.05522893, 291.51631003,
     297.56416168, 301.87345182 ,300.52356924, 295.48255161, 286.72848651,
     280.06735183, 273.11646925],
    'urban_T': [268.60282992, 269.52743073, 275.24781591, 284.63655378, 292.70571219,
     299.02080011, 302.99467084, 301.79520662, 296.7292188,  287.82713373,
     280.39375526, 273.3194097 ],
    'rural_C': [268.02942569, 268.80215692, 273.980854,   282.72435989, 290.68398939,
     296.88888584, 300.75139488, 299.71552656, 294.98162341, 286.44732573,
     279.31392802, 272.5839526 ],
    'rural_T': [268.04635862, 268.82928252, 274.01160004, 282.74527983, 290.74036887,
     296.96078614, 300.80359989, 299.77379612, 295.04463653, 286.49319702,
     279.33202344, 272.59619806]}

monthly['tasmin'] = {'urban_C': [259.09966619, 259.70201183, 264.65685081, 273.15135782, 281.06101098,
     286.67789149, 289.89034612, 289.27670785, 285.27737427, 279.41463019,
     272.4760457,  264.86663878],
    'urban_T': [260.14967692, 260.85197841, 265.98173539, 274.44521619, 282.50938514,
     288.17540351, 291.4969987,  290.81738612, 286.67679247, 280.47838157,
     273.3465526,  265.72800007],
    'rural_C': [258.88073341, 259.46837291, 264.38117519, 272.66200072, 280.63462457,
     286.12105359, 289.18600894, 288.69211784, 284.82168589, 278.95329868,
     271.93018235, 264.47989216],
    'rural_T': [258.96354311, 259.5665445,  264.48787166, 272.75053372, 280.70875322,
     286.20052637, 289.27525715, 288.76926325, 284.88360566, 278.98800622,
     271.96098765, 264.52436374]}

# Convert all temperatures to C
sim_keys = ['urban_C','urban_T','rural_C','rural_T']
for kelvin_temps in ['tas','tasmin','tasmax']:
    for sim_key in sim_keys:
        monthly[kelvin_temps][sim_key] -= 273.15*np.ones(len(monthly[kelvin_temps][sim_key]))

# # Convert fractions to percentages (ie hrss)
for sim_key in sim_keys:
    for i in range(len(monthly['hrss'][sim_key])):
        monthly['hrss'][sim_key][i] *= 100 # sets as percentage


urban_minus_rural = {}
teb_minus_class = {}

for field in ['tas','tasmax','tasmin','hdx','hdxmax','hdxmin','hrss']:
    urban_minus_rural[field] = {}
    urban_minus_rural[field]['CLASS'] = np.array(monthly[field]['urban_C']) -  np.array(monthly[field]['rural_C'])
    urban_minus_rural[field]['TEB+CLASS'] = np.array(monthly[field]['urban_T']) - np.array(monthly[field]['rural_T'])
    
    teb_minus_class[field] = {}
    teb_minus_class[field]['Urban'] = np.array(monthly[field]['urban_T']) - np.array(monthly[field]['urban_C'])
    teb_minus_class[field]['Non-Urban'] = np.array(monthly[field]['rural_T']) - np.array(monthly[field]['rural_C'])

units = {'tas':'(°C)','tasmin':'(°C)','tasmax':'(°C)','hdx':'','hdxmin':'','hdxmax':'','hrss':'(%)'}
titles = {'tas':'Hourly Temperature',
          'tasmin':'Minimum Daily Temperature',
          'tasmax':'Maximum Daily Temperature',
          'hdx':'Humidex',
          'hdxmin':'Minimum Daily Humidex',
          'hdxmax':'Maximum Daily Humidex',
          'hrss': '3-Hourly Relative Humidity'}
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_nums = list(range(1,13))


colour_map = {'urban_T': 'red', 'urban_C': 'blue', 'rural_T' : 'red' , 'rural_C': 'blue'}
dash_map =  {'urban_T': 'solid',  'urban_C': 'solid',   'rural_T' : 'dash' , 'rural_C': 'dash' }
width_map = {'urban_T': 2.,  'urban_C': 2.,   'rural_T' : 2. , 'rural_C': 2.}

axis_title_map = {'tas':'Temperature (°C)',
            'tasmin':'Temperature (°C)',
            'tasmax':'Temperature (°C)',
            'hdx': 'Humidex',
            'hdxmin': 'Humidex',
            'hdxmax': 'Humidex',
            'hrss': 'Relative Humidity (%)'}

axis_map = {'tas':'y1',
            'tasmin':'y1',
            'tasmax':'y1',
            'hrss':'y1',
            'hdx': 'y2',
            'hdxmin': 'y2',
            'hdxmax': 'y2'}



"""
Annual Charts
"""
for field in ['tas', 'tasmax','tasmin','hdx','hdxmax','hdxmin','hrss']:
    fig = go.Figure()
    for model,model_code in [('Urban TEB+CLASS','urban_T'),
                             ('Non-Urban TEB+CLASS','rural_T'),
                             ('Urban CLASS','urban_C'),
                             ('Non-Urban CLASS', 'rural_C')]:
        fig.add_trace(go.Scatter(
                    x=month_nums,
                    y=monthly[field][model_code],
                    name=f'{model}',
                    yaxis=axis_map[field],
                    line=dict(color=colour_map[model_code], width=width_map[model_code],dash=dash_map[model_code])
        ))


    fig.update_layout(
        title=f'Monthly Averaged Annual Series of {titles[field]} ({field})',
        yaxis=dict(
            title=f'{axis_title_map[field]}'
        ),

        xaxis=dict(
            range=[0, 11],
            tickmode='array',
            tickvals=month_nums,
            ticktext=months,
            autorange=False
        ),

        legend=dict(
            orientation='h',    
            yanchor='bottom',
            y=-0.35, # Just below the plot             
            xanchor='center',
            x=0.5
        ),
        # margin=dict(l=0, r=0), 
        template='plotly_white'
    )
    # fig.show()
    fig.write_html(f'{field}_monthly_average_annual_series.html')


"""
Urban-Rural
"""
# Add these to the colour map
colour_map['CLASS'] = 'blue'
colour_map['TEB+CLASS'] = 'red'


for field in ['tas', 'tasmax','tasmin','hdx','hdxmax','hdxmin','hrss']:
    fig = go.Figure()
    for model in ['CLASS','TEB+CLASS']:
        fig.add_trace(go.Scatter(
            x=months,
            y=urban_minus_rural[field][model],
            name=f'{model} UHI',
            line=dict(color=colour_map[model], dash='solid')
        ))
    fig.update_layout(
        title=f'Monthly Averaged Annual Urban-Rural Values of {titles[field]} ({field})',
        yaxis=dict(
            title=f'Difference in {axis_title_map[field]}'
        ),

        xaxis=dict(
            range=[0, 11],
            tickmode='array',
            tickvals=month_nums,
            ticktext=months,
            autorange=False
        ),
        
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.35,
            xanchor='center',
            x=0.5
        )
    )
    # fig.show()
    fig.write_html(f'{field}_UHI_monthly_average_annual_series.html')

"""
TEBdiff
"""
colour_map['Urban'] = 'slategrey'
colour_map['Non-Urban'] = 'yellowgreen'
for field in ['tas', 'tasmax','tasmin','hdx','hdxmax','hdxmin','hrss']:
    fig = go.Figure()
    for urban_level in ['Urban','Non-Urban']:
        fig.add_trace(go.Scatter(
            x=months,
            y=teb_minus_class[field][urban_level],
            name=f'{urban_level} TEB Diff',
            line=dict(color=colour_map[urban_level], dash='solid')
        ))
    fig.update_layout(
        title=f'Monthly Averaged Annual (TEB+CLASS)-(CLASS) Values of {titles[field]} ({field})',
        yaxis=dict(
            title=f'Difference in {axis_title_map[field]}'
        ),

        xaxis=dict(
            range=[0, 11],
            tickmode='array',
            tickvals=month_nums,
            ticktext=months,
            autorange=False
        ),
        
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.35,
            xanchor='center',
            x=0.5
        )
    )
    # fig.show()
    fig.write_html(f'{field}_TEBdiff_monthly_average_annual_series.html')
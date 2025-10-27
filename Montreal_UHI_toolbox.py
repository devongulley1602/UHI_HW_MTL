"""
Montreal_UHI_toolbox.py

Stores useful variables and functions used across many analysis scripts in the project.
Kept here for organisational purposes.


Public variables:

    out_dir_C : string
        - Directory of the CLASS alone model output 

    out_dir_T : string
            - Directory of the CLASS+TEB model output 

    rotated_pole : cartopy.crs.RotatedPole
        - Location of rotated pole
        
    rlat,rlon : numpy.ndarray
        - Rotated pole grid values each with shape (280,)
        
    field_keys : array (string)
        - Keys of fields to analyse such as 
        - Includes 'tas', 'tasmax', 'tasmin', 'hfss', 'hfls'
        
    veg_fields : xarray.core.dataarray.DataArray
        - CLASS vegetation and urban fields to analyse
        
    veg_levs : array of type string
        - Tranlsation from the arbitrary level to the names of the veg_fields

    class_fields : array of type xarray.core.dataarray.DataArray
        - combined and cleaned static CLASS vegetation and urban field data

    TEB_fieldnames : array of type string
        - Names of the static driving fields for TEB

    TEB_geophys : xarray.core.dataarray.DataArray
        - combined and cleaned static TEB driving data

    static_fields_C, static_fields_T : xarray.core.dataarray.DataArray
        - Fixed fields used in CLASS and CLASS+TEB simulations respectively
    
    is_rural, is_urban : xarray.core.dataarray.DataArray
        - Boolean mask for urban (>50% urban fraction) and rural (<1% urban fraction) areas

    pavics : xarray.core.dataset.Dataset
        - ECCC station data available on pavics

    stations : xarray.core.dataset.Dataset
        - pavics subset within the simulation domain

    urban_stations, rural_stations : xarray.core.dataset
        - Stations within the domain masked by urban/reduced-urban fraction  

    stand_chunk : Dict
        - 91-time unit chunking applied in time and 280 standard grid units applied in space
    

Public functions:

    add_map_features(plt) : matplotlib.pyplot.subplot
        - Takes a matplotlib.pyplot, adds relevant cartopy borders, lakes, rivers, and costline features.

    get_outputs : xarray.core.dataarray.DataArray 
        - Returns the DataArray(s) corresponding to the CLASS alone or CLASS+TEB output respectively

    standard_rechunk : xarray.core.dataarray.DataArray
        - Returns the DataArray with stand_chunk applied to all variables which can accept it

    draw_map : folium.folium.Map
        - Focuses on the simulation area, draws an instantaneous field

    draw_map_layers : folium.folium.Map
        - Focuses on the simulation area 
        - Draws a set of fields and includes colourbars as floating toggled by the selected field layer 
        
    save_zarr(ds) : None
        - Takes an xarray.core.dataset.DataArray and saves to zarr
        
"""
from glob import glob
import numpy as np
from numcodecs import Blosc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import rcParams
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import folium
from folium.raster_layers import ImageOverlay
from folium import FeatureGroup
from folium.plugins import FloatImage, GroupedLayerControl
from branca.element import Element
from folium.elements import MacroElement
from jinja2 import Template
import re
import io
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cmocean
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
import xarray as xr
from scipy.ndimage import gaussian_filter

# Set matplotlib
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Open Sans']

"""
Section 1. Static and dynamic geospatial fields
"""
out_dir_C = '/runoff/gulley/St_Laurent/StLaurent_1km_SL2.5_ERA5_advHU/'
out_dir_T = '/runoff/gulley/St_Laurent/StLaurent_1km_SL2.5_ERA5_advHU_TEB/'
intermediates_dir = '/runoff/gulley/St_Laurent/intermediates'

rotated_pole = ccrs.RotatedPole(pole_longitude=106.425, pole_latitude=44.5)

field_keys = ['tas','tasmax','tasmin','hrss','hfss','hfls']
veg_fields = xr.open_mfdataset('/runoff/gulley/St_Laurent/StLaurent_1km_SL2.5_ERA5_advHU/Fix_Fields/StLaurent_1km_SL2.5_ERA5_advHU_step0.nc')['furban'].assign_attrs({'long_name':'Vegetation Fields','standard_name':'veg_fields'}).rename('veg_fields')

# Populate the dictionary holding fixed fields from the {experiment_name - TEB or CLASS+TEB}/Fix_Fields
static_fields_C = {}
static_fields_T = {}
for f in glob(f'{out_dir_C}/Fix_Fields/*.nc'):
    fix_field_key = f[f.rfind('advHU_')+6:f.rfind('.nc')]
    try :
        static_fields_C[fix_field_key] = xr.open_mfdataset(f)[fix_field_key]
    except KeyError:
        static_fields_C[fix_field_key] = xr.open_mfdataset(f)

static_fields_C['urban_frac'].attrs['units'] = '0-1' # A small fix
for f in glob(f'{out_dir_T}/Fix_Fields/*.nc'):
    fix_field_key = f[f.rfind('TEB_')+4:f.rfind('.nc')]
    try:
        static_fields_T[fix_field_key] = xr.open_dataset(f)[fix_field_key]
    except KeyError:
        static_fields_T[fix_field_key] = xr.open_mfdataset(f)
        
rlat = veg_fields['rlat'].values
rlon = veg_fields['rlon'].values
lats = veg_fields['lat'].values
lons = veg_fields['lon'].values
veg_levs = { '1':'salt water, ocean',
                '2':'glacier',
                '3':'inland lake',
                '4':'evergreen needle-leaf trees',
                '5':'evergreen broadleaf trees',
                '6':'deciduous needle-leaf trees',
                '7':'deciduous broadleaf trees',
                '8':'tropical broadleaf trees',
                '9':'drought deciduous trees',
                '10':'evergreen broadleaf shrubs',
                '11':'deciduous shrubs',
                '12':'thorn shrubs',
                '13':'short grass and forbs',
                '14':'long grass',
                '15':'crops',
                '16':'rice',
                '17':'sugar',
                '18':'maize',
                '19':'cotton',
                '20':'irrigated crops',
                '21':'urban',
                '22':'tundra',
                '23':'swamp',
                '24':'desert / bare soil',
                '25':'mixed wood forests',
                '26':'mixed shrubs'}

# Based on metric used by Michau et al. (2023) with nonzero water fractions below 5%
no_lakes =  veg_fields.sel(lev=3)  < 0.05
is_rural = (veg_fields.sel(lev=21) < 0.01).where(no_lakes)
is_urban = (veg_fields.sel(lev=21) > 0.50).where(no_lakes)
class_fields = []
for num_key in veg_levs.keys():
    # Format:
    # veg_fields.sel(lev='5.').rename('evergreen broadleaf trees')
    class_fields.append(veg_fields.sel(lev=num_key).rename(veg_levs[num_key]))
for i in range(len(class_fields)):
    class_fields[i].attrs['units'] = '0-1'

# Names of the static driving fields for TEB
TEB_fieldnames = ['natural_frac',
    'building_frac',
    'building_height',
    'road_frac',
    'roof_roughness',
    'road_roughness',
    'roof_albedo',
    'road_albedo',
    'wall_albedo',
    'roof_emiss',
    'road_emiss',
    'wall_emiss',
    'roof_heat_cap',
    'road_heat_cap',
    'wall_heat_cap',
    'roof_thermal_cond',
    'road_thermal_cond',
    'wall_thermal_cond',
    'roof_layer_depth',
    'road_layer_depth',
    'wall_layer_depth',
    'traffic_shfx',
    'traffic_lhfx',
    'industry_shfx',
    'industry_lhfx',
    'town_roughness',
    'ratio_vert_hor']
TEB_geophys = [] # Stores the static driving fields for TEB
for field in TEB_fieldnames:
    TEB_geophys.append(static_fields_T['geophys'][field])
field = None

def get_outputs(field,extension='.zarr',canopy='both'):
    """
    Attempts to load the CLASS alone and CLASS+TEB simulations into a dask-compatible DataArray

    Parameters:
        field : string
            - Name of the 4D field to load ['tas','huss','tasmax','tasmin','hfss','hfls']
        extension : string
            - .nc or .zarr files to load
        canopy : string
            - 'C' for CLASS alone, 'T' for CLASS+TEB, or 'both'
    
    Returns:
        ds_C : xarray.core.dataarray.DataArray
            - CLASS alone
        ds_T : xarray.core.dataarray.DataArray
            - CLASS+TEB
    
    Example usage:
        T_max_CLASS, T_max_TEB = get_outputs(field='tasmax',extension='.zarr',canopy='both')
        
    """

    if '.nc' in extension:
        # Searching for the appropriate folders to load
        path_C = f'{out_dir_C}*{field}*{extension}'
        path_T = f'{out_dir_T}*{field}*{extension}'
        match canopy:
            case 'both':
                return xr.open_mfdataset(path_C)[field].sel(time=slice('2000','2023')),xr.open_mfdataset(path_T)[field].sel(time=slice('2000','2023'))
            case 'C':
                return xr.open_mfdataset(path_C)[field].sel(time=slice('2000','2023'))
            case 'T':
                return xr.open_mfdataset(path_T)[field].sel(time=slice('2000','2023'))
    elif '.zarr' in extension:
        path_C = f'{out_dir_C}StLaurent_1km_SL2.5_ERA5_advHU_{field}.zarr'
        path_T = f'{out_dir_T}StLaurent_1km_SL2.5_ERA5_advHU_TEB_{field}.zarr'
        match canopy:
            case 'both':
                return xr.open_zarr(path_C)[field].sel(time=slice('2000','2023')),xr.open_zarr(path_T)[field].sel(time=slice('2000','2023'))
            case 'C':
                return xr.open_zarr(path_C)[field].sel(time=slice('2000','2023'))
            case 'T':
                return xr.open_zarr(path_T)[field].sel(time=slice('2000','2023'))
    else:
        print('Unsupported file type.')
        return -1

"""
Section 2. Map generation, ECC_AHCCD_gen3 station data, and related variables
"""
# Map formatting properties for field projection
# bbox = [[44.23337936401367, -75.53164672851562], [46.7599983215332, -71.8677978515625]]
lat_min = min(lats.flatten())
lat_max = max(lats.flatten())
lon_min = min(lons.flatten())
lon_max = max(lons.flatten())

centre_lat = (lat_min + lat_max) / 2
centre_lon = (lon_min + lon_max) / 2
bounds = [[lat_min, lon_min], [lat_max, lon_max]]
extent = [lon_min, lon_max, lat_min, lat_max]

# Load ECCC_AHCCD_gen3_temperature data for stations in Canada
try : 
    pavics = xr.open_dataset("https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/dodsC/datasets/station_obs/ECCC_AHCCD_gen3_temperature.ncml")
    # Restrict the selection by masking for only stations within the domain
    station_is_in_domain = (
        (pavics.lat>44.23337936401367)    & 
        (pavics.lat <46.7599983215332)    & 
        (pavics.lon > -75.53164672851562) & 
        (pavics.lon < -71.8677978515625)
    )
    
    # Set the stations into a pandas dataframe
    stations = pavics.sel(station=station_is_in_domain).set_coords(['lat', 'lon', 'station_name'])
    # To project station data onto the map
    gdf = pd.read_pickle('/runoff/gulley/misc/polygons_gdf.pkl')
    patches = [MplPolygon(np.array(poly.exterior.coords), closed=True) for poly in gdf.geometry]
    
    station_locations = stations[['lat', 'lon', 'station_name']].to_dataframe().reset_index()
    geojson_stations = gpd.GeoDataFrame(
        station_locations, geometry=gpd.points_from_xy(station_locations['lon'], station_locations['lat'])
    ).to_json()  

    """
    Using the lat/lon pairs given by the pavics stations (stations xarray dataset) within the domain: 
        1. Interpolate to the closest rlat/rlon grid point in the simulation
        2. Compare this point with the blurred urban fraction field, smoothing out stark discontinuities
            2.1 A simple Gaussian blur with a 1.5-cell standard deviation
        3. Classify station as urban or reduced urban (ie rural) with >=0.5 and <0.01 respectively
        4. Assign coordinate of the mask, as well as the urban fraction itself to each station set
        
    Using this method on the PAVICS dataset, on 2025-09-01, this gives:
        - 2 urban stations
        - 14 suburban stations
        - 21 rural stations

    This method has been generalised: 
        Any field can be appended to a station dataset simply by using 
        add_field_to_stations or add_blurred_field_to_stations 
    """
    station_rotated_points = rotated_pole.transform_points(ccrs.PlateCarree(), stations['lon'].values, stations['lat'].values)
    station_rlon = station_rotated_points[:, 0]
    station_rlat = station_rotated_points[:, 1]

    def gaussian_blur_xarray(da,sigma=1.5):
        """
        Parameters: 
            da - xarray.DataArray of a particular field
            sigma - float standard deviation
    
        Returns: 
            xarray.DataArray - Gaussian blur of da 
        
        Smoothing filter that replaces each value with a Gaussian-weighted average of its (original da)
        neighbouring cells. Full width half max of the blurring weight is sqrt(2 ln2)*sigma in # of cells.
        """
        blurred = gaussian_filter(da.values, sigma=sigma, mode='nearest')
        return xr.DataArray(
            blurred,
            dims=da.dims,
            coords=da.coords,
            name=f'{da.name}_blurred_std{str(sigma).replace('.','p')}'
        )
    urban_fraction = veg_fields.sel(lev='21').rename('urban_fraction')
    blurred_urban_fraction = gaussian_blur_xarray(urban_fraction,sigma=1.5)
    lake_fraction = veg_fields.sel(lev='3').rename('lake_fraction')
    
    def add_field_to_stations(da,stations=stations,name=None,method='nearest'):
        """
        Sample a model field at station locations and attach it as a coordinate.
    
        Parameters
        ----------
        da : xarray.DataArray
            The gridded model field to sample from (must have 'rlat' and 'rlon' coordinates).
        stations : xarray.Dataset or DataArray, optional
            Station dataset with a 'station' dimension. Default is global stations.
        name : str, optional
            Name to give the new coordinate. If None, uses da.name.
        method : {"nearest", "linear"}, default "nearest"
            Interpolation method passed to xarray.DataArray.sel.
    
        Returns
        -------
        stations_with_field : xarray.Dataset or DataArray
            Copy of stations with a new coordinate named name
            containing values of da interpolated at station locations.
        """
        
        if name == None:
            name = da.name.replace(' ','_').replace(',','').replace('/','or').replace('.','')
            
        field_at_station = da.sel(
            rlat=xr.DataArray(station_rlat, dims='points'),
            rlon=xr.DataArray(station_rlon, dims='points'),
            method=method
        ).values
        # return stations.assign_coords(
        #     field_at_station=("station", np.asarray(field_at_station))
        # ).rename(field_at_station=name)
        return stations.assign_coords(
            {name: ('station', np.asarray(field_at_station))}
        )
    
    def add_blurred_field_to_stations(da,stations,name=None,method='nearest',sigma=1.5):
        """
        Sample a Gaussian-smoothed model field at station locations.
    
        Parameters
        ----------
        da : xarray.DataArray
            The gridded model field to sample from.
        stations : xarray.Dataset or DataArray, optional
            Station dataset with a 'station' dimension.
        name : str, optional
            Name to give the new coordinate. If None, uses da.name.
        method : {"nearest", "linear"}, default "nearest"
            Interpolation method passed to xarray.DataArray.sel.
        sigma : float, default 1.5
            Standard deviation (in grid cells) for Gaussian smoothing
            applied before sampling.
    
        Returns
        -------
        stations_with_field : xarray.Dataset or DataArray
            Copy of stations with a new coordinate named name
            containing values of the blurred da at station locations.
    
        Notes
        -----
        - Wraps add_field_to_stations after applying
          gaussian_blur_xarray(da, sigma).
        - Useful for including non-local context around each station
          when sampling model fields.
        """
        return add_field_to_stations(gaussian_blur_xarray(da,sigma=sigma),stations,name=name,method=method)

    # stations = add_field_to_stations(urban_fraction,stations=stations)
    # stations = add_field_to_stations(lake_fraction,stations=stations)

    # 1.5 stdev blurring
    stations = add_blurred_field_to_stations(urban_fraction,sigma=1.5,stations=stations)
    stations = add_blurred_field_to_stations(lake_fraction,sigma=1.5,stations=stations)

    # Classify stations based on blurred urban fraction and blurred lake fraction
    dry_stations = stations.where(stations.lake_fraction_blurred_std1p5 < 0.10,drop=True)
    
    urban_stations = dry_stations.where(dry_stations.urban_fraction_blurred_std1p5 > 0.5, drop=True)
    suburban_stations = dry_stations.where(dry_stations.urban_fraction_blurred_std1p5 <= 0.5,drop=True).where(dry_stations.urban_fraction_blurred_std1p5 >0.01,drop=True)
    rural_stations = dry_stations.where(dry_stations.urban_fraction_blurred_std1p5 < 0.01,drop=True)

    
except OSError:
    print('Error loading PAVICS')
    stations = []

def pad_list(lst, min_length, default_value=None):
    """Ensure lst has at least min_length items by appending default_value."""
    return lst + [default_value] * (min_length - len(lst)) if len(lst) < min_length else lst

def draw_map(field=None,cmap_name='bwr',vmin=None,vmax=None,num_levels=None,stations_visible=False,opacity=0.9,cmocean_CMAP=None):
    """
    Creates a map and displays the station_locations within the bounds of the simulation on it. 
    It then generates an ImageOverlay of some the static field generated.

    parameters:
        field - xarray.core.dataarray.DataArray
            2D temperature, humidity, or other data 
        cmap - string
            matplotlib colormap indicator
        vmin, vmax - float 
            minimum and maximum 
    returns:
        m - folium.folium.Map

    Example usage:
        > f = get_outputs(field='tasmax',extension='.zarr',canopy='C').sel(time='1999-03-08')
        > m = draw_map(f)
        > display(m)
    """
    
    # Initialize the map
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=8)

    folium.TileLayer('cartodb positron').add_to(m)

    if stations_visible:
        # Add station markers
        folium.GeoJson(
            geojson_stations,
            popup=folium.GeoJsonPopup(fields=['station_name'], aliases=['Station Name']),
            marker=folium.CircleMarker(radius=3, color='grey', fill=True, fill_color='grey', fill_opacity=opacity)
        ).add_to(m)

    # Overlay field data
    if field is not None:
        # Create an image to store the field data
        image_buffer = io.BytesIO()

        # Field properties
        norm = Normalize(vmin=vmin, vmax=vmax)

        
        cmap = plt.get_cmap(cmap_name)
        if cmocean_CMAP is not None:
            cmap = cmocean_CMAP
        sm = ScalarMappable(cmap=cmap, norm=norm)
        
        # Create the figure to save to the map
        fig, ax = plt.subplots(figsize=(6, 6), dpi=800)
        
        bm = Basemap(projection='merc', 
            llcrnrlat=lat_min, 
            urcrnrlat=lat_max, 
            llcrnrlon=lon_min, 
            urcrnrlon=lon_max,ax=ax)

        # This works fine, but it inserts resolution where none should exist!
        # x, y = bm(lons, lats)
        # bm.contourf(x, y, field.values, cmap=cmap_name,vmin= vmin,vmax= vmax, levels=num_levels,antialiased=False)
        # XOR
        # bm.pcolormesh(x, y, field.values, cmap=cmap_name,vmin= vmin,vmax= vmax)
        
        flat_field = np.ravel(field.values)
        gdf['field'] = flat_field

        fig, ax = plt.subplots(figsize=(8, 8))
        collection = PatchCollection(
            patches,
            linewidth=0,
            edgecolor='none',
            antialiased=False
        )
        collection.set_array(gdf['field'].values)
        collection.set_cmap(cmap)
        collection.set_clim(vmin, vmax)
        
        
        ax.add_collection(collection)
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.axis('off')

        minx, miny, maxx, maxy = gdf.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.axis('off')  # Remove axes for clean image
        
        # Save image to buffer
        plt.savefig(image_buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        plt.close()
        image_buffer.seek(0)
        image_base64 = base64.b64encode(image_buffer.read()).decode()
        image_uri = f"data:image/png;base64,{image_base64}"
        
        # Read image from buffer and project onto map
        ImageOverlay(
            image=image_uri,
            bounds=bounds,
            opacity=0.6,
            interactive=True,
            cross_origin=False,
            pixelated=True
        ).add_to(m)
    
    m.fit_bounds(bounds)
    return m


class FloatImageWithID(MacroElement):
    """
    Allows Folium maps to have MacroElements associated with a particular ID for displaying the colorbars on fields.
    """
    
    _template = Template("""
        {% macro script(this, kwargs) %}
        var img = L.DomUtil.create('img', '');
        img.setAttribute('src', '{{ this.image }}');
        img.style.width = '{{ this.width }}';
        img.style.height = '{{ this.height }}';

        var div = L.DomUtil.create('div', 'leaflet-control float-image');
        div.id = '{{ this.div_id }}';
        div.style.display = 'none';  // hide on load
        div.appendChild(img);

        L.Control.FloatImage{{ this.div_id | replace('-', '_') }} = L.Control.extend({
            onAdd: function(map) {
                return div;
            },
            onRemove: function(map) {}
        });

        L.control.floatImage{{ this.div_id | replace('-', '_') }} = function(opts) {
            return new L.Control.FloatImage{{ this.div_id | replace('-', '_') }}(opts);
        };

        L.control.floatImage{{ this.div_id | replace('-', '_') }}({ position: '{{ this.position }}' }).addTo({{ this._parent.get_name() }});
        {% endmacro %}
    """)

    def __init__(self, image, position='bottomleft', width='auto', height='auto', div_id='float-image'):
        super().__init__()
        self._name = 'FloatImage'
        self.image = image
        self.position = position
        self.width = width
        self.height = height
        self.div_id = div_id
        
class ColorbarToggleScript(MacroElement):
    """
    Allows insertion of Folium MacroElement into displayed maps in order to load colorbars when clicking on corresponding field.
    """
    def __init__(self, field_names):
        super().__init__()
        self._name = 'ColorbarToggleScript'
        self._template = Template(f"""
            {{% macro script(this, kwargs) %}}
            function findMap(retries = 10, delay = 300) {{
                return new Promise((resolve, reject) => {{
                    let attempts = 0;
                    function tryFind() {{
                        const map = Object.values(window).find(obj => obj instanceof L.Map);
                        if (map) {{
                            resolve(map);
                        }} else if (++attempts < retries) {{
                            setTimeout(tryFind, delay);
                        }} else {{
                            console.error("Map object not found after retries.");
                            reject("Map not found");
                        }}
                    }}
                    tryFind();
                }});
            }}

            findMap().then((map) => {{
                console.log("Map found!", map);

                const colorbarMap = {{
                    {', '.join([f'"{name}": "colorbar-{name}"' for name in field_names])}
                }};

                function show(id) {{
                    const el = document.getElementById(id);
                    if (el) {{
                        el.style.display = "block";
                        console.log("SHOWING", id);
                    }}
                }}

                function hide(id) {{
                    const el = document.getElementById(id);
                    if (el) {{
                        el.style.display = "none";
                        console.log("HIDING", id);
                    }}
                }}

                Object.values(colorbarMap).forEach(hide);

                map.on('overlayadd', (e) => {{
                    console.log("overlayadd:", e.name);
                    const id = colorbarMap[e.name];
                    if (id) show(id);
                }});

                map.on('overlayremove', (e) => {{
                    console.log("overlayremove:", e.name);
                    const id = colorbarMap[e.name];
                    if (id) hide(id);
                }});

                document.querySelectorAll(
                    ".leaflet-control-layers-group input[type='checkbox'], \
                     .leaflet-control-layers-group input[type='radio']"
                ).forEach((input) => {{
                    input.addEventListener('change', function () {{
                        const label = this.closest("label");
                        if (!label) return;

                        const name = label.innerText.trim();
                        const eventType = this.checked ? "overlayadd" : "overlayremove";
                        map.fire(eventType, {{ name: name }});
                    }});
                }});
            }});
            {{% endmacro %}}
        """)

def draw_map_layers(fields=[],cmap_name_array=[],vmins=[],vmaxs=[],num_level_array=[],stations_visible=True,opacity=0.7,cmocean_CMAP_array=[]):
    """
    Creates a map and displays the station_locations within the bounds of the simulation on it. 
    It then generates an ImageOverlay of some the static field generated.
    Example usage:
        > m = draw_map_layers([static_fields_C['orog'],static_fields_C['urban_frac']])
        > display(m)

    parameters:
        fields - xarray.core.dataarray.DataArray
            2D temperature, humidity, or other data 
        cmaps - string
            matplotlib colormap indicator
        vmin, vmax - float 
            minimum and maximum 
    returns:
        m - folium.folium.Map
    """
    # Ensure other parameters are the same length as fields
    cmap_name_array = pad_list(cmap_name_array,len(fields),'bwr')
    vmins = pad_list(vmins,len(fields),None)
    vmaxs = pad_list(vmaxs,len(fields),None)
    num_level_array = pad_list(num_level_array,len(fields),None)
    cmocean_CMAP_array = pad_list(cmocean_CMAP_array,len(fields),None)
    
    # Initialize the map
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=8)

    folium.TileLayer('cartodb positron').add_to(m)

    if stations_visible:
        # Add station markers
        folium.GeoJson(
            geojson_stations,
            popup=folium.GeoJsonPopup(fields=['station_name'], aliases=['Station Name']),
            marker=folium.CircleMarker(radius=3, color='grey', fill=True, fill_color='grey', fill_opacity=opacity),
            show=False,
            name='stations'
        ).add_to(m)

    # Overlay field data
    overlays = []
    field_names = []
    for field,vmin,vmax,cmap_name,num_levels,cmocean_CMAP in zip(fields,vmins,vmaxs,cmap_name_array,num_level_array,cmocean_CMAP_array):
        if field is not None:
            # Create an image to store the field data
            image_buffer = io.BytesIO()
            field_ravel = np.ravel(field.values)
            field.name = re.sub(r'\W|^(?=\d)', '_', field.name)
            field_names.append(field.name)
            if vmin == None:
                vmin = min(field_ravel)
            if vmax == None:
                vmax = max(field_ravel)
            
            # Colorbar properties
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(cmap_name)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            
            # Create the figure to save to the map
            fig, ax = plt.subplots(figsize=(8, 8))
            collection = PatchCollection(
                patches,
                linewidth=0,
                edgecolor='none',
                antialiased=False
            )
            collection.set_array(field_ravel)
            collection.set_cmap(cmap)
            collection.set_clim(vmin, vmax)
            
            ax.add_collection(collection)
            ax.autoscale_view()
            ax.set_aspect('equal')
            ax.axis('off')
    
            minx, miny, maxx, maxy = gdf.total_bounds
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)    
            ax.axis('off')
            # Save image to buffer
            plt.savefig(image_buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
            plt.close()
            image_buffer.seek(0)
            image_base64 = base64.b64encode(image_buffer.read()).decode()
            image_uri = f"data:image/png;base64,{image_base64}"
            
            overlays.append(ImageOverlay(
                image=image_uri,
                bounds=bounds,
                opacity=opacity,
                interactive=True,
                cross_origin=False,
                pixelated=True,
                show=False,
                name=field.name,
                group='Field'
            ))
            overlays[-1].add_to(m)
            
            # Add the colourbar
            fig, ax = plt.subplots(figsize=(4, 0.2))  # Width x Height in inches

            # Create the colorbar from ScalarMappable
            cb = fig.colorbar(sm, cax=ax,orientation='horizontal')
            cb.set_ticks([vmin,vmax])
            
            cb.set_label(f'{field.name} ( {field.attrs['units']})')
            
            # Save colorbar to buffer
            image_buffer_cbar = io.BytesIO()
            plt.savefig(image_buffer_cbar, format='png',bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
            image_buffer_cbar.seek(0)
            image_base64 = base64.b64encode(image_buffer_cbar.read()).decode()
            image_uri_cbar = f"data:image/png;base64,{image_base64}"

            # FloatImage(image_uri_cbar, bottom=90+adjB, left=5).add_to(m)
            FloatImageWithID(
                image=image_uri_cbar,
                div_id=f"colorbar-{field.name}"
            ).add_to(m)

           
    
    folium.LayerControl(collapsed=False).add_to(m)
    overlays.append(
        folium.TileLayer(
        tiles='',
        name='Clear',
        overlay=True,
        attr='none',
        control=True
    ))
    overlays[-1].add_to(m)
    
    GroupedLayerControl(groups={
                'Fields': overlays,
            }, collapsed=False).add_to(m)
    m.fit_bounds(bounds)
    m.get_root().add_child(ColorbarToggleScript(field_names))
    return m

def plot_field(field,cmap='viridis',levels=None,vmin=None,vmax=None,cbar_label='',title='',proj=None,feature_colours='grey',extend='both',spacing='proportional',bins=None,labelsize=6):
    """
    Plots a 2D xarray DataArray field over a map with a discrete colorbar and regional political and lake features.
    
    Parameters
    ----------
    field : xarray.DataArray
        2D field to plot over the map domain.
    cmap : str
        Name of the matplotlib colormap to use.
    vmin, vmax : float
        Minimum and maximum values for color normalization.
    cbar_label : str
        Label for the colorbar.
    title : str
        Title to display above the map.
    proj : cartopy.crs.Projection
        Cartopy coordinate reference system (CRS) for the map projection.
    feature_colours : str
        Colour for map features such as coastlines, gridlines, and borders.
    extend : str
        Colorbar extension type. Options: 'min', 'max', 'both', or 'neither'.
    spacing : str
        Colorbar spacing mode. Options: 'uniform' or 'proportional'.
    bins : list of float
        Discrete bin edges to categorize field values for the colormap.
    labelsize : float
        Font size for colorbar tick labels.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the map.
    ax : cartopy.mpl.geoaxes.GeoAxes
        The Cartopy GeoAxes on which the data are plotted.
    cb : matplotlib.colorbar.Colorbar
        The colorbar associated with the plot.
    
    Notes
    -----
    - If bins is not provided, they can be inferred using vmin, vmax, and number of levels.
    - Designed for use with CLASS lake features compatible with Cartopy projections.
    """

    
    if proj == None:
        proj = ccrs.NearsidePerspective(central_longitude=centre_lon, central_latitude=centre_lat)
        # proj = ccrs.RotatedPole(pole_longitude=106.425, pole_latitude=44.5)
        # proj = ccrs.Orthographic(central_longitude=centre_lon, central_latitude=centre_lat)
    
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection': proj})
    
    # Add features from smoothed CLASS contours
    # This was selected arbitrarily until a map emerged that was regionally recognisable
    lake_features = ax.contour(lons,lats,veg_fields.sel(lev='3'),transform=ccrs.PlateCarree(),
                               levels=[0.2,1.0],#np.linspace(0.2,1.0,1),
                               colors=feature_colours,zorder=2,linewidths=[0.5])
    ax.add_feature(cfeature.BORDERS,edgecolor=feature_colours,linewidth=0.5,zorder=2)
    
    if levels is None:
        levels = 10
    if vmin is None:
        vmin = np.nanmin(field) 
    if vmax is None:
        vmax = np.nanmax(field)
    if bins == None: # Bins overrides vmin,vmax,levels if not None
        bins = np.linspace(vmin, vmax, levels + 1)
        
    # Colourbar is discrete and ranged based on vmin and vmax
    cmap = plt.get_cmap(cmap,levels)
    cbar_norm = mpl.colors.BoundaryNorm(bins, cmap.N)

    # Holds the field itself
    mesh = ax.pcolormesh(lons, lats, field, transform=ccrs.PlateCarree(),cmap=cmap,norm=cbar_norm,zorder=1, rasterized=True)
    cb = fig.colorbar(ScalarMappable(norm=cbar_norm, cmap=cmap),ax=ax,
                      spacing=spacing,
                      orientation='vertical',
                      extend=extend,
                      shrink=0.73)
    cb.set_label(label=cbar_label,
                 rotation=0)
                 # labelpad=20)
    cb.ax.tick_params(labelsize=labelsize)
    
    if extend=='both' or extend=='max':
        cb.ax.yaxis.set_label_coords(0.5, 1.08) 
    else:
        cb.ax.yaxis.set_label_coords(0.5, 1.03) 

    # Gridline configuration
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color=feature_colours, linestyle='--')
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 1))  # longitude ticks
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 1))    # latitude ticks
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    ax.axis('off')
    fig.tight_layout()

    ax.add_artist(ScaleBar(1,box_alpha=0,color=feature_colours,location='lower right'))

    plt.title(title)
    return fig,ax,cb


def plot_stations(fig,ax,stations=stations,field_fontcolour='black',field_fontsize=6,features_colour='grey',features_fontsize=5,sigfigs=2,field=None,linear_field=None,units=''):
    """
    Annotates a Cartopy map with station names and optional numerical field values.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object containing the map.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object with a Cartopy map projection.
    stations : xarray.Dataset
        A dataset containing station metadata, with required coordinates:
        - stations.lon : array-like, station longitudes
        - stations.lat : array-like, station latitudes
        - stations.station_name : array-like, station name strings
    field : array-like or None, optional
        Optional numeric values to annotate below each station marker.
        If None, no field values are shown, or linear_field values are shown.
    field_fontcolour : str, optional
        Colour used for the numeric field text (default: 'black').
    field_fontsize : int, optional
        Font size for the field value annotations (default: 6).
    features_colour : str, optional
        Colour used for the station name labels (default: 'grey').
    features_fontsize : int, optional
        Font size for station name labels (default: 5).
    sigfigs : int, optional
        Number of significant figures for the field values (default: 2).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The input figure, with station annotations added.
    ax : matplotlib.axes._subplots.AxesSubplot
        The input axes, with text labels added.

    Notes:
    ------
    - Station names are positioned slightly above each marker, with a longitude offset applied 
      to prevent text overflow near the eastern edge of the map.
    - Field values are shown slightly below each station marker if provided.
    """
    station_rotated_points = rotated_pole.transform_points(ccrs.PlateCarree(), stations['lon'].values, stations['lat'].values)
    station_rlon = station_rotated_points[:, 0]
    station_rlat = station_rotated_points[:, 1]

    if field is None:
        field = [None for i in range(len(stations.station_name))]
    else:
        field = field.sel(rlat=xr.DataArray(station_rlat, dims='points'),rlon=xr.DataArray(station_rlon, dims='points'),method='nearest').values
        
    # A linear_field is just one that is simply tied to the stations, a field is tied to the grid
    if linear_field is not None:
        field = linear_field
    
    for lon, lat, name,field_value in zip(stations.lon.values, stations.lat.values, stations.station_name.values, field):

        if field_value is not None:
            label = ''
            if type(field_value) != str:
                # Label the stations on the map with the field value
                label = f'{round(field_value*10**sigfigs)/(10**sigfigs)}{units}'
            else:
                label = field_value
                
            ax.text(lon, lat-0.02, label, transform=ccrs.PlateCarree(),
                        ha='center', va='top', fontsize=field_fontsize,color=field_fontcolour)
    
        # Display the name of the station wihout allowing the Eastmost/Northmost edge to have words spilling out
        x_offset=0
        if lon>-72:
            x_offset=0.10
        y_offset=0
        if lat>46.70:
            y_offset=0.01
        label = name
        ax.text(lon-x_offset, lat+0.01-y_offset, label, transform=ccrs.PlateCarree(),
                ha='center', va='bottom', fontsize=features_fontsize,color=features_colour)
    
    return fig,ax

"""
Section 3. Chunking and file conversion tools
"""
stand_chunk = {'time': 91, 'rlat': 280, 'rlon': 280}
def standard_rechunk(ds,verbose=False,standard_chunk=stand_chunk):
    """
    Chunks an xarray Dataset or DataArray for Zarr compatibility.
    
    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset to be chunked in-place for Zarr storage.
    verbose : bool, optional
        If True, prints detailed information about the chunking process (default: False).
    standard_chunk : dict, optional
        A dictionary defining the target chunk sizes (e.g., {'x': 280, 'y': 280, 'time': 48}).
    
    Returns
    -------
    ds_chunked : xarray.Dataset or xarray.DataArray
        A version of the input dataset with adjusted chunking.
    """

    ds.encoding = {
        'scale_factor': 1.0,
        'add_offset': 0.,
        '_FillValue': 0,
        'dtype': 'float64'
    }
    
    if type(ds) == xr.core.dataarray.Dataset:
        fields = []
        for var in ds.variables:
            try:
                ds[var] = ds[var].chunk(standard_chunk)
                fields.append(var)
                if verbose:
                    print(f'{var} : rechunked as {ds[var].chunks}')
            except ValueError: 
                # Variable lacks at least one dimension of the standard_chunk
                if verbose:
                    print(f'{var} : not rechunked, kept as {ds[var].chunks}')
                pass
    elif type(ds) == xr.core.dataarray.DataArray:
        ds = ds.chunk(standard_chunk) # Well that was easier than I thought it would be
    else:
        print('Dataset type not recognised or implemented for this tool, not rechunked.')
    return ds

def save_zarr(ds,canopy=None,store=None,mode='w-',region=None,append_dim='time'):
    """
    Attempts to save a Dask-backed xarray DataArray in Zarr format.
    
    Parameters
    ----------
    ds : xarray.DataArray
        The data to be saved in Zarr format.
    
    store : str, path-like, or MutableMapping, optional
        Target location for the Zarr store. Can be a directory path (local or remote) or a Zarr-compatible storage object.
        If not provided, a default path will be inferred based on the canopy argument.
    
    canopy : str
        If store is not specified, determines the default directory based on canopy type:
        - 'C' for CLASS
        - 'T' for CLASS + TEB
    
    mode : str, optional
        Zarr write mode. Options:
        - 'w'   : overwrite existing store
        - 'w-'  : write only if store does not exist (default)
        - 'a'   : append to existing store
        - 'a-'  : append only if store exists
        - 'r+'  : read/write, must exist
    
    Returns
    -------
    None
    
    Notes
    -----
    - This function is intended for use with chunked (Dask-backed) arrays.
    - ds will be chunked appropriately, if not already, using standard_rechunk().
    """

    
    # In the event that a store location was not specified
    if store==None:
        # save based on canopy type
        if canopy == 'C':
            store = f'{out_dir_C}StLaurent_1km_SL2.5_ERA5_advHU_{ds.name}.zarr'
        elif canopy == 'T':
            store = f'{out_dir_T}StLaurent_1km_SL2.5_ERA5_advHU_TEB_{ds.name}.zarr'
        else: # save to current working directory as untitled.zarr
            store = 'untitled.zarr'
    try:
        ds.chunksizes
    except ValueError:
        print('Dataset\'s chunking is not zarr compliant, automatically attempting rechunk...')
        ds = standard_rechunk(ds)
        print(f'Chunks:\n {ds.chunksizes}')

        # Ensures no encoding issues or inconsistent data
    ds.encoding = {
        'scale_factor': 1.0,
        'add_offset': 0.,
        'compressor': Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
        'dtype': 'float64'
    }
    
    # Attempt to save ds by appending to time, if this doesn't work, the .zarr file must be created
    try:
        ds.to_zarr(store, mode=mode)#, append_dim=append_dim)
    except ValueError:
        # ds.to_zarr(store, mode='w')
        if region is not None:
            ds.to_zarr(store, mode='r+', region=region)
        else:
            ds.to_zarr(store, mode='r+')


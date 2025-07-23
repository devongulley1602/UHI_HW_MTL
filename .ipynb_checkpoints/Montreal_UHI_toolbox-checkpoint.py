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

    stand_chunk : Dict
        - 91-time unit chunking applied in time and 280 standard grid units applied in space

    pavics : xarray.core.dataset.Dataset
        - ECCC station data available on pavics

    stations : xarray.core.dataset.Dataset
        - pavics subset within the simulation domain
    

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
import matplotlib.pyplot as plt
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
import cmocean
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
import xarray as xr
"""
Section 1. Static and dynamic geospatial fields
"""
out_dir_C = '/runoff/gulley/St_Laurent/StLaurent_1km_SL2.5_ERA5_advHU/'
out_dir_T = '/runoff/gulley/St_Laurent/StLaurent_1km_SL2.5_ERA5_advHU_TEB/'
intermediates_dir = '/runoff/gulley/St_Laurent/intermediates'

rotated_pole = ccrs.RotatedPole(pole_longitude=106.425, pole_latitude=44.5)

field_keys = ['tas','tasmax','tasmin','hrss','hfss','hfls']
veg_fields = xr.open_mfdataset('/runoff/gulley/St_Laurent/StLaurent_1km_SL2.5_ERA5_advHU/Fix_Fields/StLaurent_1km_SL2.5_ERA5_advHU_step0.nc')['furban']#.assign_attrs({'long_name':'Static Fields', 'standard_name':'static_fields'})


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
is_rural = veg_fields.sel(lev=21) < 0.01 # Based on metric used by Roberge and Sushama (2018)
is_urban = veg_fields.sel(lev=21) > 0.5
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
except OSError:
    print('Error loading PAVICS')

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


def add_map_features(plt):
    """
    Set axis to include map features as a subplot on a matplotlib.plot.
    
    Parameters:
        plt : matplotlib.pyplot
    
    Returns: 
        ax : matplotlib.pyplot.subplot
    
    Example usage:
        ax = add_map_features(plt)
    """
    ax = plt.subplot(projection=rotated_pole)
    ax.add_feature(cfeature.BORDERS,edgecolor='grey')
    ax.add_feature(cfeature.LAKES, edgecolor='grey', facecolor='none')
    ax.add_feature(cfeature.RIVERS, edgecolor='grey', facecolor='none')
    ax.add_feature(cfeature.COASTLINE,edgecolor='grey')
    return ax


"""
Section 3. Chunking and file conversion tools
"""
stand_chunk = {'time': 91, 'rlat': 280, 'rlon': 280}
def standard_rechunk(ds,verbose=False,standard_chunk=stand_chunk):
    """
    Chunks dataset for zarr saving compatibility (consistent chunks until final)
    
    parameters
        ds             : xarray Dataset/array - dataset to be converted/modified by value
        varbose        : boolean              - toggle to print detailed chunking information
        standard_chunk : dict                 - default chunked by 280x280 spatial domain and 4-year time domain
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
    Attempts to save dask-compatible DataArray in zarr format
    
    Parameters:
        
        ds    : xarray DataArray 
            - data to be saved in zarr format
            
        store : MutableMapping, str or path-like, optional 
            – Store or path to directory in local or remote file system
            
        canopy : string
            - If store not specified, uses default directory based on canopy type, 'C' for CLASS or 'T' for CLASS+TEB
        
        mode  : string
            - 'w', 'w-', 'a', 'a-', 'r+', None 
            - Default 'w-' write, fail if exists
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


from Montreal_UHI_toolbox import *
"""
Print all static driving fields for TEB selecting layer 1.0 if there are more than one layer. 
When there are more layers, saves a separate plot to view the difference between layers to later verify if this exists.

Note:
    Currently, the observed differences between layers are zero within floating point accuracy.
"""

TEB_geophys = []
TEB_layerd  = []
for field in TEB_fieldnames:
    try:
        # If we're dealing with multiple layers, simply take the last value, but note that it's layered
        for lev in static_fields_T['geophys'][field].lev.values:
            TEB_layered.append(static_fields_T['geophys'][field].rename(f'{field}_{lev}'))
        TEB_geophys.append(static_fields_T['geophys'].sel(lev=1.0)[field])
    except AttributeError:
        TEB_geophys.append(static_fields_T['geophys'][field])

cmap_names_layered = ['copper',  # roof_heat_cap_3.0
                'copper',  # roof_heat_cap_2.0
                'copper',  # roof_heat_cap_1.0
                'copper', # road_heat_cap_3.0
                'copper', # road_heat_cap_2.0
                'copper', # road_heat_cap_1.0
                'copper',  # wall_heat_cap_3.0
                'copper',  # wall_heat_cap_2.0
                'copper',  # wall_heat_cap_1.0
                'plasma', # roof_thermal_cond_3.0
                'plasma', # roof_thermal_cond_2.0
                'plasma', # roof_thermal_cond_1.0
                'plasma',  # road_thermal_cond_3.0
                'plasma',  # road_thermal_cond_2.0
                'plasma',  # road_thermal_cond_1.0
                'plasma', # wall_thermal_cond_3.0
                'plasma', # wall_thermal_cond_2.0
                'plasma', # wall_thermal_cond_1.0
                'Purples',   # roof_layer_depth_3.0
                'Purples',   # roof_layer_depth_2.0
                'Purples',   # roof_layer_depth_1.0
                'Purples',  # road_layer_depth_3.0
                'Purples',  # road_layer_depth_2.0
                'Purples',  # road_layer_depth_1.0
                'Purples',   # wall_layer_depth_3.0
                'Purples',   # wall_layer_depth_2.0
                'Purples',   # wall_layer_depth_1.0
            ]

cmap_names = cmap_names = [  'Greens',  # natural_frac
                'BuPu',   # building_frac
                'BuPu',   # building_height
                'BuPu',   # road_frac
                'Purples', # roof_roughness
                'Purples', # road_roughness
                'gist_gray',  # roof_albedo
                'gist_gray',  # road_albedo
                'gist_gray',  # wall_albedo
                'hot',    # roof_emiss
                'hot',    # road_emiss
                'hot',    # wall_emiss
                'copper',  # roof_heat_cap_1.0
                'copper', # road_heat_cap_1.0
                'copper',  # wall_heat_cap_1.0
                'plasma', # roof_thermal_cond_1.0
                'plasma',  # road_thermal_cond_1.0
                'plasma', # wall_thermal_cond_1.0
                'Purples',   # roof_layer_depth_1.0
                'Purples',  # road_layer_depth_1.0
                'Purples',   # wall_layer_depth_1.0
                'hot',    # traffic_shfx
                'hot',    # traffic_lhfx
                'hot',    # industry_shfx
                'hot',    # industry_lhfx
                'Purples', # town_roughness
                'RdBu'    # ratio_vert_hor
            ]

m = draw_map_layers(TEB_layered,cmap_name_array=cmap_names_layered)
m.save('/home/gulley/UHI_HW_MTL/graphs/TEB_layered_fields.html')

m = draw_map_layers(TEB_geophys,cmap_name_array=cmap_names)
m.save('/home/gulley/UHI_HW_MTL/graphs/TEB_fields.html')
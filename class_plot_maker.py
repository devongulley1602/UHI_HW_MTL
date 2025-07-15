from Montreal_UHI_toolbox import class_fields, draw_map_layers
cmap_name_array = [
    'Blues',  # Salt water, ocean
    'Blues',  # Glacier
    'Blues',  # Inland lake
    'Greens',  # Evergreen needle-leaf trees
    'Greens',  # Evergreen broadleaf trees
    'Greens',  # Deciduous needle-leaf trees
    'Greens',  # Deciduous broadleaf trees
    'Greens',  # Tropical broadleaf trees
    'Greens',  # Drought deciduous trees
    'Greens',  # Evergreen broadleaf shrubs
    'Greens',  # Deciduous shrubs
    'Greens',  # Thorn shrubs
    'YlGn',  # Short grass and forbs
    'YlGn',  # Long grass
    'YlGn',  # Crops
    'YlOrBr',  # Rice
    'YlOrBr',  # Sugar
    'YlOrBr',  # Maize
    'YlOrBr',  # Cotton
    'YlGn',  # Irrigated crops
    'BuPu',  # Urban
    'Greens',  # Tundra
    'Blues',  # Swamp
    'OrRd',  # Desert / bare soil
    'Greens',  # Mixed wood forests
    'Greens'   # Mixed shrubs
]
m = draw_map_layers(class_fields,cmap_name_array=cmap_name_array)
m.save('/home/gulley/UHI_HW_MTL/graphs/class_fields.html')

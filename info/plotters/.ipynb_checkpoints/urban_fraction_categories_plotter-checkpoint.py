from Montreal_UHI_toolbox import *
# Plot the field using cartopy and matplot
fig,ax,cb = plot_field(urban_fraction,
                       bins=[0.0,0.01,0.5,1.0],
                       extend='neither',
                       cmap='YlGn_r',
                       title='Urban Fraction Categories',
                       cbar_label='Fraction (0-1)')

# Add the stations and names to the plot with their corresponding urban fraction
fig,ax = plot_stations(fig,ax,field=stations.urban_fraction.values)

# Scatter each station kind on the map, red for urban, orange for suburban, green for rural
ax.scatter(urban_stations.lon, urban_stations.lat, transform=ccrs.PlateCarree(),
           color='red', marker='s',label='urban',s=3)
ax.scatter(subur_stations.lon, subur_stations.lat, transform=ccrs.PlateCarree(),
           color='orange', marker='s',label='suburban',s=3)
ax.scatter(rural_stations.lon, rural_stations.lat, transform=ccrs.PlateCarree(),
           color='yellow', marker='s',label='reduced urban',s=3)
ax.legend(title='Stations',alignment='left',framealpha=0.4)

# plt.show()
plt.savefig('/home/gulley/UHI_HW_MTL/info/plots/urban_fraction_categories.pdf', bbox_inches='tight')

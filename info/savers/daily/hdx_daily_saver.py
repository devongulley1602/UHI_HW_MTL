"""
Saves max and min daily humidex
"""
import Montreal_UHI_toolbox as tools
hdx_C,hdx_T = tools.get_outputs('hdx') # Retrieve 3-hourly gridded humidex output
# Resample for maximum daily humidex values
hdxmax_C = hdx_C.resample(time='1D').max().rename('hdxmax')
hdxmax_T = hdx_T.resample(time='1D').max().rename('hdxmax')
hdxmin_C = hdx_C.resample(time='1D').min().rename('hdxmin')
hdxmin_T = hdx_T.resample(time='1D').min().rename('hdxmin')
# Save the maximum daily humidex
tools.save_zarr(hdxmax_C,canopy='C')
tools.save_zarr(hdxmax_T,canopy='T')
tools.save_zarr(hdxmin_C,canopy='C')
tools.save_zarr(hdxmin_T,canopy='T')

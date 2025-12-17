"""
Loads xarray DataArray from .nc, saves as .zarr for faster retrieval another time 
"""
from Montreal_UHI_toolbox import save_zarr, get_outputs

da_C, da_T = get_outputs('hrss','.nc') # ['tas','huss','tasmax','tasmin','hfss','hfls']
save_zarr(da_C,'C')
save_zarr(da_T,'T')

from Montreal_UHI_toolbox import save_zarr, get_outputs
print('Loading data...')
da_C, da_T = get_outputs('hfss','.nc') # ['tas','tasmax','tasmin','hfss','hfls']
print('Saving to .zarr')
save_zarr(da_C,'C')
print('Saving to .zarr')
save_zarr(da_T,'T')

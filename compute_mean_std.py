# %%
import xarray as xr
import numpy as np

# %%
orography = xr.open_dataset('orography.nc').orog
zarr_store = 'data/NYSM.zarr'
dates_range = ['2023-01-01T00','2023-12-31T23']
variables = ['i10fg']
mask = xr.open_dataset('mask_2d.nc').mask

# %%
# Dictionary to hold results
stats = {'mean': {}, 'std': {}, 'min': {}, 'max': {}}

# Loop through each variable and compute stats
for variable in variables:
    ds = xr.open_zarr(zarr_store)[variable]
    ds = ds.sel(time=slice(dates_range[0], dates_range[1]))
    ds = ds.where(mask)
    
    print(f"Processing {variable} with {len(ds.time)} samples")
    mean = ds.mean(dim=['time', 'y', 'x'])
    std = ds.std(dim=['time', 'y', 'x'])
    min_val = ds.min(dim=['time', 'y', 'x'])
    max_val = ds.max(dim=['time', 'y', 'x'])
    
    stats['mean'][variable] = mean.values
    stats['std'][variable] = std.values
    stats['min'][variable] = min_val.values
    stats['max'][variable] = max_val.values

# %%
variable = 'orography'
ds = xr.open_dataset('orography.nc').orog
print(f"Processing {variable}")
mean = ds.mean(dim=[ 'y', 'x'])
std = ds.std(dim=['y', 'x'])
min_val = ds.min(dim=[ 'y', 'x'])
max_val = ds.max(dim=[ 'y', 'x'])

stats['mean'][variable] = mean.values
stats['std'][variable] = std.values
stats['min'][variable] = min_val.values
stats['max'][variable] = max_val.values

# %%
# Create xarray Dataset
stats_ds = xr.Dataset({
    'mean': xr.DataArray(
        data=[stats['mean'][v] for v in variables+['orography']],
        dims='variable',
        coords={'variable': variables+['orography']}
    ),
    'std': xr.DataArray(
        data=[stats['std'][v] for v in variables+['orography']],
        dims='variable',
        coords={'variable': variables+['orography']}
    ),
    'min': xr.DataArray(
        data=[stats['min'][v] for v in variables+['orography']],
        dims='variable',
        coords={'variable': variables+['orography']}
    ),
    'max': xr.DataArray(
        data=[stats['max'][v] for v in variables+['orography']],
        dims='variable',
        coords={'variable': variables+['orography']}
    )
})

# %%
# Save to NetCDF
stats_ds.to_netcdf('NYSM_variable_stats.nc')
print("Saved variable stats to RTMA_variable_stats.nc")
# %%

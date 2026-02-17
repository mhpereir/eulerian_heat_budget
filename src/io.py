'''
Docstring for eulerian_heat_budget.src.io

Responsibilities:

- Load datasets (ERA5, CMIP6, etc.)
- Harmonize variable names
- Enforce pressure units (Pa)
- Return standardized `xarray.Dataset`

Should not perform calculations.


### Required Dimensions

- `time`
- `p`
- `y`
- `x`

------

### Required Coordinates

- `p` [Pa]
- `lat(y)`
- `lon(x)`
- `time`

------

### Required Variables

| Variable | Dimensions   | Units  |
| -------- | ------------ | ------ |
| `T`      | (time,p,y,x) | K      |
| `u`      | (time,p,y,x) | m s⁻¹  |
| `v`      | (time,p,y,x) | m s⁻¹  |
| `w`      | (time,p,y,x) | m s⁻¹  |
| `sp`     | (time,y,x)   | Pa     |

------

### Required Physical Assumptions

- Hydrostatic balance implied.
- Pressure coordinate vertical axis.
- Surface is a material boundary.
- No mass flux across lower boundary.
'''

import xarray as xr
import numpy as np

from . import config


def load_era5_T(filepath: str) -> xr.Dataset:
    ds_T = xr.open_dataset(filepath)
    ds_T = ds_T.rename({'latitude': 'lat', 'longitude': 'lon'}) 
    ds_T = ds_T.rename({'t': 'T'})  # rename temperature variable to 'T' for consistency

    #check units are in Kelvin
    if 'units' in ds_T['T'].attrs:
        if ds_T['T'].attrs['units'] in ['K', 'kelvin', 'Kelvin']:
            pass  # already in Kelvin
        elif ds_T['T'].attrs['units'] in ['C', 'celsius', 'Celsius']:
            ds_T['T'] = ds_T['T'] + 273.15  # convert to Kelvin
            ds_T['T'].attrs['units'] = 'K'
        else:
            raise ValueError(f"Unexpected temperature units: {ds_T['T'].attrs['units']}")
    else:
        raise ValueError("Temperature variable 'T' must have 'units' attribute.")

    return ds_T

def load_era5_u(filepath: str) -> xr.Dataset:
    ds_u = xr.open_dataset(filepath)
    ds_u = ds_u.rename({'latitude': 'lat', 'longitude': 'lon'})

    return ds_u #[m/s]

def load_era5_omega(filepath: str) -> xr.Dataset:
    ds_omega = xr.open_dataset(filepath)
    ds_omega = ds_omega.rename({'latitude': 'lat', 'longitude': 'lon'})

    return ds_omega #[Pa/s]


def load_era5_sfp(filepath: str) -> xr.Dataset:
    ds_sfp = xr.open_dataset(filepath)
    ds_sfp = ds_sfp.rename({'latitude': 'lat', 'longitude': 'lon'})

    return ds_sfp #[Pa]

# def load_era5_zg(filepath: str) -> xr.Dataset:
#     ds_z = xr.open_dataset(filepath)
#     ds_z = ds_z.rename({'latitude': 'lat', 'longitude': 'lon'})

#     ds_zg = ds_z * config.g  # convert geopotential to geopotential height
#     ds_zg = ds_zg.rename({'z': 'zg'})  # rename variable to 'zg' for consistency

#     return ds_zg  #[m]


def load_era5_merge_dataset(ds_T, ds_u, ds_v, ds_w, ds_sp) -> xr.DataTree:
    # Merge all datasets into a single dataset
    merged = xr.merge([ds_T, ds_u, ds_v, ds_w, ds_sp], 
                      compat='identical') 

    #Ensure dimensions are in correct order
    expected_order = ['time', 'level', 'lat', 'lon']
    merged = merged.transpose(*expected_order)

    #Enforce lat/lon monotonic ascending
    if not merged['lat'].diff('lat').min() > 0:
        merged = merged.sortby('lat')
    if not merged['lon'].diff('lon').min() > 0:
        merged = merged.sortby('lon')

    #Enforce pressure coordinates are monotonic decreasing
    dlev = merged['level'].diff('level')
    if not ((dlev < 0).all() and merged['level'].to_index().is_unique):
        merged = merged.sortby('level', ascending=False)

    return merged
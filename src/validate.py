'''
Docstring for eulerian_heat_budget.src.validate

Strict schema validation:

- Required variables present
- Required dimensions present
- Pressure monotonic decreasing
- lat/lon monotonic ascending
- Units consistent
- Time coordinate regular
- Order units (time, level, lat, lon)

Raises errors if violated.

This prevents silent scientific errors.
'''

import xarray as xr

REQUIRED_DIMS = ("time", "level", "lat", "lon")
REQUIRED_VARS_4D = ("T", "u", "v", "w")
REQUIRED_VARS_3D = ("sp",)

def validate_schema(ds: xr.Dataset) -> None:
    # 1) Required dims exist (order irrelevant)
    missing_dims = [d for d in REQUIRED_DIMS if d not in ds.dims]
    if missing_dims:
        raise ValueError(f"Missing required dims: {missing_dims}")

    # 2) Required variables exist
    for v in (*REQUIRED_VARS_4D, *REQUIRED_VARS_3D):
        if v not in ds:
            raise ValueError(f"Missing required variable: {v}")

    # 3) Validate variable dimension order (this is what matters)
    expected_4d = REQUIRED_DIMS
    for v in REQUIRED_VARS_4D:
        if ds[v].dims != expected_4d:
            raise ValueError(
                f"Variable '{v}' must have dims {expected_4d}, got {ds[v].dims}"
            )

    expected_3d = ("time", "lat", "lon")
    for v in REQUIRED_VARS_3D:
        if ds[v].dims != expected_3d:
            raise ValueError(
                f"Variable '{v}' must have dims {expected_3d}, got {ds[v].dims}"
            )

    # 4) Use ds.sizes for lengths (future-stable)
    if ds.sizes["level"] < 1 or ds.sizes["lat"] < 1 or ds.sizes["lon"] < 1:
        raise ValueError("Dataset has an empty required dimension.")
    
    
    # Check pressure monotonicity
    if not ds['level'].diff('level').min() < 0:
        raise ValueError("Pressure levels must be monotonic decreasing")
    else:
        print("Pressure levels are monotonic decreasing.")

    # Check lat/lon monotonicity
    if not ds['lat'].diff('lat').min() > 0:
        raise ValueError("Latitude must be monotonic ascending")
    else:
        print("Latitude is monotonic ascending.")

    if not ds['lon'].diff('lon').min() > 0:
        raise ValueError("Longitude must be monotonic ascending")
    else:
        print("Longitude is monotonic ascending.")

    # Check units (example for temperature)
    if ds['T'].attrs.get('units') != 'K':
        raise ValueError("Temperature must be in Kelvin")
    else:
        print("Temperature units are consistent (K).")

    # Check time coordinate regularity
    time_diff = ds['time'].diff('time')
    if not time_diff.min() == time_diff.max():
        raise ValueError("Time coordinate must be regular (constant time step)")
    
    return None
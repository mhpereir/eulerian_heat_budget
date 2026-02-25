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
import numpy as np

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
        
    for coord in REQUIRED_DIMS:
        if ds[coord].dims != (coord,):
            raise ValueError(f"Coordinate '{coord}' must be 1D over itself.")

    # 4) Use ds.sizes for lengths
    if ds.sizes["level"] < 1 or ds.sizes["lat"] < 1 or ds.sizes["lon"] < 1:
        raise ValueError("Dataset has an empty required dimension.")
    

    level = ds["level"].values
    # Check pressure monotonicity
    if not np.all(np.diff(level) < 0):
        raise ValueError("Pressure levels must be strictly monotonic decreasing")


    # Check lat/lon monotonicity
    if not ds['lat'].diff('lat').min() > 0:
        raise ValueError("Latitude must be monotonic ascending")

    if not ds['lon'].diff('lon').min() > 0:
        raise ValueError("Longitude must be monotonic ascending")

    # Check units (example for temperature)
    if ds['T'].attrs.get('units') != 'K':
        raise ValueError("Temperature must be in Kelvin")

    # Check time coordinate regularity
    time = ds["time"].values
    if len(time) > 2:
        dt = np.diff(time)
        if not np.all(dt == dt[0]):
            raise ValueError("Time coordinate must be regular (constant time step)")
    
    print("Dataset schema validation passed.")

    return None

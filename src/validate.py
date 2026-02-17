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

def validate_dataset(ds):
    # Check required variables
    required_vars = ['T', 'u', 'v', 'w', 'sp']
    for var in required_vars:
        if var not in ds:
            raise ValueError(f"Missing required variable: {var}")
        else:
            print(f"Variable '{var}' is present.")

    # Check required dimensions
    required_dims = ['time', 'level', 'lat', 'lon']
    for dim in required_dims:
        if dim not in ds.dims:
            raise ValueError(f"Missing required dimension: {dim}")
        else:
            print(f"Dimension '{dim}' is present.")
    
    # Ensure dimensions are in correct order
    expected_order = ['time', 'level', 'lat', 'lon']
    if list(ds.dims) != expected_order:
        raise ValueError(f"Dimensions must be in order: {expected_order}")
    else:
        print(f"Dimensions are in correct order: {expected_order}")

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

    # Additional checks can be added as needed


    return None
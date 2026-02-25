'''
Docstring for eulerian_heat_budget.src.integrals

Pure integration operators: # Layer B

- `area_integral(field_2d, area_2d, weights_2d)`
- `volume_integral_pcoords(field_3d, volume_3d, weights_3d)`

These functions should be pure and not depend on dataset structure. They take in arrays and return integrated values, applying masks and cell areas as needed.

Must not depend on dataset structure.
'''

import xarray as xr

def area_integral(field_2d: xr.DataArray, 
                  area_2d: xr.DataArray,
                  weights_2d: xr.DataArray) -> xr.DataArray:
    # Integrate over all non-time dimensions.
    # Works for wall-specific geometries:
    # - top/bottom: (time, lat, lon)
    # - east/west:  (time, level, lat)
    # - north/south:(time, level, lon)
    integrand = field_2d * area_2d * weights_2d

    sum_dims = [dim for dim in integrand.dims if dim != "time"]
    if len(sum_dims)!=2:
        raise ValueError("The number of dimensions for the area integral is not 2.")

    return integrand.sum(dim=sum_dims)

def volume_integral_pcoords(field_3d, volume_3d, weights_3d):
    # Integrate field over volume using pressure and area integrals
    # Spherical coordinates for the horizontal dimentions, and pressure coordinates for the vertical dimension
    integrand = field_3d * volume_3d * weights_3d

    sum_dims = [dim for dim in integrand.dims if dim != "time"]
    if len(sum_dims)!=3:
        raise ValueError("The number of dimensions for the volume integral is not 3.")

    return integrand.sum(dim=sum_dims)


#need to add a differential helper funct for dT/dt storage term, and a vertical advection term (omega*dT/dp)
def time_derivative(field_3d, time_3d):
    # Compute time derivative using finite differences
    return (field_3d.diff(dim='time') / time_3d.diff(dim='time')).pad(time=1, mode='edge')

def spatial_derivative(field_3d, coord_3d, dim):
    # Compute spatial derivative using finite differences along specified dimension
    return (field_3d.diff(dim=dim) / coord_3d.diff(dim=dim)).pad({dim: 1}, mode='edge')

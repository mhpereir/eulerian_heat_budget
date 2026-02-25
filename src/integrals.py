'''
Docstring for eulerian_heat_budget.src.integrals

Pure integration operators: # Layer B

- `area_integral(field_2d, area_2d, weights_2d)`
- `volume_integral_pcoords(field_3d, volume_3d, weights_3d)`

These functions should be pure and not depend on dataset structure. They take in arrays and return integrated values, applying masks and cell areas as needed.

Must not depend on dataset structure.
'''

def area_integral(field_2d, area_2d, weights_2d):
    # Integrate 2D field over horizontal area using cell area and weights
    return (field_2d * area_2d * weights_2d).sum()

def volume_integral_pcoords(field_3d, volume_3d, weights_3d):
    # Integrate field over volume using pressure and area integrals
    # Spherical coordinates for the horizontal dimentions, and pressure coordinates for the vertical dimension
    return (field_3d * volume_3d * weights_3d).sum()


#need to add a differential helper funct for dT/dt storage term, and a vertical advection term (omega*dT/dp)
def time_derivative(field_3d, time_3d):
    # Compute time derivative using finite differences
    return (field_3d.diff(dim='time') / time_3d.diff(dim='time')).pad(time=1, mode='edge')

def spatial_derivative(field_3d, coord_3d, dim):
    # Compute spatial derivative using finite differences along specified dimension
    return (field_3d.diff(dim=dim) / coord_3d.diff(dim=dim)).pad({dim: 1}, mode='edge')
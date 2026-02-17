'''
Docstring for eulerian_heat_budget.src.masks

Constructs:

- `region_mask(time, lat, lon)`
- `volume_mask(time,p,y,x)`
- `area_mask_vertical(dim_x, p)`
- `area_mask_horizontal(lat, lon)`

Ensures correct truncation at surface pressure.
'''

def volume_mask(ds):
    # Construct 3D mask based on surface pressure and vertical levels
    pass

def region_mask(ds, lat_min, lat_max, lon_min, lon_max):
    # Construct 2D mask for specified lat/lon region
    pass

def area_mask_vertical(ds):
    # Construct 2D vertical mask (pressure coordinates) for valid grid cells (e.g., atmosphere vs below surface)
    # horizontal dimension is either lat or lon, and vertical dimension is pressure
    # In case lat, area must account for spherical geometry, and in case of lon, doesn't matter.

    pass

def area_mask_horizontal(ds):
    # Construct 2D horizontal mask for valid grid cells (e.g., atmosphere vs below surface)
    # horizontal dimensions are both lat/lon, no vertical
    # area must account for spherical geometry 
    pass
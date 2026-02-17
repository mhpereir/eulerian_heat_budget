'''
Docstring for eulerian_heat_budget.src.grid

Computes grid metrics: # Layer A

- `cell_area(lat, lon)`
- Optional `dx`, `dy`
- Handles spherical Earth geometry

Must be deterministic and independently testable.
'''

import xarray as xr
import numpy as np

from . import config

def determine_latlon_domain(ds):
    # in order to ensure that the domain has an additional buffer, we can add a small margin to the extents
    # number of grid points to add as margin, evenly spaced on both sides
    margin = config.margin

    lat_min = ds['lat'][margin]
    lat_max = ds['lat'][-1- margin]
    lon_min = ds['lon'][margin]
    lon_max = ds['lon'][-1- margin]

    return lat_min, lat_max, lon_min, lon_max

def get_horizontal_cell_areas(ds):
    # Compute grid metrics (e.g., cell area) based on lat/lon coordinates
    lat = ds['lat']
    lon = ds['lon']
    
    # Assuming regular grid, compute grid spacing
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    
    # Compute cell area using spherical coordinates
    lat_rad = np.deg2rad(lat)
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)
    
    cell_area = (config.R_earth**2) * np.cos(lat_rad) * dlat_rad[:, np.newaxis] * dlon_rad[np.newaxis, :]

    return cell_area

def get_vertical_cell_areas(ds):
    # Compute vertical cell area (if needed)
    # For pressure coordinates, this might involve integrating over pressure levels
    pass

def area_mask(ds):
    # Create a mask for valid grid cells (e.g., atmosphere vs below surface)
    pass


def volume_mask(ds):
    # Create a mask for valid grid cells (e.g., atmosphere vs below surface)
    pass
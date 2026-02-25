'''
High-level orchestration (no I/O):

- assembles terms into a single output dataset
- computes residual / closure diagnostics
- exposes a stable programmatic API for scripts/CLI
'''

import xarray as xr

from . import grid, weights, terms
from .specs import DomainSpec

def calculate_budget(ds_domain: xr.Dataset, ds_halo: xr.Dataset, DomainSpecs: DomainSpec, integral_diagnostics_flag: bool) -> xr.Dataset:


    # Construct integand cell areas and weights for integration
    ds_horizontal_cell_areas = grid.get_horizontal_cell_areas(ds_domain).astype("float64")
    ds_vertical_cell_areas   = grid.get_vertical_cell_areas(ds_domain).astype("float64")

    # combine east, west, south, north + top (and bottom)
    ds_cell_areas = xr.merge([ds_horizontal_cell_areas, ds_vertical_cell_areas])

    ds_cell_volumes = grid.get_cell_volumes(ds_domain).astype("float64")

    ds_weights_horizontal = weights.area_weights_horizontal(ds_domain, DomainSpecs)
    ds_weights_vertical   = weights.area_weights_vertical(ds_domain, DomainSpecs)
    ds_weights_volume     = weights.volume_weights(ds_domain, DomainSpecs)

    ds_weights_areas = xr.merge([ds_weights_horizontal, ds_weights_vertical])

    # pre-compute differential terms for advective, and adiabatic components
    # advective term needs du/dx, dv/dy, dw/dp @ each wall (east, west, north, south, top, bottom)
    # adiabatic term needs dT/dp at cell centers 
    # these can be computed using finite differences and stored as new variables in the dataset for use in the budget calculations
    
    advection_terms = terms.compute_advective_term(ds_domain, ds_halo, ds_cell_areas, ds_weights_areas, DomainSpecs, integral_diagnostics_flag)

    return advection_terms

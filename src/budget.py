'''
High-level orchestration (no I/O):

- assembles terms into a single output dataset
- computes residual / closure diagnostics
- exposes a stable programmatic API for scripts/CLI
'''

import os

import xarray as xr
import numpy as np

from . import grid, weights, terms
from . import plot_diagnostics

from .specs import DomainSpec



def calculate_budget(ds_domain: xr.Dataset, ds_halo: xr.Dataset, DomainSpecs: DomainSpec, integral_diagnostics_flag: bool, plot_dir: str, plot_flag: bool) -> xr.Dataset:
    plot_diag_path = os.path.join(plot_dir, "diagnostics")
    os.makedirs(plot_diag_path, exist_ok=True)

    # Construct integand cell areas and weights for integration
    ds_horizontal_cell_areas = grid.get_horizontal_cell_areas(ds_domain).astype("float64")
    ds_vertical_cell_areas   = grid.get_vertical_cell_areas(ds_halo).astype("float64")

    # combine east, west, south, north + top (and bottom)
    ds_cell_areas = xr.merge([ds_horizontal_cell_areas, ds_vertical_cell_areas])

    ds_cell_volumes = grid.get_cell_volumes(ds_domain).astype("float64")

    ds_weights_horizontal = weights.area_weights_horizontal(ds_domain, DomainSpecs)
    ds_weights_vertical   = weights.area_weights_vertical(ds_halo, DomainSpecs)
    ds_weights_volumes    = weights.volume_weights(ds_domain, DomainSpecs)

    ds_weights_areas = xr.merge(
        [ds_weights_horizontal, ds_weights_vertical],
        compat="no_conflicts",
    )

    # pre-compute differential terms for advective, and adiabatic components
    # advective term needs du/dx, dv/dy, dw/dp @ each wall (east, west, north, south, top, bottom)
    # adiabatic term needs dT/dp at cell centers 
    # these can be computed using finite differences and stored as new variables in the dataset for use in the budget calculations

    

    dT_dt = terms.compute_storage(ds_domain['T'],
                                  ds_cell_volumes,
                                  ds_weights_volumes,
                                  DomainSpecs)

    domain_volume = terms.compute_domain_volume(ds_domain, ds_cell_volumes, ds_weights_volumes, DomainSpecs)
    dV_dt         = terms.compute_time_derivative(domain_volume)

    #extra terms:
    #average T over the domain for each time step
    T_domain_avg = terms.compute_T_domain_average(ds_domain['T'], domain_volume, ds_cell_volumes, ds_weights_volumes, DomainSpecs)
    T_domain_avg = T_domain_avg.sel(time=dT_dt['time'])

    advection_terms = terms.compute_advective_term(ds_domain, ds_halo, ds_cell_areas, ds_weights_areas, DomainSpecs, integral_diagnostics_flag)
    #time crop advection
    advection_terms = advection_terms.sel(time=dT_dt['time'])

    #estimate of uncertainty from mass continuity
    T_scale         = np.sqrt(np.mean(ds_domain['T'].sel(time=dT_dt['time']).values-T_domain_avg.values[:,None,None,None])**2.)
    # T_scale = np.mean(T_domain_avg.values)
    # print(T_scale)

    advection_error = (dV_dt + advection_terms['net_mass_advection']) * T_scale # mass * K

    if plot_flag:
        #plot diagnostics for advective integrals
        plot_diagnostics.fig1_mass_continuity(dV_dt, advection_terms, plot_diag_path)
        plot_diagnostics.fig2_mass_advection_residual_timeseries(advection_terms, dV_dt, domain_volume, plot_diag_path)
        plot_diagnostics.fig3_advection_components_timeseries(advection_terms, dV_dt, advection_error, domain_volume, plot_diag_path)
    
    adiabatic_term = terms.compute_adiabatic_term(ds_domain, ds_cell_volumes, ds_weights_volumes, DomainSpecs)
    #time crop adiabatic
    adiabatic_term = adiabatic_term.sel(time=dT_dt['time'])

    # compute the residual term (diabatic term) from the scalar net advection term
    diabatic_term = terms.compute_diabatic_term(
        dT_dt,
        advection_terms["net_heat_advection"],
        adiabatic_term,
    )
    

    domain_volume = domain_volume.sel(time=dT_dt['time']) 
    #combine all terms into a single output dataset
    out = xr.Dataset({
        'dT_dt': dT_dt,
        'dV_dt': dV_dt,
        'advection_term': advection_terms['net_heat_advection'], # use actual variable name in advection_terms dataset
        'advection_error': advection_error,
        'adiabatic_term': adiabatic_term,
        'diabatic_term': diabatic_term,
        'T_domain_avg': T_domain_avg,
        'domain_volume': domain_volume,
        'T_scale': T_scale,
    })


    return out

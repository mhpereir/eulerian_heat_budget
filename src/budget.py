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

from .specs import DomainSpec, SurfaceBehaviour



def calculate_budget(
    ds_domain: xr.Dataset,
    ds_halo: xr.Dataset,
    DomainSpecs: DomainSpec,
    SurfaceSpecs: SurfaceBehaviour,
    integral_diagnostics_flag: bool,
    plot_dir: str,
    plot_flag: bool,
    *,
    test_constant_T: bool = False
    ) -> xr.Dataset:
    
    plot_diag_path = os.path.join(plot_dir, "diagnostics")
    os.makedirs(plot_diag_path, exist_ok=True)

    # Construct integand cell areas and weights for integration
    ds_horizontal_cell_areas = grid.get_horizontal_cell_areas(ds_domain).astype("float64")
    ds_vertical_cell_areas   = grid.get_vertical_cell_areas(ds_halo).astype("float64")

    # combine east, west, south, north + top (and bottom)
    ds_cell_areas = xr.merge([ds_horizontal_cell_areas, ds_vertical_cell_areas],
                             compat="override", join='outer')

    ds_cell_volumes = grid.get_cell_volumes(ds_domain).astype("float64")

    ds_weights_horizontal = weights.area_weights_horizontal(ds_domain, DomainSpecs)
    ds_weights_vertical   = weights.area_weights_vertical(ds_halo, DomainSpecs, SurfaceSpecs)
    ds_weights_volumes    = weights.volume_weights(ds_domain, DomainSpecs, SurfaceSpecs)

    ds_weights_areas = xr.merge([ds_weights_horizontal, ds_weights_vertical], compat="override", join='outer')

    
    d_dt_T = terms.compute_storage(ds_domain['T'],
                                  ds_cell_volumes,
                                  ds_weights_volumes,
                                  DomainSpecs)

    domain_volume = terms.compute_domain_volume(ds_domain, ds_cell_volumes, ds_weights_volumes, DomainSpecs)
    dV_dt         = terms.compute_time_derivative(domain_volume)

    #extra terms:
    #average T over the domain for each time step
    T_domain_avg = terms.compute_T_domain_average(ds_domain['T'], domain_volume, ds_cell_volumes, ds_weights_volumes, DomainSpecs)
    dT_dt        = terms.compute_time_derivative(T_domain_avg) * domain_volume.sel(time=d_dt_T['time'])
    dT_dt        = dT_dt.sel(time=d_dt_T['time'])

    dT_dt_2 = d_dt_T - T_domain_avg.sel(time=d_dt_T['time']) * dV_dt


    #logic to distinguish between normal calculation and test with constant T field 
    # (to see if advection error matches estimate from mass continuity) 
    # within test, set T to domain average (constant), 
    # so that any spatial anomalies are removed and only the advective error
    # from mass continuity remains


    # # advection domain vars
    # domain_vars = ["T"]
    # halo_vars   = ["T", "u", "v", "w", "sp"]
    # if not test_constant_T:
        

    #     if SurfaceSpecs.use_surface_variables:
    #         domain_vars += ["T2m"]
    #         halo_vars   += ["T2m", "u10", "v10"]

    #     ds_domain_adv_base = ds_domain[domain_vars] # reduced dataset for advection term calculation (only includes variables needed for advection, and only at levels needed for advection)
    #     ds_halo_adv_base   = ds_halo[halo_vars]

    #     assign_domain = {"T": ds_domain_adv_base["T"] - T_domain_avg}
    #     assign_halo   = {"T": ds_halo_adv_base["T"] - T_domain_avg}

    #     if SurfaceSpecs.use_surface_variables:
    #         assign_domain["T2m"] = ds_domain_adv_base["T2m"] - T_domain_avg
    #         assign_halo["T2m"]   = ds_halo_adv_base["T2m"] - T_domain_avg

    #     ds_domain_adv = ds_domain_adv_base.assign(**assign_domain) # type: ignore
    #     ds_halo_adv   = ds_halo_adv_base.assign(**assign_halo) # type: ignore


    # else:
    #     # ds_domain_adv      = ds_domain.copy(deep=True)
    #     # ds_domain_adv['T'] = xr.full_like(ds_domain['T'], np.nanmean(T_domain_avg)) # set T to constant value equal to domain average
        
    #     # ds_halo_adv        = ds_halo.copy(deep=True)
    #     # ds_halo_adv['T']   = xr.full_like(ds_halo['T'], np.nanmean(T_domain_avg)) # set T to constant value equal to domain average

    #     ds_domain_adv      = ds_domain[domain_vars]
    #     ds_halo_adv        = ds_halo[halo_vars]

    #     if SurfaceSpecs.use_surface_variables:
    #         ds_domain_adv = ds_domain_adv.assign(
    #             T2m = xr.full_like(ds_domain["T2m"], np.nanmean(T_domain_avg)),
    #         )

    #         ds_halo_adv = ds_halo_adv.assign(
    #             T2m = xr.full_like(ds_halo["T2m"], np.nanmean(T_domain_avg)),
    #         )

    # advection_terms = terms.compute_advective_term(ds_domain_adv, ds_halo_adv, ds_cell_areas, ds_weights_areas, DomainSpecs, SurfaceSpecs, integral_diagnostics_flag)
    # #time crop advection
    # advection_terms = advection_terms.sel(time=d_dt_T['time']).compute()  #dependency on this .sel.compute further down!!!


    # advection domain vars
    domain_vars = ["T"]
    halo_vars   = ["T", "u", "v", "w", "sp"]

    if SurfaceSpecs.use_surface_variables:
        domain_vars += ["T2m"]
        halo_vars   += ["T2m", "u10", "v10"]

    if not test_constant_T:
        ds_domain_adv_base = ds_domain[domain_vars]
        ds_halo_adv_base   = ds_halo[halo_vars]

        assign_domain = {"T": ds_domain_adv_base["T"] - T_domain_avg}
        assign_halo   = {"T": ds_halo_adv_base["T"] - T_domain_avg}

        if SurfaceSpecs.use_surface_variables:
            assign_domain["T2m"] = ds_domain_adv_base["T2m"] - T_domain_avg
            assign_halo["T2m"]   = ds_halo_adv_base["T2m"] - T_domain_avg

        ds_domain_adv = ds_domain_adv_base.assign(**assign_domain)  #type: ignore
        ds_halo_adv   = ds_halo_adv_base.assign(**assign_halo) #type: ignore

    else:
        ds_domain_adv = ds_domain[domain_vars]
        ds_halo_adv   = ds_halo[halo_vars]

        if SurfaceSpecs.use_surface_variables:
            ds_domain_adv = ds_domain_adv.assign(
                T2m=xr.full_like(ds_domain["T2m"], np.nanmean(T_domain_avg)),
            )
            ds_halo_adv = ds_halo_adv.assign(
                T2m=xr.full_like(ds_halo["T2m"], np.nanmean(T_domain_avg)),
            )

    ds_domain_adv_trim, ds_faces = terms.prepare_advective_faces(
        ds_domain_adv,
        ds_halo_adv,
        DomainSpecs,
        SurfaceSpecs,
        integral_diagnostics_flag=integral_diagnostics_flag,
    )

    advection_terms = terms.compute_advective_term(
        ds_domain_adv_trim,
        ds_faces,
        ds_cell_areas,
        ds_weights_areas,
        DomainSpecs,
        integral_diagnostics_flag,
    )

    advection_terms = advection_terms.sel(time=d_dt_T["time"]).compute()





    #needed to estimate heat advection uncertainty from mass continuity
    if not test_constant_T:
        T_scale:float  = np.sqrt(
            np.nanmean( (ds_domain['T']-T_domain_avg)**2. ) 
        )
    else:
        T_scale:float = np.nanmean(T_domain_avg) #type:ignore

    # T_scale = np.mean(T_domain_avg.values)
    print('Is T_constant:', test_constant_T)
    print('T_scale:', T_scale)

    advection_error = (dV_dt + advection_terms['net_mass_advection']) * T_scale # mass * K

    ds_domain_adiab   = ds_domain[["T", "w"]] #reduced dataset for adiabatic term calculation (only needs T and w)
    adiabatic_term = terms.compute_adiabatic_term(ds_domain_adiab, ds_cell_volumes, ds_weights_volumes, DomainSpecs)
    #time crop adiabatic
    adiabatic_term = adiabatic_term.sel(time=d_dt_T['time'])

    # compute the residual term (diabatic term) from the scalar net advection term
    diabatic_term = terms.compute_diabatic_term(
        dT_dt,
        advection_terms["net_heat_advection"],
        adiabatic_term,
    )

    #combine all terms into a single output dataset
    out = xr.Dataset({
        'd_dt_T': d_dt_T.sel(time=d_dt_T['time']),
        'dT_dt': dT_dt.sel(time=d_dt_T['time']),
        'dT_dt_2': dT_dt_2.sel(time=d_dt_T['time']),
        'dV_dt': dV_dt.sel(time=d_dt_T['time']),
        # 'advection_term': advection_terms['net_heat_advection'].sel(time=d_dt_T['time']), # use actual variable name in advection_terms dataset
        'advection_error': advection_error.sel(time=d_dt_T['time']),
        'adiabatic_term': adiabatic_term.sel(time=d_dt_T['time']),
        'diabatic_term': diabatic_term.sel(time=d_dt_T['time']),
        'T_domain_avg': T_domain_avg.sel(time=d_dt_T['time']),
        'domain_volume': domain_volume.sel(time=d_dt_T['time']),
        'T_scale': T_scale,
    }).compute()

    out["advection_term"] = advection_terms["net_heat_advection"]  #already sel time and computed


    if plot_flag:
        #plot diagnostics for advective integrals
        plot_diagnostics.fig1_mass_continuity(out['dV_dt'], advection_terms, plot_diag_path)
        plot_diagnostics.fig2_mass_advection_residual_timeseries(advection_terms, out['dV_dt'], out['domain_volume'], plot_diag_path)
        plot_diagnostics.fig3_advection_components_timeseries(advection_terms, out['dV_dt'], out['advection_error'], out['domain_volume'], plot_diag_path)
        plot_diagnostics.fig4_temperature_derivative_timeseries(out['d_dt_T'], out['dT_dt'], out['dT_dt_2'], out['domain_volume'], plot_diag_path)


    return out

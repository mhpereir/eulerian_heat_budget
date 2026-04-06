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

import time

def timed_compute(name, obj):
    t0 = time.time()
    out = obj.compute()
    print(f"{name}: {time.time() - t0:.2f} s")
    return out

def describe_xarray(name, obj):
    if hasattr(obj, "data_vars"):  # Dataset
        print(f"\n{name}: DATASET")
        for v in obj.data_vars:
            arr = obj[v].data
            print(f"  {v}: shape={obj[v].shape}")
            if hasattr(arr, "chunks"):
                print(f"     chunks={arr.chunks}")
            if hasattr(arr, "__dask_graph__"):
                print(f"     ntasks={len(arr.__dask_graph__())}")
    else:  # DataArray
        arr = obj.data
        print(f"\n{name}: shape={obj.shape}")
        if hasattr(arr, "chunks"):
            print(f"  chunks={arr.chunks}")
        if hasattr(arr, "__dask_graph__"):
            print(f"  ntasks={len(arr.__dask_graph__())}")


def calculate_budget(
    ds_domain: xr.Dataset,
    ds_halo: xr.Dataset,
    DomainSpecs: DomainSpec,
    SurfaceSpecs: SurfaceBehaviour,
    integral_diagnostics_flag: bool,
    plot_dir: str,
    plot_flag: bool,
    *,
    benchmark_ds: xr.Dataset | None = None,
    test_constant_T: bool = False
    ) -> xr.Dataset:
    
    plot_diag_path = os.path.join(plot_dir, "diagnostics")

    print("Calculating cell areas")
    # Construct integand cell areas and weights for integration
    ds_horizontal_cell_areas = grid.get_horizontal_cell_areas(ds_domain).astype("float64")
    ds_vertical_cell_areas   = grid.get_vertical_cell_areas(ds_halo).astype("float64")

    # combine east, west, south, north + top (and bottom)
    ds_cell_areas = xr.merge([ds_horizontal_cell_areas, ds_vertical_cell_areas],
                             compat="override", join='outer')

    ds_cell_volumes = grid.get_cell_volumes(ds_domain).astype("float64")



    print("Calculating integration weights")
    ds_weights_horizontal = weights.area_weights_horizontal(ds_domain, DomainSpecs)
    ds_weights_vertical   = weights.area_weights_vertical(ds_halo, DomainSpecs, SurfaceSpecs)
    ds_weights_volumes    = weights.volume_weights(ds_domain, DomainSpecs, SurfaceSpecs)

    ds_weights_areas = xr.merge([ds_weights_horizontal, ds_weights_vertical], compat="override", join='outer')


    # describe_xarray("ds_horizontal_cell_areas", ds_horizontal_cell_areas)
    # describe_xarray("ds_vertical_cell_areas", ds_vertical_cell_areas)
    # describe_xarray("ds_cell_volumes", ds_cell_volumes)
    # describe_xarray("ds_weights_horizontal", ds_weights_horizontal)
    # describe_xarray("ds_weights_vertical", ds_weights_vertical)
    # describe_xarray("ds_weights_volumes", ds_weights_volumes)

    # timed_compute("horizontal areas", ds_horizontal_cell_areas)
    # timed_compute("vertical areas", ds_vertical_cell_areas)
    # timed_compute("cell volumes", ds_cell_volumes)
    # timed_compute("horizontal weights", ds_weights_horizontal)
    # timed_compute("vertical weights", ds_weights_vertical)
    # timed_compute("volume weights", ds_weights_volumes)

    # exit(0)

    print('Calculating budget terms...')
    print('\t Calculating storage term')
    d_dt_T = terms.compute_storage(ds_domain['T'],
                                  ds_cell_volumes,
                                  ds_weights_volumes,
                                  DomainSpecs)

    print('\t Calculating domain volume and its time derivative')
    domain_volume = terms.compute_domain_volume(ds_domain, ds_cell_volumes, ds_weights_volumes, DomainSpecs)
    dV_dt         = terms.compute_time_derivative(domain_volume)

    print('\t Calculating domain average temperature and its time derivative')
    #extra terms:
    #average T over the domain for each time step
    T_domain_avg = terms.compute_T_domain_average(ds_domain['T'], domain_volume, ds_cell_volumes, ds_weights_volumes, DomainSpecs)
    dT_dt        = terms.compute_time_derivative(T_domain_avg) * domain_volume.sel(time=d_dt_T['time'])
    dT_dt        = dT_dt.sel(time=d_dt_T['time'])

    dT_dt_2 = d_dt_T - T_domain_avg.sel(time=d_dt_T['time']) * dV_dt

    print('\t Preparing advective term')
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
                T2m=xr.full_like(ds_domain["T2m"], T_domain_avg.mean(dim="time", skipna=True).compute().item()),
            )
            ds_halo_adv = ds_halo_adv.assign(
                T2m=xr.full_like(ds_halo["T2m"], T_domain_avg.mean(dim="time", skipna=True).compute().item()),
            )

    print('\t\t Preparing advective faces')
    ds_domain_adv_trim, ds_faces = terms.prepare_advective_faces(
        ds_domain_adv,
        ds_halo_adv,
        DomainSpecs,
        SurfaceSpecs,
        integral_diagnostics_flag=integral_diagnostics_flag,
    )

    print('\t\t Computing advective term')
    advection_terms = terms.compute_advective_term(
        ds_domain_adv_trim,
        ds_faces,
        ds_cell_areas,
        ds_weights_areas,
        DomainSpecs,
        integral_diagnostics_flag,
    )

    if plot_flag:
        advection_terms = advection_terms.sel(time=d_dt_T["time"]).compute()
    else:
        advection_terms = advection_terms.sel(time=d_dt_T["time"])

    # needed to estimate heat advection uncertainty from mass continuity
    if not test_constant_T:
        t_var = ((ds_domain["T"] - T_domain_avg) ** 2).mean(
            dim=("time", "level", "lat", "lon"),
            skipna=True,
        )
        T_scale: float = float(np.sqrt(t_var))
    else:
        t_var = (T_domain_avg).mean(
            dim=("time"), 
            skipna=True,
        )
        T_scale: float = float(t_var)

    print("Is T_constant:", test_constant_T)
    print("T_scale:", T_scale)

    advection_error = (dV_dt + advection_terms['net_mass_advection']) * T_scale # mass * K

    print('\t Calculating adiabatic term')
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

    print('Combining terms into output dataset')
    #combine all terms into a single output dataset
    advection_terms_out = advection_terms.rename({"net_heat_advection": "advection_term"})

    out = xr.merge([
        xr.Dataset({
            'd_dt_T': d_dt_T.sel(time=d_dt_T['time']),
            'dT_dt': dT_dt.sel(time=d_dt_T['time']),
            'dT_dt_2': dT_dt_2.sel(time=d_dt_T['time']),
            'dV_dt': dV_dt.sel(time=d_dt_T['time']),
            'advection_error': advection_error.sel(time=d_dt_T['time']),
            'adiabatic_term': adiabatic_term.sel(time=d_dt_T['time']),
            'diabatic_term': diabatic_term.sel(time=d_dt_T['time']),
            'T_domain_avg': T_domain_avg.sel(time=d_dt_T['time']),
            'domain_volume': domain_volume.sel(time=d_dt_T['time']),
            'T_scale': T_scale,
        }),
        advection_terms_out,
    ], compat="equals", join="exact").compute()


    if benchmark_ds is not None:
        # need to align benchmark dataset with our ds_halo grid and time steps;
        # then we need to compute horizontal integrals of the benchmark fluxes to get them in the same form as our advection terms for comparison

        benchmark_ds = grid.crop_to_target_grid(benchmark_ds, ds_halo)
        benchmark_mass_fluxes, benchmark_heat_fluxes = terms.compute_advective_benchmark_fluxes(benchmark_ds, ds_domain, DomainSpecs)


    print('Plotting diagnostics...')
    if plot_flag:
        os.makedirs(plot_diag_path, exist_ok=True)
        #plot diagnostics for advective integrals
        plot_diagnostics.fig1_mass_continuity(out['dV_dt'], advection_terms_out, plot_diag_path)
        plot_diagnostics.fig2_mass_advection_residual_timeseries(advection_terms_out, out['dV_dt'], out['domain_volume'], plot_diag_path)
        plot_diagnostics.fig3_advection_components_timeseries(advection_terms_out, out['dV_dt'], out['advection_error'], out['domain_volume'], plot_diag_path)
        plot_diagnostics.fig4_temperature_derivative_timeseries(out['d_dt_T'], out['dT_dt'], out['dT_dt_2'], out['domain_volume'], plot_diag_path)

        if benchmark_ds is not None:
            plot_diagnostics.fig5_benchmark_comparison(benchmark_mass_fluxes, benchmark_heat_fluxes, out, advection_terms_out, plot_diag_path) # type: ignore

    return out

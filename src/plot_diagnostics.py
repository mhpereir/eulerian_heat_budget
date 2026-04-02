'''

'''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from matplotlib.lines import Line2D

#set label sizes 16
plt.rcParams.update({'font.size': 16})


def fig1_mass_continuity(dV_dt: xr.DataArray, advection_terms: xr.Dataset, plot_dir: str):
    #test how mass advection terms compare to volume changes (should be y=-x)
    fig,ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    x= dV_dt.values
    y= advection_terms['net_mass_advection'].values
    ax.scatter(x, y)
    #plot 1:-1 line
    # symmetric limits
    L = np.nanmax(np.abs(np.concatenate([x, y])))

    ax.plot([-L, L], [L, -L], linestyle='--', color='k')

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)

    ax.set_xlabel("dV/dt")
    ax.set_ylabel("Net mass advection")

    plt.savefig(plot_dir + '/fig1_mass_continuity.png', dpi=300)

    x = dV_dt.values
    y = advection_terms['net_mass_advection'].values
    r = y + x

    # bin by x
    nbin = 20
    edges = np.quantile(x, np.linspace(0, 1, nbin+1))
    bin_id = np.digitize(x, edges[1:-1])

    xb = np.array([np.nanmean(x[bin_id==i]) for i in range(nbin)])
    rb = np.array([np.nanmean(r[bin_id==i]) for i in range(nbin)])

    plt.figure(figsize=(10, 6), tight_layout=True)
    plt.plot(xb, rb, marker='o')
    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel("dV/dt")
    plt.ylabel("mean residual (net_mass_advection + dV/dt)")
    plt.savefig(plot_dir + "/fig1.1_mass_continuity_binned_residual.png", dpi=300)

    plt.close()




def fig2_mass_advection_residual_timeseries(advection_terms: xr.Dataset, dV_dt: xr.DataArray, domain_volume: xr.DataArray, plot_dir: str):

    
    norm_factor          = 1/ domain_volume
    mean_norm_factor     = 1/np.nanmean(domain_volume)
    time_rate_conversion = 3600

    fig, ax = plt.subplots(figsize=(10, 10), nrows=2, tight_layout=True, sharex=True)

    delta_mass = dV_dt + advection_terms['net_mass_advection']

    ax[0].plot(delta_mass['time'], delta_mass * mean_norm_factor * time_rate_conversion, label=r'$\delta M$', color='k', alpha=0.8)

    ax[0].plot(delta_mass['time'], advection_terms['net_mass_advection']* mean_norm_factor * time_rate_conversion, 
               label='Net Mass Advection', alpha=0.5, color='C0')
    ax[0].plot(delta_mass['time'], dV_dt * mean_norm_factor * time_rate_conversion, 
               label='dV/dt', alpha=0.5, color='C1')

    ax[0].legend()
    # ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Mass Terms (normalized) [1/hr]")
    ax[0].set_title(r"Mass Advection Residual Time Series (normalized by $\bar V$)")
    
    #check advection_terms['time'] is in the correct format:
    if np.issubdtype(advection_terms['time'].dtype, np.datetime64):
        print("Time coordinate is in datetime format.")
    else:
        print("Time coordinate is NOT in datetime format. Check your dataset.")
        raise ValueError("Time coordinate is not in datetime format.")

    dt = (advection_terms["time"][1] - advection_terms["time"][0]).values / np.timedelta64(1, 's') # convert to seconds
    cumulative_residual = np.cumsum( delta_mass * dt)
    cumulative_net_adv  = np.cumsum(advection_terms['net_mass_advection'] * dt)
    cumulative_dvdt     = np.cumsum(dV_dt * dt)

    ax[1].plot(advection_terms['time'], cumulative_residual * mean_norm_factor, color='k')
    
    ax[1].plot(advection_terms['time'], cumulative_net_adv * mean_norm_factor, color='C0')
    ax[1].plot(advection_terms['time'], cumulative_dvdt * mean_norm_factor, color='C1')

    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel(r"$\int \delta$ Mass dt/ $\bar V$(t) [unitless]")
    ax[1].set_title(r"Cumulative $\delta$ Mass / $\bar V$(t) Time Series")
    plt.savefig(plot_dir + '/fig2_mass_residual_time_series.png', dpi=300)
    plt.close()

    #parse through the vars in advection_terms
    mass_vars = []
    heat_vars = []
    for component in advection_terms.data_vars:
        if "mass" in str(component) or "net" in str(component):
            mass_vars.append(str(component))
            pass
        else:
            heat_vars.append(str(component))


    # epsilon_mass_advection = abs(net) / sum (abs ( advection_per_surface ))
    eps_mass = np.abs( delta_mass) / np.sum(np.abs([advection_terms[mass] for mass in mass_vars]))

    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    ax.plot(advection_terms['time'], eps_mass)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel("Time")
    ax.set_ylabel("relative error")
    ax.set_title(r'$|\delta M|/ \sum_{faces} |U\cdot A|$')
    plt.savefig(plot_dir + '/fig2.1_epsilon_mass_timeseries.png', dpi=300)
    plt.close()


    mass_vars = []
    heat_vars = []
    # Plot each component of advection terms
    for component in advection_terms.data_vars:
        if "mass" in str(component) and not "net" in str(component) and not "abs" in str(component):
            mass_vars.append(str(component))
        elif "heat" in str(component) and not "net" in str(component) and not "abs" in str(component):
            heat_vars.append(str(component))
        else:
            pass
    

    net_zonal_mass_advection = np.zeros_like(advection_terms['net_mass_advection'].values)
    net_meridional_mass_advection = np.zeros_like(advection_terms['net_mass_advection'].values)
    net_vertical_mass_advection   = np.zeros_like(advection_terms['net_mass_advection'].values)
    

    net_cum_zonal_mass_advection = np.zeros_like(advection_terms['net_mass_advection'].values)
    net_cum_meridional_mass_advection = np.zeros_like(advection_terms['net_mass_advection'].values)
    net_cum_vertical_mass_advection   = np.zeros_like(advection_terms['net_mass_advection'].values)
    
    
    fig,ax = plt.subplots(figsize=(10, 10), tight_layout=True, nrows=3, sharex=True)

    advection_terms_smoothed = advection_terms.rolling(time=24, center=True).mean()

    for var in mass_vars:
        ax[0].plot(advection_terms_smoothed['time'], advection_terms_smoothed[var] * mean_norm_factor * time_rate_conversion, label=var)

        if var.split('_')[-1] in ['east', 'west']:
            net_zonal_mass_advection += advection_terms_smoothed[var].values* mean_norm_factor * time_rate_conversion
            net_cum_zonal_mass_advection += np.cumsum(advection_terms[var].values * dt) * mean_norm_factor
        elif var.split('_')[-1] in ['north', 'south']:
            net_meridional_mass_advection += advection_terms_smoothed[var].values* mean_norm_factor * time_rate_conversion
            net_cum_meridional_mass_advection += np.cumsum(advection_terms[var].values * dt) * mean_norm_factor
        elif var.split('_')[-1] in ['top', 'bottom']:
            net_vertical_mass_advection += advection_terms_smoothed[var].values* mean_norm_factor * time_rate_conversion
            net_cum_vertical_mass_advection += np.cumsum(advection_terms[var].values * dt) * mean_norm_factor

    net_horizontal_mass_advection     = net_zonal_mass_advection + net_meridional_mass_advection
    net_cum_horizontal_mass_advection = net_cum_zonal_mass_advection + net_cum_meridional_mass_advection

    ax[1].plot(advection_terms_smoothed['time'], delta_mass * mean_norm_factor * time_rate_conversion, label=r'$\delta M$', color='k', linewidth=2)

    # ax[1].plot(advection_terms_smoothed['time'], net_zonal_mass_advection, label='Zonal', linestyle='--')
    # ax[1].plot(advection_terms_smoothed['time'], net_meridional_mass_advection, label='Meridional', linestyle='--')
    ax[1].plot(advection_terms_smoothed['time'], net_horizontal_mass_advection, label='Horizontal')
    ax[1].plot(advection_terms_smoothed['time'], net_vertical_mass_advection, label='Vertical')


    ax[2].plot(advection_terms_smoothed['time'], cumulative_residual * mean_norm_factor, label='Cumulative Residual', color='k', linewidth=2)

    ax[2].plot(advection_terms_smoothed['time'], net_cum_zonal_mass_advection, label='Zonal', linestyle='--')
    ax[2].plot(advection_terms_smoothed['time'], net_cum_meridional_mass_advection, label='Meridional', linestyle='--')
    ax[2].plot(advection_terms_smoothed['time'], net_cum_horizontal_mass_advection, label='Horizontal')
    ax[2].plot(advection_terms_smoothed['time'], net_cum_vertical_mass_advection, label='Vertical')

    ax[0].set_title('Mass Advection Terms (normalized)')
    ax[1].set_title('Net Mass Advection (normalized)')
    ax[2].set_title(r"Cumulative Mass Advection / $\bar V$(t)")

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax[2].xaxis.set_major_locator(locator)
    ax[2].xaxis.set_major_formatter(formatter)

    ax[2].set_xlabel("Time")
    ax[0].set_ylabel("[1/hr]")
    ax[1].set_ylabel("[1/hr]")
    ax[2].set_ylabel("[unitless]")

    ax[0].legend(fontsize=10)
    ax[1].legend(fontsize=10)
    ax[2].legend(fontsize=10)
    plt.savefig(plot_dir + '/fig2.2_mass_advection_terms_timeseries.png', dpi=300)
    plt.close()
    
    


def fig3_advection_components_timeseries(advection_terms: xr.Dataset, dV_dt: xr.DataArray, delta_heat: xr.DataArray, domain_volume:xr.DataArray, plot_dir: str):

    norm_factor = 1 / domain_volume
    time_rate_conversion = 3600 # convert from per second to per hour

    # delta_mass = dV_dt + advection_terms['net_mass_advection']


    fig, ax = plt.subplots(figsize=(10, 10), nrows=2, tight_layout=True, sharex=True)

    mass_vars = []
    heat_vars = []
    # Plot each component of advection terms
    for component in advection_terms.data_vars:
        if "mass" in str(component) or "net" in str(component):
            mass_vars.append(str(component))
            pass
        else:
            heat_vars.append(str(component))
            ax[0].plot(advection_terms['time'], advection_terms[component] * norm_factor * time_rate_conversion, label=component)

    ax[0].plot(advection_terms['time'], advection_terms['advection_term'] * norm_factor * time_rate_conversion, label='Net Heat Advection', linewidth=2, color='k')

    ax[0].set_ylabel("[K / hr]")
    ax[0].set_title("Advection Components Time Series")

    ax[1].plot(advection_terms['time'], advection_terms['advection_term'] * norm_factor * time_rate_conversion, label='Net Heat Advection', linewidth=2, color='k')
    ax[1].plot(advection_terms['time'], delta_heat * norm_factor * time_rate_conversion, label='Expected Residual', linewidth=1, color='red')

    ax[1].set_title(r"Net Heat Advection and $\delta M T_{scale}$ Time Series")
    ax[1].set_ylabel('[K / hr]')

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[0].legend(fontsize=10)
    ax[1].legend(fontsize=10)
    ax[1].set_xlabel("Time")
    plt.savefig(plot_dir + '/fig3_advection_components_timeseries.png', dpi=300)
    plt.close()




def fig4_temperature_derivative_timeseries(d_dt_T: xr.DataArray, dT_dt_1:xr.DataArray, dT_dt_2:xr.DataArray, domain_volume: xr.DataArray, plot_dir: str):

    norm_factor = 1 / domain_volume
    time_rate_conversion = 3600 # convert from per second to per hour

    # delta_mass = dV_dt + advection_terms['net_mass_advection']

    fig, ax = plt.subplots(figsize=(10, 10), nrows=2, tight_layout=True, sharex=True)

    ax[0].plot(d_dt_T['time'], d_dt_T * norm_factor * time_rate_conversion, label=r'd/dt $\int TdV$', linewidth=2, color='k')

    ax[0].set_ylabel("[K / hr]")
    ax[0].set_title("Storage term: d/dt of domain integrated T")

    ax[1].plot(dT_dt_2['time'], dT_dt_2 * norm_factor * time_rate_conversion, label=r'd/dt$\int TdV$-$\langle T \rangle $dV/dt', linewidth=2, color='b')
    ax[1].plot(dT_dt_1['time'], dT_dt_1 * norm_factor * time_rate_conversion, label=r'd$\langle T \rangle $/dt', linewidth=2, color='k')
    
    ax[1].set_title(r"Domain Average Temperature Tendency Time Series")
    ax[1].set_ylabel('[K / hr]')

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[0].legend(fontsize=10)
    ax[1].legend(fontsize=10)
    ax[1].set_xlabel("Time")
    plt.savefig(plot_dir + '/fig4_temperature_derivative_timeseries.png', dpi=300)
    plt.close()



def fig5_benchmark_comparison(
    benchmark_mass_fluxes: xr.Dataset,
    benchmark_heat_fluxes: xr.Dataset,
    results: xr.Dataset,
    advection_terms: xr.Dataset,
    plot_dir: str,
):
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True, nrows=2, sharex=True)

    wall_faces = ["north", "south", "east", "west"]
    colors = {
        "north": "blue",
        "south": "orange",
        "east": "green",
        "west": "red",
    }

    T_average = results["T_domain_avg"]

    name_mass_benchmark = "benchmark_mass_flux_"
    name_heat_benchmark = "benchmark_heat_flux_"
    name_calculated_mass = "mass_flux_contribution_"
    name_calculated_heat = "flux_contribution_"

    # lateral-only net mass/heat from your calculation
    mass_lateral = advection_terms["net_mass_advection"] - advection_terms["mass_flux_contribution_top"]
    heat_lateral_anom = advection_terms["advection_term"] - advection_terms["flux_contribution_top"]

    if "mass_flux_contribution_bottom" in advection_terms:
        mass_lateral = mass_lateral - advection_terms["mass_flux_contribution_bottom"]
    if "flux_contribution_bottom" in advection_terms:
        heat_lateral_anom = heat_lateral_anom - advection_terms["flux_contribution_bottom"]

    heat_lateral_full = heat_lateral_anom + T_average * mass_lateral
    heat_lateral_full_2 = heat_lateral_anom + T_average * benchmark_mass_fluxes["benchmark_mass_flux_net"]

    for face in wall_faces:
        # mass panel
        ax[0].plot(
            benchmark_mass_fluxes["time"],
            benchmark_mass_fluxes[name_mass_benchmark + face],
            linestyle="--",
            color=colors[face],
        )
        ax[0].plot(
            advection_terms["time"],
            advection_terms[name_calculated_mass + face],
            linestyle="-",
            color=colors[face],
        )

        # heat panel: reconstruct full-T lateral face flux from anomaly flux
        ax[1].plot(
            benchmark_heat_fluxes["time"],
            benchmark_heat_fluxes[name_heat_benchmark + face],
            linestyle="--",
            color=colors[face],
        )
        ax[1].plot(
            advection_terms["time"],
            advection_terms[name_calculated_heat + face]
            + advection_terms[name_calculated_mass + face] * T_average,
            linestyle="-",
            color=colors[face],
        )

    # net lines
    ax[0].plot(
        benchmark_mass_fluxes["time"],
        benchmark_mass_fluxes["benchmark_mass_flux_net"],
        linestyle="--",
        color="k",
    )
    ax[0].plot(
        advection_terms["time"],
        mass_lateral,
        linestyle="-",
        linewidth=2,
        color="k",
    )

    ax[1].plot(
        benchmark_heat_fluxes["time"],
        benchmark_heat_fluxes["benchmark_heat_flux_net"],
        linestyle="--",
        linewidth=2,
        color="k",
        label="Benchmark net",
    )
    ax[1].plot(
        advection_terms["time"],
        heat_lateral_full,
        linestyle="-",
        linewidth=2,
        color="k",
        label="Calculated net",
    )

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[0].set_ylabel("Mass Flux [units]")
    ax[0].set_title("Comparison of Calculated Mass Fluxes with Benchmark")

    face_handles = [
        Line2D([0], [0], color=colors["north"], lw=2, label="North"),
        Line2D([0], [0], color=colors["south"], lw=2, label="South"),
        Line2D([0], [0], color=colors["east"],  lw=2, label="East"),
        Line2D([0], [0], color=colors["west"],  lw=2, label="West"),
        Line2D([0], [0], color="k", lw=2, label="Net"),
    ]
    style_handles = [
        Line2D([0], [0], color="0.3", lw=2, linestyle="--", label="Benchmark"),
        Line2D([0], [0], color="0.3", lw=2, linestyle="-",  label="Calculated"),
    ]

    leg1 = ax[0].legend(handles=face_handles, loc="upper left", fontsize=10)
    ax[0].add_artist(leg1)
    ax[0].legend(handles=style_handles, loc="lower left", fontsize=10)

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Heat Flux [units]")
    ax[1].set_title("Comparison of Calculated Heat Fluxes with Benchmark")
    ax[1].legend(fontsize=10)

    plt.savefig(plot_dir + "/fig5_benchmark_comparison.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True, nrows=2, sharex=True)

    # net lines
    ax[0].plot(
        benchmark_mass_fluxes["time"],
        benchmark_mass_fluxes["benchmark_mass_flux_net"],
        linestyle="--",
        color="k",
    )
    ax[0].plot(
        advection_terms["time"],
        mass_lateral,
        linestyle="-",
        linewidth=2,
        color="k",
    )

    ax[1].plot(
        benchmark_heat_fluxes["time"],
        benchmark_heat_fluxes["benchmark_heat_flux_net"],
        linestyle="--",
        linewidth=1,
        color="k",
        label=r"$\mathcal{H}_{bench, full}$",
    )
    ax[1].plot(
        advection_terms["time"],
        heat_lateral_full,
        linestyle="-",
        linewidth=1,
        color="k",
        label=r"$\mathcal{H}'_{calc} + \rangle T \langle M_{calc} $"
    )

    ax[1].plot(
        advection_terms["time"],
        heat_lateral_full_2,
        linestyle="-.",
        linewidth=1,
        color="gray",
        label=r"$\mathcal{H}'_{calc} + \rangle T \langle M_{bench} $"
    )

    ax[1].plot(
        advection_terms["time"],
        heat_lateral_anom,
        linestyle="-",
        linewidth=1,
        color="red",
        label=r"$\mathcal{H}'_{calc}$"
    )


    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[0].set_ylabel("Mass Flux [units]")
    ax[0].set_title("Comparison of Calculated Mass Fluxes with Benchmark")

    # face_handles = [
    #     Line2D([0], [0], color=colors["north"], lw=2, label="North"),
    #     Line2D([0], [0], color=colors["south"], lw=2, label="South"),
    #     Line2D([0], [0], color=colors["east"],  lw=2, label="East"),
    #     Line2D([0], [0], color=colors["west"],  lw=2, label="West"),
    #     Line2D([0], [0], color="k", lw=2, label="Net"),
    # ]
    style_handles = [
        Line2D([0], [0], color="0.3", lw=2, linestyle="--", label="Benchmark"),
        Line2D([0], [0], color="0.3", lw=2, linestyle="-",  label="Calculated"),
    ]

    # leg1 = ax[0].legend(handles=face_handles, loc="upper left", fontsize=10)
    # ax[0].add_artist(leg1)
    ax[0].legend(handles=style_handles, loc="lower left", fontsize=10)

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Heat Flux [units]")
    ax[1].set_title("Comparison of Calculated Heat Fluxes with Benchmark")
    ax[1].legend(fontsize=10)

    plt.savefig(plot_dir + "/fig5.1_net_benchmark_comparison.png", dpi=300)
    plt.close()

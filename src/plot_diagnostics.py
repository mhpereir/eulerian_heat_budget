'''

'''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

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



def fig2_advection_components_timeseries(advection_terms: xr.Dataset, plot_dir: str):

    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

    # Plot each component of advection terms
    for component in advection_terms.data_vars:
        if "mass" in str(component) or "net" in str(component):
            pass
        else:
            ax.plot(advection_terms['time'], advection_terms[component], label=component)

    ax.plot(advection_terms['time'], advection_terms['net_heat_advection'], label='Net Heat Advection', linewidth=2, color='k')

    ax.legend(fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Advection Terms")
    plt.savefig(plot_dir + '/fig2_advection_components_timeseries.png', dpi=300)
    plt.close()

def fig3_mass_advection_residual_timeseries(advection_terms: xr.Dataset, dV_dt: xr.DataArray, plot_dir: str):

    fig, ax = plt.subplots(figsize=(10, 6), nrows=2, tight_layout=True)

    # Plot net mass advection and dV/dt
    ax[0].plot(advection_terms['time'], advection_terms['net_mass_advection'], label='Net Mass Advection')
    ax[0].plot(dV_dt['time'], dV_dt, label='dV/dt')

    # Plot residual (sum of net mass advection and dV/dt)
    residual = advection_terms['net_mass_advection'] + dV_dt
    ax[0].plot(residual['time'], residual, label='Residual')

    ax[0].legend()
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Mass Terms")

    # Plot residual alone in second panel
    ax[1].plot(residual['time'], advection_terms['abs_mass_advection_residual_fraction'])
    ax[1].axhline(0, linestyle='--', color='k')
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Absolute Residual Fraction")
    plt.savefig(plot_dir + '/fig3_mass_advection_residual_timeseries.png', dpi=300)
    plt.close()
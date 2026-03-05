'''

'''
import os


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 16})


# def plot_budget_terms(ds_budget: xr.Dataset, plot_dir: str) -> None:

#     #rolling average smoothing
#     ds_budget = ds_budget.rolling(time=5, center=True).mean()

#     fig, ax = plt.subplots(figsize=(10, 12), nrows=3, tight_layout=True, sharex=True)

#     #panel 1: time series of volume, average temperature
#     #use two y-axis
#     #lhs y-axis: volume
#     ds_budget["domain_volume"].plot.line(ax=ax[0], label="Volume")
#     ax[0].set_ylabel("Domain Volume (m^2 Pa)")
#     #rhs y-axis: average temperature
#     ax2 = ax[0].twinx()
#     ds_budget["T_domain_avg"].plot.line(ax=ax2, label=r"$\langle T \rangle$")
#     ax2.set_ylabel("Domain Average Temperature (K)")

#     ax[0].set_title("Domain Volume and Average Temperature Time Series")
#     ax[0].legend(loc='lower left')
#     ax2.legend(loc="upper right")

#     #panel 2: time series of dT/dt
#     dT_dt: xr.DataArray = ds_budget["dT_dt"]
#     dT_dt.plot.line(ax=ax[1], label="dT/dt")

#     ax[1].set_title("dT/dt Time Series")

#     #panel 3: time series of rest of budget terms

#     net_adv: xr.DataArray = ds_budget["advection_term"]  # use actual variable name in ds_budget
#     adiabatic: xr.DataArray = ds_budget["adiabatic_term"]
#     diabatic: xr.DataArray = ds_budget["diabatic_term"]

#     net_adv.plot.line(ax=ax[2], label="Net heat advection")
#     adiabatic.plot.line(ax=ax[2], label="Adiabatic term")
#     diabatic.plot.line(ax=ax[2], label="Diabatic term")

#     ax[2].set_title("Budget Terms Time Series")
#     ax[2].legend()

#     out_path = os.path.join(plot_dir, "budget_terms_timeseries.png")
#     plt.savefig(out_path, bbox_inches="tight")
#     plt.close(fig)


def plot_budget_terms(ds_budget: xr.Dataset, plot_dir: str) -> None:

    # rolling average smoothing
    ds_budget = ds_budget.rolling(time=24, center=True).mean() #daily smoothing, data is hourly

    fig, ax = plt.subplots(
        figsize=(10, 12),
        nrows=3,
        tight_layout=True,
        sharex=True
    )

    # ---------------- Panel 1 ----------------
    # Volume (LHS)
    line_vol = ds_budget["domain_volume"].plot.line(
        ax=ax[0],
        add_legend=False,
        color='C0'
    )
    ax[0].set_ylabel("Domain Volume (m$^2$ Pa)")

    # Temperature (RHS)
    ax2 = ax[0].twinx()
    line_T = ds_budget["T_domain_avg"].plot.line(
        ax=ax2,
        add_legend=False,
        color='orange'
    )
    ax2.set_ylabel(r"$\langle T \rangle$ (K)")

    ax[0].set_title("Domain Volume and Average Temperature")

    # Manual legend control
    ax[0].legend(
        [line_vol[0], line_T[0]],
        ["Volume", r"$\langle T \rangle$"],
        loc="best"
    )

    # ---------------- Panel 2 ---------------
    term_signs = {
        "advection_term": -1,
        "adiabatic_term": 1,  # flip sign for adiabatic term
        "diabatic_term": 1
    }

    dT_dt = ds_budget["dT_dt"]

    dT_dt_2 = term_signs["advection_term"] * ds_budget["advection_term"] + \
             term_signs["adiabatic_term"] * ds_budget["adiabatic_term"] + \
             term_signs["diabatic_term"] * ds_budget["diabatic_term"]

    line_dT = dT_dt.plot.line(
        ax=ax[1],
        add_legend=False,
        color='C1'
    )

    line_dT_2 = dT_dt_2.plot.line(
        ax=ax[1],
        add_legend=False,
        color='C1',
        linestyle='--'
    )

    ax[1].axhline(0, color='k', linestyle='-', linewidth=1)

    ax[1].set_ylabel("dT/dt")
    ax[1].set_title("Storage Term")

    # ---------------- Panel 3 ----------------
    lines = []
    
    color_terms = {'advection_term': 'k', 'adiabatic_term': 'green', 'diabatic_term': 'red'}
    for var, label in [
        ("advection_term", "Net Heat Advection"),
        ("adiabatic_term", "Adiabatic Term"),
        ("diabatic_term", "Diabatic Term"),
    ]:
        line = (term_signs[var] * ds_budget[var]).plot.line(
            ax=ax[2],
            add_legend=False,
            color=color_terms[var]
        )
        lines.append(line[0])

    ax[2].set_ylabel("Integrated Heating")
    ax[2].set_title("Budget Terms")
    ax[2].legend(lines, [
        "Net Heat Advection",
        "Adiabatic Term",
        "Diabatic Term"
    ])

    ax[2].axhline(0, color='k', linestyle='-', linewidth=1)

    # Remove duplicate x-labels from upper panels
    ax[0].set_xlabel("")
    ax[1].set_xlabel("")

    out_path = os.path.join(plot_dir, "budget_terms_timeseries.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
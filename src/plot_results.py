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


def plot_budget_terms_hourly(ds_budget: xr.Dataset, smoothing_window: int, plot_dir: str) -> None:

    # raw hourly data
    domain_volume = ds_budget["domain_volume"].rolling(time=smoothing_window, center=True).mean().copy()  # smoothing to reduce noise, data is hourly
    T_domain_avg = ds_budget["T_domain_avg"].rolling(time=smoothing_window, center=True).mean().copy()  # smoothing to reduce noise, data is hourly
    ds_budget = ds_budget.rolling(time=smoothing_window, center=True).mean() # smoothing to reduce noise, data is hourly
    units = "[K/hr]"  # since we are averaging over 24 hours, the units are K/hr

    fig, ax = plt.subplots(
        figsize=(10, 12),
        nrows=3,
        tight_layout=True,
        sharex=True
    )

    # ---------------- Panel 1 ----------------
    # Volume (LHS)
    line_vol = domain_volume.plot.line(
        ax=ax[0],
        add_legend=False,
        color='C0'
    )
    ax[0].set_ylabel("Domain Volume (m$^2$ Pa)")

    # Temperature (RHS)
    ax2 = ax[0].twinx()
    line_T = T_domain_avg.plot.line(
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
        loc="best", fontsize=10
    )

    # ---------------- Panel 2 ---------------
    term_signs = {
        "advection_term": -1,
        "adiabatic_term": 1,  # flip sign for adiabatic term
        "diabatic_term": 1
    }

    norm_factor            = 1 / domain_volume
    time_conversion_factor = 3600

    dT_dt = ds_budget["dT_dt"] * norm_factor * time_conversion_factor  # convert to K/s by dividing by volume and multiplying by T scale (using domain average T as scale)

    dT_dt_2 = (term_signs["advection_term"] * ds_budget["advection_term"] + \
             term_signs["adiabatic_term"] * ds_budget["adiabatic_term"] + \
             term_signs["diabatic_term"] * ds_budget["diabatic_term"] ) * norm_factor * time_conversion_factor

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

    ax[1].set_ylabel(rf"dT/dt {units}")
    ax[1].set_title("Storage Term (normalized by volume)")

    # ---------------- Panel 3 ----------------
    lines = []
    
    color_terms = {'advection_term': 'k', 'adiabatic_term': 'green', 'diabatic_term': 'red'}

    error_var = None
    if "advection_error" in ds_budget.data_vars:
        error_var = "advection_error"


    for var, label in [
        ("advection_term", "Net Heat Advection"),
        ("adiabatic_term", "Adiabatic Term"),
        ("diabatic_term", "Diabatic Term"),
    ]:
        term = term_signs[var] * ds_budget[var] * norm_factor * time_conversion_factor
        line = term.plot.line(
            ax=ax[2],
            add_legend=False,
            color=color_terms[var]
        )


        if error_var is not None and var in {"advection_term", "diabatic_term"}:
            error = np.abs(ds_budget[error_var]) * norm_factor  * time_conversion_factor
            ax[2].fill_between(
                term["time"].values,
                (term - error).values,
                (term + error).values,
                color=color_terms[var],
                alpha=0.2,
                linewidth=0,
            )

        lines.append(line[0])

    ax[2].set_ylabel(f" {units}")
    ax[2].set_title("Budget Terms (normalized by volume)")
    ax[2].legend(lines, [
        "Net Heat Advection",
        "Adiabatic Term",
        "Diabatic Term"
    ], fontsize=10)

    ax[2].axhline(0, color='k', linestyle='-', linewidth=1)

    # Remove duplicate x-labels from upper panels
    ax[0].set_xlabel("")
    ax[1].set_xlabel("")

    out_path = os.path.join(plot_dir, f"budget_terms_timeseries_hourly_smoothwindow_{smoothing_window}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_budget_terms_day_bin(ds_budget: xr.Dataset, plot_dir: str) -> None:

    #alternative to smooth, sum over the rolling time window instead of averaging, to preserve the total budget over the window
    domain_volume = ds_budget["domain_volume"].resample(time="1D").mean()  # sum over 24 hours, data is hourly
    T_domain_avg  = ds_budget["T_domain_avg"].resample(time="1D").mean()  # sum over 24 hours, data is hourly
    ds_budget = ds_budget.resample(time="1D").sum()
    units = "[K/day]"  # since we are summing over 24 hours, the units  are K/day
    # not all terms within ds_budget are time-rate-of-change, so I need to apply the unit conversion from s->hr in the individual terms.

    fig, ax = plt.subplots(
        figsize=(10, 12),
        nrows=3,
        tight_layout=True,
        sharex=True
    )

    # ---------------- Panel 1 ----------------
    # Volume (LHS)
    line_vol = domain_volume.plot.line(
        ax=ax[0],
        add_legend=False,
        color='C0',
        drawstyle="steps-post"
    )
    ax[0].set_ylabel("Domain Volume (m$^2$ Pa)")

    # Temperature (RHS)
    ax2 = ax[0].twinx()
    line_T = T_domain_avg.plot.line(
        ax=ax2,
        add_legend=False,
        color='orange',
        drawstyle="steps-post"
    )
    ax2.set_ylabel(r"$\langle T \rangle$ (K)")

    ax[0].set_title("Domain Volume and Average Temperature")

    # Manual legend control
    ax[0].legend(
        [line_vol[0], line_T[0]],
        ["Volume", r"$\langle T \rangle$"],
        loc="best", fontsize=10
    )

    # ---------------- Panel 2 ---------------
    term_signs = {
        "advection_term": -1,
        "adiabatic_term": 1,  # flip sign for adiabatic term
        "diabatic_term": 1
    }

    norm_factor            = 1 / domain_volume
    time_conversion_factor = 3600

    #storage term
    ddt_TV = ds_budget["dT_dt"] * norm_factor * time_conversion_factor  # convert to K/s by dividing by volume and multiplying by T scale (using domain average T as scale)

    #change in internal energy from volume change

    dT_from_dV = (T_domain_avg * norm_factor) * ds_budget['dV_dt'] * time_conversion_factor

    #change in average energy
    # d<T>/dt
    dTT_dt = ddt_TV - dT_from_dV

    lines = []

    line_ddt_TV = ddt_TV.plot.line(
        ax=ax[1],
        add_legend=False,
        color='C1',
        drawstyle="steps-post"
    )
    lines.append(line_ddt_TV)

    line_dT_from_dV = dT_from_dV.plot.line(
        ax=ax[1],
        add_legend=False,
        color='C0',
        drawstyle="steps-post"
    )
    lines.append(line_dT_from_dV)

    line_dTT_dt = dTT_dt.plot.line(
        ax=ax[1],
        add_legend=False,
        color='C2',
        drawstyle="steps-post"
    )
    lines.append(line_dTT_dt)

    ax[1].legend(lines, [
        r"d/dt$\int T dV$",
        r"$\langle T \rangle$/V dV/dt",
        r"d$\langle T \rangle$/dt"
    ], fontsize=10)

    ax[1].axhline(0, color='k', linestyle='-', linewidth=1)

    ax[1].set_ylabel(rf"dT/dt {units}")
    ax[1].set_title("Storage Term (normalized by volume)")

    # ---------------- Panel 3 ----------------
    lines = []
    
    color_terms = {'advection_term': 'k', 'adiabatic_term': 'green', 'diabatic_term': 'red'}

    error_var = None
    if "advection_error" in ds_budget.data_vars:
        error_var = "advection_error"


    for var, label in [
        ("advection_term", "Net Heat Advection"),
        ("adiabatic_term", "Adiabatic Term"),
        ("diabatic_term", "Diabatic Term"),
    ]:
        term = term_signs[var] * ds_budget[var] * norm_factor * time_conversion_factor
        line = term.plot.line(
            ax=ax[2],
            add_legend=False,
            color=color_terms[var],
            drawstyle="steps-post"
        )


        if error_var is not None and var in {"advection_term", "diabatic_term"}:
            error = np.abs(ds_budget[error_var]) * norm_factor  * time_conversion_factor
            ax[2].fill_between(
                term["time"].values,
                (term - error).values,
                (term + error).values,
                color=color_terms[var],
                alpha=0.2,
                linewidth=0,
                step="post"
            )

        lines.append(line[0])


    # add faint vertical lines
    day_boundaries = ds_budget.time.values
    for a in ax:
        for t in day_boundaries:
            a.axvline(
                t,
                color="k",
                linewidth=0.3,
                alpha=0.2,
                zorder=0
            )



    ax[2].set_ylabel(f" {units}")
    ax[2].set_title("Budget Terms (normalized by volume)")
    ax[2].legend(lines, [
        "Net Heat Advection",
        "Adiabatic Term",
        "Diabatic Term"
    ], fontsize=10)

    ax[2].axhline(0, color='k', linestyle='-', linewidth=1)

    # Remove duplicate x-labels from upper panels
    ax[0].set_xlabel("")
    ax[1].set_xlabel("")

    ymax = max(
        abs(ax[1].get_ylim()[0]), abs(ax[1].get_ylim()[1]),
        abs(ax[2].get_ylim()[0]), abs(ax[2].get_ylim()[1]),
    )
    ax[1].set_ylim(-ymax, ymax)
    ax[2].set_ylim(-ymax, ymax)


    out_path = os.path.join(plot_dir, "budget_terms_timeseries_daily.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def plot_constant_T_results(ds_budget: xr.Dataset, ds_test:xr.Dataset, plot_dir: str) -> None:
    # comparison plot between original budget "uncertainty in advection" and test budget net heat advection

    norm_factor            = 1 / ds_budget["domain_volume"]
    time_conversion_factor = 3600

    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True, nrows=2, sharex=True)

    ax[0].plot(ds_budget["time"], ds_budget["advection_error"] * norm_factor * time_conversion_factor, label=r"$\delta M T_{scale}$", color='red')
    ax[0].plot(ds_test["time"], ds_test["advection_term"] * norm_factor * time_conversion_factor, label=r"$\mathcal{F}_{advection}$", color='k')

    ax[0].legend(fontsize=10)

    #integrated quantities
    dt = (ds_budget["time"][1] - ds_budget["time"][0]).values / np.timedelta64(1, 's')  # time step in seconds
    integrated_advection_error = (ds_budget["advection_error"] * norm_factor * dt).cumsum(dim="time")
    integrated_net_heat_advection = (ds_test["advection_term"] * norm_factor * dt).cumsum(dim="time")

    ax[1].plot(ds_budget["time"], integrated_advection_error, label=r"$T_{scale} \int \delta M dt$", color='red')
    ax[1].plot(ds_test["time"], integrated_net_heat_advection, label=r"$\int \mathcal{F}_{advection} dt$", color='k')

    ax[1].legend(fontsize=10)

    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Advection (K/hr)")
    ax[1].set_ylabel("Integrated Advection (K)")
    fig.suptitle("Comparison of Advection Error and Net Heat Advection (Constant T Test)")
    plt.savefig(os.path.join(plot_dir, "constant_T_advection_comparison.png"), bbox_inches="tight")
    plt.close(fig)

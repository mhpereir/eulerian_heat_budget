"""
Summary plots for completed budget runs.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .time_utils import integrate_rate_over_time, samples_for_duration


plt.rcParams.update({"font.size": 16})

SECONDS_PER_HOUR = 3600.0


def _rolling_window_samples(ds_budget: xr.Dataset, smoothing_duration_hours: float | None) -> int:
    if smoothing_duration_hours is None:
        return 1
    return samples_for_duration(ds_budget["time"], smoothing_duration_hours * SECONDS_PER_HOUR)


def _rate_to_per_hour(rate: xr.DataArray) -> xr.DataArray:
    return rate * SECONDS_PER_HOUR


def _daily_total(rate: xr.DataArray) -> xr.DataArray:
    return integrate_rate_over_time(rate).resample(time="1D").sum()


def _timeseries_suffix(smoothing_duration_hours: float | None) -> str:
    if smoothing_duration_hours is None:
        return "raw"
    duration_text = f"{smoothing_duration_hours:g}".replace(".", "p")
    return f"smooth_{duration_text}h"


def plot_budget_terms_timeseries(
    ds_budget: xr.Dataset,
    plot_dir: str,
    *,
    smoothing_duration_hours: float | None = None,
) -> None:
    norm_factor = 1 / ds_budget["domain_volume"]
    smoothing_window = _rolling_window_samples(ds_budget, smoothing_duration_hours)
    units = "[K/hr]"

    domain_volume = ds_budget["domain_volume"].rolling(time=smoothing_window, center=True).mean().copy()
    T_domain_avg = ds_budget["T_domain_avg"].rolling(time=smoothing_window, center=True).mean().copy()

    fig, ax = plt.subplots(figsize=(10, 12), nrows=3, tight_layout=True, sharex=True)

    line_vol = domain_volume.plot.line(ax=ax[0], add_legend=False, color="C0")
    ax[0].set_ylabel("Domain Volume (m$^2$ Pa)")

    ax2 = ax[0].twinx()
    line_T = T_domain_avg.plot.line(ax=ax2, add_legend=False, color="orange")
    ax2.set_ylabel(r"$\langle T \rangle$ (K)")

    ax[0].set_title("Domain Volume and Average Temperature")
    ax[0].legend(
        [line_vol[0], line_T[0]],
        ["Volume", r"$\langle T \rangle$"],
        loc="best",
        fontsize=10,
    )

    term_signs = {
        "advection_term": -1,
        "adiabatic_term": 1,
        "diabatic_term": 1,
    }

    dT_dt = _rate_to_per_hour(ds_budget["dT_dt"] * norm_factor)
    dT_dt = dT_dt.rolling(time=smoothing_window, center=True).mean()

    dT_dt_2 = (
        term_signs["advection_term"] * ds_budget["advection_term"]
        + term_signs["adiabatic_term"] * ds_budget["adiabatic_term"]
        + term_signs["diabatic_term"] * ds_budget["diabatic_term"]
    ) * norm_factor
    dT_dt_2 = _rate_to_per_hour(dT_dt_2).rolling(time=smoothing_window, center=True).mean()

    line_dT = dT_dt.plot.line(ax=ax[1], add_legend=False, color="C3")
    line_dT_2 = dT_dt_2.plot.line(ax=ax[1], add_legend=False, color="C0", linestyle="--")

    ax[1].axhline(0, color="k", linestyle="-", linewidth=1)
    ax[1].set_ylabel(rf"dT/dt {units}")
    ax[1].set_title("Storage Term (normalized by volume)")
    ax[1].legend(
        [line_dT[0], line_dT_2[0]],
        [r"d$\langle T \rangle$/dt", "Budget sum"],
        fontsize=10,
    )

    lines = []
    color_terms = {"advection_term": "k", "adiabatic_term": "green", "diabatic_term": "red"}

    error_var = "advection_error" if "advection_error" in ds_budget.data_vars else None

    for var in ("advection_term", "adiabatic_term", "diabatic_term"):
        term = _rate_to_per_hour(term_signs[var] * ds_budget[var] * norm_factor)
        term = term.rolling(time=smoothing_window, center=True).mean()

        line = term.plot.line(ax=ax[2], add_legend=False, color=color_terms[var])

        if error_var is not None and var in {"advection_term", "diabatic_term"}:
            error = _rate_to_per_hour(ds_budget[error_var].abs() * norm_factor)
            error = error.rolling(time=smoothing_window, center=True).mean()  # type: ignore[assignment]
            ax[2].fill_between(
                term["time"].values,
                (term - error).values,
                (term + error).values,
                color=color_terms[var],
                alpha=0.2,
                linewidth=0,
            )

        lines.append(line[0])

    ax[2].set_ylabel(units)
    ax[2].set_title("Budget Terms (normalized by volume)")
    ax[2].legend(
        lines,
        ["Net Heat Advection", "Adiabatic Term", "Diabatic Term"],
        fontsize=10,
    )
    ax[2].axhline(0, color="k", linestyle="-", linewidth=1)

    ax[0].set_xlabel("")
    ax[1].set_xlabel("")

    out_path = os.path.join(
        plot_dir,
        f"budget_terms_timeseries_{_timeseries_suffix(smoothing_duration_hours)}.png",
    )
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_budget_terms_daily(ds_budget: xr.Dataset, plot_dir: str) -> None:
    norm_factor = 1 / ds_budget["domain_volume"]
    units = "[K/day]"

    domain_volume = ds_budget["domain_volume"].resample(time="1D").mean()
    T_domain_avg = ds_budget["T_domain_avg"].resample(time="1D").mean()

    fig, ax = plt.subplots(figsize=(10, 12), nrows=3, tight_layout=True, sharex=True)

    line_vol = domain_volume.plot.line(ax=ax[0], add_legend=False, color="C0", drawstyle="steps-post")
    ax[0].set_ylabel("Domain Volume (m$^2$ Pa)")

    ax2 = ax[0].twinx()
    line_T = T_domain_avg.plot.line(ax=ax2, add_legend=False, color="orange", drawstyle="steps-post")
    ax2.set_ylabel(r"$\langle T \rangle$ (K)")

    ax[0].set_title("Domain Volume and Average Temperature")
    ax[0].legend(
        [line_vol[0], line_T[0]],
        ["Volume", r"$\langle T \rangle$"],
        loc="best",
        fontsize=10,
    )

    term_signs = {
        "advection_term": -1,
        "adiabatic_term": 1,
        "diabatic_term": 1,
    }

    ddt_TV = _daily_total(ds_budget["d_dt_T"] * norm_factor)
    dT_from_dV = _daily_total((ds_budget["T_domain_avg"] * norm_factor) * ds_budget["dV_dt"])
    dTT_dt = _daily_total(ds_budget["dT_dt"] * norm_factor)

    lines_dT = []
    line_ddt_TV = ddt_TV.plot.line(ax=ax[1], add_legend=False, color="C1", drawstyle="steps-post")
    lines_dT.append(line_ddt_TV[0])

    line_dT_from_dV = dT_from_dV.plot.line(ax=ax[1], add_legend=False, color="C0", drawstyle="steps-post")
    lines_dT.append(line_dT_from_dV[0])

    line_dTT_dt = dTT_dt.plot.line(ax=ax[1], add_legend=False, color="C2", drawstyle="steps-post")
    lines_dT.append(line_dTT_dt[0])

    ax[1].legend(
        lines_dT,
        [r"d/dt$\int T dV$", r"$\langle T \rangle$/V dV/dt", r"d$\langle T \rangle$/dt"],
        fontsize=10,
    )
    ax[1].axhline(0, color="k", linestyle="-", linewidth=1)
    ax[1].set_ylabel(units)
    ax[1].set_title("Storage Term (normalized by volume)")

    lines = []
    color_terms = {"advection_term": "k", "adiabatic_term": "green", "diabatic_term": "red"}
    error_var = "advection_error" if "advection_error" in ds_budget.data_vars else None

    for var in ("advection_term", "adiabatic_term", "diabatic_term"):
        term = _daily_total(term_signs[var] * ds_budget[var] * norm_factor)
        line = term.plot.line(ax=ax[2], add_legend=False, color=color_terms[var], drawstyle="steps-post")

        if error_var is not None and var in {"advection_term", "diabatic_term"}:
            error = _daily_total(ds_budget[error_var].abs() * norm_factor)
            ax[2].fill_between(
                term["time"].values,
                (term - error).values,
                (term + error).values,
                color=color_terms[var],
                alpha=0.2,
                linewidth=0,
                step="post",
            )

        lines.append(line[0])

    day_boundaries = domain_volume.time.values
    for axis in ax:
        for t in day_boundaries:
            axis.axvline(t, color="k", linewidth=0.3, alpha=0.2, zorder=0)

    ax[2].set_ylabel(units)
    ax[2].set_title("Budget Terms (normalized by volume)")
    ax[2].legend(
        lines,
        ["Net Heat Advection", "Adiabatic Term", "Diabatic Term"],
        fontsize=10,
    )
    ax[2].axhline(0, color="k", linestyle="-", linewidth=1)

    ax[0].set_xlabel("")
    ax[1].set_xlabel("")

    ymax = max(
        abs(ax[1].get_ylim()[0]),
        abs(ax[1].get_ylim()[1]),
        abs(ax[2].get_ylim()[0]),
        abs(ax[2].get_ylim()[1]),
    )
    ax[1].set_ylim(-ymax, ymax)
    ax[2].set_ylim(-ymax, ymax)

    out_path = os.path.join(plot_dir, "budget_terms_timeseries_daily.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_constant_T_results(ds_budget: xr.Dataset, ds_test: xr.Dataset, plot_dir: str) -> None:
    norm_factor = 1 / ds_budget["domain_volume"]

    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True, nrows=2, sharex=True)

    advection_error_rate = _rate_to_per_hour(ds_budget["advection_error"] * norm_factor)
    advection_term_rate = _rate_to_per_hour(ds_test["advection_term"] * norm_factor)

    ax[0].plot(ds_budget["time"], advection_error_rate, label=r"$\delta M T_{scale}$", color="red")
    ax[0].plot(ds_test["time"], advection_term_rate, label=r"$\mathcal{F}_{advection}$", color="k")
    ax[0].legend(fontsize=10)

    integrated_advection_error = integrate_rate_over_time(ds_budget["advection_error"] * norm_factor).cumsum(dim="time")
    integrated_net_heat_advection = integrate_rate_over_time(ds_test["advection_term"] * norm_factor).cumsum(dim="time")

    ax[1].plot(ds_budget["time"], integrated_advection_error, label=r"$T_{scale} \int \delta M dt$", color="red")
    ax[1].plot(ds_test["time"], integrated_net_heat_advection, label=r"$\int \mathcal{F}_{advection} dt$", color="k")
    ax[1].legend(fontsize=10)

    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Advection (K/hr)")
    ax[1].set_ylabel("Integrated Advection (K)")
    fig.suptitle("Comparison of Advection Error and Net Heat Advection (Constant T Test)")
    plt.savefig(os.path.join(plot_dir, "constant_T_advection_comparison.png"), bbox_inches="tight")
    plt.close(fig)

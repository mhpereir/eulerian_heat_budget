'''

'''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText

from . import terms

SINGLE_COLUMN_WIDTH_IN = 6
FULL_TWO_COLUMN_WIDTH_IN = 12

PAPER_FONT_SIZE_PT = 14
LEGEND_FONT_SIZE_PT = 10
LINE_WIDTH_PT = 1
SCATTER_SIZE_PT2 = 16
SCATTER_ALPHA = 0.3

SINGLE_PANEL_ASPECT = 0.6
TWO_PANEL_STACK_ASPECT = 0.95
THREE_PANEL_STACK_ASPECT = 1.2
SQUARE_PANEL_ASPECT = TWO_PANEL_STACK_ASPECT
MASS_FLUX_UNITS = r"m$^2$ Pa s$^{-1}$"
HEAT_FLUX_UNITS = r"K m$^2$ Pa s$^{-1}$"
ONE_TO_ONE_XLABEL_X = 0.43
ONE_TO_ONE_FIGURE_LAYOUT = {
    "left": 0.15,
    "right": 0.97,
    "bottom": 0.20,
    "top": 0.88,
}
STACKED_FIGURE_LAYOUTS = {
    2: {"left": 0.2, "right": 0.97, "bottom": 0.17, "top": 0.88, "hspace": 0.30},
    3: {"left": 0.18, "right": 0.97, "bottom": 0.15, "top": 0.89, "hspace": 0.40},
}

plt.rcParams.update(
    {
        "font.size": PAPER_FONT_SIZE_PT,
        "axes.labelsize": PAPER_FONT_SIZE_PT,
        "axes.titlesize": PAPER_FONT_SIZE_PT,
        "xtick.labelsize": PAPER_FONT_SIZE_PT,
        "ytick.labelsize": PAPER_FONT_SIZE_PT,
        "legend.fontsize": LEGEND_FONT_SIZE_PT,
        "legend.title_fontsize": LEGEND_FONT_SIZE_PT,
        "figure.titlesize": PAPER_FONT_SIZE_PT,
        "lines.linewidth": LINE_WIDTH_PT,
    }
)


def _publication_figsize(width: str = "single", aspect: float = SINGLE_PANEL_ASPECT):
    widths = {
        "single": SINGLE_COLUMN_WIDTH_IN,
        "full": FULL_TWO_COLUMN_WIDTH_IN,
    }
    figure_width = widths[width]
    return figure_width, figure_width * aspect


def _date_locator_formatter():
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    return locator, formatter


def _apply_stacked_figure_layout(fig, nrows: int):
    fig.subplots_adjust(**STACKED_FIGURE_LAYOUTS[nrows])


def _pad_upper_ylim(ax, fraction: float = 0.15):
    lower, upper = ax.get_ylim()
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        return
    ax.set_ylim(lower, upper + span * fraction)


def _set_square_1_to_1_axis(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.xaxis.get_offset_text().set_fontsize(PAPER_FONT_SIZE_PT)
    ax.yaxis.get_offset_text().set_fontsize(PAPER_FONT_SIZE_PT)


def _set_square_xlabel(ax, label: str):
    ax.set_xlabel(label)
    ax.xaxis.set_label_coords(ONE_TO_ONE_XLABEL_X, -0.14)


def _add_correlation_text(ax, text: str):
    anchored_text = AnchoredText(
        text,
        loc="upper left",
        prop={"size": LEGEND_FONT_SIZE_PT},
        frameon=True,
        borderpad=0.6,
        pad=0.2,
    )
    anchored_text.patch.set_facecolor("white")
    anchored_text.patch.set_edgecolor("none")
    anchored_text.patch.set_alpha(0.8)
    ax.add_artist(anchored_text)


def _prepare_square_figure_for_save(fig):
    fig.subplots_adjust(**ONE_TO_ONE_FIGURE_LAYOUT)
    fig.canvas.draw()


def _save_square_figure(fig, path: str):
    _prepare_square_figure_for_save(fig)
    fig.savefig(path, dpi=300)


def _mass_flux_wall_label(var_name: str) -> str:
    prefix = "mass_flux_contribution_"
    if var_name.startswith(prefix):
        return var_name[len(prefix):].replace("_", " ").title()
    return var_name


def fig1_mass_continuity(dV_dt: xr.DataArray, advection_terms: xr.Dataset, plot_dir: str):
    #test how mass advection terms compare to volume changes (should be y=-x)
    fig,ax = plt.subplots(
        figsize=_publication_figsize("single", SQUARE_PANEL_ASPECT),
    )
    x= dV_dt.values
    y= advection_terms['net_mass_advection'].values
    ax.scatter(x, y, s=SCATTER_SIZE_PT2)
    #plot 1:-1 line
    # symmetric limits
    L = np.nanmax(np.abs(np.concatenate([x, y])))

    ax.plot([-L, L], [-L, L], linestyle='--', color='k')

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    _set_square_1_to_1_axis(ax)

    _set_square_xlabel(ax, f"dV/dt [{MASS_FLUX_UNITS}]")
    ax.set_ylabel(f"Net mass flux [{MASS_FLUX_UNITS}]")

    _save_square_figure(fig, plot_dir + '/fig1_mass_continuity.png')
    plt.close(fig)

    x = dV_dt.values
    y = advection_terms['net_mass_advection'].values
    r = y - x

    # bin by x
    nbin = 20
    edges = np.quantile(x, np.linspace(0, 1, nbin+1))
    bin_id = np.digitize(x, edges[1:-1])

    xb = np.array([np.nanmean(x[bin_id==i]) for i in range(nbin)])
    rb = np.array([np.nanmean(r[bin_id==i]) for i in range(nbin)])

    plt.figure(
        figsize=_publication_figsize("single", SINGLE_PANEL_ASPECT),
        tight_layout=True,
    )
    plt.plot(xb, rb, marker='o')
    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel("dV/dt")
    plt.ylabel(rf"[{MASS_FLUX_UNITS}]")
    plt.title("Net mass flux residual")
    plt.savefig(plot_dir + "/fig1.3_mass_continuity_binned_residual.png", dpi=300)

    plt.close()


def fig1_benchmark_mass_continuity(
    dV_dt: xr.DataArray,
    advection_terms: xr.Dataset,
    dV_dt_true: xr.DataArray,
    benchmark_mass_flux_net: xr.DataArray,
    plot_dir: str,
):
    def _finite_pair(x: xr.DataArray, y: xr.DataArray):
        x_values = x.values.ravel()
        y_values = y.values.ravel()
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        return x_values[mask], y_values[mask]

    def _axis_limits(x: np.ndarray, y: np.ndarray):
        all_values = np.concatenate([x, y])
        if all_values.size == 0:
            return -1.0, 1.0

        lower = np.nanmin(all_values)
        upper = np.nanmax(all_values)
        if not np.isfinite(lower) or not np.isfinite(upper):
            return -1.0, 1.0

        if lower == upper:
            pad = max(np.abs(lower) * 0.05, 1.0)
        else:
            pad = (upper - lower) * 0.05
        return lower - pad, upper + pad

    def _correlation_text(x: np.ndarray, y: np.ndarray):
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            return f"Pearson r: n/a\nn = {x.size}"
        r = np.corrcoef(x, y)[0, 1]
        return f"Pearson r: {r:.3f}\nn = {x.size}"

    def _plot_benchmark_vs_calculated(
        benchmark: xr.DataArray,
        calculated: xr.DataArray,
        xlabel: str,
        ylabel: str,
        title: str,
        filename: str,
    ):
        benchmark_aligned, calculated_aligned = xr.align(
            benchmark,
            calculated,
            join="inner",
        )
        x, y = _finite_pair(benchmark_aligned, calculated_aligned)
        lower, upper = _axis_limits(x, y)

        fig, ax = plt.subplots(
            figsize=_publication_figsize("single", SQUARE_PANEL_ASPECT),
        )
        ax.scatter(x, y, alpha=SCATTER_ALPHA, s=SCATTER_SIZE_PT2)
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="k", label="1:1")

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        _set_square_1_to_1_axis(ax)
        _set_square_xlabel(ax, xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _add_correlation_text(ax, _correlation_text(x, y))
        ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE_PT)

        _save_square_figure(fig, plot_dir + "/" + filename)
        plt.close(fig)

    eulerian_x, eulerian_y = xr.align(
        dV_dt,
        advection_terms["net_mass_advection"],
        join="inner",
    )
    benchmark_x, benchmark_y = xr.align(
        dV_dt_true,
        benchmark_mass_flux_net,
        join="inner",
    )

    x_e = eulerian_x.values.ravel()
    y_e = eulerian_y.values.ravel()
    x_b = benchmark_x.values.ravel()
    y_b = benchmark_y.values.ravel()

    mask_e = np.isfinite(x_e) & np.isfinite(y_e)
    mask_b = np.isfinite(x_b) & np.isfinite(y_b)

    all_values = np.concatenate(
        [x_e[mask_e], y_e[mask_e], x_b[mask_b], y_b[mask_b]]
    )
    L = np.nanmax(np.abs(all_values)) if all_values.size else 1.0
    if not np.isfinite(L) or L == 0:
        L = 1.0

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", SQUARE_PANEL_ASPECT),
    )
    ax.scatter(
        x_e[mask_e],
        y_e[mask_e],
        alpha=SCATTER_ALPHA,
        label="Eulerian",
        s=SCATTER_SIZE_PT2,
    )
    ax.scatter(
        x_b[mask_b],
        y_b[mask_b],
        alpha=SCATTER_ALPHA,
        label="Benchmark",
        s=SCATTER_SIZE_PT2,
    )
    ax.plot([-L, L], [-L, L], linestyle="--", color="k", label="1:1")

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    _set_square_1_to_1_axis(ax)
    _set_square_xlabel(ax, f"dV/dt [{MASS_FLUX_UNITS}]")
    ax.set_ylabel(f"Net mass flux [{MASS_FLUX_UNITS}]")
    ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE_PT)

    _save_square_figure(fig, plot_dir + "/fig1_benchmark_mass_continuity.png")
    plt.close(fig)

    calculated_mass_flux_lateral = advection_terms["net_mass_advection"]
    for vertical_flux in (
        "mass_flux_contribution_top",
        "mass_flux_contribution_bottom",
    ):
        if vertical_flux in advection_terms:
            calculated_mass_flux_lateral = (
                calculated_mass_flux_lateral - advection_terms[vertical_flux]
            )

    _plot_benchmark_vs_calculated(
        benchmark_mass_flux_net,
        calculated_mass_flux_lateral,
        f"Benchmark [{MASS_FLUX_UNITS}]",
        f"Calculated [{MASS_FLUX_UNITS}]",
        "Net Mass Flux",
        "fig1.1_benchmark_vs_calculated_mass_flux.png",
    )
    _plot_benchmark_vs_calculated(
        dV_dt_true,
        dV_dt,
        f"Benchmark [{MASS_FLUX_UNITS}]",
        f"Calculated [{MASS_FLUX_UNITS}]",
        "dV/dt",
        "fig1.2_benchmark_vs_calculated_dV_dt.png",
    )




def fig2_mass_advection_residual_timeseries(advection_terms: xr.Dataset, dV_dt: xr.DataArray, domain_volume: xr.DataArray, plot_dir: str):

    
    norm_factor          = 1/ domain_volume
    mean_norm_factor     = 1/np.nanmean(domain_volume)
    time_rate_conversion = 3600

    neg_dV_dt = - dV_dt

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

    delta_mass = advection_terms['net_mass_advection'] - dV_dt

    ax[0].plot(delta_mass['time'], delta_mass * mean_norm_factor * time_rate_conversion, label=r'$\delta M$', color='k', alpha=0.8)

    ax[0].plot(delta_mass['time'], advection_terms['net_mass_advection']* mean_norm_factor * time_rate_conversion, 
               label='Net Mass Advection', alpha=0.5, color='C0')
    ax[0].plot(delta_mass['time'], neg_dV_dt * mean_norm_factor * time_rate_conversion, 
               label='-dV/dt', alpha=0.5, color='C1')

    ax[0].legend(fontsize=LEGEND_FONT_SIZE_PT)
    # ax[0].set_xlabel("Time")
    ax[0].set_ylabel("[1/hr]")
    ax[0].set_title("Mass residual (normalized)")
    
    #check advection_terms['time'] is in the correct format:
    if np.issubdtype(advection_terms['time'].dtype, np.datetime64):
        print("Time coordinate is in datetime format.")
    else:
        print("Time coordinate is NOT in dneg_atetime format. Check your dataset.")
        raise ValueError("Time coordinate is not in datetime format.")

    dt = (advection_terms["time"][1] - advection_terms["time"][0]).values / np.timedelta64(1, 's') # convert to seconds
    cumulative_residual = np.cumsum( delta_mass * dt)
    cumulative_net_adv  = np.cumsum(advection_terms['net_mass_advection'] * dt)
    cumulative_dvdt     = np.cumsum(neg_dV_dt * dt)

    ax[1].plot(advection_terms['time'], cumulative_residual * mean_norm_factor, color='k')
    
    ax[1].plot(advection_terms['time'], cumulative_net_adv * mean_norm_factor, color='C0')
    ax[1].plot(advection_terms['time'], cumulative_dvdt * mean_norm_factor, color='C1')

    locator, formatter = _date_locator_formatter()

    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("[unitless]")
    ax[1].set_title("Cumulative mass residual (normalized)")
    _apply_stacked_figure_layout(fig, 2)
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

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", SINGLE_PANEL_ASPECT),
        tight_layout=True,
    )
    ax.plot(advection_terms['time'], eps_mass)

    locator, formatter = _date_locator_formatter()

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

    fig,ax = plt.subplots(
        figsize=_publication_figsize("single", TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

    advection_terms_smoothed = advection_terms.rolling(time=24, center=True).mean()

    for var in mass_vars:
        ax[0].plot(
            advection_terms_smoothed['time'],
            advection_terms_smoothed[var] * mean_norm_factor * time_rate_conversion,
            label=_mass_flux_wall_label(var),
        )

        if var.split('_')[-1] in ['east', 'west']:
            net_zonal_mass_advection += advection_terms_smoothed[var].values* mean_norm_factor * time_rate_conversion
        elif var.split('_')[-1] in ['north', 'south']:
            net_meridional_mass_advection += advection_terms_smoothed[var].values* mean_norm_factor * time_rate_conversion
        elif var.split('_')[-1] in ['top', 'bottom']:
            net_vertical_mass_advection += advection_terms_smoothed[var].values* mean_norm_factor * time_rate_conversion

    net_horizontal_mass_advection     = net_zonal_mass_advection + net_meridional_mass_advection

    ax[1].plot(advection_terms_smoothed['time'], delta_mass * mean_norm_factor * time_rate_conversion, label=r'$\delta M$', color='k', linewidth=LINE_WIDTH_PT)

    # ax[1].plot(advection_terms_smoothed['time'], net_zonal_mass_advection, label='Zonal', linestyle='--')
    # ax[1].plot(advection_terms_smoothed['time'], net_meridional_mass_advection, label='Meridional', linestyle='--')
    ax[1].plot(advection_terms_smoothed['time'], net_horizontal_mass_advection, label='Horizontal')
    ax[1].plot(advection_terms_smoothed['time'], net_vertical_mass_advection, label='Vertical')

    ax[0].set_title('Mass Advection Terms (normalized)')
    ax[1].set_title('Net Mass Advection (normalized)')

    locator, formatter = _date_locator_formatter()

    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("[1/hr]")
    ax[1].set_ylabel("[1/hr]")

    _pad_upper_ylim(ax[0], fraction=0.15)
    ax[0].legend(
        fontsize=LEGEND_FONT_SIZE_PT,
        ncol=min(3, max(1, len(mass_vars))),
        loc="upper center",
    )
    ax[1].legend(fontsize=LEGEND_FONT_SIZE_PT)
    _apply_stacked_figure_layout(fig, 2)
    plt.savefig(plot_dir + '/fig2.2_mass_advection_terms_timeseries.png', dpi=300)
    plt.close()
    
    


def fig3_advection_components_timeseries(advection_terms: xr.Dataset, dV_dt: xr.DataArray, delta_heat: xr.DataArray, domain_volume:xr.DataArray, plot_dir: str):

    norm_factor = 1 / domain_volume
    time_rate_conversion = 3600 # convert from per second to per hour

    # delta_mass = dV_dt + advection_terms['net_mass_advection']


    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

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

    ax[0].plot(advection_terms['time'], advection_terms['advection_term'] * norm_factor * time_rate_conversion, label='Net Heat Advection', linewidth=LINE_WIDTH_PT, color='k')

    ax[0].set_ylabel("[K / hr]")
    ax[0].set_title("Heat advection components")

    ax[1].plot(advection_terms['time'], advection_terms['advection_term'] * norm_factor * time_rate_conversion, label='Net Heat Advection', linewidth=LINE_WIDTH_PT, color='k')
    ax[1].plot(advection_terms['time'], delta_heat * norm_factor * time_rate_conversion, label='Expected Residual', linewidth=LINE_WIDTH_PT, color='red')

    ax[1].set_title("Net heat advection")
    ax[1].set_ylabel('[K / hr]')

    locator, formatter = _date_locator_formatter()

    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[0].legend(fontsize=LEGEND_FONT_SIZE_PT)
    ax[1].legend(fontsize=LEGEND_FONT_SIZE_PT)
    ax[1].set_xlabel("Time")
    _apply_stacked_figure_layout(fig, 2)
    plt.savefig(plot_dir + '/fig3_advection_components_timeseries.png', dpi=300)
    plt.close()




def fig4_temperature_derivative_timeseries(d_dt_T: xr.DataArray, dT_dt_1:xr.DataArray, dT_dt_2:xr.DataArray, domain_volume: xr.DataArray, plot_dir: str):

    norm_factor = 1 / domain_volume
    time_rate_conversion = 3600 # convert from per second to per hour

    # delta_mass = dV_dt + advection_terms['net_mass_advection']

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

    ax[0].plot(d_dt_T['time'], d_dt_T * norm_factor * time_rate_conversion, label=r'd/dt $\int TdV$', linewidth=LINE_WIDTH_PT, color='k')

    ax[0].set_ylabel("[K / hr]")
    ax[0].set_title("Storage term")

    ax[1].plot(dT_dt_2['time'], dT_dt_2 * norm_factor * time_rate_conversion, label=r'd/dt$\int TdV$-$\langle T \rangle $dV/dt', linewidth=LINE_WIDTH_PT, color='b')
    ax[1].plot(dT_dt_1['time'], dT_dt_1 * norm_factor * time_rate_conversion, label=r'd$\langle T \rangle $/dt', linewidth=LINE_WIDTH_PT, color='k')
    
    ax[1].set_title("Temperature tendency")
    ax[1].set_ylabel('[K / hr]')

    locator, formatter = _date_locator_formatter()

    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[0].legend(fontsize=LEGEND_FONT_SIZE_PT)
    ax[1].legend(fontsize=LEGEND_FONT_SIZE_PT)
    ax[1].set_xlabel("Time")
    _apply_stacked_figure_layout(fig, 2)
    plt.savefig(plot_dir + '/fig4_temperature_derivative_timeseries.png', dpi=300)
    plt.close()



def fig5_benchmark_comparison(
    benchmark_mass_fluxes: xr.Dataset,
    benchmark_heat_fluxes: xr.Dataset,
    results: xr.Dataset,
    advection_terms: xr.Dataset,
    plot_dir: str,
):
    benchmark_diagnostic_totals = terms.compute_benchmark_diagnostic_totals(
        benchmark_mass_fluxes,
        benchmark_heat_fluxes,
        results,
        advection_terms,
    )

    benchmark_mass_fluxes = - benchmark_mass_fluxes
    benchmark_heat_fluxes = - benchmark_heat_fluxes
    
    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

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
        benchmark_diagnostic_totals["time"],
        benchmark_diagnostic_totals["benchmark_mass_flux_net"],
        linestyle="--",
        linewidth=LINE_WIDTH_PT,
        color="k",
    )
    ax[0].plot(
        benchmark_diagnostic_totals["time"],
        benchmark_diagnostic_totals["calculated_mass_flux_net_lateral"],
        linestyle="-",
        linewidth=LINE_WIDTH_PT,
        color="k",
    )

    ax[1].plot(
        benchmark_diagnostic_totals["time"],
        benchmark_diagnostic_totals["benchmark_heat_flux_net"],
        linestyle="--",
        linewidth=LINE_WIDTH_PT,
        color="k",
        label="Benchmark net",
    )
    ax[1].plot(
        benchmark_diagnostic_totals["time"],
        benchmark_diagnostic_totals["calculated_heat_flux_net_lateral_full"],
        linestyle="-",
        linewidth=LINE_WIDTH_PT,
        color="k",
        label="Calculated net",
    )

    locator, formatter = _date_locator_formatter()
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[0].set_ylabel(rf"[{MASS_FLUX_UNITS}]")
    ax[0].set_title("Mass flux comparison")

    face_handles = [
        Line2D([0], [0], color=colors["north"], lw=LINE_WIDTH_PT, label="North"),
        Line2D([0], [0], color=colors["south"], lw=LINE_WIDTH_PT, label="South"),
        Line2D([0], [0], color=colors["east"],  lw=LINE_WIDTH_PT, label="East"),
        Line2D([0], [0], color=colors["west"],  lw=LINE_WIDTH_PT, label="West"),
        Line2D([0], [0], color="k", lw=LINE_WIDTH_PT, label="Net"),
    ]
    style_handles = [
        Line2D([0], [0], color="0.3", lw=LINE_WIDTH_PT, linestyle="--", label="Benchmark"),
        Line2D([0], [0], color="0.3", lw=LINE_WIDTH_PT, linestyle="-",  label="Calculated"),
    ]

    leg1 = ax[0].legend(
        handles=face_handles,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE_PT,
    )
    ax[0].add_artist(leg1)
    ax[0].legend(
        handles=style_handles,
        loc="lower left",
        fontsize=LEGEND_FONT_SIZE_PT,
    )

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel(rf"[{HEAT_FLUX_UNITS}]")
    ax[1].set_title("Heat-flux comparison")
    ax[1].legend(fontsize=LEGEND_FONT_SIZE_PT)

    _apply_stacked_figure_layout(fig, 2)
    plt.savefig(plot_dir + "/fig5_benchmark_comparison.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", THREE_PANEL_STACK_ASPECT),
        nrows=3,
        sharex=True,
    )

    # net lines
    ax[0].plot(
        benchmark_diagnostic_totals["time"],
        benchmark_diagnostic_totals["benchmark_mass_flux_net"],
        linestyle="--",
        color="k",
    )
    ax[0].plot(
        benchmark_diagnostic_totals["time"],
        benchmark_diagnostic_totals["calculated_mass_flux_net_lateral"],
        linestyle="-",
        linewidth=LINE_WIDTH_PT,
        color="k",
    )

    ax[1].plot(
        benchmark_diagnostic_totals["time"],
        benchmark_diagnostic_totals["benchmark_heat_flux_net"],
        linestyle="--",
        linewidth=LINE_WIDTH_PT,
        color="k",
        label=r"$\mathcal{H}_{bench, full}$",
    )
    ax[1].plot(
        benchmark_diagnostic_totals["time"],
        benchmark_diagnostic_totals["calculated_heat_flux_net_lateral_full"],
        linestyle="-",
        linewidth=LINE_WIDTH_PT,
        color="k",
        label=r"$\mathcal{H}'_{calc} + \langle T \rangle M_{calc} $"
    )

    ax[2].plot(
        benchmark_diagnostic_totals["time"],
        benchmark_diagnostic_totals["calculated_heat_flux_net_lateral"],
        linestyle="-",
        linewidth=LINE_WIDTH_PT,
        color="red",
        label=r"$\mathcal{H}'_{calc}$",
    )
    ax[2].plot(
        benchmark_diagnostic_totals["time"],
        benchmark_diagnostic_totals["benchmark_heat_flux_net_lateral_prime"],
        linestyle="--",
        linewidth=LINE_WIDTH_PT,
        color="red",
        label=r"$\mathcal{H}'_{bench}$",
    )

    locator, formatter = _date_locator_formatter()
    ax[2].xaxis.set_major_locator(locator)
    ax[2].xaxis.set_major_formatter(formatter)

    ax[0].set_ylabel(rf"[{MASS_FLUX_UNITS}]")
    ax[0].set_title("Mass flux comparison")

    # face_handles = [
    #     Line2D([0], [0], color=colors["north"], lw=LINE_WIDTH_PT, label="North"),
    #     Line2D([0], [0], color=colors["south"], lw=LINE_WIDTH_PT, label="South"),
    #     Line2D([0], [0], color=colors["east"],  lw=LINE_WIDTH_PT, label="East"),
    #     Line2D([0], [0], color=colors["west"],  lw=LINE_WIDTH_PT, label="West"),
    #     Line2D([0], [0], color="k", lw=LINE_WIDTH_PT, label="Net"),
    # ]
    style_handles = [
        Line2D([0], [0], color="0.3", lw=LINE_WIDTH_PT, linestyle="--", label="Benchmark"),
        Line2D([0], [0], color="0.3", lw=LINE_WIDTH_PT, linestyle="-",  label="Calculated"),
    ]

    # leg1 = ax[0].legend(
    #     handles=face_handles,
    #     loc="upper left",
    #     fontsize=LEGEND_FONT_SIZE_PT,
    # )
    # ax[0].add_artist(leg1)
    ax[0].legend(
        handles=style_handles,
        loc="lower left",
        fontsize=LEGEND_FONT_SIZE_PT,
    )

    ax[1].set_ylabel(r"[K m$^2$ Pa s$^{-1}$]")
    ax[1].set_title(r"Total Heat-flux ($\mathcal{H}$) comparison")
    ax[1].legend(fontsize=LEGEND_FONT_SIZE_PT)
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel(r"[K m$^2$ Pa s$^{-1}$]")
    ax[2].set_title(r"T' Heat-flux ($\mathcal{H}'$) comparison")
    ax[2].legend(fontsize=LEGEND_FONT_SIZE_PT)

    _apply_stacked_figure_layout(fig, 3)
    plt.savefig(plot_dir + "/fig5.1_net_benchmark_comparison.png", dpi=300)
    plt.close()

    def _plot_heat_flux_correlation(
        benchmark: xr.DataArray,
        calculated: xr.DataArray,
        title: str,
        filename: str,
    ):
        benchmark_aligned, calculated_aligned = xr.align(
            benchmark,
            calculated,
            join="inner",
        )
        x = benchmark_aligned.values.ravel()
        y = calculated_aligned.values.ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        all_values = np.concatenate([x, y])
        if all_values.size:
            lower = np.nanmin(all_values)
            upper = np.nanmax(all_values)
            if not np.isfinite(lower) or not np.isfinite(upper):
                lower, upper = -1.0, 1.0
        else:
            lower, upper = -1.0, 1.0

        if lower == upper:
            pad = max(np.abs(lower) * 0.05, 1.0)
        else:
            pad = (upper - lower) * 0.05
        lower -= pad
        upper += pad

        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            correlation_label = f"Pearson r: n/a\nn = {x.size}"
        else:
            correlation_label = (
                f"Pearson r: {np.corrcoef(x, y)[0, 1]:.3f}\nn = {x.size}"
            )

        fig, ax = plt.subplots(
            figsize=_publication_figsize("single", SQUARE_PANEL_ASPECT),
        )
        ax.scatter(x, y, alpha=SCATTER_ALPHA, color="red", s=SCATTER_SIZE_PT2)
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="k", label="1:1")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        _set_square_1_to_1_axis(ax)
        _set_square_xlabel(ax, f"Benchmark [{HEAT_FLUX_UNITS}]")
        ax.set_ylabel(f"Calculated [{HEAT_FLUX_UNITS}]")
        ax.set_title(title)
        _add_correlation_text(ax, correlation_label)
        ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE_PT)

        _save_square_figure(fig, plot_dir + "/" + filename)
        plt.close()

    _plot_heat_flux_correlation(
        benchmark_diagnostic_totals["benchmark_heat_flux_net_lateral_prime"],
        benchmark_diagnostic_totals["calculated_heat_flux_net_lateral"],
        r"T' Heat-flux ($\mathcal{H}'$)",
        "fig5.2_benchmark_vs_calculated_heat_flux_lateral_prime.png",
    )
    _plot_heat_flux_correlation(
        benchmark_diagnostic_totals["benchmark_heat_flux_net"],
        benchmark_diagnostic_totals["calculated_heat_flux_net_lateral_full"],
        r"Total Heat-flux ($\mathcal{H}$)",
        "fig5.3_benchmark_vs_calculated_heat_flux_total.png",
    )


def _plot_benchmark_term_correlation(
    benchmark: xr.DataArray,
    calculated: xr.DataArray,
    *,
    title: str,
    filename: str,
    plot_dir: str,
    color: str,
) -> None:
    benchmark_aligned, calculated_aligned = xr.align(
        benchmark,
        calculated,
        join="inner",
    )
    x = benchmark_aligned.values.ravel()
    y = calculated_aligned.values.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    all_values = np.concatenate([x, y])
    if all_values.size:
        lower = np.nanmin(all_values)
        upper = np.nanmax(all_values)
        if not np.isfinite(lower) or not np.isfinite(upper):
            lower, upper = -1.0, 1.0
    else:
        lower, upper = -1.0, 1.0

    if lower == upper:
        pad = max(np.abs(lower) * 0.05, 1.0)
    else:
        pad = (upper - lower) * 0.05
    lower -= pad
    upper += pad

    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        correlation_label = f"Pearson r: n/a\nn = {x.size}"
    else:
        correlation_label = (
            f"Pearson r: {np.corrcoef(x, y)[0, 1]:.3f}\nn = {x.size}"
        )

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", SQUARE_PANEL_ASPECT),
    )
    ax.scatter(x, y, alpha=SCATTER_ALPHA, color=color, s=SCATTER_SIZE_PT2)
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="k", label="1:1")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    _set_square_1_to_1_axis(ax)
    _set_square_xlabel(ax, f"Benchmark [{HEAT_FLUX_UNITS}]")
    ax.set_ylabel(f"Calculated [{HEAT_FLUX_UNITS}]")
    ax.set_title(title)
    _add_correlation_text(ax, correlation_label)
    ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE_PT)

    _save_square_figure(fig, plot_dir + "/" + filename)
    plt.close(fig)


def fig6_benchmark_heating_comparison(
    results: xr.Dataset,
    plot_dir: str,
) -> None:
    """Plot ERA5 full-column adiabatic and diabatic benchmark comparisons."""
    required = (
        "adiabatic_term",
        "diabatic_term",
        "calculated_diabatic_term_physical",
        "benchmark_adiabatic_term",
        "benchmark_diabatic_term",
        "benchmark_diabatic_term_physical",
    )
    missing = [name for name in required if name not in results]
    if missing:
        raise ValueError(
            "Figure 6 requires full-column benchmark output variables: "
            f"{missing}"
        )

    panels = (
        (
            "benchmark_adiabatic_term",
            "adiabatic_term",
            r"Adiabatic pressure work ($\mathcal{C}$)",
            "tab:green",
        ),
        (
            "benchmark_diabatic_term",
            "diabatic_term",
            r"Workflow diabatic residual ($\mathcal{D}_0$)",
            "tab:orange",
        ),
        (
            "benchmark_diabatic_term_physical",
            "calculated_diabatic_term_physical",
            r"Physical diabatic residual ($\mathcal{D}_{phys}$)",
            "tab:purple",
        ),
    )

    fig, axes = plt.subplots(
        figsize=_publication_figsize("single", THREE_PANEL_STACK_ASPECT),
        nrows=3,
        sharex=True,
    )
    for axis, (benchmark_name, calculated_name, title, color) in zip(
        axes,
        panels,
    ):
        benchmark, calculated = xr.align(
            results[benchmark_name],
            results[calculated_name],
            join="inner",
        )
        axis.plot(
            benchmark["time"],
            benchmark,
            linestyle="--",
            color=color,
            label="Benchmark",
        )
        axis.plot(
            calculated["time"],
            calculated,
            linestyle="-",
            color=color,
            label="Calculated",
        )
        axis.set_ylabel(rf"[{HEAT_FLUX_UNITS}]")
        axis.set_title(title)
        axis.legend(fontsize=LEGEND_FONT_SIZE_PT)

    locator, formatter = _date_locator_formatter()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel("Time")
    _apply_stacked_figure_layout(fig, 3)
    plt.savefig(plot_dir + "/fig6_benchmark_heating_comparison.png", dpi=300)
    plt.close(fig)

    scatter_specs = (
        (
            "benchmark_adiabatic_term",
            "adiabatic_term",
            r"Adiabatic pressure work ($\mathcal{C}$)",
            "fig6.1_benchmark_vs_calculated_adiabatic.png",
            "tab:green",
        ),
        (
            "benchmark_diabatic_term",
            "diabatic_term",
            r"Workflow diabatic residual ($\mathcal{D}_0$)",
            "fig6.2_benchmark_vs_calculated_diabatic_workflow.png",
            "tab:orange",
        ),
        (
            "benchmark_diabatic_term_physical",
            "calculated_diabatic_term_physical",
            r"Physical diabatic residual ($\mathcal{D}_{phys}$)",
            "fig6.3_benchmark_vs_calculated_diabatic_physical.png",
            "tab:purple",
        ),
    )
    for benchmark_name, calculated_name, title, filename, color in scatter_specs:
        _plot_benchmark_term_correlation(
            results[benchmark_name],
            results[calculated_name],
            title=title,
            filename=filename,
            plot_dir=plot_dir,
            color=color,
        )

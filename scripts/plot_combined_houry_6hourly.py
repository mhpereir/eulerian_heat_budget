from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
CENTERED_SIX_HOUR_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
CENTERED_SIX_HOUR_MEAN_WEIGHTS = np.array((0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5)) / 6.0
CENTERED_AVERAGE_PERIOD_HOURS = (12, 24, 48, 72)

DEFAULT_SIX_HOURLY_PATH = Path(
    "/home/mhpereir/eulerian_heat_budget_6hr_test/results/plots/"
    "2587628.venus/heat_budget_6hr_phases.nc"
)
DEFAULT_HOURLY_PATH = Path(
    "/home/mhpereir/eulerian_heat_budget_6hr_test/results/plots/"
    "2587629.venus/heat_budget_hourly.nc"
)
DEFAULT_OUTPUT_DIR = Path("/home/mhpereir/eulerian_heat_budget_6hr_test/results/diagnostics")

TENDENCY_TERMS = ("dT_dt", "adiabatic_term", "diabatic_term", "advection_term")
TERM_SIGNS = {
    "advection_term": -1,
    "adiabatic_term": 1,
    "diabatic_term": 1,
}
TERM_COLORS = {
    "advection_term": "k",
    "adiabatic_term": "green",
    "diabatic_term": "red",
}
PHASE_LINESTYLE = "--"
PHASE_ALPHA = 0.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hourly Eulerian heat-budget tendency terms against "
            "the equivalent terms computed from six-hourly phase-offset inputs."
        )
    )
    parser.add_argument(
        "--six-hourly-path",
        type=Path,
        default=DEFAULT_SIX_HOURLY_PATH,
        help="Path to the phase-combined six-hourly budget NetCDF.",
    )
    parser.add_argument(
        "--hourly-path",
        type=Path,
        default=DEFAULT_HOURLY_PATH,
        help="Path to the hourly budget NetCDF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for diagnostic plot outputs.",
    )
    parser.add_argument(
        "--time-start",
        type=str,
        default=None,
        help=(
            "Optional inclusive start time for all metrics and plots, "
            "for example '1941-07-01' or '1941-07-01T06'."
        ),
    )
    parser.add_argument(
        "--time-end",
        type=str,
        default=None,
        help=(
            "Optional exclusive end time for all metrics and plots, "
            "for example '1941-07-08' or '1941-07-08T06'."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=float,
        default=7.0,
        help=(
            "Window length in days when only --time-start or only --time-end is supplied."
        ),
    )
    return parser.parse_args()


def load_dataset(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        return ds.load()


def six_hourly_phases_to_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" not in ds:
        raise ValueError("Six-hourly dataset must contain a valid_time variable.")

    if "phase" not in ds.dims or "sample" not in ds.dims:
        raise ValueError("Six-hourly dataset must have phase and sample dimensions.")

    stacked = ds.stack(observation=("phase", "sample"))
    valid_time = stacked["valid_time"]
    valid_mask = ~xr.apply_ufunc(np.isnat, valid_time)

    if "valid_sample" in stacked:
        valid_mask = valid_mask & stacked["valid_sample"]

    stacked = stacked.where(valid_mask, drop=True)
    stacked = stacked.assign_coords(time=("observation", stacked["valid_time"].values))
    stacked = stacked.swap_dims({"observation": "time"}).sortby("time")
    stacked = stacked.drop_vars(["observation", "valid_time"], errors="ignore")

    time_index = stacked.indexes["time"]
    if time_index.has_duplicates:
        keep = ~time_index.duplicated(keep="first")
        stacked = stacked.isel(time=keep)

    return stacked


def parse_time_arg(value: str | None, *, name: str) -> np.datetime64 | None:
    if value is None:
        return None

    try:
        parsed = np.datetime64(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse {name}={value!r} as a datetime.") from exc

    if np.isnat(parsed):
        raise ValueError(f"{name} cannot be NaT.")
    return parsed


def resolve_time_range(args: argparse.Namespace) -> tuple[np.datetime64 | None, np.datetime64 | None]:
    start = parse_time_arg(args.time_start, name="--time-start")
    end = parse_time_arg(args.time_end, name="--time-end")

    if args.window_days <= 0:
        raise ValueError("--window-days must be positive.")

    window_seconds = int(round(args.window_days * SECONDS_PER_DAY))
    window = np.timedelta64(window_seconds, "s")

    if start is not None and end is None:
        end = start + window
    elif start is None and end is not None:
        start = end - window

    if start is not None and end is not None and end <= start:
        raise ValueError("--time-end must be later than --time-start.")

    return start, end


def subset_time_range(
    ds: xr.Dataset,
    *,
    start: np.datetime64 | None,
    end: np.datetime64 | None,
    label: str,
) -> xr.Dataset:
    if start is None and end is None:
        return ds

    if "time" not in ds.coords:
        raise ValueError(f"{label} dataset must have a time coordinate for subsetting.")

    time = ds["time"]
    mask = xr.ones_like(time, dtype=bool)
    if start is not None:
        mask = mask & (time >= start)
    if end is not None:
        mask = mask & (time < end)

    subset = ds.sel(time=mask)
    if subset.sizes.get("time", 0) == 0:
        raise ValueError(f"{label} dataset has no samples in the requested time range.")

    return subset


def time_range_output_suffix(start: np.datetime64 | None, end: np.datetime64 | None) -> str:
    if start is None and end is None:
        return ""

    def format_time(value: np.datetime64 | None, fallback: str) -> str:
        if value is None:
            return fallback
        text = np.datetime_as_string(value, unit="h")
        return text.replace("-", "").replace(":", "")

    return f"_{format_time(start, 'start')}_to_{format_time(end, 'end')}"


def print_time_subset(
    hourly: xr.Dataset,
    six_hourly: xr.Dataset,
    *,
    start: np.datetime64 | None,
    end: np.datetime64 | None,
) -> None:
    if start is None and end is None:
        return

    start_text = "dataset start" if start is None else np.datetime_as_string(start, unit="h")
    end_text = "dataset end" if end is None else np.datetime_as_string(end, unit="h")
    print("\nApplied time subset")
    print(f"  start inclusive: {start_text}")
    print(f"  end exclusive:   {end_text}")
    print(f"  hourly samples:  {hourly.sizes['time']}")
    print(f"  6-hour samples:  {six_hourly.sizes['time']}")


def normalize_rate_to_k_per_hour(ds: xr.Dataset, variable: str) -> xr.DataArray:
    if variable not in ds:
        raise ValueError(f"Dataset is missing required variable {variable!r}.")
    if "domain_volume" not in ds:
        raise ValueError("Dataset is missing required variable 'domain_volume'.")

    out = ds[variable] / ds["domain_volume"] * SECONDS_PER_HOUR
    out.attrs["units"] = "K/hr"
    return out


def budget_sum_to_k_per_hour(ds: xr.Dataset) -> xr.DataArray:
    for variable in TERM_SIGNS:
        if variable not in ds:
            raise ValueError(f"Dataset is missing required variable {variable!r}.")
    if "domain_volume" not in ds:
        raise ValueError("Dataset is missing required variable 'domain_volume'.")

    budget_sum = sum(sign * ds[variable] for variable, sign in TERM_SIGNS.items())
    out = budget_sum / ds["domain_volume"] * SECONDS_PER_HOUR
    out.attrs["units"] = "K/hr"
    return out


def require_regular_time(time: xr.DataArray) -> float:
    values = np.asarray(time.values)
    if values.size < 2:
        raise ValueError("At least two time points are required for regularity checks.")

    diffs = np.diff(values)
    first = diffs[0]
    if np.issubdtype(np.asarray(first).dtype, np.timedelta64):
        diffs_seconds = diffs / np.timedelta64(1, "s")
        first_seconds = float(first / np.timedelta64(1, "s"))
    else:
        diffs_seconds = diffs.astype(float)
        first_seconds = float(first)

    if not np.allclose(diffs_seconds, first_seconds):
        raise ValueError("Time coordinate must be regular.")

    return first_seconds


def daily_total(rate: xr.DataArray) -> xr.DataArray:
    dt_seconds = require_regular_time(rate["time"])
    return (rate * dt_seconds).resample(time="1D").sum()


def centered_trapezoid_offsets_and_weights(
    window_hours: int,
    *,
    spacing_hours: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    if window_hours <= 0:
        raise ValueError("Centered window length must be positive.")
    if spacing_hours <= 0:
        raise ValueError("Centered window spacing must be positive.")
    if window_hours % (2 * spacing_hours) != 0:
        raise ValueError(
            f"A {window_hours}-hour centered window cannot be represented "
            f"with {spacing_hours}-hour samples."
        )

    half_window = window_hours // 2
    offsets = tuple(range(-half_window, half_window + spacing_hours, spacing_hours))
    weights = np.ones(len(offsets), dtype=float)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights /= window_hours / spacing_hours
    return offsets, weights


def centered_weighted_mean_rate(
    rate: xr.DataArray,
    target_times: xr.DataArray,
    offsets_hours: tuple[int, ...],
    weights: np.ndarray,
    *,
    expected_spacing_hours: float,
    integration_window: str,
    check_same_phase: bool = False,
) -> xr.DataArray:
    dt_seconds = require_regular_time(rate["time"])
    expected_seconds = expected_spacing_hours * SECONDS_PER_HOUR
    if not np.isclose(dt_seconds, expected_seconds):
        raise ValueError(
            f"{integration_window} requires {expected_spacing_hours:g}-hour input data "
            f"but found a {dt_seconds / SECONDS_PER_HOUR:g}-hour spacing."
        )

    weights = np.asarray(weights, dtype=float)
    if len(offsets_hours) != weights.size:
        raise ValueError("Offsets and weights must have the same length.")
    if not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("Centered-window weights must sum to 1.")

    target_values = np.asarray(target_times.values)
    target_phase_values = None
    if check_same_phase:
        if "phase" not in rate.coords:
            raise ValueError("Same-phase checking requires a phase coordinate.")
        target_phase_values = np.asarray(rate["phase"].reindex(time=target_values).values)

    weighted_terms = []
    for offset_hours, weight in zip(offsets_hours, weights):
        sample_times = target_values + np.timedelta64(offset_hours, "h")
        sampled = rate.reindex(time=sample_times)

        if check_same_phase:
            if "phase" not in sampled.coords:
                raise ValueError("Sampled data is missing the phase coordinate.")
            sampled_phase_values = np.asarray(sampled["phase"].values)
            present = np.isfinite(sampled_phase_values) & np.isfinite(target_phase_values) # type: ignore
            mismatch = present & (sampled_phase_values != target_phase_values)
            if np.any(mismatch):
                bad_index = int(np.flatnonzero(mismatch)[0])
                bad_time = np.datetime_as_string(target_values[bad_index], unit="h")
                raise ValueError(
                    f"{integration_window} sampled offset {offset_hours:+d}h crosses phase "
                    f"at {bad_time}: target phase {target_phase_values[bad_index]}, " #type: ignore
                    f"sample phase {sampled_phase_values[bad_index]}."
                )

        sampled = sampled.reset_coords(drop=True).assign_coords(time=target_values)
        if check_same_phase:
            sampled = sampled.assign_coords(phase=("time", target_phase_values))
        weighted_terms.append(sampled * weight)

    if not weighted_terms:
        raise ValueError("No weighted terms were generated.")

    out = weighted_terms[0].copy()
    for term in weighted_terms[1:]:
        out = out + term

    out.name = rate.name
    out.attrs.update(rate.attrs)
    out.attrs["units"] = "K/hr"
    out.attrs["integration_window"] = integration_window
    return out


def centered_six_hour_mean_rate(hourly_rate: xr.DataArray, target_times: xr.DataArray) -> xr.DataArray:
    return centered_weighted_mean_rate(
        hourly_rate,
        target_times,
        CENTERED_SIX_HOUR_OFFSETS,
        CENTERED_SIX_HOUR_MEAN_WEIGHTS,
        expected_spacing_hours=1.0,
        integration_window="centered six-hour weighted mean",
    )


def centered_period_hourly_mean_rate(
    hourly_rate: xr.DataArray,
    target_times: xr.DataArray,
    *,
    window_hours: int,
) -> xr.DataArray:
    offsets, weights = centered_trapezoid_offsets_and_weights(window_hours, spacing_hours=1)
    return centered_weighted_mean_rate(
        hourly_rate,
        target_times,
        offsets,
        weights,
        expected_spacing_hours=1.0,
        integration_window=f"centered {window_hours}-hour weighted mean from hourly data",
    )


def centered_period_six_hourly_mean_rate(
    six_hourly_rate: xr.DataArray,
    *,
    window_hours: int,
) -> xr.DataArray:
    offsets, weights = centered_trapezoid_offsets_and_weights(window_hours, spacing_hours=6)
    return centered_weighted_mean_rate(
        six_hourly_rate,
        six_hourly_rate["time"],
        offsets,
        weights,
        expected_spacing_hours=6.0,
        integration_window=f"centered {window_hours}-hour weighted mean from six-hourly data",
        check_same_phase=True,
    )


def build_centered_hourly_reference(hourly: xr.Dataset, six_hourly: xr.Dataset) -> xr.Dataset:
    if "time" not in six_hourly.coords:
        raise ValueError("Six-hourly dataset must have a time coordinate.")

    data_vars = {}
    for term in TENDENCY_TERMS:
        hourly_rate = normalize_rate_to_k_per_hour(hourly, term)
        data_vars[term] = centered_six_hour_mean_rate(hourly_rate, six_hourly["time"])

    return xr.Dataset(data_vars=data_vars, coords={"time": six_hourly["time"].values})


def build_centered_period_hourly_reference(
    hourly: xr.Dataset,
    six_hourly: xr.Dataset,
    *,
    window_hours: int,
) -> xr.Dataset:
    if "time" not in six_hourly.coords:
        raise ValueError("Six-hourly dataset must have a time coordinate.")

    data_vars = {}
    for term in TENDENCY_TERMS:
        hourly_rate = normalize_rate_to_k_per_hour(hourly, term)
        data_vars[term] = centered_period_hourly_mean_rate(
            hourly_rate,
            six_hourly["time"],
            window_hours=window_hours,
        )

    return xr.Dataset(data_vars=data_vars, coords={"time": six_hourly["time"].values})


def build_centered_period_six_hourly_reference(
    six_hourly: xr.Dataset,
    *,
    window_hours: int,
) -> xr.Dataset:
    if "time" not in six_hourly.coords:
        raise ValueError("Six-hourly dataset must have a time coordinate.")
    if "phase" not in six_hourly.coords:
        raise ValueError("Six-hourly dataset must retain phase coordinates.")

    phase_references = []
    for _, phase_ds in six_hourly_phase_slices(six_hourly):
        data_vars = {}
        for term in TENDENCY_TERMS:
            six_hourly_rate = normalize_rate_to_k_per_hour(phase_ds, term)
            data_vars[term] = centered_period_six_hourly_mean_rate(
                six_hourly_rate,
                window_hours=window_hours,
            )
        phase_references.append(xr.Dataset(data_vars=data_vars))

    if not phase_references:
        raise ValueError("No six-hourly phase slices were available.")

    return xr.concat(phase_references, dim="time").sortby("time")


def finite_pair(hourly: xr.DataArray, six_hourly: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    hourly_values = np.asarray(hourly.values, dtype=float)
    six_hourly_values = np.asarray(six_hourly.values, dtype=float)
    mask = np.isfinite(hourly_values) & np.isfinite(six_hourly_values)
    return hourly_values[mask], six_hourly_values[mask]


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def print_alignment_diagnostics(
    hourly: xr.Dataset,
    six_hourly: xr.Dataset,
    hourly_aligned: xr.Dataset,
    six_aligned: xr.Dataset,
) -> None:
    six_times = six_hourly["time"].values
    six_unique_times = np.unique(six_times)
    hourly_times = hourly["time"].values
    hourly_only = np.setdiff1d(hourly_times, six_unique_times)

    print("Input time coverage")
    print(f"  hourly:    {hourly_times.min()} to {hourly_times.max()} ({hourly_times.size} samples)")
    print(
        "  six-hourly:"
        f" {six_unique_times.min()} to {six_unique_times.max()}"
        f" ({six_unique_times.size} unique valid samples)"
    )
    print(f"  aligned:   {hourly_aligned.sizes['time']} exact-match samples")

    if hourly_only.size:
        print("  hourly-only timestamps:")
        for value in hourly_only:
            print(f"    {value}")

    print("\nComparable samples by six-hour phase")
    phase_counts = six_aligned["phase"].groupby(six_aligned["phase"]).count()
    for phase, count in zip(phase_counts["phase"].values, phase_counts.values):
        print(f"  phase {int(phase)}: {int(count)}")


def print_degradation_metrics(hourly: xr.Dataset, six_hourly: xr.Dataset) -> None:
    print("\nDegradation metrics for normalized tendency terms [K/hr]")
    print(
        "  term       n     hourly_mean     six_mean        bias"
        "           MAE           RMSE          corr     nRMSE"
    )

    for term in TENDENCY_TERMS:
        hourly_rate = normalize_rate_to_k_per_hour(hourly, term)
        six_rate = normalize_rate_to_k_per_hour(six_hourly, term)
        hourly_values, six_values = finite_pair(hourly_rate, six_rate)

        diff = six_values - hourly_values
        hourly_std = float(np.std(hourly_values))
        rmse = float(np.sqrt(np.mean(diff**2)))
        nrmse = rmse / hourly_std if hourly_std != 0.0 else np.nan

        print(
            f"  {term:<8} {hourly_values.size:5d}"
            f" {np.mean(hourly_values):14.6e}"
            f" {np.mean(six_values):14.6e}"
            f" {np.mean(diff):14.6e}"
            f" {np.mean(np.abs(diff)):14.6e}"
            f" {rmse:14.6e}"
            f" {correlation(hourly_values, six_values):9.5f}"
            f" {nrmse:9.5f}"
        )


def print_centered_window_degradation_metrics(
    hourly_centered: xr.Dataset,
    six_hourly: xr.Dataset,
) -> None:
    print("\nCentered six-hour weighted-mean degradation metrics [K/hr]")
    print(
        "  term       n     hourly_mean     six_mean        bias"
        "           MAE           RMSE          corr     nRMSE"
    )

    for term in TENDENCY_TERMS:
        hourly_rate = hourly_centered[term]
        six_rate = normalize_rate_to_k_per_hour(six_hourly, term)
        hourly_values, six_values = finite_pair(hourly_rate, six_rate)

        if hourly_values.size == 0:
            hourly_mean = six_mean = bias = mae = rmse = nrmse = corr = np.nan
        else:
            diff = six_values - hourly_values
            hourly_std = float(np.std(hourly_values))
            hourly_mean = float(np.mean(hourly_values))
            six_mean = float(np.mean(six_values))
            bias = float(np.mean(diff))
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            nrmse = rmse / hourly_std if hourly_std != 0.0 else np.nan
            corr = correlation(hourly_values, six_values)

        print(
            f"  {term:<8} {hourly_values.size:5d}"
            f" {hourly_mean:14.6e}"
            f" {six_mean:14.6e}"
            f" {bias:14.6e}"
            f" {mae:14.6e}"
            f" {rmse:14.6e}"
            f" {corr:9.5f}"
            f" {nrmse:9.5f}"
        )


def print_centered_period_degradation_metrics(
    hourly_centered: xr.Dataset,
    six_hourly_centered: xr.Dataset,
    *,
    window_hours: int,
) -> None:
    print(f"\nCentered {window_hours}-hour weighted-mean degradation metrics [K/hr]")
    print(
        "  term       n     hourly_mean     six_mean        bias"
        "           MAE           RMSE          corr     nRMSE"
    )

    for term in TENDENCY_TERMS:
        hourly_rate = hourly_centered[term]
        six_rate = six_hourly_centered[term]
        hourly_values, six_values = finite_pair(hourly_rate, six_rate)

        if hourly_values.size == 0:
            hourly_mean = six_mean = bias = mae = rmse = nrmse = corr = np.nan
        else:
            diff = six_values - hourly_values
            hourly_std = float(np.std(hourly_values))
            hourly_mean = float(np.mean(hourly_values))
            six_mean = float(np.mean(six_values))
            bias = float(np.mean(diff))
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            nrmse = rmse / hourly_std if hourly_std != 0.0 else np.nan
            corr = correlation(hourly_values, six_values)

        print(
            f"  {term:<8} {hourly_values.size:5d}"
            f" {hourly_mean:14.6e}"
            f" {six_mean:14.6e}"
            f" {bias:14.6e}"
            f" {mae:14.6e}"
            f" {rmse:14.6e}"
            f" {corr:9.5f}"
            f" {nrmse:9.5f}"
        )


def variable_label(variable: str) -> str:
    labels = {
        "dT_dt": r"d$\langle T \rangle$/dt",
        "adiabatic_term": "Adiabatic term",
        "diabatic_term": "Diabatic term",
        "advection_term": "Advection term",
    }
    return labels.get(variable, variable)


def plot_tendency_scatter(
    hourly: xr.Dataset,
    six_hourly: xr.Dataset,
    output_dir: Path,
    *,
    output_suffix: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(11, 10), constrained_layout=True)

    for axis, term in zip(axes.ravel(), TENDENCY_TERMS):
        hourly_rate = normalize_rate_to_k_per_hour(hourly, term)
        six_rate = normalize_rate_to_k_per_hour(six_hourly, term)
        hourly_values, six_values = finite_pair(hourly_rate, six_rate)

        axis.scatter(hourly_values, six_values, s=9, alpha=0.35, linewidths=0)

        limits = np.array([hourly_values.min(), hourly_values.max(), six_values.min(), six_values.max()])
        lo = float(limits.min())
        hi = float(limits.max())
        padding = 0.05 * (hi - lo) if hi > lo else 1.0
        lo -= padding
        hi += padding

        axis.plot([lo, hi], [lo, hi], color="k", linewidth=1, linestyle="--")
        axis.set_xlim(lo, hi)
        axis.set_ylim(lo, hi)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linewidth=0.5, alpha=0.35)
        axis.set_title(variable_label(term))
        axis.set_xlabel("Hourly [K/hr]")
        axis.set_ylabel("6-hourly [K/hr]")

        diff = six_values - hourly_values
        rmse = float(np.sqrt(np.mean(diff**2)))
        axis.text(
            0.04,
            0.96,
            f"n={hourly_values.size}\nr={correlation(hourly_values, six_values):.3f}\nRMSE={rmse:.2e}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 3},
        )

    out_path = output_dir / f"hourly_vs_six_hourly_tendency_scatter{output_suffix}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_centered_tendency_scatter(
    hourly_centered: xr.Dataset,
    six_hourly: xr.Dataset,
    output_dir: Path,
    *,
    output_suffix: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(11, 10), constrained_layout=True)

    for axis, term in zip(axes.ravel(), TENDENCY_TERMS):
        hourly_rate = hourly_centered[term]
        six_rate = normalize_rate_to_k_per_hour(six_hourly, term)
        hourly_values, six_values = finite_pair(hourly_rate, six_rate)

        axis.set_title(variable_label(term))
        axis.set_xlabel("Centered hourly 6-hour mean [K/hr]")
        axis.set_ylabel("6-hourly [K/hr]")
        axis.grid(True, linewidth=0.5, alpha=0.35)

        if hourly_values.size == 0:
            axis.text(
                0.5,
                0.5,
                "No complete centered windows",
                transform=axis.transAxes,
                va="center",
                ha="center",
                fontsize=9,
            )
            continue

        axis.scatter(hourly_values, six_values, s=9, alpha=0.35, linewidths=0)

        limits = np.array([hourly_values.min(), hourly_values.max(), six_values.min(), six_values.max()])
        lo = float(limits.min())
        hi = float(limits.max())
        padding = 0.05 * (hi - lo) if hi > lo else 1.0
        lo -= padding
        hi += padding

        axis.plot([lo, hi], [lo, hi], color="k", linewidth=1, linestyle="--")
        axis.set_xlim(lo, hi)
        axis.set_ylim(lo, hi)
        axis.set_aspect("equal", adjustable="box")

        diff = six_values - hourly_values
        rmse = float(np.sqrt(np.mean(diff**2)))
        axis.text(
            0.04,
            0.96,
            f"n={hourly_values.size}\nr={correlation(hourly_values, six_values):.3f}\nRMSE={rmse:.2e}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 3},
        )

    out_path = output_dir / f"hourly_centered_6hr_mean_vs_six_hourly_tendency_scatter{output_suffix}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_centered_period_tendency_scatter(
    hourly_centered: xr.Dataset,
    six_hourly_centered: xr.Dataset,
    output_dir: Path,
    *,
    window_hours: int,
    output_suffix: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(11, 10), constrained_layout=True)

    for axis, term in zip(axes.ravel(), TENDENCY_TERMS):
        hourly_rate = hourly_centered[term]
        six_rate = six_hourly_centered[term]
        hourly_values, six_values = finite_pair(hourly_rate, six_rate)

        axis.set_title(variable_label(term))
        axis.set_xlabel(f"Centered hourly {window_hours}-hour mean [K/hr]")
        axis.set_ylabel(f"Centered 6-hourly {window_hours}-hour mean [K/hr]")
        axis.grid(True, linewidth=0.5, alpha=0.35)

        if hourly_values.size == 0:
            axis.text(
                0.5,
                0.5,
                "No complete centered windows",
                transform=axis.transAxes,
                va="center",
                ha="center",
                fontsize=9,
            )
            continue

        axis.scatter(hourly_values, six_values, s=9, alpha=0.35, linewidths=0)

        limits = np.array([hourly_values.min(), hourly_values.max(), six_values.min(), six_values.max()])
        lo = float(limits.min())
        hi = float(limits.max())
        padding = 0.05 * (hi - lo) if hi > lo else 1.0
        lo -= padding
        hi += padding

        axis.plot([lo, hi], [lo, hi], color="k", linewidth=1, linestyle="--")
        axis.set_xlim(lo, hi)
        axis.set_ylim(lo, hi)
        axis.set_aspect("equal", adjustable="box")

        diff = six_values - hourly_values
        rmse = float(np.sqrt(np.mean(diff**2)))
        axis.text(
            0.04,
            0.96,
            f"n={hourly_values.size}\nr={correlation(hourly_values, six_values):.3f}\nRMSE={rmse:.2e}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 3},
        )

    out_path = output_dir / (
        f"hourly_centered_{window_hours}hr_mean_vs_"
        f"six_hourly_centered_{window_hours}hr_mean_tendency_scatter{output_suffix}.png"
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def six_hourly_phase_slices(six_hourly: xr.Dataset) -> list[tuple[int, xr.Dataset]]:
    phase_values = np.unique(six_hourly["phase"].values)
    return [
        (int(phase), six_hourly.where(six_hourly["phase"] == phase, drop=True).sortby("time"))
        for phase in phase_values
    ]


def plot_phase_line(
    axis: plt.Axes, # type: ignore
    time: xr.DataArray,
    values: xr.DataArray,
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 1.0,
) -> None:
    axis.plot(
        time.values,
        values.values,
        color=color,
        linestyle=PHASE_LINESTYLE,
        alpha=PHASE_ALPHA,
        linewidth=linewidth,
        label=label,
    )


def plot_budget_terms_timeseries_comparison(
    hourly: xr.Dataset,
    six_hourly: xr.Dataset,
    output_dir: Path,
    *,
    output_suffix: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_slices = six_hourly_phase_slices(six_hourly)

    fig, axes = plt.subplots(nrows=3, figsize=(12, 12), sharex=True, constrained_layout=True)

    line_vol = axes[0].plot(
        hourly["time"].values,
        hourly["domain_volume"].values,
        color="C0",
        linewidth=2,
        label="Hourly volume",
    )[0]
    phase_vol_line = None
    for phase, phase_ds in phase_slices:
        label = "6-hour phase volume" if phase_vol_line is None else None
        phase_vol_line = axes[0].plot(
            phase_ds["time"].values,
            phase_ds["domain_volume"].values,
            color="C0",
            linestyle=PHASE_LINESTYLE,
            alpha=PHASE_ALPHA,
            linewidth=1,
            label=label,
        )[0]

    axes[0].set_ylabel("Domain Volume (m$^2$ Pa)")

    temp_axis = axes[0].twinx()
    line_temp = temp_axis.plot(
        hourly["time"].values,
        hourly["T_domain_avg"].values,
        color="orange",
        linewidth=2,
        label=r"Hourly $\langle T \rangle$",
    )[0]
    phase_temp_line = None
    for phase, phase_ds in phase_slices:
        label = r"6-hour phase $\langle T \rangle$" if phase_temp_line is None else None
        phase_temp_line = temp_axis.plot(
            phase_ds["time"].values,
            phase_ds["T_domain_avg"].values,
            color="orange",
            linestyle=PHASE_LINESTYLE,
            alpha=PHASE_ALPHA,
            linewidth=1,
            label=label,
        )[0]

    temp_axis.set_ylabel(r"$\langle T \rangle$ (K)")
    axes[0].set_title("Domain Volume and Average Temperature")

    top_handles = [line_vol, line_temp]
    top_labels = ["Hourly volume", r"Hourly $\langle T \rangle$"]
    if phase_vol_line is not None and phase_temp_line is not None:
        top_handles.extend([phase_vol_line, phase_temp_line])
        top_labels.extend(["6-hour phase volume", r"6-hour phase $\langle T \rangle$"])
    axes[0].legend(top_handles, top_labels, loc="best", fontsize=9, ncols=2)

    hourly_dT = normalize_rate_to_k_per_hour(hourly, "dT_dt")
    hourly_budget_sum = budget_sum_to_k_per_hour(hourly)
    axes[1].plot(
        hourly["time"].values,
        hourly_dT.values,
        color="C3",
        linewidth=2,
        label=r"Hourly d$\langle T \rangle$/dt",
    )
    axes[1].plot(
        hourly["time"].values,
        hourly_budget_sum.values,
        color="C0",
        linewidth=2,
        label="Hourly budget sum",
    )
    for phase, phase_ds in phase_slices:
        label_dT = r"6-hour phase d$\langle T \rangle$/dt" if phase == phase_slices[0][0] else None
        label_sum = "6-hour phase budget sum" if phase == phase_slices[0][0] else None
        plot_phase_line(
            axes[1],
            phase_ds["time"],
            normalize_rate_to_k_per_hour(phase_ds, "dT_dt"),
            color="C3",
            label=label_dT,
        )
        plot_phase_line(
            axes[1],
            phase_ds["time"],
            budget_sum_to_k_per_hour(phase_ds),
            color="C0",
            label=label_sum,
        )

    axes[1].axhline(0, color="k", linestyle="-", linewidth=1)
    axes[1].set_ylabel("dT/dt [K/hr]")
    axes[1].set_title("Storage Term (normalized by volume)")
    axes[1].legend(fontsize=9, ncols=2)

    term_labels = {
        "advection_term": "Net Heat Advection",
        "adiabatic_term": "Adiabatic Term",
        "diabatic_term": "Diabatic Term",
    }
    for term in ("advection_term", "adiabatic_term", "diabatic_term"):
        hourly_term = TERM_SIGNS[term] * normalize_rate_to_k_per_hour(hourly, term)
        axes[2].plot(
            hourly["time"].values,
            hourly_term.values,
            color=TERM_COLORS[term],
            linewidth=2,
            label=f"Hourly {term_labels[term]}",
        )
        for phase, phase_ds in phase_slices:
            label = f"6-hour phase {term_labels[term]}" if phase == phase_slices[0][0] else None
            phase_term = TERM_SIGNS[term] * normalize_rate_to_k_per_hour(phase_ds, term)
            plot_phase_line(
                axes[2],
                phase_ds["time"],
                phase_term,
                color=TERM_COLORS[term],
                label=label,
            )

    axes[2].axhline(0, color="k", linestyle="-", linewidth=1)
    axes[2].set_ylabel("[K/hr]")
    axes[2].set_title("Budget Terms (normalized by volume)")
    axes[2].legend(fontsize=9, ncols=3)

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("Time")
    for axis in axes:
        axis.grid(True, linewidth=0.4, alpha=0.3)

    out_path = output_dir / f"hourly_vs_six_hourly_budget_terms_timeseries{output_suffix}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_daily_phase_line(
    axis: plt.Axes, # type: ignore
    values: xr.DataArray,
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 1.0,
) -> None:
    axis.plot(
        values["time"].values,
        values.values,
        color=color,
        linestyle=PHASE_LINESTYLE,
        alpha=PHASE_ALPHA,
        linewidth=linewidth,
        drawstyle="steps-post",
        label=label,
    )


def plot_budget_terms_daily_comparison(
    hourly: xr.Dataset,
    six_hourly: xr.Dataset,
    output_dir: Path,
    *,
    output_suffix: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_slices = six_hourly_phase_slices(six_hourly)

    fig, axes = plt.subplots(nrows=3, figsize=(12, 12), sharex=True, constrained_layout=True)

    hourly_volume = hourly["domain_volume"].resample(time="1D").mean()
    hourly_temperature = hourly["T_domain_avg"].resample(time="1D").mean()
    line_vol = axes[0].plot(
        hourly_volume["time"].values,
        hourly_volume.values,
        color="C0",
        linewidth=2,
        drawstyle="steps-post",
        label="Hourly volume",
    )[0]
    phase_vol_line = None
    for _, phase_ds in phase_slices:
        phase_volume = phase_ds["domain_volume"].resample(time="1D").mean()
        label = "6-hour phase volume" if phase_vol_line is None else None
        phase_vol_line = axes[0].plot(
            phase_volume["time"].values,
            phase_volume.values,
            color="C0",
            linestyle=PHASE_LINESTYLE,
            alpha=PHASE_ALPHA,
            linewidth=1,
            drawstyle="steps-post",
            label=label,
        )[0]

    axes[0].set_ylabel("Domain Volume (m$^2$ Pa)")

    temp_axis = axes[0].twinx()
    line_temp = temp_axis.plot(
        hourly_temperature["time"].values,
        hourly_temperature.values,
        color="orange",
        linewidth=2,
        drawstyle="steps-post",
        label=r"Hourly $\langle T \rangle$",
    )[0]
    phase_temp_line = None
    for _, phase_ds in phase_slices:
        phase_temperature = phase_ds["T_domain_avg"].resample(time="1D").mean()
        label = r"6-hour phase $\langle T \rangle$" if phase_temp_line is None else None
        phase_temp_line = temp_axis.plot(
            phase_temperature["time"].values,
            phase_temperature.values,
            color="orange",
            linestyle=PHASE_LINESTYLE,
            alpha=PHASE_ALPHA,
            linewidth=1,
            drawstyle="steps-post",
            label=label,
        )[0]

    temp_axis.set_ylabel(r"$\langle T \rangle$ (K)")
    axes[0].set_title("Daily Domain Volume and Average Temperature")

    top_handles = [line_vol, line_temp]
    top_labels = ["Hourly volume", r"Hourly $\langle T \rangle$"]
    if phase_vol_line is not None and phase_temp_line is not None:
        top_handles.extend([phase_vol_line, phase_temp_line])
        top_labels.extend(["6-hour phase volume", r"6-hour phase $\langle T \rangle$"])
    axes[0].legend(top_handles, top_labels, loc="best", fontsize=9, ncols=2)

    hourly_norm_factor = 1 / hourly["domain_volume"]
    hourly_storage = daily_total(hourly["d_dt_T"] * hourly_norm_factor)
    hourly_volume_term = daily_total((hourly["T_domain_avg"] * hourly_norm_factor) * hourly["dV_dt"])
    hourly_temperature_tendency = daily_total(hourly["dT_dt"] * hourly_norm_factor)
    axes[1].plot(
        hourly_storage["time"].values,
        hourly_storage.values,
        color="C1",
        linewidth=2,
        drawstyle="steps-post",
        label=r"Hourly d/dt$\int T dV$",
    )
    axes[1].plot(
        hourly_volume_term["time"].values,
        hourly_volume_term.values,
        color="C0",
        linewidth=2,
        drawstyle="steps-post",
        label=r"Hourly $\langle T \rangle$/V dV/dt",
    )
    axes[1].plot(
        hourly_temperature_tendency["time"].values,
        hourly_temperature_tendency.values,
        color="C2",
        linewidth=2,
        drawstyle="steps-post",
        label=r"Hourly d$\langle T \rangle$/dt",
    )

    for phase, phase_ds in phase_slices:
        phase_norm_factor = 1 / phase_ds["domain_volume"]
        label_suffix = "6-hour phase " if phase == phase_slices[0][0] else None
        plot_daily_phase_line(
            axes[1],
            daily_total(phase_ds["d_dt_T"] * phase_norm_factor),
            color="C1",
            label=(label_suffix + r"d/dt$\int T dV$") if label_suffix else None,
        )
        plot_daily_phase_line(
            axes[1],
            daily_total((phase_ds["T_domain_avg"] * phase_norm_factor) * phase_ds["dV_dt"]),
            color="C0",
            label=(label_suffix + r"$\langle T \rangle$/V dV/dt") if label_suffix else None,
        )
        plot_daily_phase_line(
            axes[1],
            daily_total(phase_ds["dT_dt"] * phase_norm_factor),
            color="C2",
            label=(label_suffix + r"d$\langle T \rangle$/dt") if label_suffix else None,
        )

    axes[1].axhline(0, color="k", linestyle="-", linewidth=1)
    axes[1].set_ylabel("[K/day]")
    axes[1].set_title("Daily Storage Term (normalized by volume)")
    axes[1].legend(fontsize=9, ncols=2)

    term_labels = {
        "advection_term": "Net Heat Advection",
        "adiabatic_term": "Adiabatic Term",
        "diabatic_term": "Diabatic Term",
    }
    for term in ("advection_term", "adiabatic_term", "diabatic_term"):
        hourly_term = daily_total(TERM_SIGNS[term] * hourly[term] * hourly_norm_factor)
        axes[2].plot(
            hourly_term["time"].values,
            hourly_term.values,
            color=TERM_COLORS[term],
            linewidth=2,
            drawstyle="steps-post",
            label=f"Hourly {term_labels[term]}",
        )

        for phase, phase_ds in phase_slices:
            phase_norm_factor = 1 / phase_ds["domain_volume"]
            phase_term = daily_total(TERM_SIGNS[term] * phase_ds[term] * phase_norm_factor)
            label = f"6-hour phase {term_labels[term]}" if phase == phase_slices[0][0] else None
            plot_daily_phase_line(
                axes[2],
                phase_term,
                color=TERM_COLORS[term],
                label=label,
            )

    axes[2].axhline(0, color="k", linestyle="-", linewidth=1)
    axes[2].set_ylabel("[K/day]")
    axes[2].set_title("Daily Budget Terms (normalized by volume)")
    axes[2].legend(fontsize=9, ncols=3)

    day_boundaries = hourly_volume["time"].values
    for axis in axes:
        axis.grid(True, linewidth=0.4, alpha=0.3)
        for time in day_boundaries:
            axis.axvline(time, color="k", linewidth=0.3, alpha=0.15, zorder=0)

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("Time")

    ymax = max(
        abs(axes[1].get_ylim()[0]),
        abs(axes[1].get_ylim()[1]),
        abs(axes[2].get_ylim()[0]),
        abs(axes[2].get_ylim()[1]),
    )
    axes[1].set_ylim(-ymax, ymax)
    axes[2].set_ylim(-ymax, ymax)

    out_path = output_dir / f"hourly_vs_six_hourly_budget_terms_daily{output_suffix}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    six_raw = load_dataset(args.six_hourly_path)
    hourly_raw = load_dataset(args.hourly_path)

    six_time = six_hourly_phases_to_time(six_raw)
    hourly_time = hourly_raw.sortby("time")
    time_start, time_end = resolve_time_range(args)
    output_suffix = time_range_output_suffix(time_start, time_end)
    hourly_time = subset_time_range(
        hourly_time,
        start=time_start,
        end=time_end,
        label="hourly",
    )
    six_time = subset_time_range(
        six_time,
        start=time_start,
        end=time_end,
        label="six-hourly",
    )
    hourly_aligned, six_aligned = xr.align(hourly_time, six_time, join="inner")

    print_time_subset(hourly_time, six_time, start=time_start, end=time_end)
    print_alignment_diagnostics(hourly_time, six_time, hourly_aligned, six_aligned)
    print_degradation_metrics(hourly_aligned, six_aligned)
    scatter_path = plot_tendency_scatter(
        hourly_aligned,
        six_aligned,
        args.output_dir,
        output_suffix=output_suffix,
    )
    print(f"\nSaved scatter diagnostics to {scatter_path}")

    if time_start is None and time_end is None:
        hourly_centered = build_centered_hourly_reference(hourly_time, six_time)
        hourly_centered_aligned, six_centered_aligned = xr.align(hourly_centered, six_time, join="inner")
        print_centered_window_degradation_metrics(hourly_centered_aligned, six_centered_aligned)
        centered_scatter_path = plot_centered_tendency_scatter(
            hourly_centered_aligned,
            six_centered_aligned,
            args.output_dir,
            output_suffix=output_suffix,
        )
        print(f"Saved centered six-hour scatter diagnostics to {centered_scatter_path}")

        for window_hours in CENTERED_AVERAGE_PERIOD_HOURS:
            hourly_period_centered = build_centered_period_hourly_reference(
                hourly_time,
                six_time,
                window_hours=window_hours,
            )
            six_period_centered = build_centered_period_six_hourly_reference(
                six_time,
                window_hours=window_hours,
            )
            hourly_period_aligned, six_period_aligned = xr.align(
                hourly_period_centered,
                six_period_centered,
                join="inner",
            )
            print_centered_period_degradation_metrics(
                hourly_period_aligned,
                six_period_aligned,
                window_hours=window_hours,
            )
            period_scatter_path = plot_centered_period_tendency_scatter(
                hourly_period_aligned,
                six_period_aligned,
                args.output_dir,
                window_hours=window_hours,
                output_suffix=output_suffix,
            )
            print(f"Saved centered {window_hours}-hour scatter diagnostics to {period_scatter_path}")
    else:
        print(
            "\nSkipping centered 6/12/24/48/72-hour integrated diagnostics because "
            "--time-start or --time-end was supplied."
        )

    timeseries_path = plot_budget_terms_timeseries_comparison(
        hourly_time,
        six_time,
        args.output_dir,
        output_suffix=output_suffix,
    )
    print(f"Saved time series diagnostics to {timeseries_path}")
    daily_path = plot_budget_terms_daily_comparison(
        hourly_time,
        six_time,
        args.output_dir,
        output_suffix=output_suffix,
    )
    print(f"Saved daily diagnostics to {daily_path}")


if __name__ == "__main__":
    main()

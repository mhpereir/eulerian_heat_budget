import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import cli, config


run_budget = importlib.import_module("scripts.run_budget")


def _make_stub_budget_result(t_scale: float = 273.0) -> xr.Dataset:
    return xr.Dataset({"T_scale": xr.DataArray(t_scale)})


def _make_stub_budget_result_for_domain(ds_domain: xr.Dataset, t_scale: float = 273.0) -> xr.Dataset:
    return xr.Dataset(
        {
            "d_dt_T": xr.DataArray(
                range(ds_domain.sizes["time"]),
                dims=("time",),
                coords={"time": ds_domain["time"]},
            ),
            "T_scale": xr.DataArray(t_scale),
        }
    )


def _make_stub_domain_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "T": xr.DataArray(
                [[[300.0]]],
                dims=("time", "level", "cell"),
            )
        }
    )


def _make_time_dataset(periods: int = 24) -> xr.Dataset:
    time = pd.date_range("2000-01-01", periods=periods, freq="1h")
    return xr.Dataset(
        {"T": xr.DataArray([300.0] * periods, dims=("time",), coords={"time": time})},
        coords={"time": time},
    )


def _make_time_dataset_for_source_cfg(source_cfg) -> xr.Dataset:
    ds = _make_time_dataset()
    if source_cfg.temporal_phase_hour is None:
        return ds

    mask = (ds["time"].dt.hour % source_cfg.temporal_stride_hours) == source_cfg.temporal_phase_hour
    return ds.sel(time=mask)


def _parse_local_region_args(*extra: str):
    return cli.parse_args(["--data-source", "local_era5", "--region", "pnw_bartusek", *extra])


def test_cli_runtime_flags_default_to_none():
    args = cli.parse_args(["--data-source", "local_era5"])

    assert args.diagnostic_plots is None
    assert args.constant_temperature_test is None
    assert args.include_benchmark_variables is False


def test_cli_parses_region():
    args = cli.parse_args(["--region", "pnw_bartusek"])

    assert args.region == "pnw_bartusek"


def test_build_request_uses_region_bbox():
    args = cli.parse_args(["--region", "pnw_bartusek"])

    request = run_budget.build_request_from_cli(args)

    assert request.bbox == config.REGIONS["pnw_bartusek"]


def test_build_request_requires_region_or_full_explicit_bbox():
    args = cli.parse_args([])

    with pytest.raises(ValueError, match="Provide --region or all explicit bbox flags"):
        run_budget.build_request_from_cli(args)


def test_build_request_rejects_partial_explicit_bbox():
    args = cli.parse_args(["--lat-min", "40", "--lat-max", "60", "--lon-min", "-130"])

    with pytest.raises(ValueError, match="Provide --region or all explicit bbox flags"):
        run_budget.build_request_from_cli(args)


def test_build_request_accepts_full_explicit_bbox():
    args = cli.parse_args(
        [
            "--lat-min",
            "40",
            "--lat-max",
            "60",
            "--lon-min",
            "-130",
            "--lon-max",
            "-110",
        ]
    )

    request = run_budget.build_request_from_cli(args)

    assert request.bbox == (40.0, 60.0, -130.0, -110.0)


def test_build_request_rejects_region_with_explicit_bbox_flags():
    args = cli.parse_args(["--region", "pnw_bartusek", "--lat-min", "40"])

    with pytest.raises(ValueError, match="--region cannot be combined"):
        run_budget.build_request_from_cli(args)


def test_build_runtime_controls_use_config_defaults():
    args = cli.parse_args(["--data-source", "local_era5"])

    diagnostic_plots, constant_temperature_test = run_budget.build_runtime_controls_from_cli(args)

    assert diagnostic_plots is config.DEFAULT_DIAGNOSTIC_PLOTS
    assert constant_temperature_test is config.DEFAULT_CONSTANT_TEMPERATURE_TEST


def test_cli_runtime_flags_parse_explicit_values():
    args = cli.parse_args(
        [
            "--data-source",
            "local_era5",
            "--diagnostic-plots",
            "--constant-temperature-test",
            "--include-benchmark-variables",
        ]
    )

    diagnostic_plots, constant_temperature_test = run_budget.build_runtime_controls_from_cli(args)

    assert args.diagnostic_plots is True
    assert args.constant_temperature_test is True
    assert args.include_benchmark_variables is True
    assert diagnostic_plots is True
    assert constant_temperature_test is True

    args = cli.parse_args(
        [
            "--data-source",
            "local_era5",
            "--no-diagnostic-plots",
            "--no-constant-temperature-test",
        ]
    )

    diagnostic_plots, constant_temperature_test = run_budget.build_runtime_controls_from_cli(args)

    assert args.diagnostic_plots is False
    assert args.constant_temperature_test is False
    assert diagnostic_plots is False
    assert constant_temperature_test is False


def test_cli_parses_production_arguments():
    args = cli.parse_args(
        [
            "--data-source",
            "local_era5",
            "--production-output-dir",
            "/tmp/production",
            "--init-production-manifest",
            "--production-start-year",
            "1940",
            "--production-end-year",
            "2025",
            "--overwrite-output",
        ]
    )

    assert args.production_output_dir == "/tmp/production"
    assert args.init_production_manifest is True
    assert args.production_start_year == 1940
    assert args.production_end_year == 2025
    assert args.overwrite_output is True


def test_cli_parses_six_hourly_arguments():
    args = _parse_local_region_args("--six-hourly-phases")

    assert args.six_hourly_phases is True
    assert args.six_hourly_phase is None
    assert run_budget.selected_six_hourly_phases_from_cli(args) == [0, 1, 2, 3, 4, 5]

    args = _parse_local_region_args("--six-hourly-phase", "3")

    assert args.six_hourly_phases is False
    assert args.six_hourly_phase == 3
    assert run_budget.selected_six_hourly_phases_from_cli(args) == [3]


def test_cli_rejects_invalid_six_hourly_phase():
    with pytest.raises(SystemExit):
        _parse_local_region_args("--six-hourly-phase", "6")


def test_build_production_options_requires_manifest_year_bounds():
    args = cli.parse_args(
        [
            "--data-source",
            "local_era5",
            "--production-output-dir",
            "/tmp/production",
            "--init-production-manifest",
        ]
    )

    with pytest.raises(ValueError, match="requires --production-start-year and --production-end-year"):
        run_budget.build_production_options_from_cli(args)


def test_build_production_options_rejects_cross_year_slices():
    args = cli.parse_args(
        [
            "--data-source",
            "local_era5",
            "--production-output-dir",
            "/tmp/production",
            "--time-start",
            "1940-12-31T00:00:00",
            "--time-end",
            "1941-01-01T00:00:00",
        ]
    )

    with pytest.raises(ValueError, match="same calendar year"):
        run_budget.build_production_options_from_cli(args)


def test_main_default_run_skips_plots_and_constant_temperature(monkeypatch):
    _configure_main_stubs(monkeypatch, _parse_local_region_args())

    calculate_calls = []
    plot_calls = []

    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: calculate_calls.append(kwargs) or _make_stub_budget_result(),
    )
    _patch_plot_recorders(monkeypatch, plot_calls)

    run_budget.main()

    assert len(calculate_calls) == 1
    assert calculate_calls[0]["plot_flag"] is False
    assert calculate_calls[0].get("test_constant_T", False) is False
    assert calculate_calls[0]["benchmark_ds"] is None
    assert plot_calls == []


def test_main_arco_default_skips_benchmark_flux_loading(monkeypatch):
    _configure_main_stubs(
        monkeypatch,
        cli.parse_args(["--data-source", "arco_era5", "--region", "pnw_bartusek"]),
    )

    benchmark_args = []

    monkeypatch.setattr(
        run_budget.io,
        "load_arco_benchmark_fluxes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("benchmark fluxes should not be loaded by default")
        ),
    )
    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: benchmark_args.append(kwargs["benchmark_ds"]) or _make_stub_budget_result(),
    )
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert benchmark_args == [None]


def test_main_hourly_loads_inputs_once_without_temporal_sampling(monkeypatch):
    _configure_main_stubs(monkeypatch, _parse_local_region_args())

    loaded_configs = []
    written_outputs = []

    monkeypatch.setattr(
        run_budget,
        "_load_inputs",
        lambda source_cfg, surface_specs, include_benchmark_variables: loaded_configs.append(source_cfg) or (xr.Dataset(), None),
    )
    monkeypatch.setattr(
        run_budget,
        "_run_one_budget",
        lambda *args, **kwargs: (_make_stub_budget_result(), object()),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: written_outputs.append((output_path, overwrite)) or str(output_path),
    )

    run_budget.main()

    assert len(loaded_configs) == 1
    assert loaded_configs[0].temporal_stride_hours is None
    assert loaded_configs[0].temporal_phase_hour is None
    assert written_outputs == [("/tmp/test-run/heat_budget_hourly.nc", False)]


def test_main_with_diagnostic_plots_restores_main_plot_generation(monkeypatch):
    _configure_main_stubs(
        monkeypatch,
        _parse_local_region_args("--diagnostic-plots"),
    )

    calculate_calls = []
    plot_calls = []

    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: calculate_calls.append(kwargs) or _make_stub_budget_result(),
    )
    _patch_plot_recorders(monkeypatch, plot_calls)

    run_budget.main()

    assert len(calculate_calls) == 1
    assert calculate_calls[0]["plot_flag"] is True
    assert calculate_calls[0].get("test_constant_T", False) is False
    assert calculate_calls[0]["benchmark_ds"] is None
    assert plot_calls == [
        ("timeseries", None, "/tmp/test-plots"),
        ("timeseries", 24, "/tmp/test-plots"),
        ("daily", "/tmp/test-plots"),
    ]


def test_main_with_constant_temperature_test_runs_second_budget_without_plots(monkeypatch):
    _configure_main_stubs(
        monkeypatch,
        _parse_local_region_args("--constant-temperature-test"),
    )

    calculate_calls = []
    plot_calls = []
    makedirs_calls = []

    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: calculate_calls.append(kwargs) or _make_stub_budget_result(),
    )
    _patch_plot_recorders(monkeypatch, plot_calls)
    monkeypatch.setattr(run_budget.os, "makedirs", lambda path, exist_ok=False: makedirs_calls.append((path, exist_ok)))

    run_budget.main()

    assert len(calculate_calls) == 2
    assert calculate_calls[0]["plot_flag"] is False
    assert calculate_calls[0].get("test_constant_T", False) is False
    assert calculate_calls[0]["benchmark_ds"] is None
    assert calculate_calls[1]["plot_flag"] is False
    assert calculate_calls[1]["test_constant_T"] is True
    assert "benchmark_ds" not in calculate_calls[1]
    assert plot_calls == []
    assert makedirs_calls == []


def test_main_with_both_flags_restores_current_behavior(monkeypatch):
    _configure_main_stubs(
        monkeypatch,
        cli.parse_args(
            [
                "--data-source",
                "local_era5",
                "--region",
                "pnw_bartusek",
                "--diagnostic-plots",
                "--constant-temperature-test",
            ]
        ),
    )

    calculate_calls = []
    plot_calls = []
    makedirs_calls = []

    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: calculate_calls.append(kwargs) or _make_stub_budget_result(),
    )
    _patch_plot_recorders(monkeypatch, plot_calls)
    monkeypatch.setattr(run_budget.os, "makedirs", lambda path, exist_ok=False: makedirs_calls.append((path, exist_ok)))

    run_budget.main()

    assert len(calculate_calls) == 2
    assert calculate_calls[0]["plot_flag"] is True
    assert calculate_calls[0].get("test_constant_T", False) is False
    assert calculate_calls[0]["benchmark_ds"] is None
    assert calculate_calls[1]["plot_flag"] is True
    assert calculate_calls[1]["test_constant_T"] is True
    assert "benchmark_ds" not in calculate_calls[1]
    assert plot_calls == [
        ("timeseries", None, "/tmp/test-plots"),
        ("timeseries", 24, "/tmp/test-plots"),
        ("daily", "/tmp/test-plots"),
        ("daily", "/tmp/test-plots/constant_T"),
        ("constant_T", "/tmp/test-plots/constant_T"),
    ]
    assert makedirs_calls == [("/tmp/test-plots/constant_T", True)]


def test_main_six_hourly_phases_runs_budget_six_times(monkeypatch):
    _configure_six_hourly_stubs(
        monkeypatch,
        _parse_local_region_args("--six-hourly-phases"),
    )

    calculate_calls = []
    loaded_phases = []
    written_outputs = []

    monkeypatch.setattr(
        run_budget.io,
        "load_dataset",
        lambda source_cfg, surface_specs: loaded_phases.append(source_cfg.temporal_phase_hour) or _make_time_dataset_for_source_cfg(source_cfg),
    )
    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda ds_domain, *args, **kwargs: calculate_calls.append((ds_domain["time"].dt.hour.values.tolist(), kwargs)) or _make_stub_budget_result_for_domain(ds_domain),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: written_outputs.append((output_path, overwrite, ds_budget.dims)) or str(output_path),
    )
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert loaded_phases == [0, 1, 2, 3, 4, 5]
    assert len(calculate_calls) == 6
    assert [call[0][0] for call in calculate_calls] == [0, 1, 2, 3, 4, 5]
    assert calculate_calls[3][0] == [3, 9, 15, 21]
    assert written_outputs[0][0] == "/tmp/test-run/heat_budget_6hr_phases.nc"


def test_main_six_hourly_single_phase_runs_selected_phase(monkeypatch):
    _configure_six_hourly_stubs(
        monkeypatch,
        _parse_local_region_args("--six-hourly-phase", "3"),
    )

    calculate_calls = []
    loaded_phases = []
    written_outputs = []

    monkeypatch.setattr(
        run_budget.io,
        "load_dataset",
        lambda source_cfg, surface_specs: loaded_phases.append(source_cfg.temporal_phase_hour) or _make_time_dataset_for_source_cfg(source_cfg),
    )
    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda ds_domain, *args, **kwargs: calculate_calls.append((ds_domain["time"].dt.hour.values.tolist(), kwargs)) or _make_stub_budget_result_for_domain(ds_domain),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: written_outputs.append(output_path) or str(output_path),
    )
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert loaded_phases == [3]
    assert len(calculate_calls) == 1
    assert calculate_calls[0][0] == [3, 9, 15, 21]
    assert written_outputs == ["/tmp/test-run/heat_budget_6hr_phase_r3.nc"]


def test_main_six_hourly_single_phase_loads_one_phase_config(monkeypatch):
    _configure_main_stubs(
        monkeypatch,
        _parse_local_region_args("--six-hourly-phase", "3"),
    )

    loaded_configs = []

    monkeypatch.setattr(
        run_budget,
        "_load_inputs",
        lambda source_cfg, surface_specs, include_benchmark_variables: loaded_configs.append(source_cfg) or (_make_time_dataset_for_source_cfg(source_cfg), None),
    )
    monkeypatch.setattr(
        run_budget,
        "_run_one_budget",
        lambda ds_merged, *args, **kwargs: (_make_stub_budget_result_for_domain(ds_merged), object()),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: str(output_path),
    )

    run_budget.main()

    assert [(cfg.temporal_stride_hours, cfg.temporal_phase_hour) for cfg in loaded_configs] == [(6, 3)]


def test_main_six_hourly_all_phases_loads_each_phase_config(monkeypatch):
    _configure_main_stubs(
        monkeypatch,
        _parse_local_region_args("--six-hourly-phases"),
    )

    loaded_configs = []

    monkeypatch.setattr(
        run_budget,
        "_load_inputs",
        lambda source_cfg, surface_specs, include_benchmark_variables: loaded_configs.append(source_cfg) or (_make_time_dataset_for_source_cfg(source_cfg), None),
    )
    monkeypatch.setattr(
        run_budget,
        "_run_one_budget",
        lambda ds_merged, *args, **kwargs: (_make_stub_budget_result_for_domain(ds_merged), object()),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: str(output_path),
    )

    run_budget.main()

    assert [(cfg.temporal_stride_hours, cfg.temporal_phase_hour) for cfg in loaded_configs] == [
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 3),
        (6, 4),
        (6, 5),
    ]


def test_main_six_hourly_phase_selects_benchmark_fluxes(monkeypatch):
    _configure_six_hourly_stubs(
        monkeypatch,
        cli.parse_args(
            [
                "--data-source",
                "arco_era5",
                "--region",
                "pnw_bartusek",
                "--six-hourly-phase",
                "2",
                "--include-benchmark-variables",
            ]
        ),
    )
    benchmark_configs = []
    monkeypatch.setattr(
        run_budget.io,
        "load_arco_benchmark_fluxes",
        lambda source_cfg, var_map: benchmark_configs.append((source_cfg.temporal_stride_hours, source_cfg.temporal_phase_hour)) or _make_time_dataset_for_source_cfg(source_cfg),
    )

    benchmark_hours = []

    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda ds_domain, *args, **kwargs: benchmark_hours.append(kwargs["benchmark_ds"]["time"].dt.hour.values.tolist()) or _make_stub_budget_result_for_domain(ds_domain),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: str(output_path),
    )
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert benchmark_hours == [[2, 8, 14, 20]]
    assert benchmark_configs == [(6, 2)]


def test_main_six_hourly_plots_use_phase_directories(monkeypatch):
    _configure_six_hourly_stubs(
        monkeypatch,
        _parse_local_region_args("--six-hourly-phase", "3", "--diagnostic-plots"),
    )

    calculate_calls = []
    plot_calls = []
    makedirs_calls = []

    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda ds_domain, *args, **kwargs: calculate_calls.append(kwargs) or _make_stub_budget_result_for_domain(ds_domain),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: str(output_path),
    )
    monkeypatch.setattr(run_budget.os, "makedirs", lambda path, exist_ok=False: makedirs_calls.append((path, exist_ok)))
    _patch_plot_recorders(monkeypatch, plot_calls)

    run_budget.main()

    assert calculate_calls[0]["plot_dir"] == "/tmp/test-plots/phase_r3"
    assert plot_calls == [
        ("timeseries", None, "/tmp/test-plots/phase_r3"),
        ("timeseries", 24, "/tmp/test-plots/phase_r3"),
        ("daily", "/tmp/test-plots/phase_r3"),
    ]
    assert makedirs_calls == [("/tmp/test-plots/phase_r3", True)]


def test_main_six_hourly_constant_temperature_runs_per_phase_directory(monkeypatch):
    _configure_six_hourly_stubs(
        monkeypatch,
        cli.parse_args(
            [
                "--data-source",
                "local_era5",
                "--region",
                "pnw_bartusek",
                "--six-hourly-phase",
                "3",
                "--constant-temperature-test",
            ]
        ),
    )

    calculate_calls = []

    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda ds_domain, *args, **kwargs: calculate_calls.append(kwargs) or _make_stub_budget_result_for_domain(ds_domain),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: str(output_path),
    )
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert len(calculate_calls) == 2
    assert calculate_calls[0]["plot_dir"] == "/tmp/test-plots/phase_r3"
    assert calculate_calls[1]["plot_dir"] == "/tmp/test-plots/phase_r3/constant_T"
    assert calculate_calls[1]["test_constant_T"] is True


def test_main_init_production_manifest_exits_before_loading_data(monkeypatch, tmp_path):
    manifest_calls = []
    production_dir = tmp_path / "production"

    _configure_core_stubs(
        monkeypatch,
        cli.parse_args(
            [
                "--data-source",
                "local_era5",
                "--region",
                "pnw_bartusek",
                "--production-output-dir",
                str(production_dir),
                "--init-production-manifest",
                "--production-start-year",
                "1940",
                "--production-end-year",
                "2025",
            ]
        ),
    )
    monkeypatch.setattr(
        run_budget.io,
        "load_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_dataset should not be called")),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_production_manifest",
        lambda *args, **kwargs: manifest_calls.append(kwargs) or str(production_dir / "production_run.json"),
    )

    run_budget.main()

    assert len(manifest_calls) == 1
    assert manifest_calls[0]["production_start_year"] == 1940
    assert manifest_calls[0]["production_end_year"] == 2025


def test_main_production_yearly_run_writes_single_output_without_run_info(monkeypatch, tmp_path):
    production_dir = tmp_path / "production"
    production_paths = run_budget.run_outputs.prepare_production_paths(str(production_dir), year=1940)
    Path(production_paths.manifest_path).write_text("{}\n")

    _configure_core_stubs(
        monkeypatch,
        cli.parse_args(
            [
                "--data-source",
                "local_era5",
                "--region",
                "pnw_bartusek",
                "--production-output-dir",
                str(production_dir),
                "--time-start",
                "1940-01-01T00:00:00",
                "--time-end",
                "1940-12-31T23:00:00",
            ]
        ),
    )

    calculate_calls = []
    written_outputs = []
    run_info_calls = []

    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: calculate_calls.append(kwargs) or _make_stub_budget_result(),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: written_outputs.append((output_path, overwrite)) or str(output_path),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_run_info",
        lambda *args, **kwargs: run_info_calls.append((args, kwargs)) or "/tmp/unused.json",
    )
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert len(calculate_calls) == 1
    assert calculate_calls[0]["plot_dir"] == str(production_dir / "plots" / "1940")
    assert calculate_calls[0]["benchmark_ds"] is None
    assert written_outputs == [(str(production_dir / "annual" / "heat_budget_1940.nc"), False)]
    assert run_info_calls == []


def test_main_production_six_hourly_outputs_use_distinct_filenames(monkeypatch, tmp_path):
    production_dir = tmp_path / "production"
    production_paths = run_budget.run_outputs.prepare_production_paths(str(production_dir), year=1940)
    Path(production_paths.manifest_path).write_text("{}\n")

    _configure_six_hourly_stubs(
        monkeypatch,
        cli.parse_args(
            [
                "--data-source",
                "local_era5",
                "--region",
                "pnw_bartusek",
                "--production-output-dir",
                str(production_dir),
                "--time-start",
                "1940-01-01T00:00:00",
                "--time-end",
                "1940-12-31T23:00:00",
                "--six-hourly-phase",
                "3",
            ]
        ),
    )

    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda ds_domain, *args, **kwargs: _make_stub_budget_result_for_domain(ds_domain),
    )
    written_outputs = []
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: written_outputs.append((output_path, overwrite)) or str(output_path),
    )
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert written_outputs == [
        (str(production_dir / "annual" / "heat_budget_1940_6hr_phase_r3.nc"), False)
    ]


def test_main_production_mode_fails_when_manifest_is_missing(monkeypatch, tmp_path):
    production_dir = tmp_path / "production"

    _configure_core_stubs(
        monkeypatch,
        cli.parse_args(
            [
                "--data-source",
                "local_era5",
                "--region",
                "pnw_bartusek",
                "--production-output-dir",
                str(production_dir),
                "--time-start",
                "1940-01-01T00:00:00",
                "--time-end",
                "1940-12-31T23:00:00",
            ]
        ),
    )
    monkeypatch.setattr(
        run_budget.io,
        "load_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_dataset should not be called")),
    )

    with pytest.raises(FileNotFoundError, match="Production manifest not found"):
        run_budget.main()


def test_main_production_plots_use_year_specific_directory(monkeypatch, tmp_path):
    production_dir = tmp_path / "production"
    production_paths = run_budget.run_outputs.prepare_production_paths(str(production_dir), year=1940)
    Path(production_paths.manifest_path).write_text("{}\n")

    _configure_core_stubs(
        monkeypatch,
        cli.parse_args(
            [
                "--data-source",
                "local_era5",
                "--region",
                "pnw_bartusek",
                "--production-output-dir",
                str(production_dir),
                "--time-start",
                "1940-01-01T00:00:00",
                "--time-end",
                "1940-12-31T23:00:00",
                "--diagnostic-plots",
            ]
        ),
    )

    calculate_calls = []
    plot_calls = []

    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: calculate_calls.append(kwargs) or _make_stub_budget_result(),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: str(output_path),
    )
    _patch_plot_recorders(monkeypatch, plot_calls)

    run_budget.main()

    assert calculate_calls[0]["plot_dir"] == str(production_dir / "plots" / "1940")
    assert calculate_calls[0]["benchmark_ds"] is None
    assert plot_calls == [
        ("timeseries", None, str(production_dir / "plots" / "1940")),
        ("timeseries", 24, str(production_dir / "plots" / "1940")),
        ("daily", str(production_dir / "plots" / "1940")),
    ]


def test_main_arco_run_skips_benchmark_load_without_flag(monkeypatch):
    _configure_main_stubs(
        monkeypatch,
        cli.parse_args(["--data-source", "arco_era5", "--region", "pnw_bartusek"]),
    )

    calculate_calls = []
    benchmark_load_calls = []

    monkeypatch.setattr(
        run_budget.io,
        "load_arco_benchmark_fluxes",
        lambda *args, **kwargs: benchmark_load_calls.append((args, kwargs)) or xr.Dataset(),
    )
    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: calculate_calls.append(kwargs) or _make_stub_budget_result(),
    )
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert benchmark_load_calls == []
    assert calculate_calls[0]["benchmark_ds"] is None


def test_main_arco_run_loads_benchmark_with_flag(monkeypatch):
    benchmark_ds = xr.Dataset({"Fx_heat": xr.DataArray([1.0], dims=("time",))})
    _configure_main_stubs(
        monkeypatch,
        cli.parse_args(
            [
                "--data-source",
                "arco_era5",
                "--region",
                "pnw_bartusek",
                "--include-benchmark-variables",
            ]
        ),
    )

    calculate_calls = []
    benchmark_load_calls = []

    monkeypatch.setattr(
        run_budget.io,
        "load_arco_benchmark_fluxes",
        lambda *args, **kwargs: benchmark_load_calls.append((args, kwargs)) or benchmark_ds,
    )
    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: calculate_calls.append(kwargs) or _make_stub_budget_result(),
    )
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert len(benchmark_load_calls) == 1
    assert benchmark_load_calls[0][0][0].kind == "arco_era5"
    assert set(benchmark_load_calls[0][0][1].values()) == {
        "Fx_heat",
        "Fy_heat",
        "Fx_mass",
        "Fy_mass",
    }
    assert calculate_calls[0]["benchmark_ds"] is benchmark_ds


def _configure_main_stubs(monkeypatch, args):
    _configure_core_stubs(monkeypatch, args)
    monkeypatch.setattr(
        run_budget.run_outputs,
        "prepare_run_paths",
        lambda base_plot_dir: run_budget.run_outputs.RunPaths(
            run_id="test-run",
            run_root="/tmp/test-run",
            plot_dir="/tmp/test-plots",
            metadata_path="/tmp/test-run/run_info.json",
        ),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "resolve_git_provenance",
        lambda repo_dir: run_budget.run_outputs.GitProvenance(
            branch="test-branch",
            commit="abc123",
            dirty=False,
        ),
    )
    monkeypatch.setattr(run_budget.run_outputs, "write_run_info", lambda *args, **kwargs: "/tmp/test-run/run_info.json")
    monkeypatch.setattr(run_budget.run_outputs, "write_budget_result", lambda ds_budget, output_path, overwrite=False: str(output_path))


def _configure_six_hourly_stubs(monkeypatch, args):
    _configure_main_stubs(monkeypatch, args)
    monkeypatch.setattr(run_budget.io, "load_dataset", lambda source_cfg, surface_specs: _make_time_dataset_for_source_cfg(source_cfg))
    monkeypatch.setattr(
        run_budget.grid,
        "determine_domain",
        lambda ds_merged, request, eager_loading=True: (ds_merged, ds_merged, object()),
    )


def _configure_core_stubs(monkeypatch, args):
    ds_domain = _make_stub_domain_dataset()

    monkeypatch.setattr(run_budget.cli, "parse_args", lambda: args)
    monkeypatch.setattr(run_budget.io, "load_dataset", lambda source_cfg, surface_specs: xr.Dataset())
    monkeypatch.setattr(run_budget.validate, "validate_schema", lambda ds: None)
    monkeypatch.setattr(
        run_budget.grid,
        "determine_domain",
        lambda ds_merged, request, eager_loading=True: (ds_domain, ds_domain, object()),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "resolve_git_provenance",
        lambda repo_dir: run_budget.run_outputs.GitProvenance(
            branch="test-branch",
            commit="abc123",
            dirty=False,
        ),
    )


def _patch_plot_recorders(monkeypatch, plot_calls):
    monkeypatch.setattr(
        run_budget.plot_results,
        "plot_budget_terms_timeseries",
        lambda ds_budget, plot_dir, smoothing_duration_hours=None: plot_calls.append(
            ("timeseries", smoothing_duration_hours, plot_dir)
        ),
    )
    monkeypatch.setattr(
        run_budget.plot_results,
        "plot_budget_terms_daily",
        lambda ds_budget, plot_dir: plot_calls.append(("daily", plot_dir)),
    )
    monkeypatch.setattr(
        run_budget.plot_results,
        "plot_constant_T_results",
        lambda ds_budget, ds_test, plot_dir: plot_calls.append(("constant_T", plot_dir)),
    )

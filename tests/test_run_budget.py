import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import cli, config


run_budget = importlib.import_module("scripts.run_budget")


def _make_stub_budget_result(t_scale: float = 273.0) -> xr.Dataset:
    return xr.Dataset({"T_scale": xr.DataArray(t_scale)})


def _make_stub_domain_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "T": xr.DataArray(
                [[[300.0]]],
                dims=("time", "level", "cell"),
            )
        }
    )


def _parse_local_region_args(*extra: str):
    return cli.parse_args(["--data-source", "local_era5", "--region", "pnw_bartusek", *extra])


def test_cli_runtime_flags_default_to_none():
    args = cli.parse_args(["--data-source", "local_era5"])

    assert args.diagnostic_plots is None
    assert args.constant_temperature_test is None
    assert args.local_data_path is None
    assert args.output_root is None
    assert args.include_benchmark_variables is False
    assert args.write_netcdf is False


def test_cli_parses_staged_arco_cache_source():
    args = cli.parse_args(
        [
            "--data-source",
            "staged_arco_cache",
            "--staged-cache-root",
            "/tmp/ehb-cache",
            "--region",
            "pnw_bartusek",
        ]
    )

    assert args.data_source == "staged_arco_cache"
    assert args.staged_cache_root == "/tmp/ehb-cache"


def test_cli_parses_region():
    args = cli.parse_args(["--region", "pnw_bartusek"])

    assert args.region == "pnw_bartusek"


@pytest.mark.parametrize(
    ("region", "expected_bbox"),
    [
        ("alaska", (59.5, 69.5, -160, -150)),
        ("western_usa", (34, 44, -120, -110)),
        ("central_usa", (36, 46, -105, -95)),
        ("gulf_usa", (31, 41, -90, -80)),
        ("western_eu", (43, 53, -2, 8)),
        ("central_china", (25, 35, 105, 115)),
    ],
)
def test_new_google_regions_resolve_to_expected_bbox(region, expected_bbox):
    args = cli.parse_args(["--region", region])

    request = run_budget.build_request_from_cli(args)

    assert request.bbox == expected_bbox


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


def test_build_local_data_source_prefers_cli_path(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_LOCAL_PATH", "/environment/era5")
    args = cli.parse_args(
        [
            "--data-source",
            "local_era5",
            "--local-data-path",
            "/cli/era5",
        ]
    )

    source = run_budget.build_data_source_from_cli(args)

    assert source.path_data == "/cli/era5"


def test_build_local_data_source_uses_environment_default(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_LOCAL_PATH", "/environment/era5")
    args = cli.parse_args(["--data-source", "local_era5"])

    source = run_budget.build_data_source_from_cli(args)

    assert source.path_data == "/environment/era5"


def test_build_local_data_source_requires_configured_path(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_LOCAL_PATH", None)
    args = cli.parse_args(["--data-source", "local_era5"])

    with pytest.raises(ValueError, match="--local-data-path or EHB_DATA_ROOT"):
        run_budget.build_data_source_from_cli(args)


def test_build_data_source_accepts_staged_arco_cache():
    args = cli.parse_args(
        [
            "--data-source",
            "staged_arco_cache",
            "--staged-cache-root",
            "/tmp/ehb-cache",
            "--time-start",
            "1940-06-01T00:00:00",
            "--time-end",
            "1940-06-02T00:00:00",
        ]
    )

    source = run_budget.build_data_source_from_cli(args)

    assert source.kind == "staged_arco_cache"
    assert source.staged_cache_root == "/tmp/ehb-cache"
    assert source.time_start == "1940-06-01T00:00:00"
    assert source.time_end == "1940-06-02T00:00:00"


def test_build_data_source_requires_staged_cache_root():
    args = cli.parse_args(["--data-source", "staged_arco_cache"])

    with pytest.raises(ValueError, match="--staged-cache-root"):
        run_budget.build_data_source_from_cli(args)


def test_staged_arco_cache_rejects_surface_variables():
    args = cli.parse_args(
        [
            "--data-source",
            "staged_arco_cache",
            "--staged-cache-root",
            "/tmp/ehb-cache",
            "--use-surface-variables",
        ]
    )

    with pytest.raises(NotImplementedError, match="T2m, u10, and v10"):
        run_budget.build_surface_behaviour_from_cli(args)


def test_cli_runtime_flags_parse_explicit_values():
    args = cli.parse_args(
        [
            "--data-source",
            "local_era5",
            "--diagnostic-plots",
            "--constant-temperature-test",
            "--include-benchmark-variables",
            "--write-netcdf",
        ]
    )

    diagnostic_plots, constant_temperature_test = run_budget.build_runtime_controls_from_cli(args)

    assert args.diagnostic_plots is True
    assert args.constant_temperature_test is True
    assert args.include_benchmark_variables is True
    assert args.write_netcdf is True
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


def test_build_production_options_rejects_write_netcdf_in_production_mode():
    args = _parse_local_region_args(
        "--production-output-dir",
        "/tmp/production",
        "--time-start",
        "1940-01-01T00:00:00",
        "--time-end",
        "1940-12-31T23:00:00",
        "--write-netcdf",
    )

    with pytest.raises(ValueError, match="--write-netcdf cannot be combined"):
        run_budget.build_production_options_from_cli(args)


def test_build_production_options_rejects_ad_hoc_output_root():
    args = _parse_local_region_args(
        "--production-output-dir",
        "/tmp/production",
        "--time-start",
        "1940-01-01T00:00:00",
        "--time-end",
        "1940-12-31T23:00:00",
        "--output-root",
        "/tmp/ad-hoc",
    )

    with pytest.raises(ValueError, match="--output-root cannot be combined"):
        run_budget.build_production_options_from_cli(args)


def test_build_production_options_rejects_overwrite_without_output_mode():
    args = _parse_local_region_args("--overwrite-output")

    with pytest.raises(ValueError, match="--overwrite-output requires"):
        run_budget.build_production_options_from_cli(args)


def test_build_production_options_accepts_overwrite_for_ad_hoc_netcdf():
    args = _parse_local_region_args("--write-netcdf", "--overwrite-output")

    assert run_budget.build_production_options_from_cli(args) is None


def test_main_default_run_skips_plots_and_constant_temperature(monkeypatch):
    _configure_main_stubs(monkeypatch, _parse_local_region_args())

    calculate_calls = []
    plot_calls = []
    written_outputs = []

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
    _patch_plot_recorders(monkeypatch, plot_calls)

    run_budget.main()

    assert len(calculate_calls) == 1
    assert calculate_calls[0]["plot_flag"] is False
    assert calculate_calls[0].get("test_constant_T", False) is False
    assert calculate_calls[0]["benchmark_ds"] is None
    assert plot_calls == []
    assert written_outputs == []


def test_main_ad_hoc_run_prefers_cli_output_root(monkeypatch):
    args = _parse_local_region_args("--output-root", "/cli/output")
    _configure_core_stubs(monkeypatch, args)
    output_roots = []

    monkeypatch.setattr(
        run_budget.run_outputs,
        "prepare_run_paths",
        lambda output_root: output_roots.append(output_root)
        or run_budget.run_outputs.RunPaths(
            run_id="test-run",
            run_root="/tmp/test-run",
            plot_dir="/tmp/test-plots",
            metadata_path="/tmp/test-run/run_info.json",
            output_path="/tmp/test-run/heat_budget.nc",
            constant_t_output_path="/tmp/test-run/heat_budget_constant_T.nc",
        ),
    )
    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: _make_stub_budget_result(),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_run_info",
        lambda *args, **kwargs: "/tmp/test-run/run_info.json",
    )

    run_budget.main()

    assert output_roots == ["/cli/output"]


def test_main_ad_hoc_run_uses_environment_output_root(monkeypatch):
    args = _parse_local_region_args()
    _configure_core_stubs(monkeypatch, args)
    monkeypatch.setattr(config, "DEFAULT_OUTPUT_ROOT", "/environment/output")
    output_roots = []

    monkeypatch.setattr(
        run_budget.run_outputs,
        "prepare_run_paths",
        lambda output_root: output_roots.append(output_root)
        or run_budget.run_outputs.RunPaths(
            run_id="test-run",
            run_root="/tmp/test-run",
            plot_dir="/tmp/test-plots",
            metadata_path="/tmp/test-run/run_info.json",
            output_path="/tmp/test-run/heat_budget.nc",
            constant_t_output_path="/tmp/test-run/heat_budget_constant_T.nc",
        ),
    )
    monkeypatch.setattr(
        run_budget.budget,
        "calculate_budget",
        lambda *args, **kwargs: _make_stub_budget_result(),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_run_info",
        lambda *args, **kwargs: "/tmp/test-run/run_info.json",
    )

    run_budget.main()

    assert output_roots == ["/environment/output"]


def test_main_ad_hoc_write_netcdf_writes_primary_result(monkeypatch):
    _configure_main_stubs(monkeypatch, _parse_local_region_args("--write-netcdf"))

    validated_outputs = []
    written_outputs = []

    monkeypatch.setattr(
        run_budget.run_outputs,
        "require_output_path",
        lambda output_path, overwrite=False: validated_outputs.append((output_path, overwrite)) or str(output_path),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: written_outputs.append((output_path, overwrite)) or str(output_path),
    )
    monkeypatch.setattr(run_budget.budget, "calculate_budget", lambda *args, **kwargs: _make_stub_budget_result())
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert validated_outputs == [("/tmp/test-run/heat_budget.nc", False)]
    assert written_outputs == [("/tmp/test-run/heat_budget.nc", False)]


def test_main_ad_hoc_write_netcdf_fails_on_collision_before_loading_data(monkeypatch):
    _configure_main_stubs(monkeypatch, _parse_local_region_args("--write-netcdf"))

    monkeypatch.setattr(
        run_budget.run_outputs,
        "require_output_path",
        lambda output_path, overwrite=False: (_ for _ in ()).throw(FileExistsError(output_path)),
    )
    monkeypatch.setattr(
        run_budget.io,
        "load_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_dataset should not be called")),
    )

    with pytest.raises(FileExistsError, match="heat_budget.nc"):
        run_budget.main()


def test_main_ad_hoc_write_netcdf_writes_constant_temperature_result(monkeypatch):
    _configure_main_stubs(
        monkeypatch,
        _parse_local_region_args("--write-netcdf", "--constant-temperature-test", "--overwrite-output"),
    )

    validated_outputs = []
    written_outputs = []

    monkeypatch.setattr(
        run_budget.run_outputs,
        "require_output_path",
        lambda output_path, overwrite=False: validated_outputs.append((output_path, overwrite)) or str(output_path),
    )
    monkeypatch.setattr(
        run_budget.run_outputs,
        "write_budget_result",
        lambda ds_budget, output_path, overwrite=False: written_outputs.append((output_path, overwrite)) or str(output_path),
    )
    monkeypatch.setattr(run_budget.budget, "calculate_budget", lambda *args, **kwargs: _make_stub_budget_result())
    _patch_plot_recorders(monkeypatch, [])

    run_budget.main()

    assert validated_outputs == [
        ("/tmp/test-run/heat_budget.nc", True),
        ("/tmp/test-run/heat_budget_constant_T.nc", True),
    ]
    assert written_outputs == [
        ("/tmp/test-run/heat_budget.nc", False),
        ("/tmp/test-run/heat_budget_constant_T.nc", False),
    ]


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
        ("hourly", 1, "/tmp/test-plots"),
        ("hourly", 24, "/tmp/test-plots"),
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
        ("hourly", 1, "/tmp/test-plots"),
        ("hourly", 24, "/tmp/test-plots"),
        ("daily", "/tmp/test-plots"),
        ("daily", "/tmp/test-plots/constant_T"),
        ("constant_T", "/tmp/test-plots/constant_T"),
    ]
    assert makedirs_calls == [("/tmp/test-plots/constant_T", True)]


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
        ("hourly", 1, str(production_dir / "plots" / "1940")),
        ("hourly", 24, str(production_dir / "plots" / "1940")),
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


def test_main_staged_cache_loads_local_benchmark_with_flag(monkeypatch):
    benchmark_ds = xr.Dataset({"Fx_heat": xr.DataArray([1.0], dims=("time",))})
    _configure_main_stubs(
        monkeypatch,
        cli.parse_args(
            [
                "--data-source",
                "staged_arco_cache",
                "--staged-cache-root",
                "/tmp/ehb-cache",
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
        "load_staged_arco_benchmark_fluxes",
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
    assert benchmark_load_calls[0][0][0].kind == "staged_arco_cache"
    assert calculate_calls[0]["benchmark_ds"] is benchmark_ds


def test_single_run_schedulers_share_cli_settings():
    scheduler_dir = Path(PROJECT_ROOT) / "schedulers"
    settings = (scheduler_dir / "single_run_cli_settings").read_text()
    run_scheduler = (scheduler_dir / "schedule_run_budget.sh").read_text()
    retrieval_scheduler = (scheduler_dir / "schedule_staged_arco_retrieval.sh").read_text()

    assert "STAGED_CACHE_ROOT=" in settings
    assert "STAGED_ARCO_TIME_CHUNK=" in settings
    assert "STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS=" in settings
    assert "TIME_START=" in settings
    assert "ehb_build_run_budget_args" in settings
    assert "ehb_build_staged_arco_retrieval_args" in settings
    assert "PROJECT_ROOT=\"${PROJECT_ROOT:-/home/mhpereir/eulerian_heat_budget}\"" in run_scheduler
    assert "PROJECT_ROOT=\"${PROJECT_ROOT:-/home/mhpereir/eulerian_heat_budget}\"" in retrieval_scheduler
    assert "SCHEDULER_DIR=\"${SCHEDULER_DIR:-${PROJECT_ROOT}/schedulers}\"" in run_scheduler
    assert "SCHEDULER_DIR=\"${SCHEDULER_DIR:-${PROJECT_ROOT}/schedulers}\"" in retrieval_scheduler
    assert "SETTINGS_FILE=\"${SINGLE_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/single_run_cli_settings}\"" in run_scheduler
    assert "SETTINGS_FILE=\"${SINGLE_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/single_run_cli_settings}\"" in retrieval_scheduler
    assert "source \"${SETTINGS_FILE}\"" in run_scheduler
    assert "source \"${SETTINGS_FILE}\"" in retrieval_scheduler
    assert "BASH_SOURCE" not in run_scheduler
    assert "BASH_SOURCE" not in retrieval_scheduler
    assert "#PBS -o /home/mhpereir/eulerian_heat_budget/logs/" in run_scheduler
    assert "#PBS -o /home/mhpereir/eulerian_heat_budget/logs/" in retrieval_scheduler
    assert "#PBS -l select=1:ncpus=8:mem=8gb" in retrieval_scheduler
    assert "export PYTHONUNBUFFERED=\"${PYTHONUNBUFFERED:-1}\"" in retrieval_scheduler
    assert "/dev/null" not in run_scheduler
    assert "/dev/null" not in retrieval_scheduler
    assert run_scheduler.index("exec > >(tee -a") < run_scheduler.index(
        "source \"${SETTINGS_FILE}\""
    )
    assert retrieval_scheduler.index("exec > >(tee -a") < retrieval_scheduler.index(
        "source \"${SETTINGS_FILE}\""
    )
    assert "python run_budget.py \"${RUN_ARGS[@]}\"" in run_scheduler
    assert "python -u staged_arco_retrieval.py \"${RETRIEVAL_ARGS[@]}\"" in retrieval_scheduler
    assert "--stage-time-chunk \"${STAGED_ARCO_TIME_CHUNK}\"" in settings
    assert "--stage-attempt-timeout-seconds \"${STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS}\"" in settings


def test_pbs_production_scheduler_uses_cli_settings(tmp_path):
    scheduler_dir = Path(PROJECT_ROOT) / "schedulers"
    settings = (scheduler_dir / "production_run_cli_settings.sh").read_text()
    production_scheduler = (scheduler_dir / "schedule_run_budget_production.sh").read_text()
    staging_scheduler = (scheduler_dir / "schedule_staged_arco_retrieval_production.sh").read_text()
    consolidation_scheduler = (
        scheduler_dir / "schedule_consolidate_staged_arco_cache.sh"
    ).read_text()
    submission_script = (
        scheduler_dir / "submit_staged_arco_production.sh"
    ).read_text()
    run_submission_script = (
        scheduler_dir / "submit_run_budget_production.sh"
    ).read_text()

    assert "START_YEAR=" in settings
    assert "END_YEAR=" in settings
    assert "CAMPAIGN_ID=" in settings
    assert "RUN_ID=" in settings
    assert "EHB_WORKSPACE_ROOT=" in settings
    assert "EHB_CAMPAIGN_DATA_ROOT=" in settings
    assert "EHB_STAGED_ZARR_ROOT=" in settings
    assert "EHB_RUN_BUDGET_ROOT=" in settings
    assert "EHB_LOG_ROOT=" in settings
    assert "STAGED_CACHE_BASE_ROOT=" in settings
    assert "ZG_BOTTOM=" in settings
    assert "USE_SURFACE_AS_BOTTOM" not in settings
    assert "STAGED_ARCO_TIME_CHUNK=" in settings
    assert "STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS=" in settings
    assert "ehb_build_production_run_budget_args" in settings
    assert "ehb_build_production_staged_retrieval_args" in settings
    assert "ehb_build_staged_campaign_init_args" in settings
    assert "ehb_year_shard_root" in settings
    assert "ehb_require_consolidated_staged_cache" in settings
    assert "ehb_require_venus_production_checkout" in settings
    assert "ehb_verify_production_submission_checkout" in settings
    assert "refs/heads/production_development_staged" in settings
    assert "schedule_run_budget_production.sh" in settings
    assert "schedule_staged_arco_retrieval_production.sh" in settings
    assert "SETTINGS_FILE=\"${PRODUCTION_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/production_run_cli_settings.sh}\"" in production_scheduler
    assert "SETTINGS_FILE=\"${PRODUCTION_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/production_run_cli_settings.sh}\"" in staging_scheduler
    assert "source \"${SETTINGS_FILE}\"" in production_scheduler
    assert "source \"${SETTINGS_FILE}\"" in staging_scheduler
    assert "ehb_verify_runtime_checkout" in production_scheduler
    assert "ehb_require_external_production_paths" in production_scheduler
    assert "export PYTHONUNBUFFERED=\"${PYTHONUNBUFFERED:-1}\"" in staging_scheduler
    assert "#PBS -l select=1:ncpus=8:mem=8gb" in staging_scheduler
    assert "#PBS -l walltime=48:00:00" in staging_scheduler
    assert "#PBS -o /dev/null" not in production_scheduler
    assert "ehb_build_production_run_budget_args COMMON_RUN_ARGS" in production_scheduler
    assert "ehb_production_year_for_task \"${PBS_ARRAY_INDEX}\"" in production_scheduler
    assert "ehb_build_production_time_window \"${YEAR}\" TIME_START TIME_END" in production_scheduler
    assert "ehb_production_year_for_task \"${PBS_ARRAY_INDEX}\"" in staging_scheduler
    assert "ehb_build_production_time_window \"${YEAR}\" TIME_START TIME_END" in staging_scheduler
    assert "YEAR_CACHE_ROOT=$(ehb_year_shard_root \"${YEAR}\")" in staging_scheduler
    assert "\"${YEAR_CACHE_ROOT}\"" in staging_scheduler
    assert "staged_arco_campaign.py \"${CAMPAIGN_ARGS[@]}\"" in staging_scheduler
    assert "staged_arco_campaign.py finalize-year" in staging_scheduler
    assert "staged_arco_retrieval.py \"${RETRIEVAL_ARGS[@]}\"" in staging_scheduler
    assert "#PBS -l select=1:ncpus=1:mem=4gb" in consolidation_scheduler
    assert "consolidate_staged_arco_cache.py" in consolidation_scheduler
    assert "depend=afterok:${RETRIEVAL_JOB_ID}" in submission_script
    assert 'LOG_DIR="${LOG_DIR:-${EHB_LOG_ROOT}/${REGION}/${RUN_ID}}"' in settings
    assert '-o "${LOG_DIR}/"' in submission_script
    assert '-o "${LOG_DIR}/"' in run_submission_script
    assert (
        "EXPECTED_COMMIT=$(ehb_verify_production_submission_checkout)"
        in submission_script
    )
    assert (
        "EXPECTED_COMMIT=$(ehb_verify_production_submission_checkout)"
        in run_submission_script
    )
    assert "schedule_run_budget_production.sh" in run_submission_script
    assert "CAMPAIGN_ID=${CAMPAIGN_ID}" in submission_script
    assert "STAGED_CACHE_ROOT=${STAGED_CACHE_ROOT}" in submission_script

    script = """
set -euo pipefail
source schedulers/production_run_cli_settings.sh
YEAR=$(ehb_production_year_for_task 5)
ehb_build_production_time_window "${YEAR}" TIME_START TIME_END
YEAR_CACHE_ROOT=$(ehb_year_shard_root "${YEAR}")
RUN_ARGS=()
ehb_build_production_run_budget_args RUN_ARGS
RETRIEVAL_ARGS=()
ehb_build_production_staged_retrieval_args RETRIEVAL_ARGS "${TIME_START}" "${TIME_END}" "${YEAR_CACHE_ROOT}"
CAMPAIGN_ARGS=()
ehb_build_staged_campaign_init_args CAMPAIGN_ARGS
printf 'YEAR=%s\n' "${YEAR}"
printf 'TIME_START=%s\n' "${TIME_START}"
printf 'TIME_END=%s\n' "${TIME_END}"
printf 'YEAR_CACHE_ROOT=%s\n' "${YEAR_CACHE_ROOT}"
printf 'RUN_ARGS=%s\n' "${RUN_ARGS[*]}"
printf 'RETRIEVAL_ARGS=%s\n' "${RETRIEVAL_ARGS[*]}"
printf 'CAMPAIGN_ARGS=%s\n' "${CAMPAIGN_ARGS[*]}"
"""
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": PROJECT_ROOT,
            "PRODUCTION_OUTPUT_DIR": str(tmp_path / "production"),
            "STAGED_CACHE_ROOT": str(tmp_path / "cache"),
        }
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    for required in ("campaign.json", "cache.sqlite", "consolidation.json"):
        (cache_root / required).touch()

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout
    assert "YEAR=1945" in output
    assert "TIME_START=1945-05-01T00:00:00" in output
    assert "TIME_END=1945-10-31T23:00:00" in output
    assert f"YEAR_CACHE_ROOT={tmp_path / 'cache'}/shards/year=1945" in output
    assert "--data-source staged_arco_cache" in output
    assert f"--production-output-dir {tmp_path / 'production'}" in output
    assert f"--staged-cache-root {tmp_path / 'cache'}" in output
    assert "--region alaska" in output
    assert "--zg-top-pa 70000" in output
    assert "--zg-bottom surface_pressure" in output
    assert "--diagnostic-plots" in output
    assert "--no-constant-temperature-test" in output
    assert "--time-start 1945-05-01T00:00:00 --time-end 1945-10-31T23:00:00" in output
    assert "--stage-time-chunk month" in output
    assert "--stage-attempt-timeout-seconds 20800" in output
    assert f"--cache-root {tmp_path / 'cache'}" in output
    assert "--campaign-id alaska-surface-700hpa-1940-2025" in output
    assert "--start-year 1940 --end-year 2025" in output
    assert "--margin-n 1" in output

    pressure_env = env.copy()
    pressure_env.update({"ZG_BOTTOM": "pressure_level", "ZG_BOTTOM_PA": "85000"})
    pressure_result = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        env=pressure_env,
        text=True,
        capture_output=True,
        check=True,
    )
    pressure_output = pressure_result.stdout
    assert "--zg-bottom pressure_level" in pressure_output
    assert "--zg-bottom-pa 85000" in pressure_output

    (cache_root / "consolidation.json").unlink()
    incomplete = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert incomplete.returncode != 0
    assert "staged campaign is not consolidated" in incomplete.stderr


def _write_fake_production_git(path: Path) -> None:
    path.write_text(
        """#!/bin/bash
case "$*" in
  *"rev-parse --show-toplevel"*)
    printf '%s\\n' "${PROJECT_ROOT:?}"
    ;;
  *"symbolic-ref --quiet --short HEAD"*)
    if [[ "${FAKE_DETACHED:-0}" == 1 ]]; then exit 1; fi
    printf '%s\\n' "${FAKE_BRANCH:-production_development_staged}"
    ;;
  *"rev-parse --abbrev-ref --symbolic-full-name @{upstream}"*)
    printf '%s\\n' "${FAKE_UPSTREAM:-origin/production_development_staged}"
    ;;
  *"status --porcelain"*)
    printf '%s' "${FAKE_STATUS:-}"
    ;;
  *"rev-parse HEAD"*)
    printf '%s\\n' "${FAKE_HEAD:-0000000000000000000000000000000000000000}"
    ;;
  *"ls-remote --exit-code origin refs/heads/production_development_staged"*)
    printf '%s\\trefs/heads/production_development_staged\\n' \
      "${FAKE_REMOTE_HEAD:-0000000000000000000000000000000000000000}"
    ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("checkout_parent", "branch", "head", "remote_head", "expected_error"),
    [
        (
            "development",
            "production_development_staged",
            "0" * 40,
            "0" * 40,
            "must be below",
        ),
        (
            "production",
            "codex/extra-production-change",
            "0" * 40,
            "0" * 40,
            "requires branch production_development_staged",
        ),
        (
            "production",
            "production_development_staged",
            "1" * 40,
            "0" * 40,
            "is not the authoritative remote tip",
        ),
    ],
)
def test_venus_production_preflight_rejects_non_authoritative_checkout(
    tmp_path, checkout_parent, branch, head, remote_head, expected_error
):
    scheduler_dir = Path(PROJECT_ROOT) / "schedulers"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_production_git(fake_bin / "git")
    workspace_root = tmp_path / "eulerian-heat-budget"
    project_root = workspace_root / checkout_parent / "eulerian_heat_budget-test"
    project_root.mkdir(parents=True)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROJECT_ROOT": str(project_root),
            "EHB_WORKSPACE_ROOT": str(workspace_root),
            "FAKE_BRANCH": branch,
            "FAKE_HEAD": head,
            "FAKE_REMOTE_HEAD": remote_head,
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f"source {scheduler_dir / 'production_run_cli_settings.sh'}; "
                "ehb_verify_production_submission_checkout"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_staged_production_submission_links_consolidation_to_array(tmp_path):
    scheduler_dir = Path(PROJECT_ROOT) / "schedulers"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    _write_fake_production_git(fake_git)
    fake_qsub = fake_bin / "qsub"
    fake_qsub.write_text(
        """#!/bin/bash
printf '%s\\n' "$*" >> "${QSUB_LOG:?}"
case "$*" in
  *schedule_staged_arco_retrieval_production.sh) echo "123[].venus" ;;
  *schedule_consolidate_staged_arco_cache.sh) echo "124.venus" ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    fake_qsub.chmod(0o755)
    qsub_log = tmp_path / "qsub.log"
    log_dir = tmp_path / "logs"
    workspace_root = tmp_path / "eulerian-heat-budget"
    project_root = workspace_root / "production" / "eulerian_heat_budget-test"
    project_root.mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROJECT_ROOT": str(project_root),
            "EHB_WORKSPACE_ROOT": str(workspace_root),
            "SCHEDULER_DIR": str(scheduler_dir),
            "PRODUCTION_RUN_CLI_SETTINGS": str(
                scheduler_dir / "production_run_cli_settings.sh"
            ),
            "QSUB_BIN": str(fake_qsub),
            "QSUB_LOG": str(qsub_log),
            "LOG_DIR": str(log_dir),
        }
    )

    result = subprocess.run(
        ["bash", str(scheduler_dir / "submit_staged_arco_production.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    submitted = qsub_log.read_text(encoding="utf-8").splitlines()
    assert len(submitted) == 2
    assert "-J 0-85%5" in submitted[0]
    assert f"-o {log_dir}/" in submitted[0]
    assert "schedule_staged_arco_retrieval_production.sh" in submitted[0]
    assert "-W depend=afterok:123[].venus" in submitted[1]
    assert f"-o {log_dir}/" in submitted[1]
    assert "schedule_consolidate_staged_arco_cache.sh" in submitted[1]
    assert "retrieval_job_id=123[].venus" in result.stdout
    assert "consolidation_job_id=124.venus" in result.stdout


def test_run_budget_production_submission_uses_external_paths(tmp_path):
    scheduler_dir = Path(PROJECT_ROOT) / "schedulers"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    _write_fake_production_git(fake_git)
    fake_qsub = fake_bin / "qsub"
    fake_qsub.write_text(
        """#!/bin/bash
printf '%s\\n' "$*" >> "${QSUB_LOG:?}"
echo "200[].venus"
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    fake_qsub.chmod(0o755)

    cache_root = tmp_path / "campaign-data" / "staged-zarr" / "alaska" / "campaign"
    cache_root.mkdir(parents=True)
    for required in ("campaign.json", "cache.sqlite", "consolidation.json"):
        (cache_root / required).touch()
    output_dir = tmp_path / "campaign-data" / "run-budget" / "alaska" / "run"
    log_dir = tmp_path / "campaign-data" / "logs" / "alaska" / "run"
    qsub_log = tmp_path / "qsub.log"
    workspace_root = tmp_path / "eulerian-heat-budget"
    project_root = workspace_root / "production" / "eulerian_heat_budget-test"
    project_root.mkdir(parents=True)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROJECT_ROOT": str(project_root),
            "EHB_WORKSPACE_ROOT": str(workspace_root),
            "SCHEDULER_DIR": str(scheduler_dir),
            "PRODUCTION_RUN_CLI_SETTINGS": str(
                scheduler_dir / "production_run_cli_settings.sh"
            ),
            "QSUB_BIN": str(fake_qsub),
            "QSUB_LOG": str(qsub_log),
            "STAGED_CACHE_ROOT": str(cache_root),
            "PRODUCTION_OUTPUT_DIR": str(output_dir),
            "LOG_DIR": str(log_dir),
        }
    )

    result = subprocess.run(
        ["bash", str(scheduler_dir / "submit_run_budget_production.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    submitted = qsub_log.read_text(encoding="utf-8")
    assert "-J 0-85%5" in submitted
    assert f"-o {log_dir}/" in submitted
    assert "schedule_run_budget_production.sh" in submitted
    assert f"STAGED_CACHE_ROOT={cache_root}" in submitted
    assert f"PRODUCTION_OUTPUT_DIR={output_dir}" in submitted
    assert "production_job_id=200[].venus" in result.stdout


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
            output_path="/tmp/test-run/heat_budget.nc",
            constant_t_output_path="/tmp/test-run/heat_budget_constant_T.nc",
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


def _configure_core_stubs(monkeypatch, args):
    ds_domain = _make_stub_domain_dataset()

    monkeypatch.setattr(run_budget.config, "DEFAULT_LOCAL_PATH", "/tmp/test-data")
    monkeypatch.setattr(run_budget.cli, "parse_args", lambda: args)
    monkeypatch.setattr(run_budget.io, "load_dataset", lambda *args, **kwargs: xr.Dataset())
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
        "plot_budget_terms_hourly",
        lambda ds_budget, smoothing_window, plot_dir: plot_calls.append(("hourly", smoothing_window, plot_dir)),
    )
    monkeypatch.setattr(
        run_budget.plot_results,
        "plot_budget_terms_day_bin",
        lambda ds_budget, plot_dir: plot_calls.append(("daily", plot_dir)),
    )
    monkeypatch.setattr(
        run_budget.plot_results,
        "plot_constant_T_results",
        lambda ds_budget, ds_test, plot_dir: plot_calls.append(("constant_T", plot_dir)),
    )

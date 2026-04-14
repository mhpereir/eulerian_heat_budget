import sys
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import check_pbl


def _make_test_fields():
    levels = np.array([60000.0, 80000.0, 100000.0])
    Z_flipped = xr.DataArray(
        np.array(
            [
                [
                    [[1000.0, 1000.0], [1000.0, 1000.0]],
                    [[2000.0, 2000.0], [2000.0, 2000.0]],
                    [[3000.0, 3000.0], [3000.0, 3000.0]],
                ],
                [
                    [[1000.0, 1000.0], [1000.0, 1000.0]],
                    [[2000.0, 2000.0], [2000.0, 2000.0]],
                    [[3000.0, 3000.0], [3000.0, 3000.0]],
                ],
            ]
        ),
        dims=("time", "level", "lat", "lon"),
        coords={
            "time": [0, 1],
            "level": levels[::-1],
            "lat": [0, 1],
            "lon": [0, 1],
        },
    )
    pbl = xr.DataArray(
        np.array(
            [
                [[1500.0, 1500.0], [1500.0, 1500.0]],
                [[1500.0, 1500.0], [2500.0, 2500.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0, 1], "lon": [0, 1]},
    )
    lnp_vals = np.log(levels[::-1])
    return Z_flipped, pbl, lnp_vals


def test_pressure_at_height_interpolates_local_targets():
    levels = np.array([100000.0, 80000.0, 60000.0])
    lnp_vals = np.log(levels)
    Z_chunk = np.array(
        [
            [
                [[1000.0], [1000.0]],
                [[2000.0], [2000.0]],
                [[3000.0], [3000.0]],
            ]
        ]
    )
    z_target = np.array([[[1500.0], [2500.0]]])

    out = check_pbl._pressure_at_height(z_target, Z_chunk, lnp_vals)

    expected = np.array(
        [[[np.sqrt(100000.0 * 80000.0)], [np.sqrt(80000.0 * 60000.0)]]]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0.0)


def test_domain_mean_pbl_top_pressure_chunking_is_invariant():
    Z_flipped, pbl, lnp_vals = _make_test_fields()

    ts_one, pmin_one, fields_one = check_pbl._compute_domain_mean_pbl_top_pressure(
        Z_flipped=Z_flipped,
        pbl=pbl,
        lnp_vals=lnp_vals,
        chunk_size=1,
        progress=False,
    )
    ts_all, pmin_all, fields_all = check_pbl._compute_domain_mean_pbl_top_pressure(
        Z_flipped=Z_flipped,
        pbl=pbl,
        lnp_vals=lnp_vals,
        chunk_size=10,
        progress=False,
    )

    np.testing.assert_allclose(ts_one, ts_all, rtol=1e-12, atol=0.0)
    assert pmin_one == pmin_all
    np.testing.assert_allclose(fields_one, fields_all, rtol=1e-12, atol=0.0)


def test_domain_mean_pressure_series_uses_local_heights_first():
    Z_flipped, pbl, lnp_vals = _make_test_fields()

    ts, p_min_anywhere, p_field_time_series = check_pbl._compute_domain_mean_pbl_top_pressure(
        Z_flipped=Z_flipped,
        pbl=pbl,
        lnp_vals=lnp_vals,
        chunk_size=10,
        progress=False,
    )
    summary = check_pbl._summarize_pressure_series(ts)
    local_pressure_summary_hpa = check_pbl._summarize_local_pressure_field_hpa(
        p_field_time_series
    )
    spatial_summary = check_pbl._summarize_spatial_pressure_fields(p_field_time_series)

    p_1500 = np.sqrt(100000.0 * 80000.0)
    p_2500 = np.sqrt(80000.0 * 60000.0)
    expected_ts = np.array([p_1500, 0.5 * (p_1500 + p_2500)])
    mean_height_first_t1 = 80000.0

    np.testing.assert_allclose(ts, expected_ts, rtol=1e-12, atol=0.0)
    assert not np.isclose(ts[1], mean_height_first_t1)
    assert np.isclose(p_min_anywhere, p_2500)
    assert p_min_anywhere < summary["min"]
    assert np.isclose(summary["min"], expected_ts.min())
    assert np.isclose(summary["mean"], expected_ts.mean())
    assert np.isclose(summary["p01"], np.nanpercentile(expected_ts, 1))
    assert np.isclose(summary["p05"], np.nanpercentile(expected_ts, 5))
    assert np.isclose(local_pressure_summary_hpa["min"], p_2500 / 100.0)
    assert np.isclose(
        local_pressure_summary_hpa["p01"],
        np.nanpercentile(p_field_time_series / 100.0, 1),
    )
    assert np.isclose(
        local_pressure_summary_hpa["p05"],
        np.nanpercentile(p_field_time_series / 100.0, 5),
    )
    np.testing.assert_allclose(
        spatial_summary["min"],
        np.array([[p_1500, p_1500], [p_2500, p_2500]]),
        rtol=1e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        spatial_summary["p01"],
        np.nanpercentile(p_field_time_series, 1, axis=0),
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        spatial_summary["p05"],
        np.nanpercentile(p_field_time_series, 5, axis=0),
        rtol=1e-12,
        atol=0.0,
    )


def test_plot_spatial_pressure_metrics_writes_pngs(tmp_path):
    p_1500 = np.sqrt(100000.0 * 80000.0)
    p_2500 = np.sqrt(80000.0 * 60000.0)
    pressure_maps = {
        "min": xr.DataArray(
            np.full((2, 2), p_2500),
            dims=("lat", "lon"),
            coords={"lat": [0, 1], "lon": [10, 11]},
        ),
        "p01": xr.DataArray(
            np.full((2, 2), p_2500 + 1000.0),
            dims=("lat", "lon"),
            coords={"lat": [0, 1], "lon": [10, 11]},
        ),
        "p05": xr.DataArray(
            np.full((2, 2), p_1500),
            dims=("lat", "lon"),
            coords={"lat": [0, 1], "lon": [10, 11]},
        ),
    }

    out_paths = check_pbl._plot_spatial_pressure_metrics(
        pressure_maps=pressure_maps,
        plot_dir=tmp_path,
        title_prefix="PBL-top pressure",
    )

    assert len(out_paths) == 3
    for out_path in out_paths:
        assert out_path.exists()
        assert out_path.suffix == ".png"
        assert out_path.name in {
            "pbl_top_pressure_min.png",
            "pbl_top_pressure_p01.png",
            "pbl_top_pressure_p05.png",
        }


def test_write_run_info_serializes_summary_payload(tmp_path):
    output_dir = tmp_path / "2021"
    run_info_path = output_dir / "run_info.json"
    payload = {
        "year": 2021,
        "months_included": [6, 7, 8],
        "bbox": {
            "lat_min": 40.0,
            "lat_max": 60.0,
            "lon_min": -130.0,
            "lon_max": -110.0,
        },
        "grid_points": {"time": 2208, "lat": 81, "lon": 81},
        "pbl_height_stats_m": {"max": 6603.3, "p99": 3549.7, "p95": 2486.9, "mean": 755.4},
        "local_pbl_top_pressure_hpa": {
            "min": 459.22,
            "p01": 468.11,
            "p05": 481.34,
        },
        "domain_mean_pbl_top_pressure_pa": {
            "min": 79883.0,
            "p01": 81047.0,
            "p05": 82747.0,
            "mean": 92154.0,
        },
        "recommendation": {
            "default_zg_top_pa": 45900.0,
            "lowest_local_pbl_top_pressure_pa": 45922.0,
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "run_info_json": str(run_info_path),
            "plots": {
                "min": str(output_dir / "pbl_top_pressure_min.png"),
                "p01": str(output_dir / "pbl_top_pressure_p01.png"),
                "p05": str(output_dir / "pbl_top_pressure_p05.png"),
            },
        },
    }

    written_path = check_pbl._write_run_info(payload, run_info_path)

    assert written_path == run_info_path
    assert written_path.exists()

    content = json.loads(written_path.read_text())
    assert content["year"] == 2021
    assert content["months_included"] == [6, 7, 8]
    assert isinstance(content["pbl_height_stats_m"]["max"], float)
    assert isinstance(content["local_pbl_top_pressure_hpa"]["min"], float)
    assert isinstance(content["domain_mean_pbl_top_pressure_pa"]["min"], float)
    assert content["local_pbl_top_pressure_hpa"]["p05"] == 481.34
    assert content["artifacts"]["run_info_json"] == str(run_info_path)
    assert content["artifacts"]["plots"]["min"] == str(output_dir / "pbl_top_pressure_min.png")


def test_build_parser_accepts_single_year_and_rejects_year_range_args():
    parser = check_pbl._build_parser()

    args = parser.parse_args(["--year", "2021"])

    assert args.year == 2021
    assert not hasattr(args, "year_start")
    assert not hasattr(args, "year_end")

    with pytest.raises(SystemExit):
        parser.parse_args(["--year-start", "2021", "--year-end", "2021"])

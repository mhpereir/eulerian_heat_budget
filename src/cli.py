"""
Command-line argument parser for domain request inputs.

The returned Namespace is intentionally aligned with the attributes consumed by
`scripts/run_budget.py::build_request_from_cli(args)`.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the parser used to collect DomainRequest CLI inputs."""
    parser = argparse.ArgumentParser(
        prog="run_budget",
        description="Parse domain request arguments for Eulerian heat budget runs.",
    )

    parser.add_argument(
        "--lat-min",
        dest="lat_min",
        type=float,
        default=None,
        help="Southern latitude bound (degrees).",
    )
    parser.add_argument(
        "--lat-max",
        dest="lat_max",
        type=float,
        default=None,
        help="Northern latitude bound (degrees).",
    )
    parser.add_argument(
        "--lon-min",
        dest="lon_min",
        type=float,
        default=None,
        help="Western longitude bound (degrees).",
    )
    parser.add_argument(
        "--lon-max",
        dest="lon_max",
        type=float,
        default=None,
        help="Eastern longitude bound (degrees).",
    )
    parser.add_argument(
        "--margin-n",
        dest="margin_n",
        type=int,
        default=None,
        help="Margin around the selected domain (integer grid points).",
    )
    parser.add_argument(
        "--zg-top-pa",
        dest="zg_top_pa",
        type=float,
        default=None,
        help="Top pressure boundary in Pa.",
    )
    parser.add_argument(
        "--zg-bottom",
        dest="zg_bottom",
        choices=("surface_pressure", "pressure_level"),
        default=None,
        help="Bottom boundary mode.",
    )
    parser.add_argument(
        "--zg-bottom-pa",
        dest="zg_bottom_pa",
        type=float,
        default=None,
        help="Bottom pressure boundary in Pa (used when --zg-bottom=pressure_level).",
    )

    overflow_group = parser.add_mutually_exclusive_group()
    overflow_group.add_argument(
        "--allow-bottom-overflow",
        dest="allow_bottom_overflow",
        action="store_true",
        default=None,
        help="Allow bottom-layer overflow when surface pressure exceeds grid bottom pressure.",
    )
    overflow_group.add_argument(
        "--no-allow-bottom-overflow",
        dest="allow_bottom_overflow",
        action="store_false",
        help="Clip bottom-layer contribution instead of allowing overflow.",
    )


    surface_var_group = parser.add_mutually_exclusive_group()
    surface_var_group.add_argument(
        "--use-surface-variables",
        dest="use_surface_variables",
        action="store_true",
        default=None,
        help="Include surface variables (T2m, u10, v10) in budget calculations instead of lowest model level variables.",
    )
    surface_var_group.add_argument(
        "--no-use-surface-variables",
        dest="use_surface_variables",
        action="store_false",
        help="Use lowest model level variables in budget calculations instead of surface variables.",
    )

    parser.add_argument(
        "--surface-variable-mode",
        dest="surface_variable_mode",
        choices=("none", "combined", "diagnostic_only"),
        default=None,
        help="How surface variables should be handled when they are enabled.",
    )

    parser.add_argument(
        "--data-source",
        dest="data_source",
        choices=("local_era5", "arco_era5"),
        default=None,
        help="Data source to load input dataset from.",
    )

    parser.add_argument(
        "--time-start",
        dest="time_start",
        type=str,
        default=None,
        help="Start time for data selection (ISO format, e.g. 1941-06-01T00:00:00).",
    )
    parser.add_argument(
        "--time-end",
        dest="time_end",        
        type=str,
        default=None,
        help="End time for data selection (ISO format, e.g. 1941-06-07T23:00:00).",
    )

    diagnostic_plots_group = parser.add_mutually_exclusive_group()
    diagnostic_plots_group.add_argument(
        "--diagnostic-plots",
        dest="diagnostic_plots",
        action="store_true",
        default=None,
        help="Generate diagnostic and summary plots for the run.",
    )
    diagnostic_plots_group.add_argument(
        "--no-diagnostic-plots",
        dest="diagnostic_plots",
        action="store_false",
        help="Skip all diagnostic and summary plot generation.",
    )

    constant_temperature_test_group = parser.add_mutually_exclusive_group()
    constant_temperature_test_group.add_argument(
        "--constant-temperature-test",
        dest="constant_temperature_test",
        action="store_true",
        default=None,
        help="Run the constant-temperature validation calculation.",
    )
    constant_temperature_test_group.add_argument(
        "--no-constant-temperature-test",
        dest="constant_temperature_test",
        action="store_false",
        help="Skip the constant-temperature validation calculation.",
    )

    parser.add_argument(
        "--production-output-dir",
        dest="production_output_dir",
        type=str,
        default=None,
        help="Shared output directory for production yearly runs.",
    )
    parser.add_argument(
        "--init-production-manifest",
        dest="init_production_manifest",
        action="store_true",
        default=False,
        help="Create the shared production manifest and exit.",
    )
    parser.add_argument(
        "--production-start-year",
        dest="production_start_year",
        type=int,
        default=None,
        help="First year covered by the production campaign manifest.",
    )
    parser.add_argument(
        "--production-end-year",
        dest="production_end_year",
        type=int,
        default=None,
        help="Last year covered by the production campaign manifest.",
    )
    parser.add_argument(
        "--overwrite-output",
        dest="overwrite_output",
        action="store_true",
        default=False,
        help="Overwrite an existing yearly production NetCDF output.",
    )


    return parser

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args and return a Namespace for build_request_from_cli."""
    return build_arg_parser().parse_args(argv)

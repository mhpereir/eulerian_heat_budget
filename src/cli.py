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
        dest="in_surface_variables",
        action="store_true",
        default=None,
        help="Include surface variables (T2m, u10, v10) in budget calculations instead of lowest model level variables.",
    )
    surface_var_group.add_argument(
        "--no-use-surface-variables",
        dest="in_surface_variables",
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

    return parser

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args and return a Namespace for build_request_from_cli."""
    return build_arg_parser().parse_args(argv)

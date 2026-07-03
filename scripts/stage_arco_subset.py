import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import run_budget
from src import cli, config, io, specs


def parse_args():
    parser = cli.build_arg_parser()
    parser.prog = "stage_arco_subset"
    parser.description = "Stage the minimal ARCO ERA5 subset needed for an offline budget run."
    parser.add_argument(
        "--stage-output",
        dest="stage_output",
        type=str,
        required=True,
        help="Output local Zarr store to create.",
    )
    parser.add_argument(
        "--overwrite-stage-output",
        dest="overwrite_stage_output",
        action="store_true",
        default=False,
        help="Overwrite an existing staged Zarr store.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_source not in (None, "arco_era5"):
        raise ValueError("stage_arco_subset.py can only stage from ARCO ERA5.")

    request = run_budget.build_request_from_cli(args)
    surface_specs = run_budget.build_surface_behaviour_from_cli(args)
    source_cfg = specs.DataSourceConfig(
        kind="arco_era5",
        arco_path=config.DEFAULT_ARCO_PATH,
        time_start=args.time_start if args.time_start is not None else config.DEFAULT_TIME_START,
        time_end=args.time_end if args.time_end is not None else config.DEFAULT_TIME_END,
    )

    output_path = Path(args.stage_output)
    if output_path.exists() and not args.overwrite_stage_output:
        raise FileExistsError(
            f"Staged output already exists: {output_path}. "
            "Pass --overwrite-stage-output to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] staging ARCO ERA5 subset from {source_cfg.arco_path}")
    print(f"[info] writing staged Zarr to {output_path}")
    ds = io.build_arco_staged_subset(
        source_cfg,
        surface_specs,
        request,
        include_benchmark_variables=args.include_benchmark_variables,
    )
    ds.to_zarr(str(output_path), mode="w")
    print(
        "[info] staged dataset written: "
        f"sizes={dict(ds.sizes)}, variables={list(ds.data_vars)}"
    )


if __name__ == "__main__":
    main()

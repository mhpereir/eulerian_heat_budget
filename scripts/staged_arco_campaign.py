"""Initialize a staged campaign or finalize one completed yearly shard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src_arco.campaign import (  # noqa: E402
    Campaign,
    CampaignConfigError,
    initialize_campaign,
)
from src_arco.consolidation import finalize_year_shard  # noqa: E402
from src_arco.shard_artifacts import ShardValidationError  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    init = subparsers.add_parser("init", help="Create or validate campaign.json.")
    init.add_argument("--cache-root", required=True, type=Path)
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--start-year", required=True, type=int)
    init.add_argument("--end-year", required=True, type=int)
    init.add_argument("--start-month-day", required=True)
    init.add_argument("--end-month-day", required=True)
    domain = init.add_mutually_exclusive_group(required=True)
    domain.add_argument("--region")
    domain.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
    )
    init.add_argument("--margin-n", required=True, type=int)
    init.add_argument("--zg-top-pa", required=True, type=float)
    init.add_argument(
        "--zg-bottom",
        required=True,
        choices=("surface_pressure", "pressure_level"),
    )
    init.add_argument("--zg-bottom-pa", type=float)
    init.add_argument(
        "--allow-bottom-overflow",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    init.add_argument(
        "--time-chunk",
        required=True,
        choices=("none", "day", "month"),
    )
    init.add_argument("--attempt-timeout-seconds", required=True, type=float)
    init.add_argument(
        "--include-benchmark-variables",
        action=argparse.BooleanOptionalAction,
        required=True,
    )

    finalize = subparsers.add_parser(
        "finalize-year",
        help="Validate, manifest, and mark one yearly shard complete.",
    )
    finalize.add_argument("--cache-root", required=True, type=Path)
    finalize.add_argument("--year", required=True, type=int)
    finalize.add_argument("--pbs-job-id")
    finalize.add_argument("--pbs-array-index")
    finalize.add_argument("--git-commit", required=True)
    return parser


def _campaign_from_args(args: argparse.Namespace) -> Campaign:
    domain = {
        "margin_n": args.margin_n,
        "zg_top_pa": args.zg_top_pa,
        "zg_bottom": args.zg_bottom,
        "allow_bottom_overflow": args.allow_bottom_overflow,
    }
    if args.region is not None:
        domain["region"] = args.region
    else:
        lat_min, lat_max, lon_min, lon_max = args.bbox
        domain["bbox"] = {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        }
    if args.zg_bottom_pa is not None:
        domain["zg_bottom_pa"] = args.zg_bottom_pa
    return Campaign.from_mapping(
        {
            "schema_version": 1,
            "campaign_id": args.campaign_id,
            "years": {"start": args.start_year, "end": args.end_year},
            "season": {
                "start_month_day": args.start_month_day,
                "end_month_day": args.end_month_day,
            },
            "domain": domain,
            "staging": {
                "time_chunk": args.time_chunk,
                "attempt_timeout_seconds": args.attempt_timeout_seconds,
                "include_benchmark_variables": args.include_benchmark_variables,
            },
        }
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.action == "init":
            campaign = _campaign_from_args(args)
            path = initialize_campaign(args.cache_root, campaign)
            print(f"[campaign] campaign_sha256={campaign.sha256()}", flush=True)
            print(f"[campaign] descriptor={path}", flush=True)
        else:
            finalize_year_shard(
                args.cache_root,
                args.year,
                pbs_job_id=args.pbs_job_id,
                pbs_array_index=args.pbs_array_index,
                git_commit=args.git_commit,
            )
    except (CampaignConfigError, ShardValidationError, ValueError) as exc:
        print(f"[campaign:error] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

"""Create and update campaign-level staged ARCO production provenance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config  # noqa: E402
from src_arco.campaign import Campaign, CampaignConfigError  # noqa: E402
from src_arco.run_manifest import (  # noqa: E402
    StagedRunManifestError,
    build_staged_run_manifest,
    prepare_staged_run_manifest,
    record_staged_submission,
)


def _add_campaign_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--start-month-day", required=True)
    parser.add_argument("--end-month-day", required=True)
    domain = parser.add_mutually_exclusive_group(required=True)
    domain.add_argument("--region")
    domain.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
    )
    parser.add_argument("--margin-n", required=True, type=int)
    parser.add_argument("--zg-top-pa", required=True, type=float)
    parser.add_argument(
        "--zg-bottom",
        required=True,
        choices=("surface_pressure", "pressure_level"),
    )
    parser.add_argument("--zg-bottom-pa", type=float)
    parser.add_argument(
        "--allow-bottom-overflow",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument(
        "--time-chunk",
        required=True,
        choices=("none", "day", "month"),
    )
    parser.add_argument("--attempt-timeout-seconds", required=True, type=float)
    parser.add_argument(
        "--include-benchmark-variables",
        action=argparse.BooleanOptionalAction,
        required=True,
    )


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Create or validate immutable staged production settings.",
    )
    prepare.add_argument("--manifest-path", required=True, type=Path)
    prepare.add_argument("--run-id", required=True)
    _add_campaign_arguments(prepare)
    prepare.add_argument("--source-arco-path", default=config.DEFAULT_ARCO_PATH)
    prepare.add_argument("--git-branch", required=True)
    prepare.add_argument("--git-commit", required=True)
    prepare.add_argument("--git-upstream", required=True)
    prepare.add_argument("--git-origin-url", required=True)
    prepare.add_argument("--project-root", required=True, type=Path)
    prepare.add_argument("--settings-file", required=True, type=Path)
    prepare.add_argument("--staged-cache-root", required=True, type=Path)
    prepare.add_argument("--log-dir", required=True, type=Path)
    prepare.add_argument("--mamba-environment", required=True)
    prepare.add_argument("--first-task-index", required=True, type=int)
    prepare.add_argument("--last-task-index", required=True, type=int)
    prepare.add_argument("--max-parallel", required=True, type=int)
    prepare.add_argument("--retrieval-select", required=True)
    prepare.add_argument("--retrieval-walltime", required=True)
    prepare.add_argument("--consolidation-select", required=True)
    prepare.add_argument("--consolidation-walltime", required=True)

    record = subparsers.add_parser(
        "record-submission",
        help="Append or complete one OpenPBS submission record.",
    )
    record.add_argument("--manifest-path", required=True, type=Path)
    record.add_argument("--retrieval-job-id", required=True)
    record.add_argument("--consolidation-job-id")
    record.add_argument("--submission-host")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            campaign = _campaign_from_args(args)
            payload = build_staged_run_manifest(
                campaign=campaign,
                run_id=args.run_id,
                git_branch=args.git_branch,
                git_commit=args.git_commit,
                git_upstream=args.git_upstream,
                git_origin_url=args.git_origin_url,
                project_root=args.project_root,
                settings_file=args.settings_file,
                staged_cache_root=args.staged_cache_root,
                log_dir=args.log_dir,
                mamba_environment=args.mamba_environment,
                source_arco_path=args.source_arco_path,
                first_task_index=args.first_task_index,
                last_task_index=args.last_task_index,
                max_parallel=args.max_parallel,
                retrieval_select=args.retrieval_select,
                retrieval_walltime=args.retrieval_walltime,
                consolidation_select=args.consolidation_select,
                consolidation_walltime=args.consolidation_walltime,
            )
            path = prepare_staged_run_manifest(args.manifest_path, payload)
        else:
            path = record_staged_submission(
                args.manifest_path,
                retrieval_job_id=args.retrieval_job_id,
                consolidation_job_id=args.consolidation_job_id,
                submission_host=args.submission_host,
            )
    except (CampaignConfigError, StagedRunManifestError) as exc:
        parser.error(str(exc))
    print(f"[staged-run] manifest={path}", flush=True)


if __name__ == "__main__":
    main()

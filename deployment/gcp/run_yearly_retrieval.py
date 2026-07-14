"""Google Cloud Batch entrypoint for one staged ARCO campaign year."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Mapping

from deployment.gcp.campaign import Campaign, CampaignConfigError
from deployment.gcp.shard_artifacts import (
    SHARD_MANIFEST_NAME,
    SUCCESS_MARKER_NAME,
    ShardValidationError,
    build_shard_manifest,
    sha256_file,
    utc_now,
    validate_completed_shard,
    validate_manifest_files,
    write_json,
    write_shard_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[batch] {utc_now()} {message}", flush=True)


def _required_environment(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if value is None or not value.strip():
        raise CampaignConfigError(f"Required environment variable {name} is not set.")
    return value


def _environment_int(environ: Mapping[str, str], name: str) -> int:
    raw = _required_environment(environ, name)
    try:
        return int(raw)
    except ValueError as exc:
        raise CampaignConfigError(f"{name} must be an integer, got {raw!r}.") from exc


def build_retrieval_command(
    campaign: Campaign,
    *,
    year: int,
    cache_root: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    time_start, time_end = campaign.time_window(year)
    command = [
        python_executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "staged_arco_retrieval.py"),
        "--data-source",
        "arco_era5",
        "--staged-cache-root",
        str(cache_root),
        "--time-start",
        time_start,
        "--time-end",
        time_end,
        "--stage-time-chunk",
        campaign.time_chunk,
        "--stage-attempt-timeout-seconds",
        str(campaign.attempt_timeout_seconds),
        "--margin-n",
        str(campaign.margin_n),
        "--zg-top-pa",
        str(campaign.zg_top_pa),
        "--zg-bottom",
        campaign.zg_bottom,
        "--no-use-surface-variables",
    ]
    if campaign.region is not None:
        command.extend(["--region", campaign.region])
    else:
        assert campaign.bbox is not None
        command.extend(
            [
                "--lat-min",
                str(campaign.bbox[0]),
                "--lat-max",
                str(campaign.bbox[1]),
                "--lon-min",
                str(campaign.bbox[2]),
                "--lon-max",
                str(campaign.bbox[3]),
            ]
        )
    if campaign.zg_bottom_pa is not None:
        command.extend(["--zg-bottom-pa", str(campaign.zg_bottom_pa)])
    command.append(
        "--allow-bottom-overflow"
        if campaign.allow_bottom_overflow
        else "--no-allow-bottom-overflow"
    )
    if campaign.include_benchmark_variables:
        command.append("--include-benchmark-variables")
    return command


def _copy_manifest_files(source: Path, destination: Path, manifest: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    shutil.copyfile(source / SHARD_MANIFEST_NAME, destination / SHARD_MANIFEST_NAME)


def publish_shard(
    local_root: Path,
    destination: Path,
    *,
    campaign_sha256: str,
    year: int,
    time_start: str,
    time_end: str,
    task_index: int,
    retry_attempt: int,
) -> None:
    success_path = destination / SUCCESS_MARKER_NAME
    if success_path.exists():
        validate_completed_shard(
            destination,
            campaign_sha256=campaign_sha256,
            year=year,
            verify_hashes=False,
        )
        _log(f"year={year} already has a valid success marker; skipping publication")
        return

    if destination.exists():
        _log(f"removing incomplete publication for year={year}: {destination}")
        shutil.rmtree(destination)

    manifest = build_shard_manifest(
        local_root,
        campaign_sha256=campaign_sha256,
        year=year,
        time_start=time_start,
        time_end=time_end,
    )
    manifest_path = write_shard_manifest(local_root, manifest)
    _log(f"publishing {len(manifest['files'])} files for year={year} to {destination}")
    _copy_manifest_files(local_root, destination, manifest)

    destination_manifest = json.loads(
        (destination / SHARD_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    validate_manifest_files(destination, destination_manifest, verify_hashes=False)
    if sha256_file(destination / SHARD_MANIFEST_NAME) != sha256_file(manifest_path):
        raise ShardValidationError(
            f"Published shard manifest differs from local manifest for year={year}."
        )

    success = {
        "schema_version": 1,
        "campaign_sha256": campaign_sha256,
        "year": year,
        "manifest_sha256": sha256_file(manifest_path),
        "published_at": utc_now(),
        "batch_task_index": task_index,
        "batch_task_retry_attempt": retry_attempt,
    }
    local_success = local_root / SUCCESS_MARKER_NAME
    write_json(local_success, success)
    shutil.copyfile(local_success, success_path)
    _log(f"published success marker for year={year}")


def run_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    environ = os.environ if environ is None else environ
    campaign_json = _required_environment(environ, "EHB_CAMPAIGN_JSON")
    campaign = Campaign.from_json(campaign_json)
    campaign_sha256 = campaign.sha256()
    expected_sha256 = _required_environment(environ, "EHB_CAMPAIGN_SHA256")
    if expected_sha256 != campaign_sha256:
        raise CampaignConfigError(
            "EHB_CAMPAIGN_SHA256 does not match the normalized campaign JSON."
        )

    task_index = _environment_int(environ, "BATCH_TASK_INDEX")
    task_count = _environment_int(environ, "BATCH_TASK_COUNT")
    retry_attempt_raw = environ.get("BATCH_TASK_RETRY_ATTEMPT", "0")
    try:
        retry_attempt = int(retry_attempt_raw)
    except ValueError as exc:
        raise CampaignConfigError(
            f"BATCH_TASK_RETRY_ATTEMPT must be an integer, got {retry_attempt_raw!r}."
        ) from exc
    year = campaign.year_for_task(task_index, task_count)
    time_start, time_end = campaign.time_window(year)

    scratch = Path(_required_environment(environ, "EHB_LOCAL_SCRATCH"))
    output_mount = Path(_required_environment(environ, "EHB_OUTPUT_MOUNT"))
    local_root = scratch / campaign.campaign_id / f"year={year:04d}"
    destination = output_mount / "shards" / f"year={year:04d}"

    if (destination / SUCCESS_MARKER_NAME).exists():
        validate_completed_shard(
            destination,
            campaign_sha256=campaign_sha256,
            year=year,
            verify_hashes=False,
        )
        _log(f"year={year} is already complete; task will exit successfully")
        return 0

    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True)

    command = build_retrieval_command(campaign, year=year, cache_root=local_root)
    _log(f"starting retrieval for year={year}, task={task_index}/{task_count - 1}")
    command_runner(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        env=dict(environ),
    )
    _log(f"retrieval completed for year={year}; validating and publishing shard")
    publish_shard(
        local_root,
        destination,
        campaign_sha256=campaign_sha256,
        year=year,
        time_start=time_start,
        time_end=time_end,
        task_index=task_index,
        retry_attempt=retry_attempt,
    )
    return 0


def main() -> None:
    try:
        raise SystemExit(run_from_environment())
    except (CampaignConfigError, ShardValidationError) as exc:
        print(f"[batch:error] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

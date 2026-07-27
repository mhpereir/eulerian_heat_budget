"""Campaign-level provenance for staged ARCO production submissions."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping

from src_arco.campaign import Campaign


SCHEMA_VERSION = 1
MANIFEST_NAME = "production_run.json"
LOCK_NAME = ".production_run.lock"


class StagedRunManifestError(RuntimeError):
    """Raised when staged production provenance is invalid or inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StagedRunManifestError(f"{field} must be a non-empty string.")
    return value


def _require_commit(value: str) -> str:
    commit = _require_nonempty(value, "git.commit")
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise StagedRunManifestError(
            "git.commit must be a full lowercase hexadecimal Git object ID."
        )
    return commit


def _require_absolute_path(value: str | Path, field: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise StagedRunManifestError(f"{field} must be an absolute path.")
    return str(path)


def build_staged_run_manifest(
    *,
    campaign: Campaign,
    run_id: str,
    git_branch: str,
    git_commit: str,
    git_upstream: str,
    git_origin_url: str,
    project_root: str | Path,
    settings_file: str | Path,
    staged_cache_root: str | Path,
    log_dir: str | Path,
    mamba_environment: str,
    source_arco_path: str,
    first_task_index: int,
    last_task_index: int,
    max_parallel: int,
    retrieval_select: str,
    retrieval_walltime: str,
    consolidation_select: str,
    consolidation_walltime: str,
    now: str | None = None,
) -> dict[str, Any]:
    """Build one normalized immutable staged-run identity."""
    if first_task_index < 0 or last_task_index < first_task_index:
        raise StagedRunManifestError("scheduler array indices are invalid.")
    if max_parallel < 1:
        raise StagedRunManifestError("scheduler max_parallel must be positive.")
    if campaign.task_count != last_task_index - first_task_index + 1:
        raise StagedRunManifestError(
            "scheduler array size does not match the campaign year count."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "run_type": "staged_arco_retrieval",
        "created_at": now or utc_now(),
        "run_id": _require_nonempty(run_id, "run_id"),
        "campaign": {
            "sha256": campaign.sha256(),
            "configuration": campaign.to_mapping(),
        },
        "source": {
            "kind": "arco_era5",
            "arco_path": _require_nonempty(source_arco_path, "source.arco_path"),
        },
        "git": {
            "branch": _require_nonempty(git_branch, "git.branch"),
            "commit": _require_commit(git_commit),
            "dirty": False,
            "upstream": _require_nonempty(git_upstream, "git.upstream"),
            "origin_url": _require_nonempty(git_origin_url, "git.origin_url"),
        },
        "paths": {
            "project_root": _require_absolute_path(project_root, "paths.project_root"),
            "settings_file": _require_absolute_path(
                settings_file, "paths.settings_file"
            ),
            "staged_cache_root": _require_absolute_path(
                staged_cache_root, "paths.staged_cache_root"
            ),
            "log_dir": _require_absolute_path(log_dir, "paths.log_dir"),
        },
        "runtime": {
            "mamba_environment": _require_nonempty(
                mamba_environment, "runtime.mamba_environment"
            ),
        },
        "scheduler": {
            "system": "openpbs",
            "array": {
                "first_task_index": first_task_index,
                "last_task_index": last_task_index,
                "task_count": campaign.task_count,
                "max_parallel": max_parallel,
            },
            "retrieval_resources": {
                "select": _require_nonempty(
                    retrieval_select, "scheduler.retrieval_resources.select"
                ),
                "walltime": _require_nonempty(
                    retrieval_walltime, "scheduler.retrieval_resources.walltime"
                ),
            },
            "consolidation_resources": {
                "select": _require_nonempty(
                    consolidation_select,
                    "scheduler.consolidation_resources.select",
                ),
                "walltime": _require_nonempty(
                    consolidation_walltime,
                    "scheduler.consolidation_resources.walltime",
                ),
            },
        },
        "submissions": [],
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StagedRunManifestError(f"Staged run manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StagedRunManifestError(
            f"Staged run manifest is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise StagedRunManifestError("Staged run manifest must be a JSON object.")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise StagedRunManifestError(
            f"Staged run manifest schema_version must be {SCHEMA_VERSION}."
        )
    if payload.get("run_type") != "staged_arco_retrieval":
        raise StagedRunManifestError(
            "Staged run manifest run_type must be staged_arco_retrieval."
        )
    submissions = payload.get("submissions")
    if not isinstance(submissions, list):
        raise StagedRunManifestError("Staged run manifest submissions must be a list.")
    return payload


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "submissions"}
    }


@contextmanager
def _manifest_lock(manifest_path: Path) -> Iterator[None]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.parent / LOCK_NAME
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare_staged_run_manifest(
    manifest_path: str | Path,
    payload: Mapping[str, Any],
) -> str:
    """Atomically create a manifest or validate an identical run identity."""
    path = Path(manifest_path)
    desired = dict(payload)
    paths = desired.get("paths")
    if not isinstance(paths, Mapping):
        raise StagedRunManifestError("Staged run manifest paths must be an object.")
    cache_root = paths.get("staged_cache_root")
    if not isinstance(cache_root, str):
        raise StagedRunManifestError(
            "Staged run manifest paths.staged_cache_root must be a string."
        )
    expected_path = Path(cache_root) / MANIFEST_NAME
    if path != expected_path:
        raise StagedRunManifestError(
            f"Staged run manifest must be stored at {expected_path}, not {path}."
        )
    with _manifest_lock(path):
        if path.exists():
            existing = _load_manifest(path)
            if _identity(existing) != _identity(desired):
                raise StagedRunManifestError(
                    f"Staged run manifest is bound to different production settings: {path}"
                )
            return str(path)
        _atomic_write_json(path, desired)
    return str(path)


def record_staged_submission(
    manifest_path: str | Path,
    *,
    retrieval_job_id: str,
    consolidation_job_id: str | None = None,
    submission_host: str | None = None,
    submitted_at: str | None = None,
) -> str:
    """Append or complete one idempotent OpenPBS submission record."""
    path = Path(manifest_path)
    retrieval_job_id = _require_nonempty(
        retrieval_job_id, "submission.retrieval_job_id"
    )
    if consolidation_job_id is not None:
        consolidation_job_id = _require_nonempty(
            consolidation_job_id, "submission.consolidation_job_id"
        )

    with _manifest_lock(path):
        payload = _load_manifest(path)
        submissions = payload["submissions"]
        existing = next(
            (
                entry
                for entry in submissions
                if isinstance(entry, dict)
                and entry.get("retrieval_job_id") == retrieval_job_id
            ),
            None,
        )
        if existing is None:
            existing = {
                "submitted_at": submitted_at or utc_now(),
                "submission_host": submission_host,
                "retrieval_job_id": retrieval_job_id,
                "consolidation_job_id": consolidation_job_id,
                "dependency": (
                    f"afterok:{retrieval_job_id}"
                    if consolidation_job_id is not None
                    else None
                ),
            }
            submissions.append(existing)
        else:
            recorded = existing.get("consolidation_job_id")
            if (
                recorded is not None
                and consolidation_job_id is not None
                and recorded != consolidation_job_id
            ):
                raise StagedRunManifestError(
                    f"Retrieval job {retrieval_job_id} is already paired with "
                    f"consolidation job {recorded}."
                )
            if consolidation_job_id is not None:
                existing["consolidation_job_id"] = consolidation_job_id
                existing["dependency"] = f"afterok:{retrieval_job_id}"
            if submission_host is not None and existing.get("submission_host") is None:
                existing["submission_host"] = submission_host
        _atomic_write_json(path, payload)
    return str(path)

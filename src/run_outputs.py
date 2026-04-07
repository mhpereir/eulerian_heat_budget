from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
import json
import os
import re
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .specs import DomainRequest, DomainSpec, SurfaceBehaviour, DataSourceConfig


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_root: str
    plot_dir: str
    metadata_path: str


@dataclass(frozen=True)
class GitProvenance:
    branch: str
    commit: str
    dirty: bool


@dataclass(frozen=True)
class ProductionPaths:
    root_dir: str
    manifest_path: str
    annual_dir: str
    plot_root: str
    year: int | None
    output_path: str | None
    plot_dir: str | None


def _sanitize_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    if not cleaned:
        raise ValueError("Run id cannot be empty.")
    return cleaned


def resolve_run_id(
    *,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    pid: int | None = None,
) -> str:
    active_env = os.environ if env is None else env
    pbs_job_id = active_env.get("PBS_JOBID")
    if pbs_job_id:
        return _sanitize_run_id(pbs_job_id)

    timestamp = (datetime.now() if now is None else now).strftime("%Y%m%dT%H%M%S")
    process_id = os.getpid() if pid is None else pid
    return f"manual_{timestamp}_pid{process_id}"


def prepare_run_paths(
    base_plot_dir: str,
    *,
    outputs_subdir: str = "plots",
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    pid: int | None = None,
) -> RunPaths:
    run_id = resolve_run_id(env=env, now=now, pid=pid)
    run_root = Path(base_plot_dir) / run_id
    plot_dir = run_root / outputs_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        run_root=str(run_root),
        plot_dir=str(plot_dir),
        metadata_path=str(run_root / "run_info.json"),
    )


def prepare_production_paths(
    production_output_dir: str,
    *,
    year: int | None = None,
) -> ProductionPaths:
    root_dir = Path(production_output_dir)
    annual_dir = root_dir / "annual"
    plot_root = root_dir / "plots"

    annual_dir.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    output_path = annual_dir / f"heat_budget_{year}.nc" if year is not None else None
    plot_dir = plot_root / str(year) if year is not None else None

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    return ProductionPaths(
        root_dir=str(root_dir),
        manifest_path=str(root_dir / "production_run.json"),
        annual_dir=str(annual_dir),
        plot_root=str(plot_root),
        year=year,
        output_path=str(output_path) if output_path is not None else None,
        plot_dir=str(plot_dir) if plot_dir is not None else None,
    )


def write_run_info(
    paths: RunPaths,
    *,
    request: DomainRequest,
    source_spec: DataSourceConfig,
    domain_spec: DomainSpec,
    surface_behaviour: SurfaceBehaviour,
    git_provenance: GitProvenance,
    cli_args: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> str:
    active_env = os.environ if env is None else env
    timestamp = datetime.now() if now is None else now

    payload = {
        "run_id": paths.run_id,
        "pbs_job_id": active_env.get("PBS_JOBID"),
        "generated_at": timestamp.isoformat(),
        "run_root": paths.run_root,
        "plot_dir": paths.plot_dir,
        "request": request,
        "source_spec": source_spec,
        "domain_spec": domain_spec,
        "surface_behaviour": surface_behaviour,
        "git": git_provenance,
        "cli_args": dict(cli_args),
    }

    metadata_path = Path(paths.metadata_path)
    metadata_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return str(metadata_path)


def resolve_production_year(*, time_start: str | None, time_end: str | None) -> int:
    if time_start is None or time_end is None:
        raise ValueError("Production yearly runs require both time_start and time_end.")

    start = datetime.fromisoformat(time_start)
    end = datetime.fromisoformat(time_end)

    if start.year != end.year:
        raise ValueError("Production yearly runs require time_start and time_end to be in the same calendar year.")

    return start.year


def require_output_path(output_path: str | None, *, overwrite: bool) -> str:
    if output_path is None:
        raise ValueError("Output path is required.")

    output = Path(output_path)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        output.unlink()

    output.parent.mkdir(parents=True, exist_ok=True)
    return str(output)


def write_budget_result(ds_budget: Any, output_path: str | None, *, overwrite: bool) -> str:
    resolved_output_path = require_output_path(output_path, overwrite=overwrite)
    _drop_none_attrs(ds_budget).to_netcdf(resolved_output_path)
    return resolved_output_path


def write_production_manifest(
    paths: ProductionPaths,
    *,
    production_start_year: int,
    production_end_year: int,
    request: DomainRequest,
    source_spec: DataSourceConfig,
    surface_behaviour: SurfaceBehaviour,
    git_provenance: GitProvenance,
    cli_args: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> str:
    manifest_path = Path(paths.manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f"Production manifest already exists: {manifest_path}")

    active_env = os.environ if env is None else env
    timestamp = datetime.now() if now is None else now

    payload = {
        "generated_at": timestamp.isoformat(),
        "pbs_job_id": active_env.get("PBS_JOBID"),
        "production_start_year": production_start_year,
        "production_end_year": production_end_year,
        "root_dir": paths.root_dir,
        "annual_dir": paths.annual_dir,
        "plot_root": paths.plot_root,
        "manifest_path": paths.manifest_path,
        "request": request,
        "source_spec": source_spec,
        "surface_behaviour": surface_behaviour,
        "git": git_provenance,
        "cli_args": dict(cli_args),
    }

    manifest_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return str(manifest_path)


def require_production_manifest(paths: ProductionPaths) -> str:
    manifest_path = Path(paths.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Production manifest not found: {manifest_path}")
    return str(manifest_path)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)  #type: ignore
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _drop_none_attrs(ds: Any) -> Any:
    cleaned = ds.copy(deep=False)
    cleaned.attrs = {key: value for key, value in cleaned.attrs.items() if value is not None}

    for name in cleaned.variables:
        cleaned[name].attrs = {
            key: value
            for key, value in cleaned[name].attrs.items()
            if value is not None
        }

    return cleaned


def resolve_git_provenance(repo_dir: str | Path) -> GitProvenance:
    repo_root = Path(
        _run_git_command(["git", "rev-parse", "--show-toplevel"], cwd=repo_dir)
    )
    branch = _run_git_command(["git", "branch", "--show-current"], cwd=repo_root)
    if not branch:
        raise ValueError("Git provenance requires a named branch; detached HEAD is not supported.")

    commit = _run_git_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    dirty_paths = _run_git_command(
        ["git", "diff", "--name-only", "HEAD", "--", "src", "scripts", "schedulers"],
        cwd=repo_root,
    ).splitlines()
    dirty = any(_is_runtime_source_change(path) for path in dirty_paths)
    return GitProvenance(branch=branch, commit=commit, dirty=dirty)


def _run_git_command(command: list[str], *, cwd: str | Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise ValueError(stderr or f"Git command failed: {' '.join(command)}") from exc
    return completed.stdout.strip()


def _is_runtime_source_change(path: str) -> bool:
    candidate = Path(path)
    if any(part == "__pycache__" for part in candidate.parts):
        return False
    if candidate.suffix == ".pyc":
        return False
    return True

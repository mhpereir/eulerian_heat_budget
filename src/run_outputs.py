from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .specs import DomainRequest, DomainSpec, SurfaceBehaviour, DataSourceConfig


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_root: str
    plot_dir: str
    metadata_path: str


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


def write_run_info(
    paths: RunPaths,
    *,
    request: DomainRequest,
    source_spec: DataSourceConfig,
    domain_spec: DomainSpec,
    surface_behaviour: SurfaceBehaviour,
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
        "cli_args": dict(cli_args),
    }

    metadata_path = Path(paths.metadata_path)
    metadata_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return str(metadata_path)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)  #type: ignore
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

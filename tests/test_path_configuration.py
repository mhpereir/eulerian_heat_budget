from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_path_probe(
    *,
    environment_data_root: Path,
    environment_output_root: Path,
    cli_data_root: Path | None = None,
    cli_output_root: Path | None = None,
) -> dict[str, str]:
    argv = [
        "--data-source",
        "local_era5",
        "--region",
        "pnw_bartusek",
    ]
    if cli_data_root is not None:
        argv.extend(["--local-data-path", str(cli_data_root)])
    if cli_output_root is not None:
        argv.extend(["--output-root", str(cli_output_root)])

    probe = """
import json
import sys
from datetime import datetime

from scripts import run_budget
from src import cli, config, run_outputs

args = cli.parse_args(sys.argv[1:])
source = run_budget.build_data_source_from_cli(args)
output_root = args.output_root or config.DEFAULT_OUTPUT_ROOT
paths = run_outputs.prepare_run_paths(
    output_root,
    env={},
    now=datetime(2026, 7, 26, 12, 0, 0),
    pid=123,
)
print(json.dumps({
    "config_data_root": config.DEFAULT_LOCAL_PATH,
    "config_output_root": config.DEFAULT_OUTPUT_ROOT,
    "resolved_data_root": source.path_data,
    "resolved_output_root": output_root,
    "run_root": paths.run_root,
    "plot_dir": paths.plot_dir,
}))
"""
    env = os.environ.copy()
    env["EHB_DATA_ROOT"] = str(environment_data_root)
    env["EHB_OUTPUT_ROOT"] = str(environment_output_root)

    completed = subprocess.run(
        [sys.executable, "-c", probe, *argv],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_environment_roots_are_resolved_in_a_fresh_process(tmp_path):
    data_root = tmp_path / "environment-data"
    output_root = tmp_path / "environment-output"

    resolved = _run_path_probe(
        environment_data_root=data_root,
        environment_output_root=output_root,
    )

    assert resolved["config_data_root"] == str(data_root)
    assert resolved["config_output_root"] == str(output_root)
    assert resolved["resolved_data_root"] == str(data_root)
    assert resolved["resolved_output_root"] == str(output_root)
    assert Path(resolved["run_root"]).parent == output_root
    assert Path(resolved["plot_dir"]).is_dir()


def test_cli_roots_override_environment_roots_in_a_fresh_process(tmp_path):
    environment_data_root = tmp_path / "environment-data"
    environment_output_root = tmp_path / "environment-output"
    cli_data_root = tmp_path / "cli-data"
    cli_output_root = tmp_path / "cli-output"

    resolved = _run_path_probe(
        environment_data_root=environment_data_root,
        environment_output_root=environment_output_root,
        cli_data_root=cli_data_root,
        cli_output_root=cli_output_root,
    )

    assert resolved["config_data_root"] == str(environment_data_root)
    assert resolved["config_output_root"] == str(environment_output_root)
    assert resolved["resolved_data_root"] == str(cli_data_root)
    assert resolved["resolved_output_root"] == str(cli_output_root)
    assert Path(resolved["run_root"]).parent == cli_output_root
    assert Path(resolved["plot_dir"]).is_dir()


def test_empty_environment_roots_use_safe_defaults():
    probe = """
import json

from src import config

print(json.dumps({
    "data_root": config.DEFAULT_LOCAL_PATH,
    "output_root": config.DEFAULT_OUTPUT_ROOT,
}))
"""
    env = os.environ.copy()
    env["EHB_DATA_ROOT"] = ""
    env["EHB_OUTPUT_ROOT"] = ""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = json.loads(completed.stdout)

    assert resolved["data_root"] is None
    assert resolved["output_root"] == str(PROJECT_ROOT / "results")

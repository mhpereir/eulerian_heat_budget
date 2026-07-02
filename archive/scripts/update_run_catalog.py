#!/usr/bin/env python3
"""Build a sortable CSV catalog of single-run EHB experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "results" / "plots"
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"
DEFAULT_OUTPUT = REPO_ROOT / "archive" / "run_catalog.csv"

OPERATIONAL_COLUMNS = ("pbs_id", "run_info_file", "log_file", "exit_status")
FILE_NOT_FOUND = "file not found"
EXIT_STATUS_NOT_FOUND = "exit status not found"

RUN_DIR_PATTERN = re.compile(r"^(?P<run_id>\d+(?:\.venus)?)$")
SINGLE_LOG_PATTERN = re.compile(r"^(?P<run_id>\d+(?:\.venus)?)_EHB_single\.log$")
EXIT_STATUS_PATTERN = re.compile(r"^\s*Exit status:\s*(\d+)\s*$", re.MULTILINE)


class CatalogError(RuntimeError):
    """Raised when catalog inputs cannot be safely converted to CSV."""


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise CatalogError(f"{label} directory not found: {path}")


def discover_run_ids(plots_dir: Path, logs_dir: Path) -> list[str]:
    """Return sorted scheduler run IDs present in supported single-run sources."""
    _require_directory(plots_dir, "Plots")
    _require_directory(logs_dir, "Logs")

    run_ids = {
        match.group("run_id")
        for path in plots_dir.iterdir()
        if path.is_dir() and (match := RUN_DIR_PATTERN.fullmatch(path.name))
    }
    run_ids.update(
        match.group("run_id")
        for path in logs_dir.iterdir()
        if path.is_file() and (match := SINGLE_LOG_PATTERN.fullmatch(path.name))
    )
    return sorted(run_ids, key=_run_id_sort_key)


def _run_id_sort_key(run_id: str) -> tuple[int, str]:
    job_number = run_id.partition(".")[0]
    return (int(job_number), run_id)


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, bool):
        return json.dumps(value)
    return value


def flatten_mapping(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
    flattened: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten nested dictionaries into dotted CSV column names."""
    result = {} if flattened is None else flattened
    for key, nested_value in value.items():
        column = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(nested_value, Mapping) and nested_value:
            flatten_mapping(nested_value, prefix=column, flattened=result)
            continue

        if column in result:
            raise CatalogError(f"Duplicate flattened metadata field: {column}")
        result[column] = _csv_value(nested_value)
    return result


def _load_run_info(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Invalid run metadata file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CatalogError(f"Invalid run metadata file {path}: expected a JSON object")
    return flatten_mapping(payload)


def _read_exit_status(path: Path) -> str:
    try:
        contents = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CatalogError(f"Could not read log file {path}: {exc}") from exc

    matches = EXIT_STATUS_PATTERN.findall(contents)
    return matches[-1] if matches else EXIT_STATUS_NOT_FOUND


def build_catalog(
    plots_dir: Path,
    logs_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build the complete header and rows without writing an output file."""
    rows: list[dict[str, Any]] = []
    metadata_columns: set[str] = set()

    for run_id in discover_run_ids(plots_dir, logs_dir):
        run_info_path = plots_dir / run_id / "run_info.json"
        log_path = logs_dir / f"{run_id}_EHB_single.log"

        row: dict[str, Any] = {"pbs_id": run_id}

        if run_info_path.is_file():
            row["run_info_file"] = _display_path(run_info_path, repo_root)
            metadata = _load_run_info(run_info_path)
            collisions = set(OPERATIONAL_COLUMNS).intersection(metadata)
            if collisions:
                fields = ", ".join(sorted(collisions))
                raise CatalogError(
                    f"Run metadata file {run_info_path} conflicts with operational columns: {fields}"
                )
            row.update(metadata)
            metadata_columns.update(metadata)
        else:
            row["run_info_file"] = FILE_NOT_FOUND

        if log_path.is_file():
            row["log_file"] = _display_path(log_path, repo_root)
            row["exit_status"] = _read_exit_status(log_path)
        else:
            row["log_file"] = FILE_NOT_FOUND
            row["exit_status"] = FILE_NOT_FOUND

        rows.append(row)

    fieldnames = [*OPERATIONAL_COLUMNS, *sorted(metadata_columns)]
    return fieldnames, rows


def _write_catalog_atomically(
    output: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            temporary.flush()
            os.fsync(temporary.fileno())

        os.replace(temporary_path, output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def update_catalog(
    plots_dir: Path,
    logs_dir: Path,
    output: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[int, int]:
    """Rebuild the current catalog and return its row and column counts."""
    fieldnames, rows = build_catalog(plots_dir, logs_dir, repo_root=repo_root)
    _write_catalog_atomically(output, fieldnames, rows)
    return len(rows), len(fieldnames)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the CSV catalog of single-run EHB experiments."
    )
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        row_count, column_count = update_catalog(
            args.plots_dir,
            args.logs_dir,
            args.output,
        )
    except (CatalogError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {row_count} runs and {column_count} columns to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

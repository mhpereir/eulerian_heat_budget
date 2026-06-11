import csv
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "archive" / "scripts" / "update_run_catalog.py"
SPEC = importlib.util.spec_from_file_location("update_run_catalog", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
update_run_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update_run_catalog)


def _make_input_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    plots_dir = tmp_path / "results" / "plots"
    logs_dir = tmp_path / "logs"
    output = tmp_path / "archive" / "run_catalog.csv"
    plots_dir.mkdir(parents=True)
    logs_dir.mkdir()
    return plots_dir, logs_dir, output


def _write_run_info(plots_dir: Path, run_id: str, payload: dict) -> None:
    run_dir = plots_dir / run_id
    run_dir.mkdir(exist_ok=True)
    (run_dir / "run_info.json").write_text(json.dumps(payload), encoding="utf-8")


def _read_catalog(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as catalog_file:
        reader = csv.DictReader(catalog_file)
        return list(reader.fieldnames or []), list(reader)


def test_catalog_uses_supported_source_union_and_missing_markers(tmp_path):
    plots_dir, logs_dir, output = _make_input_dirs(tmp_path)
    _write_run_info(plots_dir, "100.venus", {"run_id": "100.venus"})
    (plots_dir / "200.venus").mkdir()
    (plots_dir / "check_pbl").mkdir()
    (plots_dir / "nested" / "400.venus").mkdir(parents=True)

    (logs_dir / "100.venus_EHB_single.log").write_text(
        "Exit status: 0\nmore output\nExit status: 7\n",
        encoding="utf-8",
    )
    (logs_dir / "300.venus_EHB_single.log").write_text(
        "still running\n",
        encoding="utf-8",
    )
    (logs_dir / "400.venus.OU").write_text("Exit status: 0\n", encoding="utf-8")
    (logs_dir / "500.venus_EHB_prod.log").write_text("Exit status: 0\n", encoding="utf-8")

    update_run_catalog.update_catalog(
        plots_dir,
        logs_dir,
        output,
        repo_root=tmp_path,
    )

    _, rows = _read_catalog(output)
    rows_by_id = {row["pbs_id"]: row for row in rows}

    assert list(rows_by_id) == ["100.venus", "200.venus", "300.venus"]
    assert rows_by_id["100.venus"]["run_info_file"] == "results/plots/100.venus/run_info.json"
    assert rows_by_id["100.venus"]["log_file"] == "logs/100.venus_EHB_single.log"
    assert rows_by_id["100.venus"]["exit_status"] == "7"
    assert rows_by_id["200.venus"]["run_info_file"] == "file not found"
    assert rows_by_id["200.venus"]["log_file"] == "file not found"
    assert rows_by_id["200.venus"]["exit_status"] == "file not found"
    assert rows_by_id["300.venus"]["run_info_file"] == "file not found"
    assert rows_by_id["300.venus"]["exit_status"] == "exit status not found"


def test_catalog_flattens_evolving_metadata_schema(tmp_path):
    plots_dir, logs_dir, output = _make_input_dirs(tmp_path)
    _write_run_info(
        plots_dir,
        "100.venus",
        {
            "run_id": "100.venus",
            "request": {"bbox": [40, 60, -130, -110], "old_field": None},
            "git": {"dirty": True},
        },
    )
    _write_run_info(
        plots_dir,
        "200.venus",
        {
            "run_id": "200.venus",
            "request": {"bbox": [30, 50, -125, -105], "new_field": "new"},
        },
    )

    update_run_catalog.update_catalog(
        plots_dir,
        logs_dir,
        output,
        repo_root=tmp_path,
    )

    fieldnames, rows = _read_catalog(output)
    rows_by_id = {row["pbs_id"]: row for row in rows}

    assert fieldnames[:4] == ["pbs_id", "run_info_file", "log_file", "exit_status"]
    assert fieldnames[4:] == sorted(fieldnames[4:])
    assert rows_by_id["100.venus"]["request.bbox"] == "[40,60,-130,-110]"
    assert rows_by_id["100.venus"]["request.old_field"] == ""
    assert rows_by_id["100.venus"]["request.new_field"] == ""
    assert rows_by_id["100.venus"]["git.dirty"] == "true"
    assert rows_by_id["200.venus"]["request.old_field"] == ""
    assert rows_by_id["200.venus"]["request.new_field"] == "new"


def test_rerun_refreshes_rows_removes_vanished_ids_and_avoids_duplicates(tmp_path):
    plots_dir, logs_dir, output = _make_input_dirs(tmp_path)
    _write_run_info(plots_dir, "100.venus", {"run_id": "100.venus", "version": 1})
    log_path = logs_dir / "100.venus_EHB_single.log"
    log_path.write_text("still running\n", encoding="utf-8")

    update_run_catalog.update_catalog(plots_dir, logs_dir, output, repo_root=tmp_path)

    _write_run_info(
        plots_dir,
        "100.venus",
        {"run_id": "100.venus", "version": 2, "new_field": "added"},
    )
    log_path.write_text("Exit status: 0\n", encoding="utf-8")
    update_run_catalog.update_catalog(plots_dir, logs_dir, output, repo_root=tmp_path)

    fieldnames, rows = _read_catalog(output)
    assert len(rows) == 1
    assert rows[0]["version"] == "2"
    assert rows[0]["new_field"] == "added"
    assert rows[0]["exit_status"] == "0"
    assert "new_field" in fieldnames

    (plots_dir / "100.venus" / "run_info.json").unlink()
    (plots_dir / "100.venus").rmdir()
    log_path.unlink()
    _write_run_info(plots_dir, "200.venus", {"run_id": "200.venus"})
    update_run_catalog.update_catalog(plots_dir, logs_dir, output, repo_root=tmp_path)

    _, rows = _read_catalog(output)
    assert [row["pbs_id"] for row in rows] == ["200.venus"]


def test_invalid_run_info_does_not_replace_existing_catalog(tmp_path):
    plots_dir, logs_dir, output = _make_input_dirs(tmp_path)
    run_dir = plots_dir / "100.venus"
    run_dir.mkdir()
    (run_dir / "run_info.json").write_text("{not valid json", encoding="utf-8")
    output.parent.mkdir()
    output.write_text("existing catalog\n", encoding="utf-8")

    with pytest.raises(update_run_catalog.CatalogError, match="Invalid run metadata"):
        update_run_catalog.update_catalog(
            plots_dir,
            logs_dir,
            output,
            repo_root=tmp_path,
        )

    assert output.read_text(encoding="utf-8") == "existing catalog\n"


def test_missing_input_root_does_not_replace_existing_catalog(tmp_path):
    plots_dir, logs_dir, output = _make_input_dirs(tmp_path)
    output.parent.mkdir()
    output.write_text("existing catalog\n", encoding="utf-8")
    logs_dir.rmdir()

    with pytest.raises(update_run_catalog.CatalogError, match="Logs directory not found"):
        update_run_catalog.update_catalog(
            plots_dir,
            logs_dir,
            output,
            repo_root=tmp_path,
        )

    assert output.read_text(encoding="utf-8") == "existing catalog\n"

import json
from pathlib import Path

import pytest

from src_arco.campaign import Campaign, initialize_campaign
from src_arco.run_manifest import (
    StagedRunManifestError,
    build_staged_run_manifest,
    prepare_staged_run_manifest,
    record_staged_submission,
)


def _campaign() -> Campaign:
    return Campaign.from_mapping(
        {
            "schema_version": 1,
            "campaign_id": "alaska-test",
            "years": {"start": 1940, "end": 1941},
            "season": {
                "start_month_day": "05-01",
                "end_month_day": "10-31",
            },
            "domain": {
                "region": "alaska",
                "margin_n": 1,
                "zg_top_pa": 70000,
                "zg_bottom": "surface_pressure",
                "allow_bottom_overflow": True,
            },
            "staging": {
                "time_chunk": "month",
                "attempt_timeout_seconds": 20800,
                "include_benchmark_variables": False,
            },
        }
    )


def _payload(tmp_path: Path, *, commit: str = "a" * 40, now: str = "2026-07-27T22:00:00+00:00"):
    cache_root = tmp_path / "staged-zarr" / "alaska" / "alaska-test"
    return build_staged_run_manifest(
        campaign=_campaign(),
        run_id="alaska-test",
        git_branch="production_development_staged",
        git_commit=commit,
        git_upstream="origin/production_development_staged",
        git_origin_url="git@example.invalid:eulerian_heat_budget.git",
        project_root="/home/user/eulerian-heat-budget/production/repo-abcdef0",
        settings_file="/home/user/eulerian-heat-budget/production/repo-abcdef0/settings.sh",
        staged_cache_root=cache_root,
        log_dir=tmp_path / "logs" / "alaska-test",
        mamba_environment="dev_env",
        source_arco_path="gs://example/arco.zarr",
        first_task_index=0,
        last_task_index=1,
        max_parallel=2,
        retrieval_select="1:ncpus=8:mem=8gb",
        retrieval_walltime="48:00:00",
        consolidation_select="1:ncpus=1:mem=4gb",
        consolidation_walltime="12:00:00",
        now=now,
    )


def _manifest_path(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "staged-zarr"
        / "alaska"
        / "alaska-test"
        / "production_run.json"
    )


def test_prepare_staged_run_manifest_records_complete_provenance(tmp_path):
    manifest_path = _manifest_path(tmp_path)

    prepare_staged_run_manifest(manifest_path, _payload(tmp_path))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["run_type"] == "staged_arco_retrieval"
    assert payload["campaign"]["configuration"]["years"] == {
        "start": 1940,
        "end": 1941,
    }
    assert payload["campaign"]["sha256"] == _campaign().sha256()
    assert payload["git"] == {
        "branch": "production_development_staged",
        "commit": "a" * 40,
        "dirty": False,
        "origin_url": "git@example.invalid:eulerian_heat_budget.git",
        "upstream": "origin/production_development_staged",
    }
    assert payload["runtime"]["mamba_environment"] == "dev_env"
    assert payload["scheduler"]["array"] == {
        "first_task_index": 0,
        "last_task_index": 1,
        "max_parallel": 2,
        "task_count": 2,
    }
    assert payload["scheduler"]["retrieval_resources"]["walltime"] == "48:00:00"
    assert payload["submissions"] == []

    campaign_path = initialize_campaign(manifest_path.parent, _campaign())
    assert campaign_path == manifest_path.parent / "campaign.json"


def test_prepare_staged_run_manifest_is_idempotent_but_rejects_new_commit(tmp_path):
    manifest_path = _manifest_path(tmp_path)
    prepare_staged_run_manifest(manifest_path, _payload(tmp_path))
    record_staged_submission(
        manifest_path,
        retrieval_job_id="100[].venus",
        submitted_at="2026-07-27T22:05:00+00:00",
    )

    prepare_staged_run_manifest(
        manifest_path,
        _payload(tmp_path, now="2026-07-28T00:00:00+00:00"),
    )
    preserved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(preserved["submissions"]) == 1

    with pytest.raises(StagedRunManifestError, match="different production settings"):
        prepare_staged_run_manifest(
            manifest_path,
            _payload(tmp_path, commit="b" * 40),
        )


def test_record_staged_submission_appends_and_completes_idempotently(tmp_path):
    manifest_path = _manifest_path(tmp_path)
    prepare_staged_run_manifest(manifest_path, _payload(tmp_path))

    record_staged_submission(
        manifest_path,
        retrieval_job_id="100[].venus",
        submission_host="venus",
        submitted_at="2026-07-27T22:05:00+00:00",
    )
    record_staged_submission(
        manifest_path,
        retrieval_job_id="100[].venus",
        consolidation_job_id="101.venus",
        submission_host="venus",
    )
    record_staged_submission(
        manifest_path,
        retrieval_job_id="100[].venus",
        consolidation_job_id="101.venus",
        submission_host="venus",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["submissions"] == [
        {
            "submitted_at": "2026-07-27T22:05:00+00:00",
            "submission_host": "venus",
            "retrieval_job_id": "100[].venus",
            "consolidation_job_id": "101.venus",
            "dependency": "afterok:100[].venus",
        }
    ]

    with pytest.raises(StagedRunManifestError, match="already paired"):
        record_staged_submission(
            manifest_path,
            retrieval_job_id="100[].venus",
            consolidation_job_id="102.venus",
        )

"""Migrate one legacy staged ARCO cache year into a campaign shard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src_arco.campaign import CampaignConfigError  # noqa: E402
from src_arco.legacy_migration import migrate_legacy_year  # noqa: E402
from src_arco.shard_artifacts import ShardValidationError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-cache-root", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--pbs-job-id")
    parser.add_argument("--pbs-array-index")
    args = parser.parse_args()
    try:
        result = migrate_legacy_year(
            args.legacy_cache_root,
            args.campaign_root,
            args.year,
            git_commit=args.git_commit,
            pbs_job_id=args.pbs_job_id,
            pbs_array_index=args.pbs_array_index,
        )
    except (CampaignConfigError, ShardValidationError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"[legacy-migration] finalized year={result['year']} "
        f"manifest_sha256={result['manifest_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

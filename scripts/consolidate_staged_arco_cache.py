"""Validate yearly staged ARCO shards and build one combined cache catalog."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src_arco.campaign import CampaignConfigError  # noqa: E402
from src_arco.consolidation import consolidate  # noqa: E402
from src_arco.shard_artifacts import ShardValidationError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        consolidate(args.cache_root)
    except (CampaignConfigError, ShardValidationError, ValueError) as exc:
        print(f"[consolidate:error] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

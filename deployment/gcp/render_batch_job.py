"""Render a Google Cloud Batch job from a validated campaign document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from deployment.gcp.campaign import Campaign


IMAGE_DIGEST_PATTERN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
JOB_ID_PATTERN = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def render_job(
    campaign: Campaign,
    *,
    project: str,
    region: str,
    bucket: str,
    image_uri: str,
    service_account_email: str,
    job_id: str,
    parallelism: int = 5,
    boot_disk_gb: int = 100,
) -> dict[str, Any]:
    if not IMAGE_DIGEST_PATTERN.fullmatch(image_uri):
        raise ValueError("image_uri must be pinned with an @sha256:<64 hex> digest.")
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError(
            "job_id must start with a lowercase letter, contain lowercase letters, "
            "digits or hyphens, and have at most 63 characters."
        )
    if not project or not region or not bucket or not service_account_email:
        raise ValueError("project, region, bucket, and service_account_email are required.")
    if parallelism < 1 or parallelism > campaign.task_count:
        raise ValueError("parallelism must be between 1 and the campaign task count.")
    if boot_disk_gb < 30:
        raise ValueError("boot_disk_gb must be at least 30 GB.")

    mount_path = "/mnt/disks/arco-output"
    return {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [
                        {
                            "container": {
                                "imageUri": image_uri,
                                "entrypoint": "/usr/bin/tini",
                                "commands": [
                                    "-g",
                                    "--",
                                    "python",
                                    "-m",
                                    "deployment.gcp.run_yearly_retrieval",
                                ],
                            }
                        }
                    ],
                    "computeResource": {"cpuMilli": 2000, "memoryMib": 7000},
                    "maxRetryCount": 2,
                    "maxRunDuration": "172800s",
                    "volumes": [
                        {
                            "gcs": {
                                "remotePath": (
                                    f"{bucket}/campaigns/{campaign.campaign_id}"
                                )
                            },
                            "mountPath": mount_path,
                            "mountOptions": [
                                "--implicit-dirs",
                                "--uid=10001",
                                "--gid=10001",
                                "--file-mode=0660",
                                "--dir-mode=0770",
                            ],
                        }
                    ],
                    "environment": {
                        "variables": {
                            "EHB_CAMPAIGN_JSON": campaign.canonical_json(),
                            "EHB_CAMPAIGN_SHA256": campaign.sha256(),
                            "EHB_LOCAL_SCRATCH": "/scratch",
                            "EHB_OUTPUT_MOUNT": mount_path,
                            "PYTHONUNBUFFERED": "1",
                            "DASK_SCHEDULER": "threads",
                            "DASK_NUM_WORKERS": "2",
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "NUMEXPR_NUM_THREADS": "1",
                        }
                    },
                },
                "taskCount": campaign.task_count,
                "parallelism": parallelism,
                "taskCountPerNode": 1,
            }
        ],
        "allocationPolicy": {
            "instances": [
                {
                    "policy": {
                        "machineType": "e2-standard-2",
                        "provisioningModel": "STANDARD",
                        "bootDisk": {
                            "type": "pd-balanced",
                            "sizeGb": boot_disk_gb,
                        },
                    }
                }
            ],
            "serviceAccount": {
                "email": service_account_email,
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            },
        },
        "labels": {
            "application": "eulerian-heat-budget",
            "campaign": campaign.campaign_id,
            "workload": "arco-retrieval",
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--service-account-email", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--parallelism", type=int, default=5)
    parser.add_argument("--boot-disk-gb", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--normalized-campaign-output", type=Path)
    args = parser.parse_args()

    campaign = Campaign.from_file(args.campaign)
    job = render_job(
        campaign,
        project=args.project,
        region=args.region,
        bucket=args.bucket,
        image_uri=args.image_uri,
        service_account_email=args.service_account_email,
        job_id=args.job_id,
        parallelism=args.parallelism,
        boot_disk_gb=args.boot_disk_gb,
    )
    args.output.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.normalized_campaign_output is not None:
        campaign.write_normalized(args.normalized_campaign_output)


if __name__ == "__main__":
    main()

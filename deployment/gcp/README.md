# Google Cloud Batch deployment

The canonical deployment documentation now lives under `docs/`:

- [Production command runbook](../../docs/README.md)
- [Detailed deployment design and operations reference](../../docs/google-cloud-batch-deployment.md)

The deployment in this directory runs one ARCO ERA5 staging task per campaign
year. It publishes verified yearly cache shards to Cloud Storage. It does not
run the heat-budget calculation itself; the downloaded campaign must be
consolidated before it is passed to the offline production calculation. Run
local consolidation directly with the existing compatible Python environment;
Docker is not required for that step.

Start with the production runbook for the complete resource setup, campaign,
build, canary, submission, monitoring, resume, download, and consolidation
command sequence.

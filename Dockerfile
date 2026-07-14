FROM python:3.12.11-slim-bookworm AS base

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv

WORKDIR /app
COPY requirements-arco.txt ./requirements-arco.txt
RUN python -m pip install --no-cache-dir --require-hashes -r requirements-arco.txt

FROM base AS test
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN python -c "import tempfile; import numpy as np; import xarray as xr; path = tempfile.mkdtemp(); xr.Dataset({'value': ('time', np.arange(4.0))}, coords={'time': np.arange(4)}).to_zarr(path, mode='w'); xr.open_zarr(path, chunks=None).close()" \
    && python -m pytest -q \
    && touch /tmp/ehb-tests-passed

FROM base AS runtime
RUN groupadd --gid 10001 ehb \
    && useradd --uid 10001 --gid 10001 --home-dir /app --no-create-home ehb \
    && mkdir -p /scratch \
    && chown ehb:ehb /scratch
COPY --from=test /tmp/ehb-tests-passed /opt/ehb-tests-passed
COPY --chown=ehb:ehb . /app
USER ehb
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "python", "-m", "deployment.gcp.run_yearly_retrieval"]

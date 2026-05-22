# Multi-stage build for the rebooking serving image.
#
# Stage 1 (builder): installs deps + the package into a venv at /opt/venv.
# Stage 2 (runtime): copies just /opt/venv plus the runtime payload
# (src, configs, artifacts). Runtime image has no build tooling, no
# pip cache, no scikit-learn build deps -- only what serving needs.
#
# Assumes `python -m rebooking.models.train` has produced
# artifacts/model.joblib before `docker build` runs. Training is a
# build-time concern; the image bakes a specific model snapshot.

# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the package and install. PEP 621's `dependencies` list lives
# inside pyproject.toml, which `pip install .` reads alongside the
# source -- so a single layer covers deps + package. If dependency
# install times become painful, the next refactor is to extract deps
# to a separate requirements.txt + install that before copying src
# so source-only changes don't bust the deps layer.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install .

# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# curl is needed for HEALTHCHECK. Keep apt layer minimal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. Container processes must never run as root in prod.
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
COPY --chown=app:app artifacts ./artifacts

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "rebooking.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

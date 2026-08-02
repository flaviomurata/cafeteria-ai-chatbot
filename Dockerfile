FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HOME=/home/appuser \
    INDEX_VOLUME_READ_ONLY=true

# uv from its official image, pinned by digest for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv@sha256:606e70c71c852d03f611b1e56a195d08648507018a7057fab82c4974c4eae105 /uv /bin/uv

WORKDIR /src

# Prepare the runtime user and the persistent mount targets. Fresh named volumes
# inherit the mount target's ownership; existing volumes are repaired by the entrypoint.
RUN useradd --create-home appuser \
    && mkdir -p /src/data/partner-knowledge-index /src/data/partner-knowledge-runtime \
    && chown -R appuser:appuser /src

COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Dependency manifests first, for Docker layer caching
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

USER appuser

# Production dependencies only
RUN uv sync --frozen --no-dev

# Application code
COPY --chown=appuser:appuser src/ src/

EXPOSE 8000

# Uses the stdlib rather than curl, which is not present in python:3.13-slim
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"]

# Run uvicorn straight from the venv, avoiding a uv re-sync at runtime
USER root
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/src/.venv/bin/uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

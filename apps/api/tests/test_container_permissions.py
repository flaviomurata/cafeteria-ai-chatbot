from pathlib import Path


def test_runtime_bootstraps_the_persistent_index_before_dropping_privileges() -> None:
    api_root = Path(__file__).parents[1]
    dockerfile = (api_root / "Dockerfile").read_text()
    entrypoint = (api_root / "docker-entrypoint.sh").read_text()

    assert (
        "COPY --chmod=755 apps/api/docker-entrypoint.sh "
        "/usr/local/bin/docker-entrypoint.sh" in dockerfile
    )
    assert 'ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]' in dockerfile
    assert "USER root" in dockerfile
    assert "HOME=/home/appuser" in dockerfile
    assert "RUNTIME_DATA_PATH=/app/data/partner-knowledge-runtime" in entrypoint
    assert 'mkdir -p "$RUNTIME_DATA_PATH"' in entrypoint
    assert 'chown -R "$APP_USER:$APP_USER" "$RUNTIME_DATA_PATH"' in entrypoint
    assert 'chown -R "$APP_USER:$APP_USER" "$INDEX_VOLUME_PATH"' in entrypoint
    assert 'runuser --preserve-environment --user "$APP_USER" -- "$@"' in entrypoint


def test_serving_runtime_mounts_the_index_read_only() -> None:
    api_root = Path(__file__).parents[1]
    repository_root = Path(__file__).parents[3]
    compose = (repository_root / "compose.yaml").read_text()
    dockerfile = (api_root / "Dockerfile").read_text()
    entrypoint = (api_root / "docker-entrypoint.sh").read_text()

    assert "partner-knowledge-index:/app/data/partner-knowledge-index:ro" in compose
    assert (
        compose.count("partner-knowledge-runtime:/app/data/partner-knowledge-runtime")
        == 2
    )
    assert "INDEX_VOLUME_READ_ONLY=true" in dockerfile
    assert 'INDEX_VOLUME_READ_ONLY="${INDEX_VOLUME_READ_ONLY:-true}"' in entrypoint

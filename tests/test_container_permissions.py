from pathlib import Path


def test_runtime_bootstraps_the_persistent_index_before_dropping_privileges() -> None:
    repository_root = Path(__file__).parents[1]
    dockerfile = (repository_root / "Dockerfile").read_text()
    entrypoint = (repository_root / "docker-entrypoint.sh").read_text()

    assert (
        "COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh"
        in dockerfile
    )
    assert 'ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]' in dockerfile
    assert "USER root" in dockerfile
    assert 'chown -R "$APP_USER:$APP_USER" "$INDEX_VOLUME_PATH"' in entrypoint
    assert 'runuser --preserve-environment --user "$APP_USER" -- "$@"' in entrypoint

#!/bin/sh
set -eu

INDEX_VOLUME_PATH=/src/data/partner-knowledge-index
RUNTIME_DATA_PATH=/src/data/partner-knowledge-runtime
APP_USER=appuser
INDEX_VOLUME_READ_ONLY="${INDEX_VOLUME_READ_ONLY:-true}"

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$RUNTIME_DATA_PATH"
    chown -R "$APP_USER:$APP_USER" "$RUNTIME_DATA_PATH"
    if [ "$INDEX_VOLUME_READ_ONLY" != "true" ]; then
        mkdir -p "$INDEX_VOLUME_PATH"
        chown -R "$APP_USER:$APP_USER" "$INDEX_VOLUME_PATH"
    fi
    exec runuser --preserve-environment --user "$APP_USER" -- "$@"
fi

exec "$@"

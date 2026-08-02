#!/bin/sh
set -eu

INDEX_VOLUME_PATH=/src/data/partner-knowledge-index
APP_USER=appuser

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$INDEX_VOLUME_PATH"
    chown -R "$APP_USER:$APP_USER" "$INDEX_VOLUME_PATH"
    exec runuser --preserve-environment --user "$APP_USER" -- "$@"
fi

exec "$@"

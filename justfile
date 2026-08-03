default:
  just --list

api *args:
  cd apps/api && uv run uvicorn src.main:app --reload {{args}}

test *args:
  cd apps/api && uv run pytest {{args}}

lint:
  cd apps/api && uv run ruff format src tests
  cd apps/api && uv run ruff check --fix src tests

up:
  docker compose up -d --build --remove-orphans web

kill *args:
  docker compose kill {{args}}

build:
  docker compose build

ps:
  docker compose ps

ingest:
  docker compose run --build --rm ingest-partner-knowledge

e2e-local: e2e-local-up
  cd apps/api && RUN_E2E=1 E2E_EXPECTED_ENVIRONMENT=e2e uv run pytest tests/e2e -q

e2e-local-up:
  docker compose -f compose.yaml -f compose.e2e.yaml up --build --detach --remove-orphans agent-api

e2e-local-down:
  docker compose -f compose.yaml -f compose.e2e.yaml down --remove-orphans

e2e-live:
  cd apps/api && RUN_LIVE_EVAL=1 E2E_EXPECTED_ENVIRONMENT=staging uv run pytest tests/e2e/test_live_grounding.py -q

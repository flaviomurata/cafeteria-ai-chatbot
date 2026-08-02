default:
  just --list

run *args:
  uv run uvicorn src.main:app --reload {{args}}

ruff *args:
  uv run ruff check {{args}} src

lint:
  uv run ruff format src
  just ruff --fix

# docker
up:
  docker compose up -d --remove-orphans agent-api

kill *args:
  docker compose kill {{args}}

build:
  docker compose build

ps:
  docker compose ps

ingest:
  docker compose run --build --rm ingest-partner-knowledge

e2e-local: e2e-local-up
  RUN_E2E=1 E2E_EXPECTED_ENVIRONMENT=e2e uv run pytest tests/e2e -q

e2e-local-up:
  docker compose -f docker-compose.yml -f docker-compose.e2e.yml up --build --detach --remove-orphans agent-api

e2e-local-down:
  docker compose -f docker-compose.yml -f docker-compose.e2e.yml down --remove-orphans

e2e-live:
  RUN_LIVE_EVAL=1 E2E_EXPECTED_ENVIRONMENT=staging uv run pytest tests/e2e/test_live_grounding.py -q

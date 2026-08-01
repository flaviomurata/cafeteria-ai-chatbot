default:
  just --list

# Build the API image only; indexing is an explicit operation below.
build:
  docker compose build

# One-time/bootstrap index construction. Automatic source-update detection is out of scope.
ingest:
  docker compose run --rm ingest-partner-knowledge

start:
  docker compose up --detach --remove-orphans agent-api

e2e-local: e2e-local-up
  RUN_E2E=1 E2E_EXPECTED_ENVIRONMENT=e2e uv run pytest tests/e2e -q

e2e-local-up:
  docker compose -f docker-compose.yml -f docker-compose.e2e.yml up --build --detach --remove-orphans agent-api

e2e-local-down:
  docker compose -f docker-compose.yml -f docker-compose.e2e.yml down --remove-orphans

run *args:
  uv run uvicorn src.main:app --reload {{args}}

ruff *args:
  uv run ruff check {{args}} src

lint:
  uv run ruff format src
  just ruff --fix

default:
  just --list

run *args:
  uv run uvicorn src.main:app --reload {{args}}

ruff *args:
  uv run ruff check {{args}} src

lint:
  uv run ruff format src
  just ruff --fix

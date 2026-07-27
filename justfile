default:
  just --list

run *args:
  uv run fastapi dev src/main.py

ruff *args:
  uv run ruff check {{args}} src

lint:
  uv run ruff format src
  just ruff --fix

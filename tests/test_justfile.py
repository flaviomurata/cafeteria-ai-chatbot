from pathlib import Path


def test_ingest_recipe_rebuilds_the_container_before_running() -> None:
    justfile = Path(__file__).parents[1] / "justfile"
    recipe = justfile.read_text()

    assert "  docker compose run --build --rm ingest-partner-knowledge" in recipe

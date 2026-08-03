import pytest
from pydantic import ValidationError

from src.config import Settings


@pytest.mark.parametrize("app_env", ["development", "staging", "production"])
def test_local_e2e_mode_is_rejected_outside_test_environments(app_env: str):
    with pytest.raises(ValidationError, match="E2E_MODE=local"):
        Settings(app_env=app_env, e2e_mode="local")


@pytest.mark.parametrize("app_env", ["test", "e2e"])
def test_local_e2e_mode_is_allowed_in_test_environments(app_env: str):
    settings = Settings(app_env=app_env, e2e_mode="local")

    assert settings.e2e_mode == "local"

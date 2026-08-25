from app.config import RoleConfig, Settings
from app.models import get_model


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_model_verifier_config():
    s = _settings()
    m = get_model("verifier", s)
    assert m.model_name == s.verifier.model
    assert m.temperature == s.verifier.temperature
    assert m.openai_api_base == "https://openrouter.ai/api/v1"


def test_get_model_custom_role_values():
    s = _settings()
    s.researcher = RoleConfig(model="some/model", temperature=0.55)
    m = get_model("researcher", s)
    assert m.model_name == "some/model"
    assert abs(m.temperature - 0.55) < 1e-9


def test_invalid_role_raises():
    import pytest

    with pytest.raises(ValueError):
        get_model("nonexistent", _settings())

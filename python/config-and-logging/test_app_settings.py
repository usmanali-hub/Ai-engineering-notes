import pytest
from app_settings import Settings, setup_logging


def test_defaults():
    s = Settings.from_env(environ={})
    assert (s.model, s.timeout, s.log_level, s.api_key) == ("demo-model", 30.0, "INFO", None)


def test_reads_and_converts_values():
    env = {"APP_MODEL": "m1", "APP_TIMEOUT": "12.5", "APP_LOG_LEVEL": "debug", "APP_API_KEY": "k"}
    s = Settings.from_env(environ=env)
    assert s.model == "m1" and s.timeout == 12.5 and s.log_level == "DEBUG" and s.api_key == "k"


def test_secret_not_in_repr():
    assert "super-secret" not in repr(Settings(api_key="super-secret"))


def test_custom_prefix():
    assert Settings.from_env(prefix="X_", environ={"X_MODEL": "z"}).model == "z"


@pytest.mark.parametrize("env", [{"APP_TIMEOUT": "abc"}, {"APP_TIMEOUT": "-1"}, {"APP_LOG_LEVEL": "loud"}])
def test_invalid_values_raise(env):
    with pytest.raises(ValueError):
        Settings.from_env(environ=env)


def test_setup_logging_is_idempotent():
    a = setup_logging("INFO", "notes-test")
    b = setup_logging("DEBUG", "notes-test")
    assert a is b and len(b.handlers) == 1

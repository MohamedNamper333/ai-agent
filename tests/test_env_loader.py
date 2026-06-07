import os
import pytest
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
from env_loader import get_env, DEFAULTS, _load_dotenv, _get_all, _env_cache


def _reset_cache():
    import env_loader as mod
    mod._env_cache = None


@pytest.fixture(autouse=True)
def _clean_env():
    os.environ.pop("SECRET_KEY", None)
    os.environ.pop("API_KEY_HASH_SALT", None)
    os.environ.pop("CORS_ORIGINS", None)
    os.environ.pop("LOG_LEVEL", None)
    _reset_cache()
    yield
    os.environ.pop("SECRET_KEY", None)
    os.environ.pop("API_KEY_HASH_SALT", None)
    os.environ.pop("CORS_ORIGINS", None)
    os.environ.pop("LOG_LEVEL", None)
    _reset_cache()


class TestDefaults:
    def test_returns_default_when_no_env_or_file(self):
        assert get_env("SECRET_KEY") == "change-me-in-production"

    def test_all_defaults_are_strings(self):
        for value in DEFAULTS.values():
            assert isinstance(value, str)

    def test_default_keys_match_expected_variables(self):
        expected = {"SECRET_KEY", "API_KEY_HASH_SALT", "CORS_ORIGINS", "LOG_LEVEL"}
        assert set(DEFAULTS.keys()) == expected


class TestEnvVarOverride:
    def test_os_env_overrides_default(self):
        os.environ["SECRET_KEY"] = "my-secret"
        assert get_env("SECRET_KEY") == "my-secret"

    def test_explicit_default_parameter(self):
        assert get_env("MISSING_KEY", "fallback") == "fallback"

    def test_raises_on_missing_key_without_default(self):
        with pytest.raises(KeyError, match="MISSING_KEY"):
            get_env("MISSING_KEY")


class TestDotenvLoading:
    def test_load_dotenv_returns_dict(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('SECRET_KEY=test-secret\nLOG_LEVEL=DEBUG\n', encoding="utf-8")
        result = _load_dotenv(env_file)
        assert result["SECRET_KEY"] == "test-secret"
        assert result["LOG_LEVEL"] == "DEBUG"

    def test_load_dotenv_ignores_comments(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# this is a comment\nSECRET_KEY=abc\n", encoding="utf-8")
        result = _load_dotenv(env_file)
        assert "SECRET_KEY" in result

    def test_load_dotenv_ignores_blank_lines(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("\n\nSECRET_KEY=val\n\n", encoding="utf-8")
        result = _load_dotenv(env_file)
        assert result["SECRET_KEY"] == "val"

    def test_load_dotenv_handles_missing_file(self):
        result = _load_dotenv(Path("/nonexistent/.env"))
        assert result == {}

    def test_env_file_overrides_defaults(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text('LOG_LEVEL=DEBUG\n', encoding="utf-8")
        monkeypatch.setattr("env_loader.BASE_DIR", tmp_path)
        _reset_cache()
        assert get_env("LOG_LEVEL") == "DEBUG"


class TestOsEnvPriority:
    def test_os_env_takes_priority_over_dotenv(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text('SECRET_KEY=file-value\n', encoding="utf-8")
        monkeypatch.setattr("env_loader.BASE_DIR", tmp_path)
        _reset_cache()
        os.environ["SECRET_KEY"] = "env-value"
        assert get_env("SECRET_KEY") == "env-value"

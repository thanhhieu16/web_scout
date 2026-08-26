from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


def load_env_file(env_path: str | None = None) -> None:
    load_dotenv(dotenv_path=env_path)


load_env_file()


class RoleConfig(BaseModel):
    model: str = "stealth/ox-alpha"
    temperature: float = 0.0


class SearchConfig(BaseModel):
    max_results: int = 5
    max_uses: int = 4
    max_characters: int = 4000


class FetchConfig(BaseModel):
    timeout_seconds: float = 15.0
    max_chars: int = 20000
    user_agent: str = "WebScout/0.1 (research agent)"
    max_download_bytes: int = 2_000_000
    max_redirects: int = 5
    allow_private_hosts: bool = False


REPO_ROOT = Path(__file__).resolve().parent.parent


def _yaml_path() -> str:
    """Prefer ./config.yaml so a local override wins; fall back to the repo's own.

    Both branches return an absolute path so the result is never re-resolved
    against a *different* CWD later. It must also be called fresh at each
    Settings() construction (see settings_customise_sources below) rather than
    read once off model_config: a plain class attribute is fixed at class-body
    execution (module import) time, so a chdir before Settings() is constructed
    would otherwise be invisible to it.
    """
    local = Path("config.yaml").resolve()
    return str(local if local.is_file() else REPO_ROOT / "config.yaml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        # Superseded per-instantiation in settings_customise_sources below;
        # this is only a static fallback for introspection of model_config itself.
        yaml_file=_yaml_path(),
        extra="ignore",
    )

    openrouter_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "WEBSCOUT_OPENROUTER_API_KEY"),
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    researcher: RoleConfig = RoleConfig(temperature=0.2)
    verifier: RoleConfig = RoleConfig(temperature=0.0)
    answer: RoleConfig = RoleConfig(temperature=0.3)

    max_iterations: int = 3
    skills_enabled: bool = False
    search: SearchConfig = SearchConfig()
    fetch: FetchConfig = FetchConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=_yaml_path()),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

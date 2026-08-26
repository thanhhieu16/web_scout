from functools import lru_cache

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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        yaml_file="config.yaml",
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
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

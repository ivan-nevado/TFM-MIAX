"""Service settings, read from environment variables (prefix ``RELVAL_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RELVAL_", env_file=".env", extra="ignore")

    # Where the replication outputs live: a local folder or an ``s3://bucket/prefix`` URI.
    data_root: str = "data"
    reports_root: str = "reports"
    run_name: str = "replication"
    source: str = "lseg"
    panel_name: str = "panel"
    headline_seed: int = 20230301
    cors_origins: list[str] = ["http://localhost:5173"]
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()

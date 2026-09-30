"""
Application configuration loaded from environment variables and optional .env file.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import List


def _load_env_file(dotenv_path: Path) -> None:
    """Load key-value pairs from a .env file without external dependencies."""
    if not dotenv_path.is_file():
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_load_env_file(_PROJECT_ROOT / ".env")


class Settings:
    """Typed application settings with environment variable backing and validation."""

    def __init__(self) -> None:
        self.APP_ENV: str = os.getenv("APP_ENV", "development").lower()

        try:
            self.PORT: int = int(os.getenv("PORT", "8000"))
        except ValueError:
            raise ValueError(f"Invalid PORT environment variable: {os.getenv('PORT')}")

        try:
            self.DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8501"))
        except ValueError:
            raise ValueError(
                f"Invalid DASHBOARD_PORT environment variable: {os.getenv('DASHBOARD_PORT')}"
            )

        raw_docs = os.getenv("ENABLE_API_DOCS", "true").lower()
        self.ENABLE_API_DOCS: bool = raw_docs in ("true", "1", "yes", "on")

        raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501")
        self.ALLOWED_ORIGINS: str = raw_origins

        raw_cfg = os.getenv("ENGINE_CONFIG_PATH", "config/rotax_912_simulation.yaml")
        cfg_path = Path(raw_cfg)
        if not cfg_path.is_absolute():
            cfg_path = _PROJECT_ROOT / cfg_path
        self.ENGINE_CONFIG_PATH: str = str(cfg_path)

        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

        self.validate()

    @property
    def port(self) -> int:
        return self.PORT

    @property
    def dashboard_port(self) -> int:
        return self.DASHBOARD_PORT

    @property
    def app_env(self) -> str:
        return self.APP_ENV

    @property
    def enable_api_docs(self) -> bool:
        return self.ENABLE_API_DOCS

    @property
    def allowed_origins(self) -> str:
        return self.ALLOWED_ORIGINS

    @property
    def engine_config_path(self) -> str:
        return self.ENGINE_CONFIG_PATH

    @property
    def log_level(self) -> str:
        return self.LOG_LEVEL

    @property
    def allowed_origins_list(self) -> List[str]:
        """Return allowed CORS origins as a list of strings."""
        if not self.ALLOWED_ORIGINS:
            return ["http://localhost:8501"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    def validate(self) -> None:
        """Validate startup configuration sanity."""
        if self.PORT <= 0 or self.PORT > 65535:
            raise ValueError(f"PORT must be between 1 and 65535, got {self.PORT}")
        if self.DASHBOARD_PORT <= 0 or self.DASHBOARD_PORT > 65535:
            raise ValueError(
                f"DASHBOARD_PORT must be between 1 and 65535, got {self.DASHBOARD_PORT}"
            )

        cfg_path = Path(self.ENGINE_CONFIG_PATH)
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"ENGINE_CONFIG_PATH points to non-existent file: {cfg_path}"
            )


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


settings = get_settings()

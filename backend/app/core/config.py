"""
Configuration management for the backend application.
Uses pydantic settings for environment variable management.
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application settings
    APP_NAME: str = "Fuzzy OSS20 Trading Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"

    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://fuzzy:fuzzy123@localhost:5432/fuzzy_oss20"
    )

    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_CACHE_TTL: int = 300  # 5 minutes

    # ArcticDB settings
    ARCTIC_URI: str = os.getenv("ARCTIC_URI", "lmdb://C:/Users/maxli/PycharmProjects/PythonProject/Fuzzy_oss20/arctic_storage")
    ARCTIC_TICK_LIBRARY: str = "tick_data"
    ARCTIC_BAR_LIBRARY: str = "bars_time_bars"

    # IQFeed settings
    IQFEED_HOST: str = "127.0.0.1"
    IQFEED_PORT: int = 5009
    IQFEED_PRODUCT: str = "FUZZY_OSS20"
    IQFEED_VERSION: str = "1.0"
    IQFEED_LOGIN: str = os.getenv("IQFEED_LOGIN", "487854")
    IQFEED_PASSWORD: str = os.getenv("IQFEED_PASSWORD", "LTKKma0")

    # WebSocket settings
    WS_MESSAGE_QUEUE: str = "redis://localhost:6379/1"
    WS_HEARTBEAT_INTERVAL: int = 30

    # Monitoring settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # or "console"
    LOG_FILE: Optional[str] = "logs/backend.log"

    # Timezone settings (always ET as per Data_policy.md)
    TIMEZONE: str = "US/Eastern"

    # Performance settings
    MAX_CONNECTIONS_COUNT: int = 100
    MIN_CONNECTIONS_COUNT: int = 10
    QUERY_TIMEOUT: int = 60

    # Data validation thresholds
    MAX_TICK_DELAY_SECONDS: int = 60
    MAX_BAR_DELAY_MINUTES: int = 5
    MIN_DATA_QUALITY_SCORE: float = 0.95

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Create and cache settings instance.
    Use lru_cache to ensure we only create one instance.
    """
    return Settings()


# Create a global settings instance
settings = get_settings()
"""
Centralized configuration management for JBAC Strategy Foundry.

Handles environment variables, secrets, and application settings.
Uses .env file for local development, environment variables for production.
"""

import os
from pathlib import Path
from typing import Optional, List
from functools import lru_cache
import logging

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Find .env file (look in project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file():
    """Load .env file if it exists."""
    if ENV_FILE.exists():
        logger.info(f"Loading configuration from {ENV_FILE}")
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove inline comments (after #)
                    if "#" in value:
                        value = value.split("#")[0].strip()
                    
                    # Only set if not already in environment
                    if key and value and key not in os.environ:
                        os.environ[key] = value
    else:
        logger.warning(
            f".env file not found at {ENV_FILE}. "
            "Using environment variables or defaults."
        )


# Load .env on module import
load_env_file()


class TelegramConfig(BaseModel):
    """Telegram bot configuration."""
    
    bot_token: Optional[str] = Field(
        default=None,
        description="Telegram bot API token from @BotFather"
    )
    chat_ids: List[str] = Field(
        default_factory=list,
        description="List of Telegram chat IDs to send alerts to"
    )
    channel_id: Optional[str] = Field(
        default=None,
        description="Optional Telegram channel ID for team-wide alerts"
    )
    enabled: bool = Field(
        default=True,
        description="Enable/disable Telegram alerts"
    )
    parse_mode: str = Field(
        default="Markdown",
        description="Message formatting: Markdown, HTML, or None"
    )
    
    @field_validator("chat_ids", mode="before")
    @classmethod
    def parse_chat_ids(cls, v):
        """Parse comma-separated chat IDs."""
        if isinstance(v, str):
            return [cid.strip() for cid in v.split(",") if cid.strip()]
        return v or []
    
    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_ids and self.enabled)


class BrokerConfig(BaseModel):
    """Broker API configuration."""
    
    # Alpaca
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    
    # Interactive Brokers
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1


class DataConfig(BaseModel):
    """Data source configuration."""
    
    alpha_vantage_api_key: Optional[str] = None
    use_yfinance: bool = True


class AppConfig(BaseModel):
    """Main application configuration."""
    
    env: str = Field(
        default="development",
        description="Environment: development, staging, production"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Alert settings
    enable_telegram_alerts: bool = True
    enable_console_alerts: bool = True
    enable_email_alerts: bool = False
    min_alert_severity: str = "INFO"
    
    # Sub-configs
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls(
            env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            enable_telegram_alerts=os.getenv("ENABLE_TELEGRAM_ALERTS", "true").lower() == "true",
            enable_console_alerts=os.getenv("ENABLE_CONSOLE_ALERTS", "true").lower() == "true",
            enable_email_alerts=os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true",
            min_alert_severity=os.getenv("MIN_ALERT_SEVERITY", "INFO"),
            telegram=TelegramConfig(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
                chat_ids=os.getenv("TELEGRAM_CHAT_IDS", ""),
                channel_id=os.getenv("TELEGRAM_CHANNEL_ID"),
                enabled=os.getenv("ENABLE_TELEGRAM_ALERTS", "true").lower() == "true",
            ),
            broker=BrokerConfig(
                alpaca_api_key=os.getenv("ALPACA_API_KEY"),
                alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
                alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
                ib_host=os.getenv("IB_HOST", "127.0.0.1"),
                ib_port=int(os.getenv("IB_PORT", "7497")),
                ib_client_id=int(os.getenv("IB_CLIENT_ID", "1")),
            ),
            data=DataConfig(
                alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
                use_yfinance=os.getenv("USE_YFINANCE", "true").lower() == "true",
            ),
        )


@lru_cache()
def get_config() -> AppConfig:
    """
    Get cached application configuration.
    
    This is the main entry point for accessing configuration.
    Configuration is loaded once and cached for performance.
    
    Returns:
        AppConfig: Application configuration
        
    Example:
        >>> from jsf.config import get_config
        >>> config = get_config()
        >>> if config.telegram.is_configured:
        ...     print(f"Telegram alerts enabled for {len(config.telegram.chat_ids)} users")
    """
    return AppConfig.from_env()


def validate_config() -> List[str]:
    """
    Validate configuration and return list of warnings/issues.
    
    Returns:
        List of warning messages (empty if all OK)
    """
    warnings = []
    config = get_config()
    
    # Check Telegram configuration
    if config.enable_telegram_alerts:
        if not config.telegram.bot_token:
            warnings.append(
                "Telegram alerts enabled but TELEGRAM_BOT_TOKEN not set. "
                "Get token from @BotFather on Telegram."
            )
        if not config.telegram.chat_ids:
            warnings.append(
                "Telegram alerts enabled but TELEGRAM_CHAT_IDS not set. "
                "Get your chat ID from @userinfobot on Telegram."
            )
    
    # Check broker configuration
    if config.broker.alpaca_api_key and not config.broker.alpaca_secret_key:
        warnings.append("Alpaca API key set but secret key missing")
    
    # Check data sources
    if not config.data.use_yfinance and not config.data.alpha_vantage_api_key:
        warnings.append("No data sources configured (yfinance disabled, no Alpha Vantage key)")
    
    return warnings


# Export main objects
__all__ = [
    "get_config",
    "validate_config",
    "AppConfig",
    "TelegramConfig",
    "BrokerConfig",
    "DataConfig",
]

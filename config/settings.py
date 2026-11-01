"""
TradeAgent Configuration Module.

Loads settings from environment variables and .env file.
All API keys are optional — the agent runs in demo/paper mode without them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data_store"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """
    Central configuration loaded from environment variables / .env file.
    All API keys are optional. The agent degrades gracefully without them.
    """

    model_config = SettingsConfigDict(
        env_file=str(CONFIG_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── General ──────────────────────────────────────────────
    trade_mode: str = Field(default="paper", description="paper | live")
    log_level: str = Field(default="INFO")
    default_market: str = Field(default="forex", description="forex | crypto | equity")

    # ── OANDA ────────────────────────────────────────────────
    oanda_api_key: Optional[str] = Field(default=None)
    oanda_account_id: Optional[str] = Field(default=None)
    oanda_environment: str = Field(default="practice")

    # ── MetaTrader 5 ─────────────────────────────────────────
    mt5_login: Optional[str] = Field(default=None)
    mt5_password: Optional[str] = Field(default=None)
    mt5_server: Optional[str] = Field(default=None)
    mt5_path: Optional[str] = Field(default=None)

    # ── Crypto Exchanges ─────────────────────────────────────
    binance_api_key: Optional[str] = Field(default=None)
    binance_secret: Optional[str] = Field(default=None)
    bybit_api_key: Optional[str] = Field(default=None)
    bybit_secret: Optional[str] = Field(default=None)

    # ── Fundamental Data ─────────────────────────────────────
    fred_api_key: Optional[str] = Field(default=None)
    finnhub_api_key: Optional[str] = Field(default=None)

    # ── News & Sentiment ─────────────────────────────────────
    news_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)

    # ── Telegram ─────────────────────────────────────────────
    telegram_bot_token: Optional[str] = Field(default=None)
    telegram_chat_id: Optional[str] = Field(default=None)

    # ── Redis ────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── Database ─────────────────────────────────────────────
    duckdb_path: str = Field(default=str(DATA_DIR / "tradeagent.duckdb"))

    # ── Derived Properties ───────────────────────────────────
    @property
    def is_live(self) -> bool:
        return self.trade_mode == "live"

    @property
    def is_paper(self) -> bool:
        return self.trade_mode == "paper"

    @property
    def has_oanda(self) -> bool:
        return bool(self.oanda_api_key and self.oanda_account_id)

    @property
    def has_mt5(self) -> bool:
        return bool(self.mt5_login and self.mt5_password and self.mt5_server)

    @property
    def has_binance(self) -> bool:
        return bool(self.binance_api_key and self.binance_secret)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def has_fred(self) -> bool:
        return bool(self.fred_api_key)

    @property
    def has_finnhub(self) -> bool:
        return bool(self.finnhub_api_key)

    @property
    def has_news_api(self) -> bool:
        return bool(self.news_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @field_validator("trade_mode")
    @classmethod
    def validate_trade_mode(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("paper", "live"):
            raise ValueError(f"trade_mode must be 'paper' or 'live', got '{v}'")
        return v


class TradingParams:
    """
    Loads strategy/risk/indicator parameters from trading_params.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._path = config_path or (CONFIG_DIR / "trading_params.yaml")
        self._data: dict = {}
        self.reload()

    def reload(self) -> None:
        """Reload parameters from YAML file."""
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = {}

    # ── Convenience Accessors ────────────────────────────────
    @property
    def general(self) -> dict:
        return self._data.get("general", {})

    @property
    def risk(self) -> dict:
        return self._data.get("risk", {})

    @property
    def technical(self) -> dict:
        return self._data.get("technical", {})

    @property
    def fundamentals(self) -> dict:
        return self._data.get("fundamentals", {})

    @property
    def sentiment_config(self) -> dict:
        return self._data.get("sentiment", {})

    @property
    def ml(self) -> dict:
        return self._data.get("ml", {})

    @property
    def backtesting_config(self) -> dict:
        return self._data.get("backtesting", {})

    @property
    def notifications(self) -> dict:
        return self._data.get("notifications", {})

    @property
    def asset_strategies(self) -> dict:
        return self._data.get("asset_strategies", {})

    @property
    def watchlist(self) -> dict:
        return self.general.get("watchlist", {})

    @property
    def timeframes(self) -> dict:
        return self.general.get("timeframes", {})

    @property
    def scan_interval(self) -> int:
        return self.general.get("scan_interval_seconds", 60)

    def get(self, key_path: str, default=None):
        """
        Get a nested config value using dot notation.
        Example: params.get("risk.risk_per_trade_pct", 1.0)
        """
        keys = key_path.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value


# ── Singleton Instances ──────────────────────────────────────
settings = Settings()
trading_params = TradingParams()

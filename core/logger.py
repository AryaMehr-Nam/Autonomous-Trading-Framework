"""
TradeAgent Structured Logging.

Uses loguru for structured, colored, and file-rotated logging.
All components should use `get_logger(name)` to get a contextualized logger.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config.settings import LOGS_DIR


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure loguru for the application.

    Outputs:
    - Console: colored, human-readable format
    - File: JSON-structured, rotated daily, kept for 30 days
    """
    # Remove default handler
    logger.remove()

    # Console handler — colored, concise
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[component]}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    # File handler — structured JSON, rotated
    logger.add(
        str(LOGS_DIR / "tradeagent_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[component]} | {message}",
        rotation="1 day",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
        backtrace=True,
        diagnose=False,
    )

    # Trade-specific log (only trade events)
    logger.add(
        str(LOGS_DIR / "trades_{time:YYYY-MM-DD}.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        rotation="1 day",
        retention="90 days",
        filter=lambda record: record["extra"].get("component") == "trades",
        encoding="utf-8",
    )


def get_logger(component: str = "main"):
    """
    Get a logger instance bound to a component name.

    Usage:
        from core.logger import get_logger
        logger = get_logger("risk_manager")
        logger.info("Daily drawdown check passed")
    """
    return logger.bind(component=component)


# Initialize logging on import
_initialized = False


def init_logging(log_level: str = "INFO") -> None:
    """Initialize logging (call once at startup)."""
    global _initialized
    if not _initialized:
        setup_logging(log_level)
        _initialized = True

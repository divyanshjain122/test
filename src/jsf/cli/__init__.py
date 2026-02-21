"""Command-line interface for JSF-Core."""

from .setup_telegram import setup_telegram_bot
from .main import cli

__all__ = ['setup_telegram_bot', 'cli']

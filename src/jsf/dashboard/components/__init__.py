"""Dashboard Pages Package

Contains individual page renderers for the dashboard.
"""

from .portfolio import render_portfolio
from .pnl import render_pnl
from .trades import render_trades
from .risk import render_risk

__all__ = [
    "render_portfolio",
    "render_pnl",
    "render_trades",
    "render_risk",
]

"""
Plotting functions for backtesting results.

This module provides comprehensive visualization tools for analyzing
backtest performance, including equity curves, drawdowns, returns
distributions, and rolling metrics.
"""

import logging
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

from jsf.simulation.backtest import BacktestResult
from jsf.simulation.metrics import calculate_max_drawdown

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_equity_curve(
    result: BacktestResult,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot equity curve from backtest results.
    
    Args:
        result: BacktestResult object
        title: Plot title (default: "Equity Curve")
        show: Whether to display the plot
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot equity curve
    equity = result.equity_curve
    ax.plot(equity.index, equity.values, linewidth=2, color='#2E86AB')
    
    # Format
    ax.set_title(title or "Equity Curve", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    
    # Add stats box
    stats_text = (
        f"Total Return: {result.total_return:.2%}\n"
        f"CAGR: {result.cagr:.2%}\n"
        f"Sharpe Ratio: {result.sharpe_ratio:.2f}\n"
        f"Max Drawdown: {result.max_drawdown:.2%}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved equity curve to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_drawdown(
    result: BacktestResult,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot drawdown chart from backtest results.
    
    Args:
        result: BacktestResult object
        title: Plot title (default: "Drawdown")
        show: Whether to display the plot
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    # Calculate drawdown
    equity = result.equity_curve
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    
    # Plot equity curve
    ax1.plot(equity.index, equity.values, linewidth=2, color='#2E86AB', label='Equity')
    ax1.plot(running_max.index, running_max.values, linewidth=1, 
             color='#A23B72', linestyle='--', alpha=0.7, label='Running Max')
    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title or "Equity Curve & Drawdown", fontsize=14, fontweight='bold')
    
    # Plot drawdown
    ax2.fill_between(drawdown.index, drawdown.values, 0, 
                     color='#F18F01', alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values, linewidth=1, color='#C73E1D')
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Drawdown", fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add max drawdown line
    max_dd = drawdown.min()
    ax2.axhline(y=max_dd, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.text(drawdown.index[len(drawdown)//2], max_dd, 
             f'Max DD: {max_dd:.2%}', verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved drawdown chart to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_returns_distribution(
    result: BacktestResult,
    bins: int = 50,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot returns distribution histogram with statistics.
    
    Args:
        result: BacktestResult object
        bins: Number of histogram bins
        title: Plot title (default: "Returns Distribution")
        show: Whether to display the plot
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    returns = result.returns.dropna()
    
    # Histogram
    ax1.hist(returns, bins=bins, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(returns.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {returns.mean():.2%}')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax1.set_xlabel("Daily Returns", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(title or "Returns Distribution", fontsize=14, fontweight='bold')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Normal Distribution)", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    skew = returns.skew()
    kurt = returns.kurtosis()
    stats_text = (
        f"Mean: {returns.mean():.3%}\n"
        f"Std Dev: {returns.std():.3%}\n"
        f"Skewness: {skew:.2f}\n"
        f"Kurtosis: {kurt:.2f}\n"
        f"Min: {returns.min():.2%}\n"
        f"Max: {returns.max():.2%}"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round',
             facecolor='wheat', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved returns distribution to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_rolling_metrics(
    result: BacktestResult,
    window: int = 60,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot rolling performance metrics.
    
    Args:
        result: BacktestResult object
        window: Rolling window size in days
        title: Plot title (default: "Rolling Metrics")
        show: Whether to display the plot
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    returns = result.returns.dropna()
    equity = result.equity_curve
    
    # Calculate rolling metrics
    rolling_return = returns.rolling(window).sum()
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (returns.rolling(window).mean() * 252) / rolling_vol
    
    # Calculate rolling drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    rolling_max_dd = drawdown.rolling(window).min()
    
    # Plot 1: Rolling Returns
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rolling_return.index, rolling_return.values, linewidth=1.5, color='#2E86AB')
    ax1.set_ylabel(f"{window}-Day Return", fontsize=10)
    ax1.set_title("Rolling Returns", fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot 2: Rolling Volatility
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(rolling_vol.index, rolling_vol.values, linewidth=1.5, color='#F18F01')
    ax2.set_ylabel("Annualized Volatility", fontsize=10)
    ax2.set_title("Rolling Volatility", fontsize=12, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling Sharpe Ratio
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5, color='#A23B72')
    ax3.set_ylabel("Sharpe Ratio", fontsize=10)
    ax3.set_title("Rolling Sharpe Ratio", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax3.axhline(y=1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot 4: Rolling Max Drawdown
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(rolling_max_dd.index, rolling_max_dd.values, linewidth=1.5, color='#C73E1D')
    ax4.set_ylabel("Max Drawdown", fontsize=10)
    ax4.set_title("Rolling Max Drawdown", fontsize=12, fontweight='bold')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(rolling_max_dd.index, rolling_max_dd.values, 0, 
                     color='#C73E1D', alpha=0.3)
    
    # Plot 5: Win Rate
    ax5 = fig.add_subplot(gs[2, 0])
    rolling_wins = (returns > 0).rolling(window).sum() / window
    ax5.plot(rolling_wins.index, rolling_wins.values, linewidth=1.5, color='#06A77D')
    ax5.set_ylabel("Win Rate", fontsize=10)
    ax5.set_title("Rolling Win Rate", fontsize=12, fontweight='bold')
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax5.set_xlabel("Date", fontsize=10)
    
    # Plot 6: Cumulative Returns
    ax6 = fig.add_subplot(gs[2, 1])
    cumulative = (1 + returns).cumprod()
    ax6.plot(cumulative.index, cumulative.values, linewidth=1.5, color='#2E86AB')
    ax6.set_ylabel("Cumulative Return", fontsize=10)
    ax6.set_title("Cumulative Returns", fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlabel("Date", fontsize=10)
    
    # Format x-axis dates for all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.suptitle(title or f"Rolling Metrics ({window}-Day Window)", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved rolling metrics to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_monthly_returns(
    result: BacktestResult,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot monthly returns heatmap.
    
    Args:
        result: BacktestResult object
        title: Plot title (default: "Monthly Returns")
        show: Whether to display the plot
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    returns = result.returns.dropna()
    
    # Calculate monthly returns
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    # Create pivot table for heatmap
    monthly_returns_pivot = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    }).pivot(index='Month', columns='Year', values='Return')
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_returns_pivot.index = [month_names[i-1] for i in monthly_returns_pivot.index]
    
    # Plot heatmap
    sns.heatmap(monthly_returns_pivot, annot=True, fmt='.1%', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'Monthly Return'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title(title or "Monthly Returns Heatmap", fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Month", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved monthly returns heatmap to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_performance_summary(
    result: BacktestResult,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comprehensive performance summary dashboard.
    
    Args:
        result: BacktestResult object
        title: Plot title (default: "Performance Summary")
        show: Whether to display the plot
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    returns = result.returns.dropna()
    equity = result.equity_curve
    
    # Calculate drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    
    # Plot 1: Equity Curve (large)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity.index, equity.values, linewidth=2, color='#2E86AB')
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.set_title("Equity Curve", fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='#F18F01', alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values, linewidth=1, color='#C73E1D')
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_title("Drawdown", fontsize=12, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Returns Distribution
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(returns, bins=40, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax3.axvline(returns.mean(), color='red', linestyle='--', linewidth=1.5)
    ax3.set_xlabel("Daily Returns", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.set_title("Returns Distribution", fontsize=11, fontweight='bold')
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling Sharpe
    ax4 = fig.add_subplot(gs[2, 1])
    rolling_vol = returns.rolling(60).std() * np.sqrt(252)
    rolling_sharpe = (returns.rolling(60).mean() * 252) / rolling_vol
    ax4.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5, color='#A23B72')
    ax4.set_xlabel("Date", fontsize=10)
    ax4.set_ylabel("Sharpe Ratio", fontsize=10)
    ax4.set_title("60-Day Rolling Sharpe", fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot 5: Performance Metrics Table
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    metrics_data = [
        ['Total Return', f'{result.total_return:.2%}'],
        ['CAGR', f'{result.cagr:.2%}'],
        ['Volatility', f'{result.volatility:.2%}'],
        ['Sharpe Ratio', f'{result.sharpe_ratio:.2f}'],
        ['Max Drawdown', f'{result.max_drawdown:.2%}'],
        ['# Trades', f'{len(result.trades)}'],
        ['Win Rate', f'{(returns > 0).sum() / len(returns):.1%}'],
        ['Best Day', f'{returns.max():.2%}'],
        ['Worst Day', f'{returns.min():.2%}'],
    ]
    
    table = ax5.table(cellText=metrics_data, cellLoc='left',
                      colWidths=[0.6, 0.4], loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style table
    for i in range(len(metrics_data)):
        table[(i, 0)].set_facecolor('#E8E8E8')
        table[(i, 1)].set_facecolor('#F5F5F5')
    
    ax5.set_title("Performance Metrics", fontsize=11, fontweight='bold', pad=20)
    
    # Format x-axis dates
    for ax in [ax1, ax2, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.suptitle(title or "Performance Summary", fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance summary to {save_path}")
    
    if show:
        plt.show()
    
    return fig

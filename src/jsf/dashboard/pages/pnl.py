"""P&L Tracking Page

Profit and loss analysis with equity curve, returns charts, and performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Any
from datetime import datetime, timedelta

from ..models import DashboardState, PerformanceMetrics
from ..metrics import (
    MetricsCalculator,
    calculate_returns,
    calculate_drawdown,
)


def render_pnl(state: DashboardState, collector: Optional[Any] = None):
    """Render the P&L tracking page.
    
    Args:
        state: Current dashboard state
        collector: DataCollector instance
    """
    st.title("💰 Profit & Loss")
    
    snapshot = state.current_snapshot
    
    if snapshot is None:
        st.warning("No P&L data available.")
        return
    
    # Top metrics row
    render_pnl_summary(snapshot, state.initial_capital)
    
    st.markdown("---")
    
    # Get equity history
    equity_series = get_equity_series(state, collector)
    
    if equity_series.empty or len(equity_series) < 2:
        st.info("Insufficient historical data for charts. Data will accumulate over time.")
        return
    
    # Equity curve and drawdown
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Equity Curve")
        render_equity_curve(equity_series, state.initial_capital)
    
    with col2:
        st.subheader("Performance Metrics")
        render_performance_metrics(equity_series, state.initial_capital)
    
    st.markdown("---")
    
    # Returns analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Returns Distribution")
        render_returns_histogram(equity_series)
    
    with col2:
        st.subheader("Daily P&L")
        render_daily_pnl(equity_series)
    
    st.markdown("---")
    
    # Additional analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cumulative Returns")
        render_cumulative_returns(equity_series)
    
    with col2:
        st.subheader("Drawdown")
        render_drawdown_chart(equity_series)
    
    st.markdown("---")
    
    # Monthly returns heatmap
    st.subheader("Monthly Returns Heatmap")
    render_monthly_returns(equity_series)


def get_equity_series(state: DashboardState, collector: Optional[Any]) -> pd.Series:
    """Get equity time series from state or collector.
    
    Args:
        state: Dashboard state
        collector: DataCollector
        
    Returns:
        Equity series
    """
    # Try collector first
    if collector is not None and hasattr(collector, 'history'):
        series = collector.history.get_equity_series()
        if not series.empty:
            return series
    
    # Fall back to state
    if state.equity_history:
        return state.get_equity_series()
    
    return pd.Series(dtype=float)


def render_pnl_summary(snapshot, initial_capital: float):
    """Render P&L summary metrics row.
    
    Args:
        snapshot: PortfolioSnapshot
        initial_capital: Starting capital
    """
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Daily P&L",
            f"${snapshot.daily_pnl:+,.2f}",
            f"{snapshot.daily_return:+.2f}%",
        )
    
    with col2:
        st.metric(
            "Total P&L",
            f"${snapshot.total_pnl:+,.2f}",
            f"{snapshot.total_return:+.2f}%",
        )
    
    with col3:
        st.metric(
            "Current Equity",
            f"${snapshot.equity:,.2f}",
        )
    
    with col4:
        st.metric(
            "Initial Capital",
            f"${initial_capital:,.2f}",
        )
    
    with col5:
        # Calculate unrealized P&L from positions
        unrealized = sum(p.unrealized_pnl for p in snapshot.positions)
        st.metric(
            "Unrealized P&L",
            f"${unrealized:+,.2f}",
        )
    
    with col6:
        # Realized = Total - Unrealized
        unrealized = sum(p.unrealized_pnl for p in snapshot.positions)
        realized = snapshot.total_pnl - unrealized
        st.metric(
            "Realized P&L",
            f"${realized:+,.2f}",
        )


def render_equity_curve(equity: pd.Series, initial_capital: float):
    """Render equity curve chart.
    
    Args:
        equity: Equity time series
        initial_capital: Starting capital for reference line
    """
    fig = go.Figure()
    
    # Equity line
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='Equity',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
    ))
    
    # Initial capital reference line
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Initial: ${initial_capital:,.0f}",
    )
    
    # High water mark
    hwm = equity.expanding().max()
    fig.add_trace(go.Scatter(
        x=hwm.index,
        y=hwm.values,
        mode='lines',
        name='High Water Mark',
        line=dict(color='#00c853', width=1, dash='dot'),
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Date',
        yaxis_title='Equity ($)',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_performance_metrics(equity: pd.Series, initial_capital: float):
    """Render performance metrics card.
    
    Args:
        equity: Equity series
        initial_capital: Starting capital
    """
    calculator = MetricsCalculator()
    perf = calculator.calculate_performance_metrics(equity, initial_capital)
    
    # Display as metrics
    st.metric("1-Day Return", f"{perf.return_1d:+.2f}%")
    st.metric("1-Week Return", f"{perf.return_1w:+.2f}%")
    st.metric("1-Month Return", f"{perf.return_1m:+.2f}%")
    st.metric("Total Return", f"{perf.return_total:+.2f}%")
    
    st.markdown("---")
    
    st.metric("Best Day", f"{perf.best_day:+.2f}%")
    st.metric("Worst Day", f"{perf.worst_day:+.2f}%")
    st.metric("Win Rate", f"{perf.win_rate:.1f}%")


def render_returns_histogram(equity: pd.Series):
    """Render returns distribution histogram.
    
    Args:
        equity: Equity series
    """
    returns = calculate_returns(equity) * 100  # Convert to percentage
    
    if returns.empty:
        st.info("No returns data available")
        return
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        marker_color='#1f77b4',
        opacity=0.7,
        name='Returns',
    ))
    
    # Add mean line
    mean_return = returns.mean()
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_return:.2f}%",
    )
    
    # Add zero line
    fig.add_vline(x=0, line_dash="solid", line_color="gray")
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats below
    st.caption(
        f"Mean: {mean_return:.2f}% | "
        f"Std: {returns.std():.2f}% | "
        f"Skew: {returns.skew():.2f} | "
        f"Kurtosis: {returns.kurtosis():.2f}"
    )


def render_daily_pnl(equity: pd.Series):
    """Render daily P&L bar chart.
    
    Args:
        equity: Equity series
    """
    # Calculate daily changes
    daily_pnl = equity.diff().dropna()
    
    if daily_pnl.empty:
        st.info("No daily P&L data available")
        return
    
    # Limit to last 30 days for readability
    if len(daily_pnl) > 30:
        daily_pnl = daily_pnl.tail(30)
    
    colors = ['#00c853' if x >= 0 else '#ff1744' for x in daily_pnl.values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=daily_pnl.index,
            y=daily_pnl.values,
            marker_color=colors,
        )
    ])
    
    fig.add_hline(y=0, line_dash="solid", line_color="gray")
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Date',
        yaxis_title='P&L ($)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    wins = (daily_pnl > 0).sum()
    losses = (daily_pnl < 0).sum()
    total = len(daily_pnl)
    
    st.caption(
        f"Winning Days: {wins} | "
        f"Losing Days: {losses} | "
        f"Win Rate: {wins/total*100:.1f}%"
    )


def render_cumulative_returns(equity: pd.Series):
    """Render cumulative returns chart.
    
    Args:
        equity: Equity series
    """
    if equity.empty:
        st.info("No data available")
        return
    
    # Calculate cumulative return from start
    cum_returns = ((equity / equity.iloc[0]) - 1) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns.values,
        mode='lines',
        name='Cumulative Return',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_chart(equity: pd.Series):
    """Render drawdown chart.
    
    Args:
        equity: Equity series
    """
    drawdown, max_dd, current_dd = calculate_drawdown(equity)
    
    if drawdown.empty:
        st.info("No drawdown data available")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown',
        line=dict(color='#ff1744', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 23, 68, 0.3)',
    ))
    
    # Max drawdown line
    fig.add_hline(
        y=-max_dd,
        line_dash="dash",
        line_color="darkred",
        annotation_text=f"Max DD: {max_dd:.2f}%",
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Max Drawdown: {max_dd:.2f}% | Current Drawdown: {current_dd:.2f}%")


def render_monthly_returns(equity: pd.Series):
    """Render monthly returns heatmap.
    
    Args:
        equity: Equity series
    """
    calculator = MetricsCalculator()
    monthly = calculator.get_monthly_returns(equity)
    
    if monthly.empty:
        st.info("Insufficient data for monthly returns heatmap")
        return
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=monthly.values,
        x=monthly.columns.tolist(),
        y=monthly.index.tolist(),
        colorscale=[
            [0, '#ff1744'],      # Negative (red)
            [0.5, '#ffffff'],    # Zero (white)
            [1, '#00c853'],      # Positive (green)
        ],
        zmid=0,
        text=[[f'{v:.1f}%' if pd.notna(v) else '' for v in row] for row in monthly.values],
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{y} %{x}: %{z:.2f}%<extra></extra>',
    ))
    
    fig.update_layout(
        height=max(200, len(monthly.index) * 40),
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Month',
        yaxis_title='Year',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Yearly totals
    if not monthly.empty:
        yearly = monthly.sum(axis=1)
        st.caption("Yearly Returns: " + " | ".join([f"{y}: {r:+.2f}%" for y, r in yearly.items()]))

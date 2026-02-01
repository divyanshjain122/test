"""Risk Metrics Page

Risk analysis with drawdown, VaR, exposure monitoring, and risk-adjusted metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Any
from datetime import datetime

from ..models import DashboardState, RiskMetrics
from ..metrics import (
    MetricsCalculator,
    calculate_returns,
    calculate_drawdown,
    calculate_var,
    calculate_volatility,
    calculate_sharpe,
    calculate_sortino,
)


def render_risk(state: DashboardState, collector: Optional[Any] = None):
    """Render the risk metrics page.
    
    Args:
        state: Current dashboard state
        collector: DataCollector instance
    """
    st.title("⚠️ Risk Analysis")
    
    snapshot = state.current_snapshot
    
    if snapshot is None:
        st.warning("No risk data available.")
        return
    
    # Get equity history
    equity_series = get_equity_series(state, collector)
    
    # Calculate risk metrics
    calculator = MetricsCalculator()
    
    if not equity_series.empty and len(equity_series) >= 2:
        positions_data = [{'weight': p.weight, 'side': p.side} for p in snapshot.positions]
        risk_metrics = calculator.calculate_risk_metrics(equity_series, positions_data)
    else:
        risk_metrics = None
    
    # Risk summary
    render_risk_summary(risk_metrics, snapshot)
    
    st.markdown("---")
    
    # Risk metrics cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Drawdown Analysis")
        render_drawdown_section(equity_series)
    
    with col2:
        st.subheader("Value at Risk")
        render_var_section(equity_series)
    
    st.markdown("---")
    
    # Risk-adjusted metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk-Adjusted Returns")
        render_risk_adjusted_metrics(equity_series, risk_metrics)
    
    with col2:
        st.subheader("Volatility Analysis")
        render_volatility_section(equity_series)
    
    st.markdown("---")
    
    # Exposure analysis
    st.subheader("Portfolio Exposure")
    render_exposure_section(snapshot, risk_metrics)
    
    st.markdown("---")
    
    # Position risk
    st.subheader("Position Risk Analysis")
    render_position_risk(snapshot)


def get_equity_series(state: DashboardState, collector: Optional[Any]) -> pd.Series:
    """Get equity time series from state or collector."""
    if collector is not None and hasattr(collector, 'history'):
        series = collector.history.get_equity_series()
        if not series.empty:
            return series
    
    if state.equity_history:
        return state.get_equity_series()
    
    return pd.Series(dtype=float)


def render_risk_summary(risk_metrics: Optional[RiskMetrics], snapshot):
    """Render risk summary metrics row."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    if risk_metrics:
        with col1:
            color = "normal" if risk_metrics.max_drawdown < 10 else "inverse"
            st.metric(
                "Max Drawdown",
                f"{risk_metrics.max_drawdown:.2f}%",
                delta=None,
                delta_color=color,
            )
        
        with col2:
            st.metric(
                "Current DD",
                f"{risk_metrics.current_drawdown:.2f}%",
            )
        
        with col3:
            st.metric(
                "VaR 95%",
                f"{risk_metrics.var_95:.2f}%",
            )
        
        with col4:
            st.metric(
                "Volatility",
                f"{risk_metrics.volatility:.2f}%",
            )
        
        with col5:
            st.metric(
                "Sharpe Ratio",
                f"{risk_metrics.sharpe_ratio:.2f}",
            )
        
        with col6:
            st.metric(
                "Sortino Ratio",
                f"{risk_metrics.sortino_ratio:.2f}",
            )
    else:
        st.info("Insufficient data for risk metrics. Metrics will appear as data accumulates.")


def render_drawdown_section(equity: pd.Series):
    """Render drawdown analysis section."""
    if equity.empty or len(equity) < 2:
        st.info("Insufficient data for drawdown analysis")
        return
    
    drawdown, max_dd, current_dd = calculate_drawdown(equity)
    
    # Drawdown chart
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
    
    # Max drawdown horizontal line
    fig.add_hline(
        y=-max_dd,
        line_dash="dash",
        line_color="darkred",
        annotation_text=f"Max: {max_dd:.2f}%",
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    with col2:
        st.metric("Current Drawdown", f"{current_dd:.2f}%")
    with col3:
        # Average drawdown
        avg_dd = abs(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0
        st.metric("Average Drawdown", f"{avg_dd:.2f}%")


def render_var_section(equity: pd.Series):
    """Render Value at Risk section."""
    if equity.empty or len(equity) < 10:
        st.info("Need at least 10 data points for VaR calculation")
        return
    
    returns = calculate_returns(equity)
    
    # Calculate VaR at different confidence levels
    var_90 = calculate_var(returns, 0.90)
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    
    # VaR comparison chart
    fig = go.Figure(data=[
        go.Bar(
            x=['90%', '95%', '99%'],
            y=[var_90, var_95, var_99],
            marker_color=['#ffb74d', '#ff9800', '#f57c00'],
            text=[f'{v:.2f}%' for v in [var_90, var_95, var_99]],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Confidence Level',
        yaxis_title='Value at Risk (%)',
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # VaR interpretation
    current_equity = equity.iloc[-1]
    var_95_dollar = current_equity * (var_95 / 100)
    
    st.info(
        f"**VaR Interpretation:** With 95% confidence, the maximum expected daily loss "
        f"is {var_95:.2f}% or ${var_95_dollar:,.2f} based on historical data."
    )


def render_risk_adjusted_metrics(equity: pd.Series, risk_metrics: Optional[RiskMetrics]):
    """Render risk-adjusted metrics section."""
    if equity.empty or len(equity) < 10:
        st.info("Insufficient data for risk-adjusted metrics")
        return
    
    returns = calculate_returns(equity)
    
    # Calculate metrics
    sharpe = calculate_sharpe(returns)
    sortino = calculate_sortino(returns)
    
    if risk_metrics:
        calmar = risk_metrics.calmar_ratio
    else:
        from ..metrics import calculate_calmar
        _, max_dd, _ = calculate_drawdown(equity)
        calmar = calculate_calmar(returns, max_dd) if max_dd > 0 else 0
    
    # Gauge charts for ratios
    fig = go.Figure()
    
    # Create comparison bar chart
    metrics = ['Sharpe', 'Sortino', 'Calmar']
    values = [sharpe, sortino, calmar]
    
    # Color based on value (green if > 1, yellow if > 0, red if < 0)
    colors = []
    for v in values:
        if v >= 1:
            colors.append('#00c853')
        elif v >= 0:
            colors.append('#ffb74d')
        else:
            colors.append('#ff1744')
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color=colors,
        text=[f'{v:.2f}' for v in values],
        textposition='auto',
    ))
    
    # Add reference line at 1.0 (good threshold)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Good (1.0)")
    fig.add_hline(y=0, line_dash="solid", line_color="black")
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        yaxis_title='Ratio Value',
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.caption(
        "**Interpretation:** Values > 1.0 indicate good risk-adjusted returns. "
        "Sharpe uses total volatility, Sortino uses downside only, Calmar uses max drawdown."
    )


def render_volatility_section(equity: pd.Series):
    """Render volatility analysis section."""
    if equity.empty or len(equity) < 20:
        st.info("Need at least 20 data points for volatility analysis")
        return
    
    returns = calculate_returns(equity)
    
    # Rolling volatility
    rolling_vol_10 = returns.rolling(10).std() * np.sqrt(252) * 100
    rolling_vol_20 = returns.rolling(20).std() * np.sqrt(252) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_vol_10.index,
        y=rolling_vol_10.values,
        mode='lines',
        name='10-day Vol',
        line=dict(color='#1f77b4', width=1),
    ))
    
    fig.add_trace(go.Scatter(
        x=rolling_vol_20.index,
        y=rolling_vol_20.values,
        mode='lines',
        name='20-day Vol',
        line=dict(color='#ff7f0e', width=2),
    ))
    
    # Current volatility line
    current_vol = calculate_volatility(returns) * 100
    fig.add_hline(
        y=current_vol,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Current: {current_vol:.1f}%",
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Date',
        yaxis_title='Annualized Volatility (%)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Vol", f"{current_vol:.2f}%")
    with col2:
        avg_vol = rolling_vol_20.mean()
        st.metric("Avg Vol (20d)", f"{avg_vol:.2f}%")
    with col3:
        max_vol = rolling_vol_20.max()
        st.metric("Max Vol (20d)", f"{max_vol:.2f}%")


def render_exposure_section(snapshot, risk_metrics: Optional[RiskMetrics]):
    """Render portfolio exposure analysis."""
    col1, col2 = st.columns(2)
    
    with col1:
        # Exposure breakdown
        cash_weight = snapshot.cash_weight
        invested_weight = snapshot.invested_weight
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Cash', 'Invested'],
                y=[cash_weight, invested_weight],
                marker_color=['#90caf9', '#1976d2'],
                text=[f'{v:.1f}%' for v in [cash_weight, invested_weight]],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=30, b=40),
            yaxis_title='Allocation (%)',
            showlegend=False,
            title='Cash vs Invested',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Position concentration
        if snapshot.positions:
            sorted_pos = sorted(snapshot.positions, key=lambda p: p.weight, reverse=True)
            
            # Top positions
            labels = [p.symbol for p in sorted_pos[:5]]
            weights = [p.weight for p in sorted_pos[:5]]
            
            if len(sorted_pos) > 5:
                labels.append('Others')
                weights.append(sum(p.weight for p in sorted_pos[5:]))
            
            fig = px.pie(
                values=weights,
                names=labels,
                hole=0.4,
                title='Position Concentration',
            )
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positions to analyze")
    
    # Exposure metrics
    if risk_metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Gross Exposure", f"{risk_metrics.gross_exposure:.1f}%")
        with col2:
            st.metric("Net Exposure", f"{risk_metrics.net_exposure:.1f}%")
        with col3:
            st.metric("Max Position", f"{risk_metrics.max_position_size:.1f}%")
        with col4:
            leverage = risk_metrics.gross_exposure / 100
            st.metric("Effective Leverage", f"{leverage:.2f}x")


def render_position_risk(snapshot):
    """Render position-level risk analysis."""
    if not snapshot.positions:
        st.info("No positions to analyze")
        return
    
    # Build position risk data
    data = []
    for pos in snapshot.positions:
        # Calculate position-specific metrics
        pnl_risk = abs(pos.unrealized_pnl_pct)
        concentration_risk = pos.weight
        
        # Simple risk score (higher = more risky)
        risk_score = (pnl_risk * 0.3 + concentration_risk * 0.7)
        
        data.append({
            'Symbol': pos.symbol,
            'Weight': pos.weight,
            'P&L %': pos.unrealized_pnl_pct,
            'Value': pos.market_value,
            'Risk Score': risk_score,
        })
    
    df = pd.DataFrame(data)
    
    # Position risk scatter plot
    fig = px.scatter(
        df,
        x='Weight',
        y='P&L %',
        size='Value',
        color='Risk Score',
        text='Symbol',
        color_continuous_scale='RdYlGn_r',
    )
    
    fig.update_traces(textposition='top center')
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Portfolio Weight (%)',
        yaxis_title='Unrealized P&L (%)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk table
    st.subheader("Position Risk Table")
    
    styled_df = df.style.format({
        'Weight': '{:.1f}%',
        'P&L %': '{:+.2f}%',
        'Value': '${:,.2f}',
        'Risk Score': '{:.1f}',
    }).background_gradient(
        subset=['Risk Score'],
        cmap='RdYlGn_r',
    )
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

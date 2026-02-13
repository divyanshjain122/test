"""Trades History Page

Trade log, order history, and execution analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Any, List
from datetime import datetime, timedelta

try:
    from ..models import DashboardState, TradeRecord
except ImportError:
    from jsf.dashboard.models import DashboardState, TradeRecord


def render_trades(state: DashboardState, collector: Optional[Any] = None):
    """Render the trades history page.
    
    Args:
        state: Current dashboard state
        collector: DataCollector instance
    """
    st.title("📋 Trade History")
    
    # Get trade history
    trades = get_trade_history(state, collector)
    
    if not trades:
        st.info("No trades recorded yet. Trades will appear here as they are executed.")
        return
    
    # Summary metrics
    render_trade_summary(trades)
    
    st.markdown("---")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbols = sorted(set(t.symbol for t in trades))
        selected_symbol = st.selectbox(
            "Symbol",
            ["All"] + symbols,
        )
    
    with col2:
        selected_side = st.selectbox(
            "Side",
            ["All", "BUY", "SELL"],
        )
    
    with col3:
        date_range = st.selectbox(
            "Time Period",
            ["All Time", "Today", "Last 7 Days", "Last 30 Days"],
        )
    
    with col4:
        sort_order = st.selectbox(
            "Sort By",
            ["Newest First", "Oldest First", "Largest First"],
        )
    
    # Filter trades
    filtered_trades = filter_trades(
        trades,
        symbol=selected_symbol,
        side=selected_side,
        date_range=date_range,
    )
    
    # Sort trades
    filtered_trades = sort_trades(filtered_trades, sort_order)
    
    st.markdown("---")
    
    # Trade table
    st.subheader(f"Trades ({len(filtered_trades)} of {len(trades)})")
    render_trade_table(filtered_trades)
    
    st.markdown("---")
    
    # Trade analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trades by Symbol")
        render_trades_by_symbol(trades)
    
    with col2:
        st.subheader("Trades Over Time")
        render_trades_over_time(trades)
    
    st.markdown("---")
    
    # Volume analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trade Value Distribution")
        render_trade_value_distribution(trades)
    
    with col2:
        st.subheader("Buy vs Sell Analysis")
        render_buy_sell_analysis(trades)


def get_trade_history(state: DashboardState, collector: Optional[Any]) -> List[TradeRecord]:
    """Get trade history from state or collector.
    
    Args:
        state: Dashboard state
        collector: DataCollector
        
    Returns:
        List of TradeRecord
    """
    # Try collector first
    if collector is not None and hasattr(collector, 'trade_history'):
        if collector.trade_history:
            return collector.trade_history
    
    # Fall back to state
    return state.trade_history


def filter_trades(
    trades: List[TradeRecord],
    symbol: str = "All",
    side: str = "All",
    date_range: str = "All Time",
) -> List[TradeRecord]:
    """Filter trades based on criteria.
    
    Args:
        trades: List of trades
        symbol: Symbol filter
        side: Side filter (BUY/SELL)
        date_range: Time period filter
        
    Returns:
        Filtered list of trades
    """
    filtered = trades
    
    # Symbol filter
    if symbol != "All":
        filtered = [t for t in filtered if t.symbol == symbol]
    
    # Side filter
    if side != "All":
        filtered = [t for t in filtered if t.side.upper() == side]
    
    # Date filter
    now = datetime.now()
    if date_range == "Today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        filtered = [t for t in filtered if t.timestamp >= start]
    elif date_range == "Last 7 Days":
        start = now - timedelta(days=7)
        filtered = [t for t in filtered if t.timestamp >= start]
    elif date_range == "Last 30 Days":
        start = now - timedelta(days=30)
        filtered = [t for t in filtered if t.timestamp >= start]
    
    return filtered


def sort_trades(trades: List[TradeRecord], sort_order: str) -> List[TradeRecord]:
    """Sort trades based on order.
    
    Args:
        trades: List of trades
        sort_order: Sort order string
        
    Returns:
        Sorted list of trades
    """
    if sort_order == "Newest First":
        return sorted(trades, key=lambda t: t.timestamp, reverse=True)
    elif sort_order == "Oldest First":
        return sorted(trades, key=lambda t: t.timestamp)
    elif sort_order == "Largest First":
        return sorted(trades, key=lambda t: t.value, reverse=True)
    return trades


def render_trade_summary(trades: List[TradeRecord]):
    """Render trade summary metrics.
    
    Args:
        trades: List of trades
    """
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_value = sum(t.value for t in trades)
    total_commission = sum(t.commission for t in trades)
    buy_count = sum(1 for t in trades if t.side.upper() == 'BUY')
    sell_count = sum(1 for t in trades if t.side.upper() == 'SELL')
    
    # Calculate realized P&L from trades with P&L data
    trades_with_pnl = [t for t in trades if t.pnl is not None]
    realized_pnl = sum(t.pnl for t in trades_with_pnl) if trades_with_pnl else 0
    
    with col1:
        st.metric("Total Trades", len(trades))
    
    with col2:
        st.metric("Total Value", f"${total_value:,.2f}")
    
    with col3:
        st.metric("Buys", buy_count)
    
    with col4:
        st.metric("Sells", sell_count)
    
    with col5:
        st.metric("Commissions", f"${total_commission:,.2f}")
    
    with col6:
        st.metric("Realized P&L", f"${realized_pnl:+,.2f}")


def render_trade_table(trades: List[TradeRecord]):
    """Render trades as a formatted table.
    
    Args:
        trades: List of trades
    """
    if not trades:
        st.info("No trades match the filter criteria")
        return
    
    # Build DataFrame
    data = []
    for t in trades:
        data.append({
            'Time': t.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Symbol': t.symbol,
            'Side': t.side.upper(),
            'Quantity': t.quantity,
            'Price': t.price,
            'Value': t.value,
            'Commission': t.commission,
            'P&L': t.pnl if t.pnl is not None else None,
            'Order ID': t.order_id[:8] if t.order_id else '-',
        })
    
    df = pd.DataFrame(data)
    
    # Style function for side and P&L coloring
    def style_side(val):
        if val == 'BUY':
            return 'color: #00c853; font-weight: bold'
        elif val == 'SELL':
            return 'color: #ff1744; font-weight: bold'
        return ''
    
    def style_pnl(val):
        if isinstance(val, (int, float)) and not pd.isna(val):
            if val > 0:
                return 'color: #00c853'
            elif val < 0:
                return 'color: #ff1744'
        return ''
    
    # Apply formatting
    styled_df = df.style.format({
        'Price': '${:,.2f}',
        'Value': '${:,.2f}',
        'Commission': '${:.2f}',
        'P&L': lambda x: f'${x:+,.2f}' if x is not None and not pd.isna(x) else '-',
        'Quantity': '{:,.0f}',
    }).applymap(style_side, subset=['Side']).applymap(style_pnl, subset=['P&L'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def render_trades_by_symbol(trades: List[TradeRecord]):
    """Render trades count by symbol pie chart.
    
    Args:
        trades: List of trades
    """
    if not trades:
        return
    
    # Count by symbol
    symbol_counts = {}
    for t in trades:
        symbol_counts[t.symbol] = symbol_counts.get(t.symbol, 0) + 1
    
    df = pd.DataFrame({
        'Symbol': list(symbol_counts.keys()),
        'Count': list(symbol_counts.values()),
    })
    
    fig = px.pie(
        df,
        values='Count',
        names='Symbol',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_trades_over_time(trades: List[TradeRecord]):
    """Render trades over time chart.
    
    Args:
        trades: List of trades
    """
    if not trades:
        return
    
    # Group by date
    df = pd.DataFrame({
        'Date': [t.timestamp.date() for t in trades],
        'Value': [t.value for t in trades],
    })
    
    daily = df.groupby('Date').agg({
        'Value': ['count', 'sum']
    }).reset_index()
    daily.columns = ['Date', 'Count', 'Value']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Trade count bars
    fig.add_trace(
        go.Bar(
            x=daily['Date'],
            y=daily['Count'],
            name='Trade Count',
            marker_color='#1f77b4',
        ),
        secondary_y=False,
    )
    
    # Cumulative value line
    fig.add_trace(
        go.Scatter(
            x=daily['Date'],
            y=daily['Value'].cumsum(),
            name='Cumulative Value',
            line=dict(color='#ff7f0e', width=2),
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=40),
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    fig.update_yaxes(title_text="Trade Count", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Value ($)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)


def render_trade_value_distribution(trades: List[TradeRecord]):
    """Render trade value distribution histogram.
    
    Args:
        trades: List of trades
    """
    if not trades:
        return
    
    values = [t.value for t in trades]
    
    fig = go.Figure(data=[
        go.Histogram(
            x=values,
            nbinsx=20,
            marker_color='#1f77b4',
            opacity=0.7,
        )
    ])
    
    # Add mean line
    mean_val = sum(values) / len(values)
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: ${mean_val:,.0f}",
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title='Trade Value ($)',
        yaxis_title='Frequency',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_buy_sell_analysis(trades: List[TradeRecord]):
    """Render buy vs sell analysis.
    
    Args:
        trades: List of trades
    """
    if not trades:
        return
    
    buys = [t for t in trades if t.side.upper() == 'BUY']
    sells = [t for t in trades if t.side.upper() == 'SELL']
    
    buy_value = sum(t.value for t in buys)
    sell_value = sum(t.value for t in sells)
    
    df = pd.DataFrame({
        'Side': ['BUY', 'SELL'],
        'Count': [len(buys), len(sells)],
        'Value': [buy_value, sell_value],
    })
    
    fig = go.Figure(data=[
        go.Bar(
            name='Count',
            x=df['Side'],
            y=df['Count'],
            marker_color=['#00c853', '#ff1744'],
            text=df['Count'],
            textposition='auto',
        ),
    ])
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        yaxis_title='Trade Count',
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Value comparison
    st.caption(
        f"Buy Value: ${buy_value:,.2f} | "
        f"Sell Value: ${sell_value:,.2f} | "
        f"Net: ${buy_value - sell_value:+,.2f}"
    )


# Import for subplots
from plotly.subplots import make_subplots

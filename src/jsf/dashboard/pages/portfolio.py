"""Portfolio Page

Detailed portfolio analysis with positions breakdown and allocation charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

try:
    from ..models import DashboardState, PositionSnapshot
except ImportError:
    from jsf.dashboard.models import DashboardState, PositionSnapshot


def render_portfolio(state: DashboardState):
    """Render the portfolio details page.
    
    Args:
        state: Current dashboard state
    """
    st.title("💼 Portfolio Details")
    
    snapshot = state.current_snapshot
    
    if snapshot is None:
        st.warning("No portfolio data available.")
        return
    
    # Summary metrics row
    st.subheader("Portfolio Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Equity",
            f"${snapshot.equity:,.2f}",
        )
    
    with col2:
        st.metric(
            "Portfolio Value",
            f"${snapshot.portfolio_value:,.2f}",
        )
    
    with col3:
        st.metric(
            "Cash",
            f"${snapshot.cash:,.2f}",
        )
    
    with col4:
        st.metric(
            "Open Positions",
            snapshot.num_positions,
        )
    
    with col5:
        total_unrealized = sum(p.unrealized_pnl for p in snapshot.positions)
        st.metric(
            "Unrealized P&L",
            f"${total_unrealized:+,.2f}",
        )
    
    st.markdown("---")
    
    # Positions table and charts
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("Positions")
        render_positions_table(snapshot.positions)
    
    with col_right:
        st.subheader("Allocation")
        render_allocation_chart(snapshot)
    
    st.markdown("---")
    
    # Additional analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Position Weights")
        render_weights_bar_chart(snapshot.positions)
    
    with col2:
        st.subheader("P&L by Position")
        render_pnl_bar_chart(snapshot.positions)
    
    st.markdown("---")
    
    # Position details expanders
    st.subheader("Position Details")
    render_position_details(snapshot.positions)


def render_positions_table(positions: list):
    """Render positions as a formatted table.
    
    Args:
        positions: List of PositionSnapshot
    """
    if not positions:
        st.info("No open positions")
        return
    
    # Build DataFrame
    data = []
    for pos in positions:
        data.append({
            'Symbol': pos.symbol,
            'Side': pos.side.upper(),
            'Quantity': pos.quantity,
            'Avg Cost': pos.avg_cost,
            'Price': pos.current_price,
            'Value': pos.market_value,
            'P&L': pos.unrealized_pnl,
            'P&L %': pos.unrealized_pnl_pct,
            'Weight': pos.weight,
        })
    
    df = pd.DataFrame(data)
    
    # Style function for P&L coloring
    def color_pnl(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #00c853'
            elif val < 0:
                return 'color: #ff1744'
        return ''
    
    # Apply formatting
    styled_df = df.style.format({
        'Avg Cost': '${:,.2f}',
        'Price': '${:,.2f}',
        'Value': '${:,.2f}',
        'P&L': '${:+,.2f}',
        'P&L %': '{:+.2f}%',
        'Weight': '{:.1f}%',
        'Quantity': '{:,.0f}',
    }).applymap(color_pnl, subset=['P&L', 'P&L %'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Summary row
    total_value = sum(p.market_value for p in positions)
    total_pnl = sum(p.unrealized_pnl for p in positions)
    total_weight = sum(p.weight for p in positions)
    
    st.caption(
        f"**Total:** Value: ${total_value:,.2f} | "
        f"P&L: ${total_pnl:+,.2f} | "
        f"Weight: {total_weight:.1f}%"
    )


def render_allocation_chart(snapshot):
    """Render allocation pie chart.
    
    Args:
        snapshot: PortfolioSnapshot
    """
    allocation_data = snapshot.get_allocation_data()
    
    if allocation_data.empty:
        st.info("No allocation data")
        return
    
    fig = px.pie(
        allocation_data,
        values='Value',
        names='Asset',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=350,
        margin=dict(l=20, r=20, t=20, b=50),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_weights_bar_chart(positions: list):
    """Render position weights bar chart.
    
    Args:
        positions: List of PositionSnapshot
    """
    if not positions:
        st.info("No positions to display")
        return
    
    # Sort by weight descending
    sorted_positions = sorted(positions, key=lambda p: p.weight, reverse=True)
    
    df = pd.DataFrame({
        'Symbol': [p.symbol for p in sorted_positions],
        'Weight': [p.weight for p in sorted_positions],
    })
    
    fig = px.bar(
        df,
        x='Symbol',
        y='Weight',
        color='Weight',
        color_continuous_scale='Blues',
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=40),
        yaxis_title='Weight (%)',
        showlegend=False,
        coloraxis_showscale=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_pnl_bar_chart(positions: list):
    """Render P&L by position bar chart.
    
    Args:
        positions: List of PositionSnapshot
    """
    if not positions:
        st.info("No positions to display")
        return
    
    # Sort by P&L
    sorted_positions = sorted(positions, key=lambda p: p.unrealized_pnl, reverse=True)
    
    df = pd.DataFrame({
        'Symbol': [p.symbol for p in sorted_positions],
        'P&L': [p.unrealized_pnl for p in sorted_positions],
    })
    
    # Color based on positive/negative
    colors = ['#00c853' if pnl >= 0 else '#ff1744' for pnl in df['P&L']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Symbol'],
            y=df['P&L'],
            marker_color=colors,
        )
    ])
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=40),
        yaxis_title='Unrealized P&L ($)',
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)


def render_position_details(positions: list):
    """Render expandable position details.
    
    Args:
        positions: List of PositionSnapshot
    """
    if not positions:
        st.info("No positions to display")
        return
    
    for pos in positions:
        with st.expander(f"📊 {pos.symbol} - {pos.side.upper()}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Position Details**")
                st.write(f"Quantity: {pos.quantity:,.0f} shares")
                st.write(f"Side: {pos.side.upper()}")
                st.write(f"Average Cost: ${pos.avg_cost:,.2f}")
            
            with col2:
                st.markdown("**Current Values**")
                st.write(f"Current Price: ${pos.current_price:,.2f}")
                st.write(f"Market Value: ${pos.market_value:,.2f}")
                st.write(f"Cost Basis: ${pos.quantity * pos.avg_cost:,.2f}")
            
            with col3:
                st.markdown("**Performance**")
                pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
                st.markdown(f"Unrealized P&L: :{pnl_color}[${pos.unrealized_pnl:+,.2f}]")
                st.markdown(f"P&L %: :{pnl_color}[{pos.unrealized_pnl_pct:+.2f}%]")
                st.write(f"Portfolio Weight: {pos.weight:.1f}%")
            
            # Price vs Cost visualization
            cost_basis = pos.quantity * pos.avg_cost
            market_value = pos.market_value
            
            fig = go.Figure(data=[
                go.Bar(
                    name='Cost Basis',
                    x=['Value'],
                    y=[cost_basis],
                    marker_color='#1f77b4',
                ),
                go.Bar(
                    name='Market Value',
                    x=['Value'],
                    y=[market_value],
                    marker_color='#00c853' if market_value >= cost_basis else '#ff1744',
                )
            ])
            
            fig.update_layout(
                barmode='group',
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            
            st.plotly_chart(fig, use_container_width=True)

"""Streamlit Dashboard Application

Main entry point for the monitoring dashboard.
Run with: streamlit run src/jsf/dashboard/app.py
"""

import streamlit as st
from datetime import datetime
from typing import Optional, Any, Dict
import time

# Check if streamlit is available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Use absolute imports for standalone execution
try:
    from .models import (
        DashboardState,
        DashboardConfig,
        DashboardPage,
        RefreshRate,
    )
    from .collectors import DataCollector, MockDataCollector
except ImportError:
    from jsf.dashboard.models import (
        DashboardState,
        DashboardConfig,
        DashboardPage,
        RefreshRate,
    )
    from jsf.dashboard.collectors import DataCollector, MockDataCollector


def init_session_state():
    """Initialize Streamlit session state."""
    if 'dashboard_state' not in st.session_state:
        st.session_state.dashboard_state = DashboardState()
    
    if 'collector' not in st.session_state:
        st.session_state.collector = None
    
    if 'config' not in st.session_state:
        st.session_state.config = DashboardConfig()
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="JSF Trading Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
    /* Metric cards */
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    
    /* Positive values */
    .positive {
        color: #00c853;
    }
    
    /* Negative values */
    .negative {
        color: #ff1744;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.2em;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 5px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.9em;
    }
    
    /* Status indicator */
    .status-connected {
        color: #00c853;
        font-weight: bold;
    }
    
    .status-disconnected {
        color: #ff1744;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar navigation and controls."""
    with st.sidebar:
        st.title("JSF Dashboard")
        st.markdown("---")
        
        # Connection status
        state = st.session_state.dashboard_state
        if state.is_connected:
            st.markdown("**Status:** Connected")
        else:
            st.markdown("**Status:** Disconnected")
        
        if state.last_update:
            st.caption(f"Last update: {state.last_update.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        
        pages = {
            "Overview": DashboardPage.OVERVIEW,
            "Portfolio": DashboardPage.PORTFOLIO,
            "P&L": DashboardPage.PNL,
            "Trades": DashboardPage.TRADES,
            "Risk": DashboardPage.RISK,
            "Backtest": DashboardPage.BACKTEST,
            "Settings": DashboardPage.SETTINGS,
        }
        
        selected = st.radio(
            "Select Page",
            list(pages.keys()),
            label_visibility="collapsed",
        )
        
        state.current_page = pages[selected]
        
        st.markdown("---")
        
        # Refresh controls
        st.subheader("Refresh")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh", use_container_width=True):
                refresh_data()
        
        with col2:
            st.session_state.auto_refresh = st.checkbox(
                "Auto",
                value=st.session_state.auto_refresh,
            )
        
        if st.session_state.auto_refresh:
            refresh_rate = st.selectbox(
                "Interval",
                options=[
                    ("5 seconds", 5),
                    ("15 seconds", 15),
                    ("30 seconds", 30),
                    ("1 minute", 60),
                ],
                format_func=lambda x: x[0],
            )
            st.session_state.config.refresh_rate = RefreshRate(refresh_rate[1])
        
        st.markdown("---")
        
        # Quick stats
        if state.current_snapshot:
            st.subheader("Quick Stats")
            snapshot = state.current_snapshot
            
            st.metric(
                "Equity",
                f"${snapshot.equity:,.2f}",
                f"{snapshot.daily_return:+.2f}%",
            )
            st.metric(
                "Positions",
                snapshot.num_positions,
            )
            st.metric(
                "Cash",
                f"${snapshot.cash:,.2f}",
                f"{snapshot.cash_weight:.1f}%",
            )


def refresh_data():
    """Refresh data from collector."""
    collector = st.session_state.collector
    
    if collector is not None:
        try:
            snapshot = collector.collect_snapshot()
            st.session_state.dashboard_state.current_snapshot = snapshot
            st.session_state.dashboard_state.last_update = datetime.now()
            st.session_state.dashboard_state.is_connected = True
        except Exception as e:
            st.session_state.dashboard_state.is_connected = False
            st.error(f"Failed to refresh data: {e}")


def render_overview_page():
    """Render the overview/home page."""
    st.title("Portfolio Overview")
    
    state = st.session_state.dashboard_state
    snapshot = state.current_snapshot
    
    if snapshot is None:
        st.warning("No data available. Connect to a broker or use demo mode.")
        
        if st.button("Start Demo Mode"):
            start_demo_mode()
            st.rerun()
        return
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Equity",
            f"${snapshot.equity:,.2f}",
            f"${snapshot.daily_pnl:+,.2f} ({snapshot.daily_return:+.2f}%)",
        )
    
    with col2:
        st.metric(
            "Portfolio Value",
            f"${snapshot.portfolio_value:,.2f}",
            f"{snapshot.invested_weight:.1f}% invested",
        )
    
    with col3:
        st.metric(
            "Cash",
            f"${snapshot.cash:,.2f}",
            f"{snapshot.cash_weight:.1f}% of equity",
        )
    
    with col4:
        st.metric(
            "Total P&L",
            f"${snapshot.total_pnl:+,.2f}",
            f"{snapshot.total_return:+.2f}%",
        )
    
    st.markdown("---")
    
    # Two column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Positions")
        
        if snapshot.positions:
            # Build positions table
            import pandas as pd
            
            data = [pos.to_dict() for pos in snapshot.positions]
            df = pd.DataFrame(data)
            
            # Format columns
            if not df.empty:
                st.dataframe(
                    df.style.format({
                        'Avg Cost': '${:,.2f}',
                        'Price': '${:,.2f}',
                        'Market Value': '${:,.2f}',
                        'P&L': '${:+,.2f}',
                        'P&L %': '{:+.2f}%',
                        'Weight %': '{:.1f}%',
                    }).map(
                        lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 
                        else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
                        subset=['P&L', 'P&L %']
                    ),
                    width="stretch",
                    hide_index=True,
                )
        else:
            st.info("No open positions")
    
    with col_right:
        st.subheader("Allocation")
        
        allocation_data = snapshot.get_allocation_data()
        
        if not allocation_data.empty:
            import plotly.express as px
            
            fig = px.pie(
                allocation_data,
                values='Value',
                names='Asset',
                hole=0.4,
            )
            fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, width="stretch")


def render_portfolio_page():
    """Render detailed portfolio page."""
    try:
        from .components.portfolio import render_portfolio
    except ImportError:
        from jsf.dashboard.components.portfolio import render_portfolio
    render_portfolio(st.session_state.dashboard_state)


def render_pnl_page():
    """Render P&L tracking page."""
    try:
        from .components.pnl import render_pnl
    except ImportError:
        from jsf.dashboard.components.pnl import render_pnl
    render_pnl(st.session_state.dashboard_state, st.session_state.collector)


def render_trades_page():
    """Render trades history page."""
    try:
        from .components.trades import render_trades
    except ImportError:
        from jsf.dashboard.components.trades import render_trades
    render_trades(st.session_state.dashboard_state, st.session_state.collector)


def render_risk_page():
    """Render risk metrics page."""
    try:
        from .components.risk import render_risk
    except ImportError:
        from jsf.dashboard.components.risk import render_risk
    render_risk(st.session_state.dashboard_state, st.session_state.collector)


def render_backtest_page():
    """Render the interactive strategy backtesting page."""
    try:
        from .components.backtest import render_backtest
    except ImportError:
        from jsf.dashboard.components.backtest import render_backtest
    render_backtest()


def render_settings_page():
    """Render settings page."""
    st.title("Settings")
    
    config = st.session_state.config
    
    st.subheader("Display Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config.theme = st.selectbox(
            "Theme",
            ["light", "dark"],
            index=0 if config.theme == "light" else 1,
        )
        
        config.decimal_places = st.number_input(
            "Decimal Places",
            min_value=0,
            max_value=6,
            value=config.decimal_places,
        )
    
    with col2:
        config.chart_height = st.number_input(
            "Chart Height (px)",
            min_value=200,
            max_value=800,
            value=config.chart_height,
        )
        
        config.max_trade_history = st.number_input(
            "Max Trade History",
            min_value=10,
            max_value=10000,
            value=config.max_trade_history,
        )
    
    st.markdown("---")
    
    st.subheader("Notifications")
    
    config.show_notifications = st.checkbox(
        "Show trade notifications",
        value=config.show_notifications,
    )
    
    st.markdown("---")
    
    st.subheader("Data Source")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Demo Mode", use_container_width=True):
            start_demo_mode()
            st.success("Demo mode started!")
            st.rerun()
    
    with col2:
        if st.button("Disconnect", use_container_width=True):
            disconnect()
            st.info("Disconnected")
            st.rerun()
    
    st.markdown("---")
    
    st.subheader("About")
    try:
        from jsf import __version__ as jsf_version
    except ImportError:
        jsf_version = "0.7.2"
    st.markdown(f"""
    **JSF Trading Dashboard** v{jsf_version}
    
    Part of the JBAC Strategy Foundry framework.
    
    For documentation, visit the project repository.
    """)


def start_demo_mode():
    """Start demo mode with mock data."""
    st.session_state.collector = MockDataCollector(
        initial_capital=100000.0,
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"],
        history_days=90,
    )
    
    # Collect initial snapshot (adds to history)
    snapshot = st.session_state.collector.collect_snapshot()
    st.session_state.dashboard_state.current_snapshot = snapshot
    st.session_state.dashboard_state.is_connected = True
    st.session_state.dashboard_state.last_update = datetime.now()
    st.session_state.dashboard_state.initial_capital = 100000.0
    
    # Populate equity history in state from collector history
    st.session_state.dashboard_state.equity_history = [
        (s.timestamp, s.equity) for s in st.session_state.collector.history.snapshots
    ]
    
    # Populate trade history in state from collector
    st.session_state.dashboard_state.trade_history = list(
        st.session_state.collector.trade_history
    )


def disconnect():
    """Disconnect from data source."""
    if st.session_state.collector:
        if hasattr(st.session_state.collector, 'stop_collection'):
            st.session_state.collector.stop_collection()
        st.session_state.collector = None
    
    st.session_state.dashboard_state.is_connected = False
    st.session_state.dashboard_state.current_snapshot = None


def connect_broker(broker: Any, engine: Any = None, initial_capital: float = 100000.0):
    """Connect dashboard to a broker.
    
    Args:
        broker: Broker instance
        engine: Optional LiveTradingEngine
        initial_capital: Starting capital
    """
    st.session_state.collector = DataCollector(
        broker=broker,
        engine=engine,
        initial_capital=initial_capital,
    )
    
    # Collect initial snapshot
    snapshot = st.session_state.collector.collect_snapshot()
    st.session_state.dashboard_state.current_snapshot = snapshot
    st.session_state.dashboard_state.is_connected = True
    st.session_state.dashboard_state.last_update = datetime.now()
    st.session_state.dashboard_state.initial_capital = initial_capital


def main():
    """Main dashboard entry point."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed. Install with: pip install streamlit")
        return
    
    # Setup
    setup_page_config()
    apply_custom_css()
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Auto-refresh logic
    if st.session_state.auto_refresh and st.session_state.collector:
        config = st.session_state.config
        now = datetime.now()
        elapsed = (now - st.session_state.last_refresh).total_seconds()
        
        if elapsed >= config.refresh_rate.value and config.refresh_rate.value > 0:
            refresh_data()
            st.session_state.last_refresh = now
            time.sleep(0.1)
            st.rerun()
    
    # Render current page
    state = st.session_state.dashboard_state
    page = state.current_page
    
    if page == DashboardPage.OVERVIEW:
        render_overview_page()
    elif page == DashboardPage.PORTFOLIO:
        render_portfolio_page()
    elif page == DashboardPage.PNL:
        render_pnl_page()
    elif page == DashboardPage.TRADES:
        render_trades_page()
    elif page == DashboardPage.RISK:
        render_risk_page()
    elif page == DashboardPage.BACKTEST:
        render_backtest_page()
    elif page == DashboardPage.SETTINGS:
        render_settings_page()


# Run with: streamlit run src/jsf/dashboard/app.py
if __name__ == "__main__":
    main()

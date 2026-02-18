"""Backtest Dashboard Page

Interactive strategy backtesting UI for the JSF dashboard.
Allows users to configure, run, and inspect backtests from the browser.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_backtest_deps():
    """Import heavy backtesting dependencies (deferred to avoid slow cold start)."""
    try:
        from jsf.data import load_data
        from jsf.strategies import MomentumStrategy, MeanReversionStrategy, TrendFollowingStrategy
        from jsf.simulation import BacktestEngine, BacktestConfig, calculate_all_metrics
        return load_data, MomentumStrategy, MeanReversionStrategy, TrendFollowingStrategy, BacktestEngine, BacktestConfig, calculate_all_metrics
    except ImportError as exc:
        st.error(f"Could not import backtest dependencies: {exc}")
        return None


def _build_strategy(strategy_name: str, params: dict):
    """Instantiate the selected strategy class with UI-supplied params."""
    _, MomentumStrategy, MeanReversionStrategy, TrendFollowingStrategy, _, _, _ = _load_backtest_deps()
    if strategy_name == "Momentum":
        return MomentumStrategy(
            name="momentum",
            lookback=params["lookback"],
            long_only=params["long_only"],
        )
    elif strategy_name == "Mean Reversion":
        return MeanReversionStrategy(
            name="mean_reversion",
            lookback=params["lookback"],
            long_only=params.get("long_only", True),
        )
    elif strategy_name == "Trend Following":
        return TrendFollowingStrategy(
            name="trend_following",
            fast_period=params["fast_period"],
            slow_period=params["slow_period"],
        )
    raise ValueError(f"Unknown strategy: {strategy_name}")


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _equity_chart(equity_curve: pd.Series, initial_capital: float) -> go.Figure:
    """Return a Plotly figure for the equity curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode="lines",
        name="Equity",
        line=dict(color="#1f77b4", width=2),
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.08)",
    ))
    # Horizontal baseline at initial capital
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="grey",
        annotation_text="Initial Capital",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=350,
        margin=dict(l=60, r=30, t=50, b=50),
        yaxis=dict(tickformat="$,.0f"),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _drawdown_chart(equity_curve: pd.Series) -> go.Figure:
    """Return a Plotly figure for the rolling drawdown."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100  # as percent

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode="lines",
        name="Drawdown",
        line=dict(color="#d62728", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(214,39,40,0.12)",
    ))
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=250,
        margin=dict(l=60, r=30, t=50, b=50),
        yaxis=dict(tickformat=".1f", ticksuffix="%"),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _returns_histogram(returns: pd.Series) -> go.Figure:
    """Daily returns distribution."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns.values * 100,
        nbinsx=40,
        name="Daily Returns",
        marker_color="#1f77b4",
        opacity=0.75,
    ))
    fig.update_layout(
        title="Return Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Count",
        height=250,
        margin=dict(l=60, r=30, t=50, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def render_backtest(_state=None):
    """Render the backtest configuration and results page."""

    st.title("Strategy Backtester")
    st.markdown(
        "Configure a strategy, choose a date range and universe, then click "
        "**Run Backtest** to see full performance analytics."
    )

    deps = _load_backtest_deps()
    if deps is None:
        return
    load_data, MomentumStrategy, MeanReversionStrategy, TrendFollowingStrategy, BacktestEngine, BacktestConfig, calculate_all_metrics = deps

    # -----------------------------------------------------------------------
    # Configuration panel
    # -----------------------------------------------------------------------
    with st.expander("Backtest Configuration", expanded=True):
        col_strategy, col_universe = st.columns([1, 2])

        with col_strategy:
            st.subheader("Strategy")
            strategy_name = st.selectbox(
                "Strategy",
                ["Momentum", "Mean Reversion", "Trend Following"],
                key="bt_strategy",
            )

            if strategy_name == "Momentum":
                lookback = st.slider("Lookback Period (days)", 10, 200, 20, key="bt_lookback_mom")
                long_only = st.checkbox("Long Only", value=True, key="bt_longonly_mom")
                strategy_params = {"lookback": lookback, "long_only": long_only}

            elif strategy_name == "Mean Reversion":
                lookback = st.slider("Lookback Period (days)", 5, 60, 20, key="bt_lookback_mr")
                long_only = st.checkbox("Long Only", value=True, key="bt_longonly_mr")
                strategy_params = {"lookback": lookback, "long_only": long_only}

            else:  # Trend Following
                fast_period = st.slider("Fast MA Period (days)", 5, 100, 10, key="bt_fast")
                slow_period = st.slider("Slow MA Period (days)", 50, 300, 30, key="bt_slow")
                if fast_period >= slow_period:
                    st.warning("Fast period must be less than Slow period.")
                strategy_params = {"fast_period": fast_period, "slow_period": slow_period}

        with col_universe:
            st.subheader("Universe & Dates")

            available_symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA",
                "META", "NFLX", "AMD", "INTC", "JPM", "GS",
            ]
            symbols = st.multiselect(
                "Symbols",
                available_symbols,
                default=["AAPL", "MSFT", "GOOGL"],
                key="bt_symbols",
            )
            if not symbols:
                st.warning("Select at least one symbol.")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_date = st.date_input(
                    "Start Date",
                    value=date.today() - timedelta(days=730),
                    key="bt_start",
                )
            with col_d2:
                end_date = st.date_input(
                    "End Date",
                    value=date.today() - timedelta(days=1),
                    key="bt_end",
                )

            if start_date >= end_date:
                st.error("Start date must be before end date.")

        # -----------------------------------------------------------------------
        st.subheader("Execution Parameters")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=10_000_000,
                value=100_000,
                step=10_000,
                key="bt_capital",
            )
        with col_c2:
            transaction_cost = st.number_input(
                "Transaction Cost",
                min_value=0.0,
                max_value=0.05,
                value=0.001,
                step=0.0005,
                format="%.4f",
                key="bt_tc",
            )
        with col_c3:
            slippage = st.number_input(
                "Slippage",
                min_value=0.0,
                max_value=0.02,
                value=0.0005,
                step=0.0001,
                format="%.4f",
                key="bt_slip",
            )
        with col_c4:
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                key="bt_seed",
            )

    # -----------------------------------------------------------------------
    # Run button
    # -----------------------------------------------------------------------
    run_disabled = not symbols or start_date >= end_date
    if strategy_name == "Trend Following" and strategy_params["fast_period"] >= strategy_params["slow_period"]:
        run_disabled = True

    if st.button("Run Backtest", type="primary", disabled=run_disabled):
        with st.spinner("Loading data and running backtest…"):
            try:
                price_data = load_data(
                    source="synthetic",
                    symbols=symbols,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    seed=int(random_seed),
                )

                strategy = _build_strategy(strategy_name, strategy_params)

                config = BacktestConfig(
                    initial_capital=float(initial_capital),
                    transaction_cost=float(transaction_cost),
                    slippage=float(slippage),
                )

                engine = BacktestEngine(config)
                result = engine.run_strategy(strategy, price_data)

                # Persist in session state
                st.session_state["bt_result"] = result
                st.session_state["bt_initial_capital"] = float(initial_capital)
                st.session_state["bt_strategy_label"] = f"{strategy_name} | {symbols} | {start_date} → {end_date}"

            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                st.session_state.pop("bt_result", None)

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    if "bt_result" not in st.session_state:
        st.info("Configure the options above and click **Run Backtest** to see results.")
        return

    result = st.session_state["bt_result"]
    cap = st.session_state.get("bt_initial_capital", 100_000)
    label = st.session_state.get("bt_strategy_label", "")

    st.markdown("---")
    st.subheader(f"Results — {label}")

    # Top-level KPI row
    col1, col2, col3, col4, col5 = st.columns(5)

    def _pct(v):
        return f"{v * 100:+.2f}%"

    def _val(v, fmt=".2f"):
        return f"{v:{fmt}}"

    col1.metric("Total Return", _pct(result.total_return))
    col2.metric("CAGR", _pct(result.cagr))
    col3.metric("Sharpe Ratio", _val(result.sharpe_ratio))
    col4.metric("Max Drawdown", _pct(result.max_drawdown))
    col5.metric("Volatility (ann.)", _pct(result.volatility))

    st.markdown("---")

    # Equity + drawdown charts side by side
    chart_left, chart_right = st.columns([3, 2])
    with chart_left:
        st.plotly_chart(_equity_chart(result.equity_curve, cap), width="stretch")

    with chart_right:
        st.plotly_chart(_drawdown_chart(result.equity_curve), width="stretch")
        st.plotly_chart(_returns_histogram(result.returns), width="stretch")

    # Full metrics table
    st.markdown("---")
    st.subheader("Full Metrics")

    try:
        metrics = calculate_all_metrics(result.returns)
        metrics_display = {
            "Total Return": f"{metrics.get('total_return', 0) * 100:+.2f}%",
            "CAGR": f"{metrics.get('cagr', 0) * 100:+.2f}%",
            "Mean Daily Return": f"{metrics.get('mean_return', 0) * 100:+.4f}%",
            "Volatility (ann.)": f"{metrics.get('volatility', 0) * 100:.2f}%",
            "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.3f}",
            "Sortino Ratio": f"{metrics.get('sortino_ratio', 0):.3f}",
            "Calmar Ratio": f"{metrics.get('calmar_ratio', 0):.3f}",
            "Max Drawdown": f"{metrics.get('max_drawdown', 0) * 100:+.2f}%",
            "VaR (95%)": f"{metrics.get('var_95', 0) * 100:+.2f}%",
            "CVaR (95%)": f"{metrics.get('cvar_95', 0) * 100:+.2f}%",
            "Win Rate": f"{metrics.get('win_rate', 0) * 100:.1f}%",
            "Profit Factor": f"{metrics.get('profit_factor', 0):.3f}",
            "Best Day": f"{metrics.get('best_day', 0) * 100:+.2f}%",
            "Worst Day": f"{metrics.get('worst_day', 0) * 100:+.2f}%",
            "# Periods": str(int(metrics.get("num_periods", 0))),
            "# Positive Periods": str(int(metrics.get("num_positive", 0))),
            "# Negative Periods": str(int(metrics.get("num_negative", 0))),
            "Skewness": f"{metrics.get('skewness', 0):.3f}",
            "Kurtosis": f"{metrics.get('kurtosis', 0):.3f}",
        }

        half = len(metrics_display) // 2
        keys = list(metrics_display.keys())

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.table(pd.DataFrame(
                {"Metric": keys[:half], "Value": [metrics_display[k] for k in keys[:half]]}
            ).set_index("Metric"))
        with col_m2:
            st.table(pd.DataFrame(
                {"Metric": keys[half:], "Value": [metrics_display[k] for k in keys[half:]]}
            ).set_index("Metric"))

    except Exception as exc:
        st.warning(f"Could not compute extended metrics: {exc}")

    # Trades table
    st.markdown("---")
    st.subheader("Trade Log")

    trades_df = result.trades
    if trades_df is not None and not trades_df.empty:
        st.caption(f"{len(trades_df)} trades recorded")

        # Format the dataframe nicely before display
        display_df = trades_df.copy()
        if "cost" in display_df.columns:
            display_df["cost"] = display_df["cost"].apply(lambda x: f"${x:,.4f}")
        if "change" in display_df.columns:
            display_df["change"] = display_df["change"].apply(lambda x: f"{x:+.4f}")

        st.dataframe(display_df, width="stretch", hide_index=True)

        # Download CSV
        csv = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Trade Log (CSV)",
            csv,
            file_name="backtest_trades.csv",
            mime="text/csv",
        )
    else:
        st.info("No trades were recorded during this backtest.")

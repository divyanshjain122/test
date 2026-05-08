import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, 'src')

from jsf.data import PriceData
from jsf.data.sources.yahoo import YahooFinanceLoader
from jsf.ml import (
    FeatureExtractor, RandomForestModel, MLStrategy, 
    FinBERT, handle_missing_features, create_target_variable
)
from jsf.ml.preprocessing import TargetType
from jsf.signals import MeanReversionSignal, MomentumSignal
from jsf.portfolio import Portfolio
from jsf.simulation import BacktestConfig, BacktestEngine, calculate_all_metrics

# --- CONFIGURATION ---
warnings.filterwarnings('ignore')
OUTPUT_DIR = Path("./resultLogs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    'AAPL', 'NVDA', 'AMZN', 'TSLA', 'MSFT', 'SHW', 
    'JPM', 'BAC', 'WFC', 'HD', 'V', 'GS', 'PG', 'XOM', 'DIS'
]
INITIAL_CAPITAL = 100_000.0

def load_news_data(file_path, symbols, start_date, end_date):
    """Loads and cleans news for FinBERT."""
    if not os.path.exists(file_path):
        print(f"⚠️ News file {file_path} not found. Sentiment approaches will be zeroed.")
        return pd.DataFrame(columns=["date", "symbol", "text", "source"])

    df = pd.read_csv(file_path)
    col_map = {'Stock_symbol': 'symbol', 'Date': 'date', 'Article_title': 'text', 'Publisher': 'source'}
    df = df.rename(columns=col_map)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"].isin(symbols)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    return df.loc[mask].dropna(subset=["date", "text"])

def build_long_only_weights(signals, threshold=0.01):
    """Normalizes signals into portfolio weights (Long Only)."""
    positive = signals.clip(lower=0.0).where(signals >= threshold, 0.0)
    denom = positive.sum(axis=1)
    weights = positive.div(denom, axis=0).fillna(0.0)
    return weights.shift(1).fillna(0.0)

def run_backtest(name, signals, price_data):
    """Standardized backtest execution."""
    weights = build_long_only_weights(signals)
    portfolio = Portfolio(weights=weights, metadata={"name": name})
    engine = BacktestEngine(BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=0.001,
        slippage=0.0005
    ))
    result = engine.run(portfolio, price_data)
    metrics = calculate_all_metrics(result.returns.fillna(0.0))
    add_return_metrics(metrics, result)
    return result, metrics, portfolio

def export_daily_matrix(matrix, file_name, value_name, scale=1.0):
    """Export a date x symbol matrix as a daily long-form CSV."""
    export_df = (matrix * scale).copy()
    export_df.index.name = "date"
    export_df = export_df.stack().rename(value_name).reset_index()
    export_df.columns = ["date", "symbol", value_name]
    export_df.to_csv(OUTPUT_DIR / file_name, index=False)

def export_portfolio_weights(name, portfolio, price_index, symbols):
    """Export daily stock holding weights as percentages."""
    weights = portfolio.weights.reindex(index=price_index, columns=symbols).fillna(0.0)
    export_daily_matrix(
        weights,
        f"{name}_daily_holding_weights.csv",
        "holding_weight_pct",
        scale=100.0,
    )

def add_return_metrics(metrics, result):
    """Add gross and net return metrics to a backtest result."""
    metrics["gross_total_return"] = metrics["total_return"]
    metrics["final_value"] = result.equity_curve.iloc[-1]
    metrics["net_total_return"] = (metrics["final_value"] / INITIAL_CAPITAL) - 1
    metrics["total_return"] = metrics["net_total_return"]
    return metrics

def create_ml_strategy(warmup_days):
    """Create a fresh ML strategy so model state cannot leak between runs."""
    extractor = FeatureExtractor(
        feature_groups=['momentum', 'volatility', 'trend', 'mean_reversion'],
        lag_periods=[1, 5, 10],
        normalize=True,
        rank=True
    )
    ml_model = RandomForestModel(n_estimators=100, max_depth=5, prediction_type='regression')
    ml_strategy = MLStrategy(
        name='ml_engine',
        model=ml_model,
        feature_extractor=extractor,
        prediction_type='returns',
        retrain_frequency=63, # Quarterly retraining
        long_only=True
    )
    ml_strategy.config.warmup_period = warmup_days
    return ml_strategy

def main():
    # 1. SETUP & DATA LOADING
    start_date = "2019-01-01"
    end_date = "2019-12-31"
    warmup_days =  420 # Increased for better ML training window
    
    print(f"🚀 Initializing 4-Way Unified Pipeline: {start_date} to {end_date}")
    
    # Load Prices
    price_start = (pd.to_datetime(start_date) - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    loader = YahooFinanceLoader(symbols=SYMBOLS, start_date=price_start, end_date=end_date)
    df_prices = loader.load()
    full_price_data = PriceData(df_prices)
    
    # Slice for the backtest period
    backtest_price_data = PriceData(df_prices.loc[start_date:end_date])
    close_prices = backtest_price_data.get_close_prices()

    # Load News
    news_df = load_news_data("filtered_stock_data.csv", SYMBOLS, start_date, end_date)
    
    # 2. SENTIMENT ANALYSIS
    print("\n[Step 1] Processing Sentiment Signals...")
    finbert = FinBERT(use_mock=False)
    sentences = news_df["text"].tolist()
    
    if sentences:
        sent_results = finbert.predict(sentences)
        news_df["sentiment_score"] = [
            r.score if r.label.value == "positive" else -r.score if r.label.value == "negative" else 0.0 
            for r in sent_results
        ]
        daily_sent = news_df.groupby(["date", "symbol"])["sentiment_score"].mean().unstack()
        daily_sent = daily_sent.reindex(index=close_prices.index, columns=close_prices.columns).fillna(0.0)
        sentiment_signal = daily_sent.rolling(window=3, min_periods=1).mean()
        sentiment_signal = sentiment_signal.where(sentiment_signal.abs() >= 0.15, 0.0)
    else:
        sentiment_signal = pd.DataFrame(0.0, index=close_prices.index, columns=close_prices.columns)

    export_daily_matrix(
        sentiment_signal,
        "daily_sentiment_scores.csv",
        "sentiment_score",
    )

    # 3. TECHNICAL SIGNALS (Baseline Component)
    print("[Step 2] Generating Technical Signals...")
    mom = MomentumSignal(lookback=20).generate(full_price_data).reindex(close_prices.index).fillna(0.0)
    mrev = MeanReversionSignal(lookback=10).generate(full_price_data).reindex(close_prices.index).fillna(0.0)
    sig_baseline = (0.6 * mom + 0.4 * mrev)

    # 4. ML WALK-FORWARD SIGNALS (ML Only & ML + Sentiment Component)
    print("[Step 3] Running ML Walk-Forward Strategy...")
    ml_strategy = create_ml_strategy(warmup_days)
###############
    full_close_prices = full_price_data.get_close_prices()
    pretrade_dates = full_close_prices.index[full_close_prices.index < pd.to_datetime(start_date)]
    ml_strategy.config.warmup_period = max(0, len(pretrade_dates) - 1)

    sig_ml_only = ml_strategy.generate_signals(full_price_data).reindex(close_prices.index).fillna(0.0)

    # 5. INTEGRATE & BACKTEST ALL 4 APPROACHES
    print("[Step 4] Executing Backtests...")
    
    # Approach 1: Baseline (Momentum + Mean Reversion)
    res_1, met_1, portfolio_1 = run_backtest("Baseline_Tech", sig_baseline, backtest_price_data)
    export_portfolio_weights("baseline_tech", portfolio_1, close_prices.index, close_prices.columns)
    
    # Approach 2: Tech + Sentiment
    sig_tech_sent = (0.75 * sig_baseline + 0.25 * sentiment_signal)
    res_2, met_2, portfolio_2 = run_backtest("Tech_Sentiment", sig_tech_sent, backtest_price_data)
    export_portfolio_weights("tech_sentiment", portfolio_2, close_prices.index, close_prices.columns)
    
    # Approach 3: ML Only (No sentiment filter)
    engine = BacktestEngine(BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=0.001,
        slippage=0.0005
    ))
    portfolio_ml_only = ml_strategy.construct_portfolio(sig_ml_only, backtest_price_data)
    export_portfolio_weights("ml_only", portfolio_ml_only, close_prices.index, close_prices.columns)
    res_3 = engine.run(portfolio_ml_only, backtest_price_data)
    met_3 = calculate_all_metrics(res_3.returns.fillna(0.0))
    add_return_metrics(met_3, res_3)
    
    # Approach 4: ML + Sentiment (The Full Ensemble)
    sig_ml_sent = (0.70 * sig_ml_only + 0.30 * sentiment_signal)
    portfolio_ml_sent = ml_strategy.construct_portfolio(sig_ml_sent, backtest_price_data)
    export_portfolio_weights("ml_sentiment", portfolio_ml_sent, close_prices.index, close_prices.columns)
    res_4 = engine.run(portfolio_ml_sent, backtest_price_data)
    met_4 = calculate_all_metrics(res_4.returns.fillna(0.0))
    add_return_metrics(met_4, res_4)

    # 6. FINAL COMPARISON
    print("\n" + "="*110)
    print(f"{'STRATEGY':<25} | {'FINAL VALUE':<15} | {'SHARPE':<10} | {'GROSS RETURN':<12} | {'NET RETURN':<12}")
    print("-" * 110)
    
    strategies = [
        ("1. Baseline Tech", met_1),
        ("2. Tech + Sentiment", met_2),
        ("3. ML Only", met_3),
        ("4. ML + Sentiment", met_4)
    ]
    
    for name, m in strategies:
        print(
            f"{name:<25} | ${m['final_value']:>14,.2f} | {m['sharpe_ratio']:>9.2f} | "
            f"{m['gross_total_return']:>11.2%} | {m['net_total_return']:>11.2%}"
        )
    print("="*110)

    # Export metrics comparison
    comparison = pd.DataFrame({
        "Baseline": met_1,
        "Tech_Sentiment": met_2,
        "ML_Only": met_3,
        "ML_Sentiment": met_4
    })
    comparison.to_csv(OUTPUT_DIR / "four_way_strategy_comparison.csv")
    print(f"\n✅ All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
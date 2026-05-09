import os
import sys
import warnings
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import random

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
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
random_suffix = str(random.randint(1000, 9999))
# Construct the directory path
OUTPUT_DIR = Path("./resultLogs") / (RUN_TIMESTAMP + "rf-200-8-regression-" + random_suffix)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = Path("finbert_sentiment_cache.json") # Added Cache Reference

SYMBOLS = [
    'AAPL', 'NVDA', 'AMZN', 'TSLA', 'MSFT', 'SHW', 
    'JPM', 'BAC', 'WFC', 'HD', 'V', 'GS', 'PG', 'XOM', 'DIS'
]
INITIAL_CAPITAL = 100_000.0

# --- NEW CACHE HELPER ---
def get_text_hash(text):
    """Creates a unique MD5 hash for a string to use as a cache key."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

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

def export_date_wise_news(news_df):
    """Export cleaned news articles and daily stock-level news counts."""
    detail_columns = ["date", "symbol", "source", "text"]
    if "sentiment_score" in news_df.columns:
        detail_columns.append("sentiment_score")

    if news_df.empty:
        pd.DataFrame(columns=detail_columns).to_csv(
            OUTPUT_DIR / "date_wise_stock_news.csv",
            index=False,
        )
        pd.DataFrame(columns=["date", "symbol", "news_count", "avg_sentiment_score"]).to_csv(
            OUTPUT_DIR / "date_wise_stock_news_summary.csv",
            index=False,
        )
        return

    news_log = news_df.copy()
    news_log = news_log.sort_values(["date", "symbol", "source", "text"])
    news_log[detail_columns].to_csv(OUTPUT_DIR / "date_wise_stock_news.csv", index=False)

    aggregations = {"text": "count"}
    if "sentiment_score" in news_log.columns:
        aggregations["sentiment_score"] = "mean"

    summary = news_log.groupby(["date", "symbol"], as_index=False).agg(aggregations)
    summary = summary.rename(columns={"text": "news_count", "sentiment_score": "avg_sentiment_score"})
    if "avg_sentiment_score" not in summary.columns:
        summary["avg_sentiment_score"] = np.nan
    summary.to_csv(OUTPUT_DIR / "date_wise_stock_news_summary.csv", index=False)

def export_run_details(start_date, end_date, price_start, warmup_days):
    """Export run configuration details for reproducibility."""
    details = {
        "run_timestamp": RUN_TIMESTAMP,
        "output_directory": str(OUTPUT_DIR),
        "symbols": SYMBOLS,
        "start_date": start_date,
        "end_date": end_date,
        "price_start": price_start,
        "warmup_days": warmup_days,
        "initial_capital": INITIAL_CAPITAL,
        "transaction_cost": 0.001,
        "slippage": 0.0005,
        "news_file": "filtered_stock_data.csv",
        "strategies": [
            "Baseline Tech",
            "Tech + Sentiment",
            "ML Only",
            "ML + Sentiment",
        ],
    }
    with open(OUTPUT_DIR / "run_details.json", "w", encoding="utf-8") as file:
        json.dump(details, file, indent=2)

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
    ml_model = RandomForestModel(n_estimators=200, max_depth=8, prediction_type='regression')
    ml_strategy = MLStrategy(
        name='ml_engine',
        model=ml_model,
        feature_extractor=extractor,
        prediction_type='returns',
        retrain_frequency=63, 
        long_only=True
    )
    ml_strategy.config.warmup_period = warmup_days
    return ml_strategy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default="2019-01-01")
    parser.add_argument("--end_date", type=str, default="2019-12-31")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    warmup_days = 420 
    
    print(f"🚀 Initializing 4-Way Unified Pipeline: {start_date} to {end_date}")
    
    price_start = (pd.to_datetime(start_date) - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    export_run_details(start_date, end_date, price_start, warmup_days)
    loader = YahooFinanceLoader(symbols=SYMBOLS, start_date=price_start, end_date=end_date)
    df_prices = loader.load()
    
    full_price_data = PriceData(df_prices)
    backtest_price_data = PriceData(df_prices.loc[start_date:end_date])
    close_prices = backtest_price_data.get_close_prices()

    news_df = load_news_data("filtered_stock_data.csv", SYMBOLS, start_date, end_date)
    
    # 2. CACHE-AWARE SENTIMENT ANALYSIS
    print("\n[Step 1] Processing Sentiment Signals...")
    
    sentiment_cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            sentiment_cache = json.load(f)

    if not news_df.empty:
        news_df["text_hash"] = news_df["text"].apply(get_text_hash)
        unique_items = news_df.drop_duplicates(subset=["text_hash"])
        
        # Check for items not in cache
        missing_mask = ~unique_items["text_hash"].isin(sentiment_cache.keys())
        items_to_process = unique_items[missing_mask]

        if not items_to_process.empty:
            print(f"--- FinBERT: Processing {len(items_to_process)} new articles ---")
            finbert = FinBERT(use_mock=False)
            sentences = items_to_process["text"].tolist()
            hashes = items_to_process["text_hash"].tolist()
            
            sent_results = finbert.predict(sentences)
            
            for h, r in zip(hashes, sent_results):
                score = r.score if r.label.value == "positive" else -r.score if r.label.value == "negative" else 0.0
                sentiment_cache[h] = score
            
            with open(CACHE_FILE, 'w') as f:
                json.dump(sentiment_cache, f)
        else:
            print("--- All news sentiment retrieved from cache ---")

        # Map scores back to the full news dataframe
        news_df["sentiment_score"] = news_df["text_hash"].map(sentiment_cache)
        export_date_wise_news(news_df)
        
        daily_sent = news_df.groupby(["date", "symbol"])["sentiment_score"].mean().unstack()
        daily_sent = daily_sent.reindex(index=close_prices.index, columns=close_prices.columns).fillna(0.0)
        sentiment_signal = daily_sent.rolling(window=3, min_periods=1).mean()
        sentiment_signal = sentiment_signal.where(sentiment_signal.abs() >= 0.15, 0.0)
    else:
        export_date_wise_news(news_df)
        sentiment_signal = pd.DataFrame(0.0, index=close_prices.index, columns=close_prices.columns)

    export_daily_matrix(
        sentiment_signal,
        "daily_sentiment_scores.csv",
        "sentiment_score",
    )

    # 3. TECHNICAL SIGNALS
    print("[Step 2] Generating Technical Signals...")
    mom = MomentumSignal(lookback=20).generate(full_price_data).reindex(close_prices.index).fillna(0.0)
    mrev = MeanReversionSignal(lookback=10).generate(full_price_data).reindex(close_prices.index).fillna(0.0)
    sig_baseline = (0.6 * mom + 0.4 * mrev)

    # 4. ML WALK-FORWARD SIGNALS
    print("[Step 3] Running ML Walk-Forward Strategy...")
    ml_strategy = create_ml_strategy(warmup_days)
    full_close_prices = full_price_data.get_close_prices()
    pretrade_dates = full_close_prices.index[full_close_prices.index < pd.to_datetime(start_date)]
    ml_strategy.config.warmup_period = max(0, len(pretrade_dates) - 1)

    sig_ml_only = ml_strategy.generate_signals(full_price_data).reindex(close_prices.index).fillna(0.0)

    # 5. BACKTESTS
    print("[Step 4] Executing Backtests...")
    
    # Approach 1: Baseline
    res_1, met_1, portfolio_1 = run_backtest("Baseline_Tech", sig_baseline, backtest_price_data)
    export_portfolio_weights("baseline_tech", portfolio_1, close_prices.index, close_prices.columns)
    
    # Approach 2: Tech + Sentiment
    sig_tech_sent = (0.75 * sig_baseline + 0.25 * sentiment_signal)
    res_2, met_2, portfolio_2 = run_backtest("Tech_Sentiment", sig_tech_sent, backtest_price_data)
    export_portfolio_weights("tech_sentiment", portfolio_2, close_prices.index, close_prices.columns)
    
    # Approach 3: ML Only
    portfolio_ml_only = ml_strategy.construct_portfolio(sig_ml_only, backtest_price_data)
    export_portfolio_weights("ml_only", portfolio_ml_only, close_prices.index, close_prices.columns)
    engine_ml = BacktestEngine(BacktestConfig(initial_capital=INITIAL_CAPITAL, transaction_cost=0.001, slippage=0.0005))
    res_3 = engine_ml.run(portfolio_ml_only, backtest_price_data)
    met_3 = calculate_all_metrics(res_3.returns.fillna(0.0))
    add_return_metrics(met_3, res_3)
    
    # Approach 4: ML + Sentiment
    sig_ml_sent = (0.70 * sig_ml_only + 0.30 * sentiment_signal)
    portfolio_ml_sent = ml_strategy.construct_portfolio(sig_ml_sent, backtest_price_data)
    export_portfolio_weights("ml_sentiment", portfolio_ml_sent, close_prices.index, close_prices.columns)
    res_4 = engine_ml.run(portfolio_ml_sent, backtest_price_data)
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

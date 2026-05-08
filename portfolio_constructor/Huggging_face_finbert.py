import argparse
import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Environment setup for stability
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

from jsf.data import PriceData
from jsf.data.sources.yahoo import YFINANCE_AVAILABLE, YahooFinanceLoader
from jsf.portfolio import Portfolio
from jsf.signals import MeanReversionSignal, MomentumSignal
from jsf.simulation import BacktestConfig, BacktestEngine, calculate_all_metrics

# --- CONFIGURATION ---
OUTPUT_DIR = Path(__file__).resolve().parent / "resultLogs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = "real_data_finbert_backtest"
SUBSET_NEWS_FILE = "filtered_stock_data.csv" 

SYMBOLS = [
    'AAPL', 'NVDA', 'AMZN', 'TSLA', 'BRK.B', 'MSFT', 'SHW', 
    'JPM', 'BAC', 'WFC', 'HD', 'V', 'GS', 'PG', 'XOM', 'DIS'
]

INITIAL_CAPITAL = 100_000.0

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest FNSPID Subset with Script 1 Logic.")
    parser.add_argument("--start-date", default="2022-01-01", help="YYYY-MM-DD backtest start")
    parser.add_argument("--end-date", default="2023-01-01", help="YYYY-MM-DD backtest end")
    parser.add_argument("--dataset-file", default=SUBSET_NEWS_FILE, help="Path to your filtered CSV")
    parser.add_argument("--price-warmup-days", type=int, default=60)
    return parser.parse_args()

def load_subset_news(file_path, symbols, start_date, end_date):
    """Keep the exact logic for your local DuckDB-filtered subset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Subset news file not found: {file_path}")

    df = pd.read_csv(file_path)
    col_map = {
        'Stock_symbol': 'symbol',
        'Date': 'date',
        'Article_title': 'text',
        'Publisher': 'source'
    }
    df = df.rename(columns=col_map)
    
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"].isin(symbols)]

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_convert(None).dt.normalize()
    df = df.dropna(subset=["date"])

    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    df = df.loc[mask]

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""]
    
    if 'source' not in df.columns:
        df["source"] = "FNSPID_Subset"
        
    return df[["date", "symbol", "text", "source"]].drop_duplicates()

def build_long_only_weights(signals, threshold=0.05):
    """Script 1 style weight calculation."""
    positive = signals.clip(lower=0.0).where(signals >= threshold, 0.0)
    weights = positive.div(positive.sum(axis=1), axis=0).fillna(0.0)
    return weights.shift(1).fillna(0.0)

def run_signal_backtest(name, signals, price_data):
    """Backtest engine logic with Script 1 parameters (costs/slippage)."""
    weights = build_long_only_weights(signals)
    portfolio = Portfolio(weights=weights, metadata={"name": name})
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            transaction_cost=0.001,  # 0.1% as per Script 1
            slippage=0.0005,         # 0.05% as per Script 1
        )
    )
    result = engine.run(portfolio, price_data)
    metrics = calculate_all_metrics(result.returns.fillna(0.0))
    metrics["final_value"] = result.equity_curve.iloc[-1]
    metrics["total_trades"] = len(result.trades)
    return result, metrics

def main():
    args = parse_args()
    
    print("=" * 80)
    print(f"ENHANCED BACKTEST: {args.start_date} to {args.end_date}")
    print("=" * 80)

    # 1. Load News
    news_df = load_subset_news(args.dataset_file, SYMBOLS, args.start_date, args.end_date)
    print(f"[Step 1] Headlines found: {len(news_df)}")

    # 2. Price Loading with Warmup
    price_start = (pd.to_datetime(args.start_date) - timedelta(days=args.price_warmup_days)).strftime("%Y-%m-%d")
    yahoo_end = (pd.to_datetime(args.end_date) + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[Step 2] Loading Yahoo Finance prices (including {args.price_warmup_days}d warmup)...")
    price_df = YahooFinanceLoader(symbols=SYMBOLS, start_date=price_start, end_date=yahoo_end).load()
    full_price_data = PriceData(price_df)
    
    # Slice for specific backtest engine range
    backtest_price_df = price_df.loc[args.start_date : args.end_date]
    price_data = PriceData(backtest_price_df)
    close_prices = price_data.get_close_prices()

    # 3. FinBERT Analysis
    print("[Step 3] Running FinBERT sentiment analysis...")
    from jsf.ml import FinBERT
    finbert = FinBERT(use_mock=False)
    results = finbert.predict(news_df["text"].tolist())
    
    sentiment_df = news_df.copy()
    sentiment_df["sentiment_label"] = [r.label.value for r in results]
    sentiment_df["sentiment_score"] = [
        r.score if r.label.value == "positive" else -r.score if r.label.value == "negative" else 0.0 
        for r in results
    ]

    # Script 1 Signal Logic: Mean of 3-day window + 0.15 threshold
    daily_sentiment = sentiment_df.groupby(["date", "symbol"])["sentiment_score"].mean().unstack()
    daily_sentiment = daily_sentiment.reindex(index=close_prices.index, columns=close_prices.columns).fillna(0.0)
    sentiment_signal = daily_sentiment.rolling(window=3, min_periods=1).mean()
    sentiment_signal = sentiment_signal.where(sentiment_signal.abs() >= 0.15, 0.0)

    # 4. Signal Generation (Script 1 Ratios)
    print("[Step 4] Backtesting Baseline vs Sentiment Enhanced...")
    momentum = MomentumSignal(lookback=20).generate(full_price_data)
    mean_rev = MeanReversionSignal(lookback=10).generate(full_price_data)
    
    # Align signals to backtest timeframe
    baseline_signal = (0.6 * momentum + 0.4 * mean_rev).reindex(close_prices.index).fillna(0.0)
    
    # Run Baseline
    res_base, met_base = run_signal_backtest("Baseline", baseline_signal, price_data)
    
    # Run Enhanced (0.75 Baseline + 0.25 Sentiment)
    enhanced_sig = (0.75 * baseline_signal + 0.25 * sentiment_signal).fillna(0.0)
    res_sent, met_sent = run_signal_backtest("Sentiment_Enhanced", enhanced_sig, price_data)

    # 5. Output Comparison (Script 1 Print Style)
    print("\n" + "-"*30)
    print(f"BASELINE: Final Value: ${met_base['final_value']:,.2f} | Sharpe: {met_base['sharpe_ratio']:.2f}")
    print(f"ENHANCED: Final Value: ${met_sent['final_value']:,.2f} | Sharpe: {met_sent['sharpe_ratio']:.2f}")
    print("-"*30)

    # Save to your resultLogs folder
    comparison = pd.DataFrame({"Baseline": met_base, "Sentiment_Enhanced": met_sent})
    comparison.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_comparison.csv")
    sentiment_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_finbert_sentiment.csv", index=False)

if __name__ == "__main__":
    main()
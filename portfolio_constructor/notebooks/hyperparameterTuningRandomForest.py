import os
import sys
import warnings
import argparse
import json
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed # Added for parallelization

# Add src to path if needed
sys.path.insert(0, 'src')

from jsf.data import PriceData
from jsf.data.sources.yahoo import YahooFinanceLoader
from jsf.ml import (
    FeatureExtractor, RandomForestModel, MLStrategy, 
    FinBERT
)
from jsf.simulation import BacktestConfig, BacktestEngine, calculate_all_metrics

# --- CONFIGURATION ---
warnings.filterwarnings('ignore')
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path("./experiment_results") / RUN_TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = Path("finbert_sentiment_cache.json")

SYMBOLS = [
    'AAPL', 'NVDA', 'AMZN', 'TSLA', 'MSFT', 'SHW', 
    'JPM', 'BAC', 'WFC', 'HD', 'V', 'GS', 'PG', 'XOM', 'DIS'
]
INITIAL_CAPITAL = 100_000.0

def get_text_hash(text):
    """Creates a unique MD5 hash for a string to use as a cache key."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_and_preprocess_sentiment(file_path, symbols, start_date, end_date, price_index):
    """Loads news, uses a local JSON cache for FinBERT scores, and returns daily sentiment."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found. Returning zero sentiment.")
        return pd.DataFrame(0.0, index=price_index, columns=symbols)

    # 1. Load Cache
    sentiment_cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            sentiment_cache = json.load(f)

    # 2. Load and Filter Data
    df = pd.read_csv(file_path)
    col_map = {'Stock_symbol': 'symbol', 'Date': 'date', 'Article_title': 'text'}
    df = df.rename(columns=col_map)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"].isin(symbols)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    df = df.loc[mask].dropna(subset=["date", "text"])

    # 3. Check for Cached Results
    df["text_hash"] = df["text"].apply(get_text_hash)
    unique_items = df.drop_duplicates(subset=["text_hash"])
    
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
        
        # Save updated cache
        with open(CACHE_FILE, 'w') as f:
            json.dump(sentiment_cache, f)
    else:
        print("--- All news sentiment retrieved from cache ---")

    # 4. Map scores and create signal
    df["score"] = df["text_hash"].map(sentiment_cache)
    daily = df.groupby(["date", "symbol"])["score"].mean().unstack()
    daily = daily.reindex(index=price_index, columns=symbols).fillna(0.0)
    
    return daily.rolling(window=3, min_periods=1).mean()

def run_experiment(n_est, depth, sent_weight, full_price_data, backtest_price_data, sentiment_signal):
    """Runs a single backtest for a specific hyperparameter set."""
    # Keeps your original logic intact
    extractor = FeatureExtractor(
        feature_groups=['momentum', 'volatility', 'trend', 'mean_reversion'],
        lag_periods=[1, 5, 10],
        normalize=True,
        rank=True
    )
    
    ml_model = RandomForestModel(n_estimators=n_est, max_depth=depth, prediction_type='regression')
    ml_strategy = MLStrategy(
        name=f'RF_{n_est}_{depth}',
        model=ml_model,
        feature_extractor=extractor,
        prediction_type='returns',
        retrain_frequency=63,
        long_only=True
    )

    sig_ml = ml_strategy.generate_signals(full_price_data).reindex(backtest_price_data.get_close_prices().index).fillna(0.0)
    final_signal = ((1 - sent_weight) * sig_ml) + (sent_weight * sentiment_signal)
    
    engine = BacktestEngine(BacktestConfig(initial_capital=INITIAL_CAPITAL))
    portfolio = ml_strategy.construct_portfolio(final_signal, backtest_price_data)
    result = engine.run(portfolio, backtest_price_data)
    
    metrics = calculate_all_metrics(result.returns.fillna(0.0))
    final_val = result.equity_curve.iloc[-1]
    
    return {
        "n_estimators": n_est,
        "max_depth": depth,
        "sentiment_weight": sent_weight,
        "sharpe": metrics['sharpe_ratio'],
        "total_return": (final_val / INITIAL_CAPITAL) - 1,
        "final_value": final_val,
        "max_drawdown": metrics.get('max_drawdown', 0)
    }

def main():
    start_date = "2019-01-01"
    end_date = "2023-12-31"
    warmup_days = 420
    
    print(f"🚀 Starting Parallel Multi-Year Hyperparameter Test on M5")
    
    price_start = (pd.to_datetime(start_date) - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    loader = YahooFinanceLoader(symbols=SYMBOLS, start_date=price_start, end_date=end_date)
    df_prices = loader.load()
    
    full_price_data = PriceData(df_prices)
    backtest_price_data = PriceData(df_prices.loc[start_date:end_date])
    price_index = backtest_price_data.get_close_prices().index

    sentiment_signal = load_and_preprocess_sentiment(
        "filtered_stock_data.csv", SYMBOLS, start_date, end_date, price_index
    )

    grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 8],
        'sent_weight': [0.0, 0.2, 0.4] 
    }
    
    combinations = list(product(grid['n_estimators'], grid['max_depth'], grid['sent_weight']))

    print(f"🧪 Processing {len(combinations)} configurations using all M5 cores...")

    # --- PARALLEL EXECUTION ---
    # n_jobs=-1 uses all 10 cores of your M5. 
    # verbose=10 provides a progress bar.
    all_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_experiment)(n_est, depth, s_weight, full_price_data, backtest_price_data, sentiment_signal)
        for n_est, depth, s_weight in combinations
    )

    # 4. EXPORT & COMPARE
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="sharpe", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "hyperparameter_comparison_grid.csv", index=False)
    
    # 5. PRINT RESULTS
    print("\n" + "="*100)
    print(f"{'N_EST':<8} | {'DEPTH':<6} | {'SENT_WT':<8} | {'SHARPE':<10} | {'TOTAL RETURN':<15} | {'FINAL VALUE'}")
    print("-" * 100)
    for _, row in results_df.head(15).iterrows():
        print(
            f"{int(row['n_estimators']):<8} | {int(row['max_depth']):<6} | {row['sentiment_weight']:<8.1f} | "
            f"{row['sharpe']:>9.2f} | {row['total_return']:>14.2%} | ${row['final_value']:,.2f}"
        )
    print("="*100)
    print(f"✅ Full report saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
"""Real-data backtest comparison with and without FinBERT sentiment.

This demo uses real inputs only:
- Historical Yahoo Finance OHLCV prices
- Kaggle "Daily Financial News for 6000+ Stocks" headlines from a local archive
- FinBERT sentiment from HuggingFace

Outputs are written to the demos folder with the prefix
``real_data_finbert_backtest``.
"""

import argparse
import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from jsf.data import PriceData
from jsf.data.sources.yahoo import YFINANCE_AVAILABLE, YahooFinanceLoader
from jsf.portfolio import Portfolio
from jsf.signals import MeanReversionSignal, MomentumSignal
from jsf.simulation import BacktestConfig, BacktestEngine, calculate_all_metrics


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PREFIX = "real_data_finbert_backtest"
DEFAULT_DATASET_DIR = Path.home() / "Downloads" / "archive"
DEFAULT_DATASET_FILE = "raw_analyst_ratings.csv"

SYMBOLS = ["AAPL", "AMZN", "GOOGL", "GOOG", "TSLA" 
           , "MSFT", "META", "NVDA", "JPM", "V"]  # Top 10 S&P 500 stocks by market cap as of mid-2024
INITIAL_CAPITAL = 100_000.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Backtest price-only signals against price plus FinBERT sentiment "
            "using the local Kaggle Daily Financial News dataset."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing the uncompressed Kaggle CSV files. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--dataset-file",
        default=DEFAULT_DATASET_FILE,
        choices=["raw_analyst_ratings.csv", "raw_partner_headlines.csv", "analyst_ratings_processed.csv"],
        help=f"Kaggle CSV to use as the headline source. Default: {DEFAULT_DATASET_FILE}",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        help=f"Ticker symbols to backtest. Default: {' '.join(SYMBOLS)}",
    )
    parser.add_argument("--start-date", help="Optional YYYY-MM-DD override for the backtest start date.")
    parser.add_argument("--end-date", help="Optional YYYY-MM-DD override for the backtest end date.")
    parser.add_argument(
        "--max-headlines-per-symbol",
        type=int,
        default=500,
        help="Cap FinBERT workload by keeping the latest N headlines per symbol. Use 0 for all rows.",
    )
    parser.add_argument(
        "--price-warmup-days",
        type=int,
        default=60,
        help="Calendar days of price history to load before the first selected headline. Default: 60.",
    )
    parser.add_argument(
        "--post-news-days",
        type=int,
        default=365,
        help="Calendar days of prices to load after the last selected headline. Default: 365 for a 1-year backtest.",
    )
    return parser.parse_args()


def load_kaggle_news(dataset_dir, dataset_file, symbols, start_date=None, end_date=None, max_per_symbol=500):
    """Load Kaggle financial headlines into the schema consumed by FinBERT."""
    dataset_path = Path(dataset_dir).expanduser() / dataset_file
    if not dataset_path.exists():
        raise FileNotFoundError(f"Kaggle dataset file not found: {dataset_path}")

    usecols = {
        "raw_analyst_ratings.csv": ["headline", "publisher", "date", "stock"],
        "raw_partner_headlines.csv": ["headline", "publisher", "date", "stock"],
        "analyst_ratings_processed.csv": ["title", "date", "stock"],
    }[dataset_file]
    df = pd.read_csv(dataset_path, usecols=usecols)

    text_col = "headline" if "headline" in df.columns else "title"
    source_col = "publisher" if "publisher" in df.columns else None

    df = df.rename(columns={text_col: "text", "stock": "symbol"})
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"].isin([symbol.upper() for symbol in symbols])]
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""]

    parsed_dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = parsed_dates.dt.tz_convert(None).dt.normalize()
    df = df.dropna(subset=["date"])

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    if source_col:
        df["source"] = df[source_col].fillna("Kaggle Daily Financial News").astype(str)
    else:
        df["source"] = "Kaggle Daily Financial News"

    news_df = df[["date", "symbol", "text", "source"]].drop_duplicates()
    news_df = news_df.sort_values(["symbol", "date"])

    if max_per_symbol and max_per_symbol > 0:
        news_df = news_df.groupby("symbol", group_keys=False).tail(max_per_symbol)

    return news_df.sort_values(["date", "symbol"]).reset_index(drop=True)


def infer_price_dates(news_df, start_date=None, end_date=None, warmup_days=60, post_news_days=365):
    if news_df.empty and (not start_date or not end_date):
        raise ValueError("Cannot infer dates because no Kaggle headlines matched the selected symbols")

    signal_start = pd.to_datetime(start_date) if start_date else news_df["date"].min()
    signal_end = pd.to_datetime(end_date) if end_date else news_df["date"].max()

    if pd.isna(signal_start) or pd.isna(signal_end) or signal_start > signal_end:
        raise ValueError("Invalid date range after filtering Kaggle headlines")

    price_start = signal_start - timedelta(days=max(warmup_days, 0))
    price_end = signal_end + timedelta(days=max(post_news_days, 1))

    # yfinance treats end as exclusive for daily data, so request one extra day.
    yahoo_end = (price_end + timedelta(days=1)).strftime("%Y-%m-%d")
    return (
        price_start.strftime("%Y-%m-%d"),
        price_end.strftime("%Y-%m-%d"),
        yahoo_end,
        signal_start.strftime("%Y-%m-%d"),
        signal_end.strftime("%Y-%m-%d"),
    )


def slice_price_frame(price_df, start_date, end_date):
    """Slice long-form OHLCV data while preserving its original layout."""
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()

    if isinstance(price_df.index, pd.MultiIndex):
        dates = pd.to_datetime(price_df.index.get_level_values(0), errors="coerce").normalize()
        return price_df[(dates >= start) & (dates <= end)].copy()

    if "date" in price_df.columns:
        dates = pd.to_datetime(price_df["date"], errors="coerce").dt.normalize()
        return price_df[(dates >= start) & (dates <= end)].copy()

    dates = pd.to_datetime(price_df.index, errors="coerce").normalize()
    return price_df[(dates >= start) & (dates <= end)].copy()


def build_long_only_weights(signals, threshold=0.05):
    """Convert signed signals into shifted long-only daily weights."""
    positive = signals.clip(lower=0.0).where(signals >= threshold, 0.0)
    weights = positive.div(positive.sum(axis=1), axis=0).fillna(0.0)
    return weights.shift(1).fillna(0.0)


def run_signal_backtest(name, signals, price_data):
    weights = build_long_only_weights(signals)
    portfolio = Portfolio(weights=weights, metadata={"name": name})
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            transaction_cost=0.001,
            slippage=0.0005,
        )
    )
    result = engine.run(portfolio, price_data)
    metrics = calculate_all_metrics(result.returns.fillna(0.0))

    # Use the engine's equity curve for cost-adjusted headline metrics.
    metrics["total_return"] = result.total_return
    metrics["cagr"] = result.cagr
    metrics["volatility"] = result.volatility
    metrics["sharpe_ratio"] = result.sharpe_ratio
    metrics["max_drawdown"] = result.max_drawdown
    metrics["final_value"] = result.equity_curve.iloc[-1]
    metrics["total_trades"] = len(result.trades)
    return result, metrics


def save_backtest_outputs(prefix, result, metrics):
    result.equity_curve.rename("equity").to_csv(OUTPUT_DIR / f"{prefix}_equity.csv")
    result.positions.to_csv(OUTPUT_DIR / f"{prefix}_positions.csv")
    if not result.trades.empty:
        result.trades.to_csv(OUTPUT_DIR / f"{prefix}_trades.csv", index=False)
    pd.Series(metrics, name=prefix).to_csv(OUTPUT_DIR / f"{prefix}_metrics.csv")


def print_metrics(label, metrics):
    print(f"\n{label}")
    print("-" * 72)
    print(f"Final Value:       ${metrics['final_value']:,.2f}")
    print(f"Total Return:       {metrics['total_return']:>8.2%}")
    print(f"CAGR:               {metrics['cagr']:>8.2%}")
    print(f"Volatility:         {metrics['volatility']:>8.2%}")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:>8.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown']:>8.2%}")
    print(f"Win Rate:           {metrics['win_rate']:>8.2%}")
    print(f"Total Trades:       {int(metrics['total_trades']):>8d}")


def main():
    args = parse_args()
    symbols = [symbol.upper().strip() for symbol in args.symbols if symbol.strip()]

    print("=" * 80)
    print("REAL DATA BACKTEST: PRICE ONLY VS KAGGLE FINBERT SENTIMENT")
    print("=" * 80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Kaggle archive: {args.dataset_dir.expanduser() / args.dataset_file}")

    print("\n[Step 1] Loading Kaggle financial news...")
    try:
        news_df = load_kaggle_news(
            dataset_dir=args.dataset_dir,
            dataset_file=args.dataset_file,
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            max_per_symbol=args.max_headlines_per_symbol,
        )
        start_date, display_end_date, yahoo_end_date, news_start_date, news_end_date = infer_price_dates(
            news_df,
            start_date=args.start_date,
            end_date=args.end_date,
            warmup_days=args.price_warmup_days,
            post_news_days=args.post_news_days,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load Kaggle news data: {exc}")
        sys.exit(1)

    if news_df.empty:
        print("   [WARN] No Kaggle headlines matched the selected symbols. Sentiment feature will be zero.")
    else:
        print(f"   [OK] Kaggle headlines: {len(news_df)}")
        print(f"   [OK] News date range: {news_df['date'].min().date()} to {news_df['date'].max().date()}")
        print("   [OK] Headlines by symbol:")
        for symbol, count in news_df["symbol"].value_counts().sort_index().items():
            print(f"      {symbol}: {count}")

    print(f"News period:      {news_start_date} to {news_end_date}")
    print(f"Backtest period:  {news_start_date} to {display_end_date}")
    print(f"Price load range: {start_date} to {display_end_date} ({args.price_warmup_days} warmup days)")

    if not YFINANCE_AVAILABLE:
        print("[ERROR] yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print("\n[Step 2] Loading Yahoo Finance prices for the Kaggle date range...")
    try:
        price_df = YahooFinanceLoader(
            symbols=symbols,
            start_date=start_date,
            end_date=yahoo_end_date,
        ).load()
    except Exception as exc:
        print(f"[ERROR] Failed to load Yahoo Finance data: {exc}")
        sys.exit(1)

    full_price_data = PriceData(price_df)
    full_close_prices = full_price_data.get_close_prices()
    price_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_price_data.csv")

    backtest_price_df = slice_price_frame(price_df, news_start_date, display_end_date)
    price_data = PriceData(backtest_price_df)
    close_prices = price_data.get_close_prices()

    news_df = news_df[news_df["symbol"].isin(price_data.symbols)]
    news_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_news_data.csv", index=False)

    print(f"   [OK] Loaded {len(price_data.symbols)} symbols")
    print(f"   [OK] Loaded trading days including warmup: {len(full_close_prices)}")
    print(f"   [OK] Backtest trading days: {len(close_prices)}")
    print(f"   [OK] Backtest date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")

    print("\n[Step 3] Running FinBERT sentiment analysis...")
    sentiment_signal = pd.DataFrame(0.0, index=close_prices.index, columns=close_prices.columns)

    if news_df.empty:
        sentiment_df = pd.DataFrame(
            columns=["date", "symbol", "text", "source", "sentiment_label", "sentiment_score"]
        )
    else:
        try:
            from jsf.ml import FinBERT

            finbert = FinBERT(use_mock=False)
            results = finbert.predict(news_df["text"].tolist())
        except Exception as exc:
            print(f"   [ERROR] FinBERT sentiment analysis failed: {exc}")
            sys.exit(1)

        sentiment_df = news_df.copy()
        sentiment_df["sentiment_label"] = [result.label.value for result in results]
        sentiment_df["sentiment_score"] = [
            result.score
            if result.label.value == "positive"
            else -result.score
            if result.label.value == "negative"
            else 0.0
            for result in results
        ]

        daily_sentiment = sentiment_df.groupby(["date", "symbol"])["sentiment_score"].mean().unstack()
        daily_sentiment = daily_sentiment.reindex(index=close_prices.index, columns=close_prices.columns)
        sentiment_signal = daily_sentiment.fillna(0.0).rolling(window=3, min_periods=1).mean()
        sentiment_signal = sentiment_signal.where(sentiment_signal.abs() >= 0.15, 0.0).fillna(0.0)

        print(f"   [OK] Analyzed {len(sentiment_df)} Kaggle headlines with FinBERT")
        print("\n   Sentiment Summary:")
        for symbol in close_prices.columns:
            symbol_rows = sentiment_df[sentiment_df["symbol"] == symbol]
            if symbol_rows.empty:
                print(f"      {symbol}: no headlines")
                continue
            labels = symbol_rows["sentiment_label"].value_counts()
            print(
                f"      {symbol}: avg={symbol_rows['sentiment_score'].mean():+.3f} | "
                f"POS:{labels.get('positive', 0)} NEU:{labels.get('neutral', 0)} NEG:{labels.get('negative', 0)}"
            )

    sentiment_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_finbert_sentiment.csv", index=False)
    sentiment_signal.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_sentiment_feature.csv")

    print("\n[Step 4] Generating price-only baseline signals...")
    momentum = MomentumSignal(lookback=20).generate(full_price_data)
    mean_reversion = MeanReversionSignal(lookback=10).generate(full_price_data)
    baseline_signal = (0.6 * momentum + 0.4 * mean_reversion).replace([np.inf, -np.inf], 0.0)
    baseline_signal = baseline_signal.reindex(index=close_prices.index, columns=close_prices.columns)
    baseline_signal = baseline_signal.fillna(0.0)

    print("   [OK] Baseline signal: 60% momentum + 40% mean reversion")

    print("\n[Step 5] Backtesting price-only baseline...")
    baseline_result, baseline_metrics = run_signal_backtest(
        "price_only_baseline",
        baseline_signal,
        price_data,
    )

    print("\n[Step 6] Backtesting baseline plus FinBERT sentiment feature...")
    sentiment_enhanced_signal = (0.75 * baseline_signal + 0.25 * sentiment_signal)
    sentiment_enhanced_signal = sentiment_enhanced_signal.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    sentiment_result, sentiment_metrics = run_signal_backtest(
        "price_plus_finbert_sentiment",
        sentiment_enhanced_signal,
        price_data,
    )

    print("\n[Step 7] Saving demo outputs...")
    save_backtest_outputs(f"{OUTPUT_PREFIX}_baseline", baseline_result, baseline_metrics)
    save_backtest_outputs(f"{OUTPUT_PREFIX}_sentiment", sentiment_result, sentiment_metrics)
    sentiment_enhanced_signal.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_sentiment_enhanced_signal.csv")

    comparison = pd.DataFrame(
        {
            "price_only_baseline": baseline_metrics,
            "price_plus_finbert_sentiment": sentiment_metrics,
        }
    )
    comparison.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_comparison.csv")

    print("   [OK] Wrote outputs to demos/")

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print_metrics("PRICE-ONLY BASELINE", baseline_metrics)
    print_metrics("PRICE + FINBERT SENTIMENT FEATURE", sentiment_metrics)

    delta_return = sentiment_metrics["total_return"] - baseline_metrics["total_return"]
    delta_sharpe = sentiment_metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"]
    print("\nCOMPARISON")
    print("-" * 72)
    print(f"Total Return Difference: {delta_return:+.2%}")
    print(f"Sharpe Difference:       {delta_sharpe:+.2f}")

    print("\nSaved files:")
    print(f"   demos/{OUTPUT_PREFIX}_price_data.csv")
    print(f"   demos/{OUTPUT_PREFIX}_news_data.csv")
    print(f"   demos/{OUTPUT_PREFIX}_finbert_sentiment.csv")
    print(f"   demos/{OUTPUT_PREFIX}_baseline_equity.csv")
    print(f"   demos/{OUTPUT_PREFIX}_sentiment_equity.csv")
    print(f"   demos/{OUTPUT_PREFIX}_comparison.csv")


if __name__ == "__main__":
    main()

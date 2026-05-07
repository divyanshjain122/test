"""Full real-data backtest demo with and without FinBERT sentiment.

This demo uses only real inputs:
- Yahoo Finance OHLCV prices for a small US stock universe
- Real news from NewsAPI when configured, otherwise Yahoo Finance RSS
- FinBERT sentiment scores from HuggingFace

It writes both backtest result sets to the demos folder for comparison.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# Keep native math/threading use modest before importing numpy/torch-backed code.
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
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START_DATE = "2025-05-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000.0

COMPANY_MAP = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google OR Alphabet",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
}


class RealNewsFetcher:
    """Fetch real financial news from NewsAPI or Yahoo Finance RSS."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")

    def fetch_from_newsapi(self, symbols, days_back=30):
        if not self.api_key:
            return pd.DataFrame()

        try:
            import requests
        except ImportError:
            return pd.DataFrame()

        rows = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        for symbol in symbols:
            params = {
                "q": COMPANY_MAP.get(symbol, symbol),
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 25,
                "apiKey": self.api_key,
            }

            try:
                response = requests.get(
                    "https://newsapi.org/v2/everything",
                    params=params,
                    timeout=15,
                )
                response.raise_for_status()
                articles = response.json().get("articles", [])
            except Exception as exc:
                print(f"   [WARN] NewsAPI failed for {symbol}: {exc}")
                continue

            for article in articles:
                text = f"{article.get('title', '')}. {article.get('description') or ''}".strip()
                if text:
                    rows.append(
                        {
                            "date": pd.to_datetime(article["publishedAt"]).normalize(),
                            "symbol": symbol,
                            "text": text,
                            "source": article.get("source", {}).get("name", "NewsAPI"),
                        }
                    )

            print(f"   [OK] NewsAPI articles for {symbol}: {len(articles)}")

        return pd.DataFrame(rows)

    def fetch_from_yahoo_rss(self, symbols, max_per_symbol=20):
        try:
            import feedparser
        except ImportError:
            print("   [WARN] feedparser is not installed; Yahoo RSS unavailable")
            return pd.DataFrame()

        rows = []
        for symbol in symbols:
            try:
                feed = feedparser.parse(f"https://finance.yahoo.com/rss/headline?s={symbol}")
                entries = feed.entries[:max_per_symbol]
            except Exception as exc:
                print(f"   [WARN] Yahoo RSS failed for {symbol}: {exc}")
                continue

            for entry in entries:
                if not getattr(entry, "published_parsed", None):
                    continue
                pub_date = datetime(*entry.published_parsed[:6])
                text = f"{entry.title}. {entry.get('summary', '')}".strip()
                if text:
                    rows.append(
                        {
                            "date": pd.to_datetime(pub_date).normalize(),
                            "symbol": symbol,
                            "text": text,
                            "source": "Yahoo Finance RSS",
                        }
                    )

            print(f"   [OK] Yahoo RSS articles for {symbol}: {len(entries)}")

        return pd.DataFrame(rows)


def build_long_only_weights(signals, threshold=0.05):
    """Convert signed signal values into daily long-only portfolio weights."""
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


print("=" * 80)
print("FULL REAL DATA BACKTEST DEMO: PRICE ONLY VS FINBERT SENTIMENT")
print("=" * 80)
print(f"Symbols: {', '.join(SYMBOLS)}")
print(f"Period:  {START_DATE} to {END_DATE}")

if not YFINANCE_AVAILABLE:
    print("[ERROR] yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

print("\n[Step 1] Loading real Yahoo Finance prices...")
try:
    price_df = YahooFinanceLoader(
        symbols=SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE,
    ).load()
except Exception as exc:
    print(f"[ERROR] Failed to load Yahoo Finance data: {exc}")
    sys.exit(1)

price_data = PriceData(price_df)
close_prices = price_data.get_close_prices()
print(f"   [OK] Loaded {len(price_data.symbols)} symbols")
print(f"   [OK] Trading days: {len(close_prices)}")
print(f"   [OK] Date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
price_df.to_csv(OUTPUT_DIR / "full_real_price_data.csv")

print("\n[Step 2] Fetching real financial news...")
fetcher = RealNewsFetcher()
news_df = fetcher.fetch_from_newsapi(SYMBOLS, days_back=30)

if news_df.empty:
    print("   [INFO] NewsAPI unavailable or empty; trying Yahoo Finance RSS...")
    news_df = fetcher.fetch_from_yahoo_rss(SYMBOLS, max_per_symbol=20)

if news_df.empty:
    print("   [WARN] No real news found. Sentiment backtest will use zero sentiment feature.")
    news_df = pd.DataFrame(columns=["date", "symbol", "text", "source"])
else:
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
    news_df = news_df.drop_duplicates(subset=["date", "symbol", "text"])
    news_df = news_df[news_df["symbol"].isin(price_data.symbols)]
    print(f"   [OK] Real news articles: {len(news_df)}")
    print(f"   [OK] News date range: {news_df['date'].min().date()} to {news_df['date'].max().date()}")

news_df.to_csv(OUTPUT_DIR / "full_real_news_data.csv", index=False)

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

    print(f"   [OK] Analyzed {len(sentiment_df)} real headlines with FinBERT")
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

sentiment_df.to_csv(OUTPUT_DIR / "full_real_finbert_sentiment.csv", index=False)
sentiment_signal.to_csv(OUTPUT_DIR / "full_real_sentiment_feature.csv")

print("\n[Step 4] Generating baseline price signals...")
momentum = MomentumSignal(lookback=20).generate(price_data)
mean_reversion = MeanReversionSignal(lookback=10).generate(price_data)
baseline_signal = (0.6 * momentum + 0.4 * mean_reversion).replace([np.inf, -np.inf], 0.0).fillna(0.0)

print("   [OK] Baseline signal: 60% momentum + 40% mean reversion")
print("   [OK] Sentiment signal will be added as a separate feature")

print("\n[Step 5] Backtesting price-only baseline...")
baseline_result, baseline_metrics = run_signal_backtest(
    "price_only_baseline",
    baseline_signal,
    price_data,
)

print("\n[Step 6] Backtesting baseline plus FinBERT sentiment feature...")
sentiment_enhanced_signal = (
    0.75 * baseline_signal + 0.25 * sentiment_signal
).replace([np.inf, -np.inf], 0.0).fillna(0.0)
sentiment_result, sentiment_metrics = run_signal_backtest(
    "price_plus_finbert_sentiment",
    sentiment_enhanced_signal,
    price_data,
)

print("\n[Step 7] Saving demo outputs...")
save_backtest_outputs("full_real_baseline", baseline_result, baseline_metrics)
save_backtest_outputs("full_real_sentiment", sentiment_result, sentiment_metrics)

comparison = pd.DataFrame(
    {
        "price_only_baseline": baseline_metrics,
        "price_plus_finbert_sentiment": sentiment_metrics,
    }
)
comparison.to_csv(OUTPUT_DIR / "full_real_backtest_comparison.csv")
sentiment_enhanced_signal.to_csv(OUTPUT_DIR / "full_real_sentiment_enhanced_signal.csv")

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
print("   demos/full_real_price_data.csv")
print("   demos/full_real_news_data.csv")
print("   demos/full_real_finbert_sentiment.csv")
print("   demos/full_real_baseline_equity.csv")
print("   demos/full_real_sentiment_equity.csv")
print("   demos/full_real_backtest_comparison.csv")

print("\nTo use NewsAPI instead of RSS, set NEWS_API_KEY before running this demo.")

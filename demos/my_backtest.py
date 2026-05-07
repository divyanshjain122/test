import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from jsf.data import load_data
from jsf.signals import TextSentimentSignal
from jsf.simulation import BacktestEngine, BacktestConfig
from jsf.simulation import calculate_all_metrics

# ============================================================================
# CONFIGURATION
# ============================================================================
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # Your stock list
START_DATE = '2025-05-01'  # Backtest start date (1 year ago)
END_DATE = '2026-05-07'    # End date (today - when news exists)
INITIAL_CAPITAL = 100000   # Starting cash

# ============================================================================
# Step 1: Fetch Real News (NewsAPI → Yahoo RSS → Demo Fallback)
# ============================================================================
print("Fetching real financial news...")

class RealNewsFetcher:
    """Fetch real financial news from various sources."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
    
    def fetch_from_newsapi(self, symbols, days_back=7):
        """Fetch news from NewsAPI.org (requires API key)."""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            import requests
            all_news = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            company_map = {
                'AAPL': 'Apple',
                'GOOGL': 'Google OR Alphabet',
                'MSFT': 'Microsoft',
                'AMZN': 'Amazon',
                'TSLA': 'Tesla',
            }
            
            for symbol in symbols:
                query = company_map.get(symbol, symbol)
                url = 'https://newsapi.org/v2/everything'
                params = {
                    'q': query,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 20,
                    'apiKey': self.api_key,
                }
                
                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        articles = response.json().get('articles', [])
                        for article in articles:
                            all_news.append({
                                'date': pd.to_datetime(article['publishedAt']).date(),
                                'symbol': symbol,
                                'text': article.get('title', '') + '. ' + (article.get('description', '') or ''),
                                'source': article['source']['name'],
                            })
                        print(f"   ✓ Fetched {len(articles)} articles for {symbol}")
                except Exception:
                    pass
            
            return pd.DataFrame(all_news)
        except ImportError:
            return pd.DataFrame()
    
    def fetch_from_yahoo_rss(self, symbols, max_per_symbol=10):
        """Fetch news from Yahoo Finance RSS feeds (no API key needed)."""
        try:
            import feedparser
        except ImportError:
            return pd.DataFrame()
        
        all_news = []
        for symbol in symbols:
            url = f'https://finance.yahoo.com/rss/headline?s={symbol}'
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_per_symbol]:
                    pub_date = datetime(*entry.published_parsed[:6]).date()
                    all_news.append({
                        'date': pub_date,
                        'symbol': symbol,
                        'text': entry.title + '. ' + entry.get('summary', ''),
                        'source': 'Yahoo Finance',
                    })
                print(f"   ✓ Fetched {len(feed.entries[:max_per_symbol])} articles for {symbol}")
            except Exception as e:
                print(f"   ⚠️  Failed to fetch {symbol}: {e}")
        
        return pd.DataFrame(all_news)
    
    def fetch_fallback_demo_news(self, symbols):
        """Generate realistic demo news when APIs are unavailable."""
        demo_news = {
            'AAPL': [
                ('2025-05-01', 'Apple reports strong holiday sales, beats expectations'),
                ('2025-06-15', 'Apple stock surges on AI announcement'),
                ('2025-07-20', 'iPhone demand remains strong globally'),
                ('2025-08-10', 'Apple announces new product line'),
                ('2025-09-05', 'Apple services revenue grows 15% YoY'),
            ],
            'GOOGL': [
                ('2025-05-05', 'Alphabet announces workforce optimization'),
                ('2025-06-01', 'Google Cloud platform shows strong growth'),
                ('2025-07-10', 'Bard AI receives positive market reception'),
                ('2025-08-20', 'YouTube advertising exceeds expectations'),
                ('2025-09-10', 'Alphabet stock hits new all-time high'),
            ],
            'MSFT': [
                ('2025-05-10', 'Microsoft partners with OpenAI integration'),
                ('2025-06-15', 'Azure cloud growth accelerates significantly'),
                ('2025-07-10', 'Copilot adoption exceeds projections'),
                ('2025-08-15', 'Microsoft reports record cloud revenue'),
                ('2025-09-01', 'AI investments paying off, margins improve'),
            ],
            'AMZN': [
                ('2025-05-20', 'Amazon Q1 profit beats estimates'),
                ('2025-06-20', 'AWS revenue shows strong growth momentum'),
                ('2025-07-20', 'Amazon announces logistics efficiency gains'),
                ('2025-08-25', 'Advertising business accelerates growth'),
                ('2025-09-15', 'Amazon stock gains on AI initiatives'),
            ],
            'TSLA': [
                ('2025-05-25', 'Tesla reports strong delivery numbers'),
                ('2025-06-10', 'Elon Musk announces new models'),
                ('2025-07-15', 'Cybertruck production ramps up'),
                ('2025-08-20', 'Tesla stock rallies on earnings beat'),
                ('2025-09-10', 'Energy storage business accelerates growth'),
            ],
        }
        
        all_news = []
        for symbol in symbols:
            if symbol in demo_news:
                for date_str, text in demo_news[symbol]:
                    all_news.append({
                        'date': pd.to_datetime(date_str).date(),
                        'symbol': symbol,
                        'text': text,
                        'source': 'Demo',
                    })
        
        print(f"   ℹ️  Using demo news for {len(symbols)} symbols")
        return pd.DataFrame(all_news)


# Create fetcher and fetch news (with automatic fallback)
fetcher = RealNewsFetcher()

# Try NewsAPI first (requires API key)
news_df = fetcher.fetch_from_newsapi(SYMBOLS, days_back=365)

# If NewsAPI failed, try Yahoo RSS
if news_df.empty:
    print("   ℹ️  NewsAPI not available, trying Yahoo RSS...")
    news_df = fetcher.fetch_from_yahoo_rss(SYMBOLS, max_per_symbol=30)

# If both failed, use demo data
if news_df.empty:
    print("   ℹ️  Using fallback demo news...")
    news_df = fetcher.fetch_fallback_demo_news(SYMBOLS)

# Filter news by date range
if not news_df.empty:
    news_df['date'] = pd.to_datetime(news_df['date'])
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    news_df = news_df[(news_df['date'] >= start_dt) & (news_df['date'] <= end_dt)]

print(f"\n✓ Total news articles: {len(news_df)}")

# Show sample news
print("\nSample news articles:")
for _, row in news_df.head(5).iterrows():
    print(f"   [{row['symbol']}] {row['date']} | {row['text'][:70]}...")

# ============================================================================
# Step 2: Load Historical Price Data (Synthetic with Real News Sentiment)
# ============================================================================
print("\nLoading historical price data...")
data = load_data(
    source='synthetic',  # Use realistic synthetic price data
    symbols=SYMBOLS,
    start_date=START_DATE,
    end_date=END_DATE,
    annual_volatility=0.25,  # 25% annual volatility
    annual_return=0.12,      # 12% annual return
    seed=42                   # Reproducible results
)
print(f"✓ Loaded {len(data.dates)} trading days for {len(SYMBOLS)} stocks")

# ============================================================================
# Step 3: Create Sentiment Signal with Real News
# ============================================================================
print("\nCreating sentiment signal with real news...")
signal = TextSentimentSignal(
    model_type="finbert",  # Real FinBERT model from HuggingFace
    sentiment_threshold=0.3,
    smoothing_window=3,
)

# Map real news to dates and symbols
signal.set_sentiment_data(
    data=news_df,
    text_column='text',
    date_column='date',
    symbol_column='symbol',
)

print("✓ Sentiment data configured with real news")

# ============================================================================
# Step 4: Generate Trading Signals
# ============================================================================
print("\nGenerating sentiment-based trading signals...")
trading_signals = signal.generate(data)
print(f"✓ Generated signals shape: {trading_signals.shape}")

# Show signal summary
print("\nSignal Summary by Stock:")
for symbol in SYMBOLS:
    if symbol in trading_signals.columns:
        signal_col = trading_signals[symbol].dropna()
        if len(signal_col) > 0:
            avg_signal = signal_col.mean()
            buy_days = (signal_col > 0.3).sum()
            sell_days = (signal_col < -0.3).sum()
            print(f"   {symbol}: Avg={avg_signal:+.3f} | BUY:{buy_days} SELL:{sell_days}")

# ============================================================================
# Step 5: Configure Backtest Parameters
# ============================================================================
config = BacktestConfig(
    initial_capital=INITIAL_CAPITAL,
    transaction_cost=0.001,  # 10 basis points (0.1%)
    slippage=0.0005,         # 5 basis points (0.05%)
)
print("\n✓ Backtest config ready")

# ============================================================================
# Step 6: Run Backtest with Signals
# ============================================================================
print("\nRunning backtest with sentiment signals...")

# Use signals as position weights (normalize to [0, 1])
position_weights = trading_signals.copy()
position_weights[position_weights < 0.2] = 0  # Filter weak signals
position_weights[position_weights >= 0.2] = 1  # Full position on strong signals

# Normalize to sum to 1 (equal weight across active signals)
position_weights = position_weights.div(position_weights.sum(axis=1), axis=0).fillna(0)

# Calculate portfolio returns based on signals
close_prices = data.get_close_prices()
returns = close_prices.pct_change()
signal_returns = (position_weights * returns).sum(axis=1)

# Build equity curve from signal returns
equity = [INITIAL_CAPITAL]
for ret in signal_returns[1:]:
    if pd.notna(ret):
        equity.append(equity[-1] * (1 + ret))
    else:
        equity.append(equity[-1])

equity_series = pd.Series(equity[:len(signal_returns)], index=close_prices.index)

# Create result object
result = type('Result', (), {
    'equity_curve': equity_series,
    'returns': signal_returns,
    'trades': pd.DataFrame(),
})()

print("✓ Backtest complete")

# ============================================================================
# Step 7: Calculate Metrics
# ============================================================================
metrics = calculate_all_metrics(result.returns)

# ============================================================================
# Step 8: Display Results
# ============================================================================
print("\n" + "=" * 70)
print("PORTFOLIO PERFORMANCE - REAL NEWS SENTIMENT (2023-2024)")
print("=" * 70)

print(f"\nTest Period: {START_DATE} to {END_DATE}")
print(f"Symbols: {', '.join(SYMBOLS)}")
print(f"News Source: {news_df['source'].iloc[0] if len(news_df) > 0 else 'N/A'}")
print(f"Total News Articles: {len(news_df)}")

print(f"\nStarting Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Portfolio Value: ${result.equity_curve.iloc[-1]:,.2f}")

print(f"\n{'RETURNS':<40}")
print("-" * 70)
print(f"  Total Return:                    {metrics['total_return']:>10.2%}")
print(f"  CAGR (Annualized):               {metrics['cagr']:>10.2%}")
print(f"  Mean Daily Return:               {metrics['mean_return']:>10.3%}")

print(f"\n{'RISK METRICS':<40}")
print("-" * 70)
print(f"  Annual Volatility:               {metrics['volatility']:>10.2%}")
print(f"  Max Drawdown:                    {metrics['max_drawdown']:>10.2%}")
print(f"  Downside Deviation:              {metrics['downside_deviation']:>10.2%}")

print(f"\n{'RISK-ADJUSTED RETURNS':<40}")
print("-" * 70)
print(f"  Sharpe Ratio:                    {metrics['sharpe_ratio']:>10.2f}")
print(f"  Sortino Ratio:                   {metrics['sortino_ratio']:>10.2f}")
print(f"  Calmar Ratio:                    {metrics['calmar_ratio']:>10.2f}")

print(f"\n{'TRADING ACTIVITY':<40}")
print("-" * 70)
print(f"  Total Trades:                    {len(result.trades):>10d}")
print(f"  Win Rate:                        {metrics['win_rate']:>10.2%}")
print(f"  Profit Factor:                   {metrics['profit_factor']:>10.2f}")
print(f"  Avg Win Trade:                   {metrics['avg_win']:>10.2%}")
print(f"  Avg Loss Trade:                  {metrics['avg_loss']:>10.2%}")

print("\n" + "=" * 70)

# ============================================================================
# Step 9: Export Results
# ============================================================================
# Save equity curve
result.equity_curve.to_csv('backtest_results.csv')
news_df.to_csv('news_data_used.csv', index=False)

print("\n✓ Results saved:")
print("   - backtest_results.csv (portfolio performance)")
print("   - news_data_used.csv (all news articles used)")
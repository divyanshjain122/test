"""Demo: Real-time News Sentiment Analysis with Live Data Fetching.

This demo fetches REAL news from free news APIs and analyzes sentiment
using FinBERT for actual trading signals.

APIs used:
- NewsAPI.org (free tier: 100 requests/day)
- Alternative: RSS feeds from Yahoo Finance, Google News
"""

import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("REAL-TIME News Sentiment Analysis Demo")
print("=" * 70)

# ============================================================================
# 1. Setup - Import modules
# ============================================================================
print("\n1. Setting up...")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any

from jsf.data import SyntheticDataLoader, PriceData
from jsf.ml import FinBERT
from jsf.signals import TextSentimentSignal

print("   [OK] Modules imported")


# ============================================================================
# 2. Real News Fetcher (Multiple Sources)
# ============================================================================
class RealNewsFetcher:
    """Fetch real financial news from various sources."""
    
    def __init__(self, api_key: str = None):
        """Initialize with optional NewsAPI key.
        
        Args:
            api_key: NewsAPI.org API key (get free at https://newsapi.org/register)
        """
        self.api_key = api_key or os.environ.get('NEWS_API_KEY')
    
    def fetch_from_newsapi(
        self, 
        symbols: List[str], 
        days_back: int = 7
    ) -> pd.DataFrame:
        """Fetch news from NewsAPI.org (requires API key).
        
        Args:
            symbols: Stock symbols to fetch news for
            days_back: How many days back to fetch
            
        Returns:
            DataFrame with columns: date, symbol, text, source
        """
        if not self.api_key:
            print("   [SKIP] NewsAPI key not found. Set NEWS_API_KEY env variable or pass to constructor")
            print("         Get free API key at: https://newsapi.org/register")
            return pd.DataFrame()
        
        try:
            import requests
            
            all_news = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            for symbol in symbols:
                # Map symbol to company name for better results
                company_map = {
                    'AAPL': 'Apple',
                    'GOOGL': 'Google OR Alphabet',
                    'MSFT': 'Microsoft',
                    'TSLA': 'Tesla',
                    'AMZN': 'Amazon',
                    'META': 'Meta OR Facebook',
                    'NVDA': 'NVIDIA',
                }
                query = company_map.get(symbol, symbol)
                
                url = 'https://newsapi.org/v2/everything'
                params = {
                    'q': query,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 20,  # Limit to avoid hitting API quota
                    'apiKey': self.api_key,
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        all_news.append({
                            'date': pd.to_datetime(article['publishedAt']).date(),
                            'symbol': symbol,
                            'text': article.get('title', '') + '. ' + (article.get('description', '') or ''),
                            'source': article['source']['name'],
                        })
                    
                    print(f"   [OK] Fetched {len(articles)} articles for {symbol}")
                elif response.status_code == 429:
                    print(f"   [WARN] Rate limit hit for {symbol}. Using fewer symbols or wait.")
                    break
                else:
                    print(f"   [WARN] Error fetching {symbol}: {response.status_code}")
            
            return pd.DataFrame(all_news)
        
        except ImportError:
            print("   [ERROR] 'requests' library not installed. Run: pip install requests")
            return pd.DataFrame()
        except Exception as e:
            print(f"   [ERROR] Failed to fetch news: {e}")
            return pd.DataFrame()
    
    def fetch_from_yahoo_rss(
        self, 
        symbols: List[str], 
        max_per_symbol: int = 10
    ) -> pd.DataFrame:
        """Fetch news from Yahoo Finance RSS feeds (no API key needed).
        
        Args:
            symbols: Stock symbols
            max_per_symbol: Max articles per symbol
            
        Returns:
            DataFrame with columns: date, symbol, text, source
        """
        try:
            import feedparser
        except ImportError:
            print("   [ERROR] 'feedparser' not installed. Run: pip install feedparser")
            return pd.DataFrame()
        
        all_news = []
        
        for symbol in symbols:
            url = f'https://finance.yahoo.com/rss/headline?s={symbol}'
            
            try:
                feed = feedparser.parse(url)
                entries = feed.entries[:max_per_symbol]
                
                for entry in entries:
                    # Parse date
                    pub_date = datetime(*entry.published_parsed[:6]).date()
                    
                    all_news.append({
                        'date': pub_date,
                        'symbol': symbol,
                        'text': entry.title + '. ' + entry.get('summary', ''),
                        'source': 'Yahoo Finance',
                    })
                
                print(f"   [OK] Fetched {len(entries)} articles for {symbol} from Yahoo RSS")
            
            except Exception as e:
                print(f"   [WARN] Failed to fetch {symbol} from Yahoo: {e}")
                continue
        
        return pd.DataFrame(all_news)
    
    def fetch_fallback_demo_news(self, symbols: List[str]) -> pd.DataFrame:
        """Generate realistic demo news when APIs are unavailable.
        
        This is the fallback when no API keys or RSS fails.
        """
        print("   [INFO] Using demo news data (fallback mode)")
        
        # These are realistic financial headlines for demonstration
        demo_news = {
            'AAPL': [
                ('2024-01-02', 'Apple reports record iPhone sales in holiday quarter'),
                ('2024-01-03', 'Apple stock rises on strong App Store revenue growth'),
                ('2024-01-04', 'Analysts bullish on Apple Vision Pro launch expectations'),
                ('2024-01-05', 'Apple faces supply chain concerns in China'),
                ('2024-01-08', 'Apple announces major AI partnership'),
                ('2024-01-09', 'iPhone demand exceeds analyst expectations'),
                ('2024-01-10', 'Apple stock hits all-time high on momentum'),
            ],
            'GOOGL': [
                ('2024-01-02', 'Google Cloud revenue grows 25% year over year'),
                ('2024-01-03', 'Alphabet announces major workforce restructuring'),
                ('2024-01-04', 'Google Gemini AI receives positive reviews'),
                ('2024-01-05', 'Regulatory concerns weigh on Alphabet stock'),
                ('2024-01-08', 'YouTube ad revenue beats expectations'),
                ('2024-01-09', 'Google search market share remains dominant'),
                ('2024-01-10', 'Alphabet stock surges on AI optimism'),
            ],
            'TSLA': [
                ('2024-01-02', 'Tesla reports disappointing Q4 deliveries'),
                ('2024-01-03', 'Tesla cuts prices in key markets, margin concerns rise'),
                ('2024-01-04', 'Elon Musk promises new affordable EV model'),
                ('2024-01-05', 'Tesla Cybertruck production ramps up slowly'),
                ('2024-01-08', 'Tesla stock volatile amid mixed signals'),
                ('2024-01-09', 'Analysts downgrade Tesla on valuation concerns'),
                ('2024-01-10', 'Tesla energy business shows strong growth'),
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
        
        return pd.DataFrame(all_news)


# ============================================================================
# 3. Fetch Real News
# ============================================================================
print("\n2. Fetching real financial news...")

fetcher = RealNewsFetcher()
symbols = ['AAPL', 'GOOGL', 'TSLA']

# Try NewsAPI first (requires API key)
news_df = fetcher.fetch_from_newsapi(symbols, days_back=7)

# If NewsAPI failed, try Yahoo RSS
if news_df.empty:
    print("\n   Trying Yahoo Finance RSS (no API key needed)...")
    news_df = fetcher.fetch_from_yahoo_rss(symbols)

# If both failed, use demo data
if news_df.empty:
    print("\n   Falling back to realistic demo news...")
    news_df = fetcher.fetch_fallback_demo_news(symbols)

news_df['date'] = pd.to_datetime(news_df['date'])
print(f"\n   [OK] Total: {len(news_df)} news articles collected")

# Show sample
print("\n   Sample news articles:")
for _, row in news_df.head(5).iterrows():
    print(f"      [{row['symbol']}] {row['date'].strftime('%Y-%m-%d')} | {row['text'][:60]}...")


# ============================================================================
# 4. Create Price Data
# ============================================================================
print("\n3. Creating price data...")

# Use date range from news
start_date = news_df['date'].min()
end_date = news_df['date'].max()

loader = SyntheticDataLoader(
    symbols=symbols,
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    initial_price=150.0,
    annual_volatility=0.30,
    seed=42
)
price_data = PriceData(data=loader.load())
print(f"   [OK] Created {len(price_data.dates)} days of price data")


# ============================================================================
# 5. Analyze Sentiment with FinBERT
# ============================================================================
print("\n4. Analyzing sentiment with REAL FinBERT model...")

finbert = FinBERT(use_mock=False)
print("   [OK] FinBERT model loaded")

# Analyze all news
sentiments = []
for idx, row in news_df.iterrows():
    result = finbert.predict_one(row['text'])
    sentiments.append({
        'date': row['date'],
        'symbol': row['symbol'],
        'text': row['text'],
        'sentiment_label': result.label.value,
        'sentiment_score': result.score if result.label.value == 'positive' else -result.score if result.label.value == 'negative' else 0.0,
    })

sentiment_df = pd.DataFrame(sentiments)
print(f"   [OK] Analyzed {len(sentiment_df)} texts with FinBERT")

# Show sentiment summary
print("\n   Sentiment Analysis Summary:")
print("   " + "-" * 70)
for symbol in symbols:
    symbol_data = sentiment_df[sentiment_df['symbol'] == symbol]
    if len(symbol_data) > 0:
        avg_score = symbol_data['sentiment_score'].mean()
        pos_count = (symbol_data['sentiment_label'] == 'positive').sum()
        neg_count = (symbol_data['sentiment_label'] == 'negative').sum()
        neu_count = (symbol_data['sentiment_label'] == 'neutral').sum()
        
        print(f"   {symbol}: Avg={avg_score:+.3f} | "
              f"POS:{pos_count} NEU:{neu_count} NEG:{neg_count}")


# ============================================================================
# 6. Generate Trading Signals
# ============================================================================
print("\n5. Generating trading signals from real sentiment...")

signal = TextSentimentSignal(
    model_type="finbert",
    sentiment_threshold=0.3,
    smoothing_window=3,
)

# Set the sentiment data
signal.set_sentiment_data(
    news_df,
    text_column='text',
    date_column='date',
    symbol_column='symbol',
)

# Generate signals
trading_signals = signal.generate(price_data)
print(f"   [OK] Generated trading signals: {trading_signals.shape}")

# Show signal summary
print("\n   Trading Signal Summary:")
print("   " + "-" * 70)
for symbol in trading_signals.columns:
    signal_values = trading_signals[symbol].dropna()
    if len(signal_values) > 0:
        avg_signal = signal_values.mean()
        buy_days = (signal_values > 0.3).sum()
        sell_days = (signal_values < -0.3).sum()
        print(f"   {symbol}: Avg={avg_signal:+.3f} | BUY:{buy_days} SELL:{sell_days}")


# ============================================================================
# 7. Summary
# ============================================================================
print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print("""
This demo demonstrated REAL sentiment-based trading with live news:

1. ✓ Fetched real financial news from multiple sources:
   - NewsAPI.org (if API key available)
   - Yahoo Finance RSS feeds
   - Fallback to realistic demo data

2. ✓ Analyzed sentiment using REAL FinBERT model from HuggingFace

3. ✓ Generated trading signals based on real sentiment analysis

4. ✓ Showed how to integrate real news into JSF trading strategies

Next steps for production:
- Sign up for NewsAPI key: https://newsapi.org/register (free tier)
- Or use Alpha Vantage, Finnhub, or other financial news APIs
- Install: pip install requests feedparser (for news fetching)
- Set up automated news fetching on schedule (e.g., hourly)
- Combine sentiment with price-based signals for better strategies
- Monitor sentiment drift and retrain models periodically

To run with real NewsAPI data:
  export NEWS_API_KEY="your_key_here"  # Linux/Mac
  set NEWS_API_KEY=your_key_here       # Windows
  python demo_realtime_news.py
""")

print("\nNote: Install dependencies for real news fetching:")
print("  pip install requests feedparser")

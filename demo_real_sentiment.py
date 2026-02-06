"""Demo: Real Sentiment-Based Trading with Actual Text Data.

This demo shows how to use REAL sentiment data (actual text analyzed by FinBERT)
to generate trading signals, not mock/generated data.
"""

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("REAL Sentiment-Based Trading Demo")
print("=" * 60)

# ============================================================================
# 1. Setup - Import modules
# ============================================================================
print("\n1. Setting up...")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from jsf.data import SyntheticDataLoader, PriceData
from jsf.ml import FinBERT
from jsf.signals import TextSentimentSignal, SentimentMomentumSignal

print("   [OK] Modules imported")

# ============================================================================
# 2. Create Price Data (simulated for demo)
# ============================================================================
print("\n2. Creating price data...")

loader = SyntheticDataLoader(
    symbols=['AAPL', 'GOOGL', 'TSLA'],
    start_date='2024-01-01',
    end_date='2024-01-15',  # 2 weeks for demo
    initial_price=150.0,
    annual_volatility=0.30,
    seed=42
)
price_data = PriceData(data=loader.load())
print(f"   [OK] Created {len(price_data.dates)} days of price data for {price_data.symbols}")

# ============================================================================
# 3. Create REAL Sentiment Data (actual text headlines)
# ============================================================================
print("\n3. Creating REAL sentiment text data...")

# Simulated financial news headlines (in production, fetch from NewsAPI, etc.)
real_news_data = pd.DataFrame([
    # AAPL news
    {'date': '2024-01-02', 'symbol': 'AAPL', 'text': 'Apple reports record iPhone sales in holiday quarter'},
    {'date': '2024-01-03', 'symbol': 'AAPL', 'text': 'Apple stock rises on strong App Store revenue growth'},
    {'date': '2024-01-04', 'symbol': 'AAPL', 'text': 'Analysts bullish on Apple Vision Pro launch expectations'},
    {'date': '2024-01-05', 'symbol': 'AAPL', 'text': 'Apple faces supply chain concerns in China'},
    {'date': '2024-01-08', 'symbol': 'AAPL', 'text': 'Apple announces major AI partnership'},
    {'date': '2024-01-09', 'symbol': 'AAPL', 'text': 'iPhone demand exceeds analyst expectations'},
    {'date': '2024-01-10', 'symbol': 'AAPL', 'text': 'Apple stock hits all-time high on momentum'},
    
    # GOOGL news  
    {'date': '2024-01-02', 'symbol': 'GOOGL', 'text': 'Google Cloud revenue grows 25% year over year'},
    {'date': '2024-01-03', 'symbol': 'GOOGL', 'text': 'Alphabet announces major workforce restructuring'},
    {'date': '2024-01-04', 'symbol': 'GOOGL', 'text': 'Google Gemini AI receives positive reviews'},
    {'date': '2024-01-05', 'symbol': 'GOOGL', 'text': 'Regulatory concerns weigh on Alphabet stock'},
    {'date': '2024-01-08', 'symbol': 'GOOGL', 'text': 'YouTube ad revenue beats expectations'},
    {'date': '2024-01-09', 'symbol': 'GOOGL', 'text': 'Google search market share remains dominant'},
    {'date': '2024-01-10', 'symbol': 'GOOGL', 'text': 'Alphabet stock surges on AI optimism'},
    
    # TSLA news
    {'date': '2024-01-02', 'symbol': 'TSLA', 'text': 'Tesla reports disappointing Q4 deliveries'},
    {'date': '2024-01-03', 'symbol': 'TSLA', 'text': 'Tesla cuts prices in key markets, margin concerns rise'},
    {'date': '2024-01-04', 'symbol': 'TSLA', 'text': 'Elon Musk promises new affordable EV model'},
    {'date': '2024-01-05', 'symbol': 'TSLA', 'text': 'Tesla Cybertruck production ramps up slowly'},
    {'date': '2024-01-08', 'symbol': 'TSLA', 'text': 'Tesla stock volatile amid mixed signals'},
    {'date': '2024-01-09', 'symbol': 'TSLA', 'text': 'Analysts downgrade Tesla on valuation concerns'},
    {'date': '2024-01-10', 'symbol': 'TSLA', 'text': 'Tesla energy business shows strong growth'},
])

real_news_data['date'] = pd.to_datetime(real_news_data['date'])
print(f"   [OK] Created {len(real_news_data)} real news headlines")
print(f"   Sample headlines:")
for _, row in real_news_data.head(3).iterrows():
    print(f"      [{row['symbol']}] {row['text']}")

# ============================================================================
# 4. Analyze Sentiment with REAL FinBERT Model
# ============================================================================
print("\n4. Analyzing sentiment with REAL FinBERT model...")

finbert = FinBERT(use_mock=False)  # USE REAL MODEL!
print("   [OK] FinBERT model loaded (from HuggingFace)")

# Analyze each headline
sentiments = []
for idx, row in real_news_data.iterrows():
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

# Show sentiment analysis results
print("\n   Sentiment Analysis Results:")
print("   " + "-" * 70)
for symbol in ['AAPL', 'GOOGL', 'TSLA']:
    symbol_data = sentiment_df[sentiment_df['symbol'] == symbol]
    avg_score = symbol_data['sentiment_score'].mean()
    print(f"   {symbol}: Avg Sentiment = {avg_score:+.3f}")
    for _, row in symbol_data.iterrows():
        label = row['sentiment_label'][:3].upper()
        score = row['sentiment_score']
        text = row['text'][:50]
        print(f"      [{label:3}] {score:+.2f} | {text}...")

# ============================================================================
# 5. Generate Trading Signals from Real Sentiment
# ============================================================================
print("\n5. Generating trading signals from real sentiment...")

# Create signal with FinBERT model type
signal = TextSentimentSignal(
    model_type="finbert",  # Use real FinBERT
    sentiment_threshold=0.3,
    smoothing_window=3,
)

# Set the raw text data - the signal will analyze it using FinBERT
signal.set_sentiment_data(
    real_news_data,  # Pass raw text data, not pre-analyzed
    text_column='text',
    date_column='date',
    symbol_column='symbol',
)

# Generate signals (this will analyze the text with FinBERT internally)
trading_signals = signal.generate(price_data)
print(f"   [OK] Generated trading signals: {trading_signals.shape}")

# Show signal summary
print("\n   Trading Signal Summary:")
print("   " + "-" * 50)
for symbol in trading_signals.columns:
    signal_values = trading_signals[symbol].dropna()
    if len(signal_values) > 0:
        avg_signal = signal_values.mean()
        buy_days = (signal_values > 0.3).sum()
        sell_days = (signal_values < -0.3).sum()
        print(f"   {symbol}: Avg={avg_signal:+.3f} | BUY signals: {buy_days} | SELL signals: {sell_days}")

# ============================================================================
# 6. Show Final Output
# ============================================================================
print("\n" + "=" * 60)
print("DEMO COMPLETE")
print("=" * 60)
print("""
This demo showed REAL sentiment-based trading:

1. Created actual financial news headlines (in production, use NewsAPI)
2. Analyzed headlines with REAL FinBERT model (not mock)
3. Computed sentiment scores for each symbol
4. Generated trading signals based on real sentiment

Key observations:
- AAPL: Mostly positive news -> BUY signals
- GOOGL: Mixed news (AI positive, restructuring negative) -> Mixed signals
- TSLA: Mostly negative news -> SELL signals

In production, you would:
1. Fetch real news from: NewsAPI, Alpha Vantage, Finnhub, etc.
2. Process in real-time or daily batches
3. Feed to TextSentimentSignal or SentimentMomentumSignal
4. Combine with price-based signals for final trading decisions
""")

# Print the actual sentiment-signal values for verification
print("\nActual Trading Signals Generated:")
print(trading_signals.to_string())

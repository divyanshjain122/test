# Sentiment-Based Investing Roadmap

**Target Release**: 0.7.0  
**Status**: Planned  
**Priority**: High  

## Overview

Integrate sentiment analysis from news, social media, and financial reports into trading strategies and signals. This feature leverages the existing ML transformers infrastructure (BERT, attention models) to provide sentiment-driven decision making.

---

## Architecture

### 1. Data Sources & Collection

#### News Sources
- **Financial News APIs**:
  - Alpha Vantage News API
  - Finnhub News API
  - NewsAPI.org (business category)
  - Yahoo Finance RSS feeds
  
- **Implementation**: `src/jsf/data/sources/news.py`
  ```python
  class NewsDataSource:
      """Fetch financial news for sentiment analysis."""
      - fetch_news(symbols: List[str], lookback_days: int) -> pd.DataFrame
      - fetch_company_news(symbol: str) -> List[Article]
      - fetch_market_news(category: str) -> List[Article]
  ```

#### Social Media Sources
- **Twitter/X Financial Tweets** (via API v2)
- **Reddit** (r/wallstreetbets, r/stocks, r/investing)
- **StockTwits** community sentiment

- **Implementation**: `src/jsf/data/sources/social.py`
  ```python
  class SocialMediaSource:
      """Aggregate social media sentiment."""
      - fetch_tweets(symbols: List[str], count: int) -> pd.DataFrame
      - fetch_reddit_posts(subreddit: str, symbols: List[str]) -> pd.DataFrame
      - fetch_stocktwits(symbols: List[str]) -> pd.DataFrame
  ```

#### SEC Filings & Reports
- **10-K/10-Q filings** (Management Discussion & Analysis sections)
- **8-K filings** (Material events)
- **Earnings call transcripts**

- **Implementation**: `src/jsf/data/sources/filings.py`
  ```python
  class SECFilingsSource:
      """Extract text from SEC filings."""
      - fetch_filing(symbol: str, filing_type: str) -> str
      - extract_mda_section(filing_text: str) -> str
      - fetch_earnings_transcripts(symbol: str) -> List[str]
  ```

---

### 2. Sentiment Analysis Engine

#### Core Sentiment Models

**Location**: `src/jsf/ml/transformers/sentiment_engine.py`

```python
class SentimentEngine:
    """Multi-source sentiment aggregator."""
    
    def __init__(self, models: Dict[str, SentimentAnalyzer]):
        self.news_model = models.get("news", FinBERTSentiment())
        self.social_model = models.get("social", SimpleSentiment())
        self.filing_model = models.get("filing", FinBERTSentiment())
    
    def analyze_multi_source(
        self, 
        symbol: str, 
        sources: List[str] = ["news", "social", "filings"]
    ) -> SentimentScore:
        """Aggregate sentiment from multiple sources."""
        pass
    
    def get_sentiment_timeseries(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Historical sentiment data aligned with price data."""
        pass
```

#### Sentiment Metrics

```python
@dataclass
class SentimentScore:
    """Comprehensive sentiment measurement."""
    overall: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    
    # Source-specific scores
    news_sentiment: float
    social_sentiment: float
    filing_sentiment: float
    
    # Temporal metrics
    sentiment_trend: float  # Change over time
    sentiment_volatility: float  # Stability of sentiment
    
    # Volume metrics
    news_volume: int
    social_volume: int
    
    # Timestamp
    timestamp: datetime
```

---

### 3. Sentiment Features for ML

**Location**: `src/jsf/ml/features.py` (extend existing `FeatureExtractor`)

```python
class SentimentFeatureExtractor:
    """Convert sentiment to ML features."""
    
    def extract_sentiment_features(
        self, 
        data: pd.DataFrame, 
        symbol: str
    ) -> pd.DataFrame:
        """
        Features:
        - sentiment_score: Raw sentiment (-1 to 1)
        - sentiment_ma_7d: 7-day moving average
        - sentiment_ma_30d: 30-day moving average
        - sentiment_momentum: Rate of change
        - sentiment_divergence: Price vs sentiment gap
        - news_volume_spike: Unusual news activity
        - social_buzz: Social media activity level
        """
        pass
```

#### Integration with Existing ML Pipeline

```python
# In src/jsf/ml/models.py - extend MLStrategy

class SentimentEnhancedStrategy(MLStrategy):
    """ML strategy with sentiment features."""
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # Existing technical features
        features = super().prepare_features(data)
        
        # Add sentiment features
        sentiment_extractor = SentimentFeatureExtractor()
        sentiment_features = sentiment_extractor.extract_sentiment_features(
            data, symbol=self.symbol
        )
        
        return pd.concat([features, sentiment_features], axis=1)
```

---

### 4. Sentiment-Based Signals

**Location**: `src/jsf/signals/sentiment.py` (extend existing module)

#### New Signal Types

```python
class NewsSentimentSignal(Signal):
    """Trade on news sentiment changes."""
    
    def __init__(
        self, 
        threshold: float = 0.3,  # Sentiment threshold for signal
        lookback: int = 7,  # Days to average
        volume_filter: bool = True  # Require high news volume
    ):
        pass
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Signal logic:
        - BUY: sentiment > threshold and rising
        - SELL: sentiment < -threshold and falling
        - NEUTRAL: otherwise
        """
        pass


class SentimentDivergenceSignal(Signal):
    """Detect price-sentiment divergence for reversals."""
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Divergence patterns:
        - Bullish: Price falling, sentiment rising
        - Bearish: Price rising, sentiment falling
        """
        pass


class SocialBuzzSignal(Signal):
    """Identify unusual social media activity."""
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Signal on volume spikes:
        - Social mentions > 2 std dev above mean
        - Combined with positive/negative sentiment
        """
        pass
```

---

### 5. Strategies

**Location**: `src/jsf/strategies/sentiment_strategies.py` (new file)

```python
class SentimentMomentumStrategy(Strategy):
    """Follow strong sentiment trends."""
    
    def __init__(self):
        self.sentiment_signal = NewsSentimentSignal(threshold=0.4)
        self.technical_filter = MovingAverageCrossover(fast=20, slow=50)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Entry:
        - Strong positive sentiment (>0.4)
        - Confirmed by price above 50-day MA
        
        Exit:
        - Sentiment drops below 0.1
        - Or price crosses below 20-day MA
        """
        pass


class SentimentMeanReversionStrategy(Strategy):
    """Trade against extreme sentiment (contrarian)."""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Entry:
        - Extreme negative sentiment (<-0.6) → BUY
        - Extreme positive sentiment (>0.6) → SELL
        
        Exit:
        - Sentiment normalizes (returns to [-0.2, 0.2])
        """
        pass


class MultiFactorSentimentStrategy(Strategy):
    """Combine sentiment with fundamentals and technicals."""
    
    def __init__(self):
        self.sentiment_signal = NewsSentimentSignal()
        self.fundamental_signal = PERatioSignal()
        self.technical_signal = RSISignal()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Weighted scoring:
        - Sentiment: 40%
        - Fundamentals: 30%
        - Technicals: 30%
        
        Trade when combined score > threshold
        """
        pass
```

---

### 6. Real-Time Sentiment Updates

**Location**: `src/jsf/live/sentiment_feed.py` (new file)

```python
class SentimentFeed:
    """Real-time sentiment data for live trading."""
    
    def __init__(
        self, 
        symbols: List[str],
        update_interval: int = 300  # 5 minutes
    ):
        self.symbols = symbols
        self.engine = SentimentEngine()
        self.cache = {}
    
    async def start_feed(self):
        """Background task to update sentiment."""
        while True:
            for symbol in self.symbols:
                sentiment = await self.engine.analyze_multi_source(symbol)
                self.cache[symbol] = sentiment
            
            await asyncio.sleep(self.update_interval)
    
    def get_latest_sentiment(self, symbol: str) -> SentimentScore:
        """Get cached sentiment for symbol."""
        return self.cache.get(symbol)


# Integration with LiveEngine
class SentimentLiveEngine(LiveEngine):
    """LiveEngine with sentiment feed."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sentiment_feed = SentimentFeed(symbols=self.universe)
    
    async def start(self):
        # Start sentiment feed
        asyncio.create_task(self.sentiment_feed.start_feed())
        
        # Start normal live trading
        await super().start()
```

---

### 7. Backtesting & Evaluation

**Location**: Extend `src/jsf/simulation/backtest.py`

```python
class SentimentBacktest(Backtest):
    """Backtest with historical sentiment data."""
    
    def __init__(
        self, 
        strategy: Strategy,
        sentiment_data: pd.DataFrame,  # Historical sentiment
        **kwargs
    ):
        super().__init__(strategy, **kwargs)
        self.sentiment_data = sentiment_data
    
    def run(self) -> BacktestResult:
        """Run backtest with sentiment-aligned data."""
        # Merge price data with sentiment data by timestamp
        combined_data = self._merge_sentiment_and_price()
        
        return super().run(data=combined_data)
```

#### Performance Metrics

```python
class SentimentMetrics:
    """Analyze sentiment strategy performance."""
    
    @staticmethod
    def sentiment_accuracy(
        predictions: pd.Series,  # Sentiment predictions
        returns: pd.Series  # Actual returns
    ) -> float:
        """Correlation between sentiment and next-period returns."""
        pass
    
    @staticmethod
    def sentiment_alpha(
        strategy_returns: pd.Series,
        market_returns: pd.Series,
        sentiment_scores: pd.Series
    ) -> float:
        """Returns attributed to sentiment vs market."""
        pass
```

---

## Implementation Plan

### Phase 1: Data Infrastructure (2-3 weeks)
- [ ] Implement `NewsDataSource` (Alpha Vantage, Finnhub)
- [ ] Implement `SocialMediaSource` (Twitter, Reddit)
- [ ] Implement `SECFilingsSource` (SEC EDGAR)
- [ ] Create data pipeline to fetch and store sentiment data
- [ ] Write tests for data sources

### Phase 2: Sentiment Engine (1-2 weeks)
- [ ] Implement `SentimentEngine` with multi-source aggregation
- [ ] Create `SentimentScore` dataclass and metrics
- [ ] Integrate FinBERT and BERT models from existing transformers module
- [ ] Build sentiment time-series alignment with price data
- [ ] Write tests for sentiment analysis

### Phase 3: Features & Signals (1 week)
- [ ] Implement `SentimentFeatureExtractor`
- [ ] Add sentiment features to `FeatureExtractor`
- [ ] Create `NewsSentimentSignal`, `SentimentDivergenceSignal`, `SocialBuzzSignal`
- [ ] Write tests for signals

### Phase 4: Strategies (1 week)
- [ ] Implement `SentimentMomentumStrategy`
- [ ] Implement `SentimentMeanReversionStrategy`
- [ ] Implement `MultiFactorSentimentStrategy`
- [ ] Write backtests for strategies

### Phase 5: Live Trading Integration (1 week)
- [ ] Implement `SentimentFeed` for real-time updates
- [ ] Integrate with `LiveEngine`
- [ ] Add sentiment monitoring to dashboard
- [ ] Write integration tests

### Phase 6: Documentation & Examples (3-5 days)
- [ ] User guide for sentiment strategies
- [ ] Example notebooks demonstrating sentiment analysis
- [ ] API documentation
- [ ] Tutorial: Building a sentiment-driven strategy

---

## Dependencies

### New Python Packages
```toml
# pyproject.toml additions

[project.optional-dependencies]
sentiment = [
    "newsapi-python>=0.2.7",  # NewsAPI client
    "praw>=7.7.0",  # Reddit API
    "tweepy>=4.14.0",  # Twitter API
    "sec-edgar-downloader>=5.0.0",  # SEC filings
    "beautifulsoup4>=4.12.0",  # HTML parsing
    "feedparser>=6.0.0",  # RSS feeds
]
```

### API Keys Required
- **Alpha Vantage**: Free tier (500 calls/day)
- **Finnhub**: Free tier (60 calls/min)
- **NewsAPI**: Free tier (1000 requests/day)
- **Twitter API**: Essential tier ($100/month)
- **Reddit API**: Free (OAuth required)

---

## Data Storage

### Sentiment Database Schema

```sql
-- TimescaleDB or PostgreSQL with time-series extension

CREATE TABLE sentiment_scores (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    source VARCHAR(20) NOT NULL,  -- 'news', 'social', 'filing'
    sentiment FLOAT NOT NULL,  -- -1.0 to 1.0
    confidence FLOAT,
    volume INT,
    PRIMARY KEY (timestamp, symbol, source)
);

CREATE INDEX idx_sentiment_symbol_time ON sentiment_scores (symbol, timestamp DESC);
CREATE INDEX idx_sentiment_source ON sentiment_scores (source);
```

### Caching Strategy
- **Redis** for real-time sentiment cache (5-minute TTL)
- **PostgreSQL** for historical sentiment storage
- **Parquet files** for backtesting datasets

---

## Monitoring & Alerts

### Sentiment Alerts
```python
class SentimentAlert(Alert):
    """Alert on significant sentiment changes."""
    
    def check_condition(self, data: pd.DataFrame) -> bool:
        current_sentiment = data['sentiment'].iloc[-1]
        prev_sentiment = data['sentiment'].iloc[-2]
        
        # Alert on large sentiment swings
        if abs(current_sentiment - prev_sentiment) > 0.5:
            return True
        
        # Alert on extreme sentiment
        if abs(current_sentiment) > 0.8:
            return True
        
        return False
```

---

## Testing Strategy

### Unit Tests
- Data source fetching (mocked APIs)
- Sentiment model predictions
- Feature extraction correctness
- Signal generation logic

### Integration Tests
- End-to-end pipeline: data → sentiment → signals → trades
- Multi-source sentiment aggregation
- Live feed integration

### Backtests
- Historical sentiment strategies (2020-2024)
- Compare sentiment-enhanced vs baseline strategies
- Stress test on major events (COVID crash, 2022 bear market)

---

## Success Metrics

### Performance Targets
- **Sentiment-Market Correlation**: >0.3 for news sentiment vs next-day returns
- **Signal Win Rate**: >55% for sentiment-based entry signals
- **Alpha Generation**: >2% annualized over baseline strategy
- **Latency**: <30 seconds from news event to signal generation

### Quality Metrics
- **Model Accuracy**: >70% for FinBERT sentiment classification
- **Data Coverage**: >90% of trading days with sentiment data
- **API Reliability**: <1% failed API calls

---

## Risks & Mitigation

### Data Quality Risks
- **Risk**: Noisy social media data (spam, bots)
- **Mitigation**: Implement volume filters, account quality checks, spam detection

### Model Risks
- **Risk**: Sentiment models trained on general text, not finance-specific
- **Mitigation**: Use FinBERT (pre-trained on financial news), fine-tune on proprietary data

### Latency Risks
- **Risk**: Delayed news/sentiment data in live trading
- **Mitigation**: Use real-time APIs, implement caching, set max staleness thresholds

### Overfitting Risks
- **Risk**: Strategies overfit to historical sentiment patterns
- **Mitigation**: Walk-forward optimization, out-of-sample testing, regime detection

---

## Future Enhancements (Post-0.7.0)

### Advanced Features
1. **Entity-Specific Sentiment**: Track sentiment for specific executives, products, competitors
2. **Event Detection**: Automatically identify earnings, M&A, regulatory events
3. **Cross-Asset Sentiment**: Correlate sentiment across related assets (sector, supply chain)
4. **Multilingual Sentiment**: Analyze non-English news sources
5. **Alternative Data**: Satellite imagery, credit card data, web traffic

### ML Improvements
1. **Fine-Tuned Models**: Train custom BERT on proprietary trading data
2. **Attention Visualization**: Show which words/phrases drive sentiment
3. **Ensemble Models**: Combine multiple sentiment models (BERT, VADER, TextBlob)
4. **Sentiment Forecasting**: Predict future sentiment from current trends

### Infrastructure
1. **Distributed Crawling**: Scale news/social media scraping with Celery/Redis
2. **Real-Time Streaming**: Apache Kafka for event-driven sentiment updates
3. **GPU Acceleration**: Batch sentiment inference on GPUs for large datasets

---

## References & Resources

### Research Papers
- "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" (Araci, 2019)
- "Deep Learning for Event-Driven Stock Prediction" (Ding et al., 2015)
- "Twitter Mood Predicts the Stock Market" (Bollen et al., 2011)

### Datasets
- **Financial PhraseBank**: 5000 sentences from financial news (sentiment labels)
- **SemEval 2017 Task 5**: Financial news headline sentiment
- **StockNet**: Stock movement prediction with tweets

### Tools & Libraries
- **FinBERT**: https://huggingface.co/ProsusAI/finbert
- **VADER**: Rule-based sentiment for social media
- **spaCy**: NLP preprocessing
- **Transformers**: HuggingFace library for BERT models

---

## Version History

| Version | Changes | Date |
|---------|---------|------|
| 0.1 | Initial roadmap draft | 2024-01-XX |

---

## Maintainers

- **Primary**: Jai Ansh Bindra
- **Contributors**: Anubhav (handover target)

---

**Next Steps**: Review roadmap → Approve architecture → Begin Phase 1 implementation

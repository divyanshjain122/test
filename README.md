# JSF-Core: JBAC Strategy Foundry

[![PyPI version](https://img.shields.io/pypi/v/jsf-core.svg)](https://pypi.org/project/jsf-core/)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-635%20passing-brightgreen.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> ⚠️ **EDUCATIONAL PURPOSE ONLY** — This software is for **learning and research only**. It is NOT intended for trading with real money. See [Disclaimer](#disclaimer).

**JSF-Core** is a production-grade, open-source quantitative research engine for building, backtesting, and optimizing algorithmic trading strategies. It ships with a real-time Streamlit monitoring dashboard, FinBERT-powered sentiment analysis, live paper trading via Alpaca, and multi-channel alert support.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Running the Dashboard](#running-the-dashboard)
- [Live Paper Trading](#live-paper-trading)
- [Sentiment Analysis (NLP)](#sentiment-analysis-nlp)
- [Module Reference](#module-reference)
- [Project Structure](#project-structure)
- [Development](#development)
- [Disclaimer](#disclaimer)

---

## Features

| Category | What's Included |
|---|---|
| **Backtesting** | Transaction costs, slippage, position tracking, multi-symbol |
| **Signals** | 10+ signal types: Momentum, Mean-Reversion, Volatility, Sentiment |
| **Portfolio** | 24 components: Equal weight, Risk Parity, Kelly, Min-Variance |
| **Strategies** | Pre-built Momentum, Mean-Reversion, Trend-Following templates |
| **ML** | XGBoost, LightGBM, Neural Networks, walk-forward validation |
| **NLP** | FinBERT sentiment signals (positive/negative/neutral on news) |
| **Dashboard** | Real-time Streamlit UI: P&L, Portfolio, Trades, Risk pages |
| **Alerts** | Telegram, Console, Email notifications |
| **Live Trading** | Alpaca paper/live trading integration |
| **Metrics** | 20+ metrics: Sharpe, Sortino, Calmar, VaR, CVaR, Win Rate |

---

## Installation

### Step 1 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install from PyPI

```bash
# Full install — backtesting + dashboard + live trading + alerts (recommended)
pip install jsf-core

# + FinBERT NLP sentiment analysis and neural network strategies
# (adds PyTorch ~800 MB, TensorFlow ~500 MB, FinBERT ~400 MB)
pip install jsf-core[ml]

# Minimal install — backtesting engine only, no dashboard / trading / alerts
# Use this on CI servers or resource-constrained environments
pip install jsf-core[lite]
```

After install, verify with:

```bash
jsf version
# jsf-core version 0.7.6
```

### Install from Source (contributors)

```bash
git clone https://github.com/JaiAnshSB26/JBAC-Strategy-Foundry.git
cd JBAC-Strategy-Foundry
python -m venv venv
venv\Scripts\activate       # Windows — use "source venv/bin/activate" on Mac/Linux
pip install -e ".[dev]"
pre-commit install
```

---

## Configuration

JSF-Core supports two configuration methods — use whichever fits your workflow. Both can be used together (YAML for settings, `.env` for secrets).

### Option A — `.env` file (API keys and secrets)

```bash
cp .env.example .env    # Mac/Linux
copy .env.example .env  # Windows
```

Then edit `.env`:

```env
# Alpaca paper trading — get free keys at https://alpaca.markets
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# News / market data — get free key at https://www.alphavantage.co
ALPHA_VANTAGE_API_KEY=your_key_here

# Telegram alerts — create a bot via @BotFather on Telegram
TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
TELEGRAM_CHAT_IDS=your_chat_id

# App settings
APP_ENV=development
LOG_LEVEL=INFO
ENABLE_TELEGRAM_ALERTS=true
```

> `.env` is in `.gitignore` — your secrets never touch git.

### Option B — `config.yml` (backtesting and dashboard settings)

```bash
cp config.example.yml config.yml    # Mac/Linux
copy config.example.yml config.yml  # Windows
```

Then edit `config.yml`:

```yaml
environment: development

backtest:
  initial_capital: 100000
  transaction_cost: 0.001     # 10 basis points per trade
  slippage: 0.0005            # 5 basis points
  benchmark: "SPY"
  risk_free_rate: 0.05

broker:
  provider: alpaca
  mode: paper                 # paper | live
  alpaca:
    api_key: ""               # or set ALPACA_API_KEY in .env
    secret_key: ""
    base_url: https://paper-api.alpaca.markets

dashboard:
  symbols: [AAPL, GOOGL, MSFT, AMZN, NVDA]
  initial_capital: 100000
  history_days: 90
  refresh_interval: 30        # seconds (0 = disabled)

alerts:
  channels:
    telegram:
      enabled: true
      bot_token: ""           # or set TELEGRAM_BOT_TOKEN in .env
    console: true
```

See [`config.example.yml`](config.example.yml) for all available options including ML, email alerts, and IB.

### Getting API Keys (all free)

| Service | Purpose | Sign up |
|---|---|---|
| **Alpaca** | Paper/live trading + market data | [alpaca.markets](https://alpaca.markets) |
| **Alpha Vantage** | Historical market data (25 req/day) | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| **Telegram Bot** | Trade alerts on your phone | [@BotFather](https://t.me/BotFather) on Telegram |

---

## Quick Start

### Backtesting a Strategy

```python
from jsf.data import load_data
from jsf.strategies import MomentumStrategy
from jsf.simulation import BacktestEngine, BacktestConfig

# Load synthetic data (no API key needed)
data = load_data(
    source="synthetic",
    symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
    start_date="2020-01-01",
    end_date="2023-12-31",
)

# Create strategy
strategy = MomentumStrategy(name="momentum_60d", lookback=60, long_only=True)

# Configure and run backtest
config = BacktestConfig(
    initial_capital=100_000,
    transaction_cost=0.001,
    slippage=0.0005,
)
engine = BacktestEngine(config)
result = engine.run_strategy(strategy, data)

print(f"Total Return : {result.total_return:.2%}")
print(f"Sharpe Ratio : {result.sharpe_ratio:.2f}")
print(f"Max Drawdown : {result.max_drawdown:.2%}")
```

### Backtesting with Real Data (Alpaca)

```python
from jsf.data import load_data

# Set ALPACA_API_KEY and ALPACA_SECRET_KEY in environment or .env file
data = load_data(
    source="alpaca",
    symbols=["AAPL", "MSFT"],
    start_date="2023-01-01",
    end_date="2023-12-31",
)
```

### Custom Signal

```python
from jsf.signals import BaseSignal, SignalResult
import pandas as pd

class MyMomentumSignal(BaseSignal):
    def generate(self, data: pd.DataFrame) -> SignalResult:
        score = data["close"].pct_change(20).iloc[-1]
        return SignalResult(
            value=score,
            direction="long" if score > 0 else "short",
        )
```

### Running the Example Scripts

Example scripts are included in the repository (not in the pip package). Clone the repo to run them:

```bash
git clone https://github.com/JaiAnshSB26/JBAC-Strategy-Foundry.git
cd JBAC-Strategy-Foundry
pip install -e ".[dev]"

python examples/quickstart.py                   # Simple backtest
python examples/complete_backtest_example.py    # Full metrics + charts
python examples/ml_example.py                   # ML strategy
python examples/paper_trading_alerts.py         # Live paper trading
```

---

## Running the Dashboard

The Streamlit dashboard provides a live monitoring UI with P&L, positions, trades, risk analytics, and interactive backtesting.

### Launch

```bash
# Option 1: CLI command (recommended)
jsf dashboard

# Option 2: Python module
python -m jsf.dashboard
```

Opens at **http://localhost:8501**

### Demo Mode (no API keys needed)

Click **"Start Demo Mode"** in the sidebar. This instantly generates:
- 90 days of historical mock equity data
- 40 mock trades across AAPL, GOOGL, MSFT, AMZN, NVDA
- Full metrics: Sharpe, drawdown, VaR, win rate, monthly heatmap

### Dashboard Pages

| Page | What it shows |
|---|---|
| **Overview** | Current positions table, allocation pie chart |
| **Portfolio** | Detailed position breakdown, weights, sector allocation |
| **P&L** | Equity curve, drawdown, daily returns, monthly heatmap, date range filters |
| **Trades** | Full trade history, buy/sell breakdown, filters by symbol/date, CSV export |
| **Risk** | VaR, CVaR, volatility, Calmar ratio, exposure analysis |
| **Backtest** | Interactive strategy tester with all 3 strategy types, parameter sliders, equity/drawdown charts |
| **Settings** | Connection status, version info, auto-refresh interval |

### Connect to Your Alpaca Account

Go to **Settings** in the dashboard sidebar and enter your Alpaca API key and secret. The dashboard will display your actual paper trading portfolio and update in real time.

---

## Live Paper Trading

Connect to Alpaca paper trading to run strategies with real-time market data (no real money).

```python
from jsf.broker import AlpacaBroker
from jsf.live import LiveTradingEngine, EngineConfig
from jsf.strategies import MomentumStrategy

# Initialize broker (reads from ALPACA_API_KEY env var)
broker = AlpacaBroker(paper=True)

# Create strategy
strategy = MomentumStrategy(name="live_momentum", lookback=20)

# Configure and start engine
config = EngineConfig(symbols=["AAPL", "MSFT"], poll_interval=60)
engine = LiveTradingEngine(broker=broker, strategy=strategy, config=config)
engine.start()
```

**Set up Telegram alerts:**

```bash
jsf setup-telegram
```

This wizard guides you through creating a Telegram bot and saves the credentials to `.env`.

---

## Sentiment Analysis (NLP)

Requires `pip install jsf-core[ml]` — adds PyTorch, TensorFlow, and the HuggingFace Transformers library. The FinBERT model downloads ~400 MB on first run, then caches locally.

### Single prediction

```python
from jsf.ml.transformers.bert import FinBERT

model = FinBERT()

result = model.predict_one("Apple reports record earnings, beating analyst estimates by 20%.")
print(result.label.value)      # 'positive'
print(f"{result.score:.0%}")   # e.g. 96%
print(result.probabilities)    # {'positive': 0.96, 'negative': 0.01, 'neutral': 0.03}
```

### Batch prediction

```python
headlines = [
    "Revenue grew 25% year over year",
    "Company faces bankruptcy and mass layoffs",
    "Quarterly report was released on schedule",
]
for text, r in zip(headlines, model.predict(headlines)):
    print(f"{r.label.value:8s}  {r.score:.0%}  {text}")

# positive   94%  Revenue grew 25% year over year
# negative   97%  Company faces bankruptcy and mass layoffs
# neutral    81%  Quarterly report was released on schedule
```

### Sentiment signal in strategy

```python
from jsf.signals.sentiment import SentimentMomentumSignal

signal = SentimentMomentumSignal(
    name="finbert_momentum",
    model_type="finbert",       # "simple" for lightweight rule-based version
    lookback=7,
    sentiment_threshold=0.3,
    momentum_threshold=0.1,
)
# BUY  when sentiment > threshold AND price momentum rising
# SELL when sentiment < -threshold AND price momentum falling
```

---

## Module Reference

```
jsf/
├── config/         Configuration schemas and YAML/env loaders
├── data/           Data loaders: Synthetic, Alpaca, YFinance, AlphaVantage
├── signals/        Signal generators: Momentum, MeanReversion, Volatility, Sentiment
├── portfolio/      Position sizing, optimization, rebalancing, risk constraints
├── strategies/     Ready-made strategies: Momentum, MeanReversion, TrendFollowing
├── simulation/     BacktestEngine with transaction costs, slippage, tracking
├── optimization/   Grid search and walk-forward parameter optimization
├── evaluation/     Performance metrics (20+ metrics)
├── ml/             XGBoost, LightGBM, Neural Networks, FinBERT NLP
├── broker/         Alpaca and Interactive Brokers live trading
├── dashboard/      Streamlit real-time monitoring dashboard
├── alerts/         Telegram, Console, Email alert system
├── cli/            Command-line tools (setup_telegram, etc.)
└── utils/          Shared utilities and helpers
```

---

## Project Structure

```
JBAC-Strategy-Foundry/
├── src/jsf/                    Main package source
├── tests/                      Test suite (635 tests)
├── examples/                   Runnable example scripts
│   ├── quickstart.py
│   ├── complete_backtest_example.py
│   ├── ml_example.py
│   └── paper_trading_alerts.py
├── demos/                      Feature demo scripts
│   ├── demo_real_sentiment.py
│   ├── demo_ml_pipeline.py
│   └── demo_realtime_news.py
├── docs/                       Extended documentation
├── config.example.yml          Config template  →  copy to config.yml
├── .env.example                API key template →  copy to .env
├── pyproject.toml              Package metadata and dependencies
├── requirements.txt            Full dependency list with comments
└── Makefile                    Dev shortcuts: test, lint, format, build
```

---

## Development

```bash
make test        # Run all 635 tests with coverage report
make test-fast   # Run tests without coverage
make lint        # ruff + mypy
make format      # black formatter
make check       # lint + test (run before committing)
make build       # Build wheel + sdist into dist/
make clean       # Remove build artifacts
```

---

## Disclaimer

**EDUCATIONAL PURPOSE ONLY**

This software is provided for **educational and research purposes only**. It is designed to help students, researchers, and developers learn about quantitative trading concepts, backtesting methodologies, and algorithmic strategy development.

1. **NOT FINANCIAL ADVICE** — Nothing here constitutes investment, tax, or legal advice.
2. **NO REAL TRADING** — This software is not intended for use with real money.
3. **PAST PERFORMANCE** — Backtested results do not guarantee future returns.
4. **SUBSTANTIAL RISK** — Trading involves substantial risk of total capital loss.
5. **NO WARRANTY** — Provided "AS IS" with no liability for any losses or damages.

By using this software you accept full responsibility for any outcomes.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

**JBAC EdTech** · GitHub: [@JaiAnshSB26](https://github.com/JaiAnshSB26)

**PyPI**: [pypi.org/project/jsf-core](https://pypi.org/project/jsf-core/)

# JSF-Core: JBAC Strategy Foundry

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**JSF-Core** is a production-grade, open-source quantitative research engine for building, backtesting, and optimizing trading strategies. Built by JBAC EdTech, it provides a modular, type-safe, and extensible framework for systematic trading research.

## 🎯 Features

- **Modular Architecture**: Clean separation between data, signals, strategies, simulation, and evaluation
- **Type-Safe**: Full Pydantic validation and Python type hints throughout
- **Extensible**: Easy to add custom signals, strategies, and evaluation metrics
- **Production-Ready**: Comprehensive logging, error handling, and testing
- **Reproducible**: Deterministic backtests with seed control
- **Fast**: Vectorized operations and optional parallel processing
- **Well-Documented**: Detailed docstrings and usage examples

## 📦 Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/JaiAnshSB26/JBAC-Strategy-Foundry.git
cd JBAC-Strategy-Foundry

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### From PyPI (Coming Soon)

```bash
pip install jsf-core
```

## 🚀 Quick Start

```python
from jsf.data import load_data
from jsf.strategies import MomentumStrategy
from jsf.simulation import BacktestEngine, BacktestConfig

# Load data
data = load_data(
    source='synthetic',
    symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Create strategy
strategy = MomentumStrategy(
    name="momentum_60d",
    lookback=60,
    long_only=True
)

# Configure backtest
config = BacktestConfig(
    initial_capital=100000,
    transaction_cost=0.001,  # 10 bps
    slippage=0.0005,  # 5 bps
)

# Run backtest
engine = BacktestEngine(config)
result = engine.run_strategy(strategy, data)

# Display results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Total Trades: {len(result.trades)}")

# Full metrics
from jsf.simulation import calculate_all_metrics
metrics = calculate_all_metrics(result.returns)
print(f"\nCAGR: {metrics['cagr']:.2%}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

**See `examples/complete_backtest_example.py` for comprehensive examples including:**
- Basic backtesting
- Multi-strategy comparison
- Detailed performance metrics
- Transaction cost analysis

## 📖 Documentation

### Core Modules

- **`jsf.config`**: Configuration schemas and validation
- **`jsf.data`**: Data loading and preprocessing (synthetic & real data)
- **`jsf.signals`**: Signal generation (momentum, mean-reversion, trend, volatility, etc.)
- **`jsf.portfolio`**: Portfolio construction (position sizing, optimization, rebalancing)
- **`jsf.strategies`**: Strategy templates (momentum, mean-reversion, trend-following)
- **`jsf.simulation`**: Backtesting engine with transaction costs and performance metrics
- **`jsf.optimization`**: Parameter optimization (coming soon)

### Available Components

**Signals** (10+ types):
- Momentum, Mean Reversion, Trend Following
- Volatility, Volume, Moving Average Cross
- Breakout, Support/Resistance, Seasonality

**Portfolio Construction** (24 components):
- Position Sizing: Equal weight, signal-weighted, volatility-scaled, risk parity, Kelly
- Optimization: Min variance, max Sharpe, mean-variance, risk parity, max diversification
- Rebalancing: Periodic, threshold-based, signal-triggered, volatility-adjusted, cost-aware
- Constraints: Long-only, leverage, position limits, sector, turnover, exposure

**Strategies** (3 templates):
- Momentum Strategy (trend following)
- Mean Reversion Strategy (counter-trend)
- Trend Following Strategy (MA cross + trend strength)

**Performance Metrics** (20+ metrics):
- Returns: Total, CAGR, mean daily
- Risk: Volatility, downside deviation, max drawdown, VaR, CVaR
- Risk-adjusted: Sharpe, Sortino, Calmar
- Trading: Win rate, profit factor, avg win/loss
- Distribution: Skewness, kurtosis

## 🧪 Testing

```bash
# Run all tests
make test

# Run fast tests (no coverage)
make test-fast

# Run linting
make lint

# Format code
make format
```

## 🛠 Development

### Project Structure

```
jsf-core/
├── src/jsf/              # Main package
│   ├── config/           # Configuration schemas
│   ├── data/             # Data loading & preprocessing
│   ├── signals/          # Signal generation
│   ├── strategies/       # Strategy templates
│   ├── simulation/       # Backtest engine
│   ├── optimization/     # Parameter optimization
│   ├── evaluation/       # Performance metrics
│   ├── reporting/        # Report generation
│   └── utils/            # Utilities
├── tests/                # Test suite
├── docs/                 # Documentation
└── examples/             # Usage examples
```

### Development Roadmap

**Phase 1**: Foundation & Project Structure ✅  
**Phase 2**: Core Configuration System ✅  
**Phase 3**: Data Loading Infrastructure ✅  
**Phase 4-6**: Signal Framework ✅  
**Phase 7**: Portfolio Construction ✅  
**Phase 8**: Strategy Templates ✅  
**Phase 9**: Backtesting & Simulation Engine ✅  
**Phase 10**: Evaluation & Robustness (In Progress)  
**Phase 11**: Parameter Optimization  
**Phase 12**: Walk-Forward Analysis  
**Phase 13**: Reporting & Visualization  
**Phase 14**: High-Level API  
**Phase 15**: CLI Tool  
**Phase 16**: Advanced Strategies  
**Phase 17**: Real Data Integration  
**Phase 18**: SDK Documentation  
**Phase 19**: Performance Optimization  
**Phase 20**: Release Preparation  

### Current Status (Phase 9 Complete)

✅ **Fully Functional End-to-End System**:
- Data loading (synthetic + real data support)
- Signal generation (10+ signal types)
- Portfolio construction (24 components)
- Strategy templates (3 pre-built strategies)
- Backtesting engine (transaction costs, slippage, tracking)
- Performance metrics (20+ metrics)

**Latest Results** (from examples):
- Momentum 120d: 90.48% return, 2.11 Sharpe, -5.75% max drawdown
- Trend Following: 69.75% return, 1.85 Sharpe, -8.39% max drawdown
- Transaction cost impact: 21.72% → 3.51% (no costs → high costs)

**Next Steps**:
1. Add visualization/plotting capabilities
2. Implement walk-forward analysis
3. Parameter optimization framework
4. Real market data integration  

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run checks before committing
make check
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python best practices
- Inspired by leading quantitative research frameworks
- Powered by NumPy, Pandas, and Pydantic

## 📧 Contact

**JBAC EdTech**  
GitHub: [@JaiAnshSB26](https://github.com/JaiAnshSB26)

---

**Status**: 🚧 Active Development (v0.5.0-dev)  
**Phase**: 7/20 - Portfolio Construction Complete  
**Next**: Phase 8-9 - Strategy Templates
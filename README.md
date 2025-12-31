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
from jsf import run_experiment, ExperimentConfig

# Define your experiment
config = ExperimentConfig(
    universe="SP500",
    strategy="ts_momentum",
    start_date="2015-01-01",
    end_date="2023-01-01",
    parameters={"lookback": 60},
    initial_capital=100000,
)

# Run the backtest
result = run_experiment(config)

# Access results
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")

# Plot results
result.plot_pnl()
result.plot_drawdown()

# Generate report
result.save_report("my_experiment.html")
```

## 📖 Documentation

### Core Modules

- **`jsf.config`**: Configuration schemas and validation
- **`jsf.data`**: Data loading and preprocessing
- **`jsf.signals`**: Signal generation (time-series, cross-sectional)
- **`jsf.strategies`**: Strategy templates (momentum, mean-reversion, etc.)
- **`jsf.simulation`**: Backtesting engine and portfolio simulation
- **`jsf.optimization`**: Parameter optimization (grid search, Bayesian)
- **`jsf.evaluation`**: Performance metrics and robustness tests
- **`jsf.reporting`**: Report generation and visualization

### Strategy Templates (MVP)

1. **Time-Series Momentum**: Trend-following based on historical returns
2. **Time-Series Mean Reversion**: Counter-trend based on z-scores
3. **Cross-Sectional Momentum**: Relative strength ranking across assets

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
**Phase 4-6**: Signal Framework ✅ (Phase 4 complete)  
**Phase 7-9**: Strategy Templates  
**Phase 10-11**: Backtesting Engine  
**Phase 12-13**: Evaluation & Robustness  
**Phase 14-15**: Optimization Framework  
**Phase 16**: Reporting & Visualization  
**Phase 17**: High-Level API  
**Phase 18**: CLI Tool  
**Phase 19**: SDK Documentation  
**Phase 20**: Release Preparation  

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

**Status**: 🚧 Active Development (v0.4.0-dev)  
**Phase**: 4/20 - Signal Framework Phase 1 Complete  
**Next**: Phase 5-6 - Signal Framework Completion
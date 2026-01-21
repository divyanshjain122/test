# Changelog

All notable changes to JSF-Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 11: Parameter Optimization

#### Added
- **Grid search optimization** (`jsf.optimization`):
  - `GridSearchOptimizer` - Systematic parameter search engine
  - `ParameterGrid` - Iterator over parameter combinations
  - `OptimizationResult` - Results container with best params and summary
  - `optimize_strategy` - Convenience function for quick optimization
- **Optimization features**:
  - Support for multiple parameters simultaneously
  - Customizable optimization metrics (Sharpe, return, Calmar, etc.)
  - Progress tracking with tqdm
  - Summary DataFrame with all tested combinations
  - Sorted results by metric
- **BacktestResult enhancements**:
  - Added `calmar_ratio` property (CAGR / abs(max_drawdown))

#### Examples
- Created `optimization_example.py` with 4 comprehensive examples
- Example 1: Single-parameter optimization (lookback period)
- Example 2: Multi-parameter optimization (lookback + long_only)
- Example 3: Compare optimized strategies
- Example 4: Optimize using different metrics

#### Testing
- Tested with momentum strategy optimization
- Found optimal parameters: lookback=30 (Sharpe 0.735)
- Multi-parameter: lookback=120, long_only=True (Sharpe 2.00, 95.65% return)
- Strategy comparison: Momentum beats Mean Reversion after optimization

### Phase 10: Visualization & Reporting

#### Added
- **Comprehensive plotting module** (`jsf.visualization`):
  - `plot_equity_curve` - Portfolio value over time with key statistics
  - `plot_drawdown` - Equity + drawdown chart with max drawdown highlight
  - `plot_returns_distribution` - Histogram + Q-Q plot with normality test
  - `plot_rolling_metrics` - 6-panel dashboard (returns, volatility, Sharpe, max DD, win rate, cumulative)
  - `plot_monthly_returns` - Heatmap of monthly performance by year
  - `plot_performance_summary` - Complete dashboard with all key metrics
- **Professional styling**:
  - Seaborn-based theme
  - Publication-quality charts (300 DPI)
  - Automatic date formatting
  - Currency formatting
  - Color-coded performance indicators
- **Export capabilities**:
  - Save to PNG files
  - Configurable titles
  - Optional display control

#### Technical Details
- Uses matplotlib + seaborn for plotting
- GridSpec layouts for multi-panel dashboards
- Automatic handling of date indices
- Statistics boxes on charts
- Q-Q plots for normality analysis
- Heatmaps for periodic returns

#### Examples
- Created `visualization_example.py` demonstrating all 6 plot types
- Updated `complete_backtest_example.py` with visualization
- All plots tested and verified working

### Phase 9: Backtesting & Simulation Engine

#### Added
- **Backtesting infrastructure**:
  - `BacktestEngine` - Complete simulation engine for strategy backtesting
  - `BacktestConfig` - Configuration for capital, costs, slippage, margin
  - `BacktestResult` - Results container with equity curve, returns, positions, trades
- **Transaction cost modeling**:
  - Configurable transaction costs (default 10 bps)
  - Slippage modeling (default 5 bps)
  - Realistic execution simulation
  - Position tracking and trade logging
- **Performance metrics** (20+ metrics):
  - **Return metrics**: Total return, CAGR, mean return
  - **Risk metrics**: Volatility, downside deviation, max drawdown, VaR (95%), CVaR (95%)
  - **Risk-adjusted returns**: Sharpe ratio, Sortino ratio, Calmar ratio
  - **Trading metrics**: Win rate, profit factor, best/worst day, avg win/loss
  - **Distribution**: Skewness, kurtosis
- **Complete end-to-end examples**:
  - Basic momentum strategy backtest
  - Multi-strategy comparison (4 strategies)
  - Detailed performance metrics analysis
  - Transaction cost impact analysis

#### Technical Details
- Compound returns supported
- Equity curve generation
- Daily returns tracking
- Complete trade history
- Tested with realistic results: 2-90% returns, 0.3-2.1 Sharpe ratios
- Full integration with strategy framework

#### Testing
- Verified with 4 complete examples
- Real execution tests showing:
  - Example 1: 46.58% return, 0.76 Sharpe, -16.70% max drawdown
  - Example 2: 90.48% return, 2.11 Sharpe, -5.75% max drawdown (best)
  - Example 3: Comprehensive metrics demonstration
  - Example 4: Transaction cost impact (21.72% → 3.51% with high costs)

### Phase 8: Strategy Templates

#### Added
- **Base strategy infrastructure**:
  - `Strategy` - Abstract base class for all trading strategies
  - `StrategyType` - Enum for strategy classification (momentum, mean_reversion, trend_following, etc.)
  - `StrategyMetadata` - Metadata container for strategy documentation
- **Pre-built strategy templates** (3 templates):
  - `MomentumStrategy` - Classic momentum (buy winners, sell/avoid losers)
  - `MeanReversionStrategy` - Mean reversion (buy oversold, sell/avoid overbought)
  - `TrendFollowingStrategy` - Trend following with MA crossover + trend strength
- **Complete pipeline**:
  - Signal generation → Portfolio construction → Strategy execution
  - Configurable portfolio construction methods
  - Support for long-only and long-short strategies
  - Comprehensive parameter management

#### Technical Details
- All strategies implement run() method for complete execution
- Strategies combine multiple signals intelligently
- Full metadata tracking for reproducibility
- Extensible architecture for custom strategies

### Phase 7: Portfolio Construction

#### Added
- **Base portfolio infrastructure**:
  - `Portfolio` - Container for weights with exposure/turnover tracking
  - `PortfolioConstructor` - Abstract base for portfolio construction
  - `PositionSizer` - Abstract base for position sizing
  - `WeightOptimizer` - Abstract base for weight optimization
  - `Rebalancer` - Abstract base for rebalancing logic
  - `RebalanceFrequency` - Enum for rebalancing schedules
- **Position sizing methods** (5 methods):
  - `EqualWeightSizer` - Equal weight allocation
  - `SignalWeightedSizer` - Proportional to signal strength
  - `VolatilityScaledSizer` - Inverse volatility weighting (risk parity)
  - `RiskParitySizer` - Equal risk contribution (iterative optimization)
  - `KellyCriterionSizer` - Kelly formula with fractional sizing
- **Weight optimization methods** (5 optimizers):
  - `MinimumVarianceOptimizer` - Minimize portfolio variance
  - `MaxSharpeOptimizer` - Maximize risk-adjusted returns
  - `MeanVarianceOptimizer` - Balance expected return and risk
  - `RiskParityOptimizer` - Equalize risk contributions
  - `MaxDiversificationOptimizer` - Maximize diversification ratio
- **Rebalancing strategies** (5 strategies):
  - `PeriodicRebalancer` - Time-based rebalancing (daily/weekly/monthly/etc)
  - `ThresholdRebalancer` - Drift-based rebalancing
  - `VolatilityTargetRebalancer` - Maintain target volatility
  - `BandRebalancer` - Rebalance when outside allowed bands
  - `SmartRebalancer` - Combine multiple triggers intelligently
- **Portfolio constraints** (6 constraints):
  - `PortfolioConstraints` - Manage multiple constraints
  - `PositionLimitConstraint` - Individual position limits
  - `SectorConstraint` - Sector exposure limits
  - `TurnoverConstraint` - Limit trading turnover
  - `LeverageConstraint` - Gross/net exposure limits
  - `ConcentrationConstraint` - Herfindahl-based concentration limits
- **Portfolio constructors** (3 types):
  - `SimplePortfolioConstructor` - Basic signal-to-weights conversion
  - `OptimizedPortfolioConstructor` - Optimization-based construction
  - `HybridPortfolioConstructor` - Blend signals and optimization
- **Complete portfolio workflow**:
  - Signal → Position Sizing → Optimization → Constraints → Rebalancing → Portfolio
  - Support for long-only and long-short portfolios
  - Full exposure and turnover tracking
  - Comprehensive metadata capture

#### Technical Details
- All optimizers use scipy.optimize with SLSQP method
- Position sizing supports long-only and long-short modes
- Constraints support both check() and enforce() methods
- Rebalancing supports full and partial rebalancing
- Portfolio tracking includes gross/net/long/short exposure
- Complete end-to-end portfolio construction pipeline
- **Comprehensive test suite** (44 tests):
  - Base class tests (3 tests)
  - Position sizing tests (10 tests)
  - Optimization tests (8 tests)
  - Rebalancing tests (8 tests)
  - Constraints tests (9 tests)
  - Constructor tests (4 tests)
  - Integration workflow tests (2 tests)
  - Note: Tests reveal some API alignment needed with implementations
- **Usage examples** (`examples/portfolio_examples.py`):
  - 12 comprehensive examples demonstrating all components
  - Examples 1-3: Position sizing (equal weight, signal-weighted, volatility-scaled)
  - Examples 4-5: Optimization (minimum variance, maximum Sharpe)
  - Examples 6-7: Rebalancing (periodic, threshold-based)
  - Examples 8-10: Constraints (position limits, leverage, multi-constraint)
  - Examples 11-12: Full constructors (simple, optimized)
  - Real-world scenarios with synthetic data
- **Dependencies updated**:
  - pyarrow version constraint relaxed to >=12.0.0 (no upper bound)
  - pytest-cov added for test coverage analysis
  - All dependencies successfully installed

#### Documentation
- Full docstrings for all 24 portfolio components
- Type hints throughout codebase
- Comprehensive inline comments
- Usage examples showing real-world patterns

### Phase 5-6: Signal Framework Completion

#### Added
- **Fundamental analysis signals** (5 signals):
  - `ValueSignal` - Value investing using P/B ratios or price proxies
  - `QualitySignal` - Stability and quality metrics (Sharpe-like, downside deviation)
  - `GrowthSignal` - Multi-period growth trends (price + volume)
  - `SizeSignal` - Market cap factor (small vs large cap preference)
  - `DividendSignal` - Dividend yield strategies
- **Sentiment and market regime signals** (6 signals):
  - `MarketRegimeSignal` - Bull/bear/neutral regime detection
  - `BreadthSignal` - Market breadth (advancing vs declining stocks)
  - `RelativeStrengthSignal` - Performance vs benchmark
  - `NewHighLowSignal` - New highs/lows detection
  - `VolumeShockSignal` - Unusual volume spike detection
  - `SeasonalitySignal` - Calendar effects and patterns (monthly, quarterly, day-of-week)
- **Advanced composite patterns** (5 composite types):
  - `RotationSignal` - Top-N rotation strategies with rebalancing
  - `MultiTimeframeSignal` - Multi-timeframe confirmation
  - `AdaptiveWeightSignal` - Performance-based dynamic weight adjustment
  - `ThresholdFilterSignal` - Signal filtering by threshold (absolute/percentile)
  - `ConsensusSignal` - Require agreement across multiple signals
- **Signal transformation utilities**:
  - `normalize_signal` - Multiple normalization methods (z-score, min-max, rank, percentile, tanh)
  - `rank_signal` - Cross-sectional, time-series, and hybrid ranking
  - `smooth_signal` - Noise reduction (SMA, EMA, Gaussian)
  - `clip_signal` - Value clipping to ranges
  - `winsorize_signal` - Outlier handling
  - `demean_signal` - Cross-sectional and time-series demeaning
  - `neutralize_signal` - Factor neutralization via regression
  - `apply_decay` - Exponential decay with configurable half-life
  - `combine_signals` - Multiple signal aggregation (average, median, max, min)
  - `score_signals` - Convert signals to portfolio scores (z-score, percentile, exponential)
  - `NormalizationMethod` - Enum for normalization types
  - `RankingMethod` - Enum for ranking methods
- **Comprehensive test suite** (31 tests):
  - Fundamental signal tests (6 tests)
  - Sentiment signal tests (6 tests)
  - Advanced composite tests (5 tests)
  - Transformation utility tests (14 tests)
  - 100% test pass rate
  - 57% overall coverage (88% fundamental, 86% sentiment, 84% composites, 61% transforms)
- **Full module exports** in `signals/__init__.py`

#### Technical Details
- All signals follow consistent interface (generate method, get_metadata)
- Fundamental signals support both actual fundamental data and price-based proxies
- Sentiment signals work with price/volume data when sentiment data unavailable
- Advanced composites enable sophisticated multi-signal strategies
- Transformation utilities provide complete signal preprocessing pipeline
- Enums provide type safety for normalization and ranking methods

### Phase 4: Signal Framework (Part 1)

#### Added
- **Base signal infrastructure**:
  - `Signal` - Abstract base class for all signals
  - `SignalType` - Enumeration for signal types (technical, statistical, fundamental, sentiment, composite)
  - `SignalDirection` - Signal direction indicators (long, short, neutral)
  - `SignalMetadata` - Dataclass for signal metadata
  - `SignalError` - Custom exception for signal errors
  - `CompositeSignal` - Combine multiple signals with various methods
- **Signal caching system**:
  - Automatic caching of generated signals
  - Cache enable/disable functionality
  - Cache key generation from inputs
- **Technical indicator signals** (6 signals):
  - `MomentumSignal` - Rate of change momentum
  - `MovingAverageCrossSignal` - MA crossover strategy
  - `RSISignal` - Relative Strength Index
  - `BollingerBandsSignal` - Price position in bands
  - `MACDSignal` - MACD line crossover
  - `VolumeWeightedSignal` - Volume-confirmed momentum
- **Statistical signals** (5 signals):
  - `MeanReversionSignal` - Z-score based mean reversion
  - `PairsSignal` - Pairs trading spread
  - `TrendStrengthSignal` - Linear regression trend
  - `VolatilitySignal` - Volatility regime detection
  - `CorrelationSignal` - Rolling correlation signals
- **Signal composition**:
  - Average combination method
  - Weighted average with custom weights
  - Voting mechanism (majority vote)
  - Max/min signal selection
  - Automatic signal alignment
- **Comprehensive test suite** (26 tests):
  - Technical signal tests (12 tests)
  - Statistical signal tests (5 tests)
  - Composite signal tests (5 tests)
  - Base functionality tests (4 tests)
  - 100% test pass rate
- **Full module exports** in `__init__.py`

### Phase 3: Data Loading Infrastructure

#### Added
- **Base data infrastructure**:
  - `DataLoader` - Abstract base class for all loaders
  - `PriceData` - Container class for OHLCV data with rich functionality
  - `DataLoadError` - Custom exception for data loading errors
- **Multiple data loaders**:
  - `CSVLoader` - Load from CSV files (single file or directory)
  - `ParquetLoader` - Load from Parquet files (efficient binary format)
  - `InMemoryLoader` - Wrap existing DataFrames
  - `SyntheticDataLoader` - Generate test data with geometric Brownian motion
  - `load_data()` - Convenience function for quick loading
- **Universe management**:
  - `Universe` - Class for defining and manipulating asset universes
  - `UniverseFilter` - Dataclass for filtering criteria
  - `UNIVERSE_CONSTITUENTS` - Predefined universes (S&P 500, NASDAQ 100, DOW 30)
  - `create_universe()` - Factory function for easy universe creation
  - Set operations: intersection, union, difference
  - Sampling and filtering capabilities
- **Preprocessing utilities** (11 functions):
  - `handle_missing_data()` - ffill, bfill, drop, interpolate methods
  - `resample_data()` - Change frequency with OHLCV aggregation
  - `calculate_returns()` - Simple and log returns
  - `normalize_prices()` - Normalize to base value
  - `calculate_rolling_stats()` - Rolling mean and std
  - `align_data()` - Align multiple DataFrames
  - `remove_outliers()` - IQR and z-score methods
  - `winsorize()` - Cap extreme values
  - `forward_fill_gaps()` - Fill data gaps
  - `ensure_business_days()` - Enforce business day index
  - `calculate_volatility()` - Rolling volatility with annualization
- **PriceData container features**:
  - `get_symbol_data()` - Extract single symbol
  - `get_close_prices()` - Get close prices in wide format
  - `get_field()` - Get any field in wide format
  - `get_returns()` / `get_log_returns()` - Calculate returns
  - `summary()` - Get summary statistics
- **Comprehensive test suite** (24 tests):
  - Synthetic data generation tests
  - PriceData container tests
  - Universe operations tests
  - Preprocessing function tests
  - Loader tests (CSV, Parquet, InMemory)
  - Data validation tests
- **Examples**: data_examples.py with 7 usage patterns
- **Full module exports** in `__init__.py`

### Phase 2: Core Configuration System

#### Added
- **Pydantic configuration schemas**:
  - `JSFBaseConfig` - Base class with serialization/validation
  - `ExperimentConfig` - Complete experiment specification
  - `StrategyConfig` - Strategy parameters with validation
  - `OptimizationConfig` - Optimization settings
  - `CostConfig` - Transaction cost models
  - `RiskConfig` - Risk management settings
  - `DataConfig` - Data source configuration
  - `DateRangeConfig` - Date range validation
- **Enumerations** for all config options:
  - StrategyType, UniverseType, FrequencyType
  - RebalanceFrequency, OptimizationMethod
  - CostModel, PositionSizing, RiskMetric, DataSource
- **Default parameter presets**:
  - Strategy defaults for all 9 strategy types
  - Optimization parameter grids
  - Cost model presets (conservative, moderate, aggressive, zero_cost)
  - Risk model presets (conservative, moderate, aggressive, long_only)
- **Convenience functions**:
  - `create_experiment_config()` - Quick config builder
  - `quick_config()` - Simplified config with presets
  - `get_default_strategy_config()` - Get strategy defaults
  - `get_cost_preset()` / `get_risk_preset()` - Get presets
- **Comprehensive validation**:
  - Date format and range validation
  - Parameter JSON-serializability checks
  - Universe validation (non-empty lists)
  - Leverage and position size constraints
- **Full serialization support**:
  - JSON import/export with file support
  - Dictionary conversion
  - `copy_with()` for creating modified copies
- **Complete test suite** (70+ tests):
  - Schema validation tests
  - Serialization round-trip tests
  - Preset functionality tests
  - Error handling tests
- **Examples**: config_examples.py with 7 usage patterns

### Phase 1: Foundation & Project Structure

#### Added
- Initial project structure with modular architecture
- Core package structure (config, data, signals, strategies, simulation, optimization, evaluation, reporting, utils)
- Comprehensive pyproject.toml with dependencies and tooling configuration
- Requirements files for core and development dependencies
- pytest configuration and test structure with fixtures
- Utility modules:
  - Logging infrastructure with consistent formatting
  - Time utilities for date parsing and manipulation
  - I/O utilities for JSON, pickle, and DataFrame operations
  - Parallel processing utilities
- Pre-commit hooks configuration
- EditorConfig for consistent coding styles
- Makefile for common development tasks
- Comprehensive README with project overview
- MIT License
- Contributing guidelines
- This CHANGELOG

#### Infrastructure
- Git repository initialized
- .gitignore configured
- Code quality tools configured (Black, Ruff, mypy)
- Test coverage configuration
- EditorConfig for consistent formatting

---

## Version History

### [0.2.0-dev] - 2025-11-29

**Phase 2 Complete**: Core Configuration System

Comprehensive Pydantic-based configuration system with 7 schemas, 8 enums, presets, and 70+ tests.

### [0.1.0-dev] - 2025-11-29

**Phase 1 Complete**: Foundation & Project Structure

This is the initial development version establishing the project foundation.

---

## Roadmap

- ~~**Phase 1**: Foundation & Project Structure~~ ✅ Complete
- ~~**Phase 2**: Core Configuration System~~ ✅ Complete
- ~~**Phase 3**: Data Loading Infrastructure~~ ✅ Complete
- **Phase 4-6**: Signal Framework
- **Phase 7-9**: Strategy Templates
- **Phase 10-11**: Backtesting Engine
- **Phase 12-13**: Evaluation & Robustness
- **Phase 14-15**: Optimization Framework
- **Phase 16**: Reporting & Visualization
- **Phase 17**: High-Level API
- **Phase 18**: CLI Tool
- **Phase 19**: SDK Documentation
- **Phase 20**: Release Preparation

[unreleased]: https://github.com/JaiAnshSB26/JBAC-Strategy-Foundry/compare/v0.2.0-dev...HEAD
[0.2.0-dev]: https://github.com/JaiAnshSB26/JBAC-Strategy-Foundry/compare/v0.1.0-dev...v0.2.0-dev
[0.1.0-dev]: https://github.com/JaiAnshSB26/JBAC-Strategy-Foundry/releases/tag/v0.1.0-dev

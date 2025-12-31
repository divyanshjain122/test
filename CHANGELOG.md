# Changelog

All notable changes to JSF-Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

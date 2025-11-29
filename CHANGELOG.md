# Changelog

All notable changes to JSF-Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### [0.1.0-dev] - 2025-11-29

**Phase 1 Complete**: Foundation & Project Structure

This is the initial development version establishing the project foundation.

---

## Roadmap

- **Phase 2**: Core Configuration System
- **Phase 3**: Data Loading Infrastructure
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

[unreleased]: https://github.com/JaiAnshSB26/JBAC-Strategy-Foundry/compare/v0.1.0-dev...HEAD
[0.1.0-dev]: https://github.com/JaiAnshSB26/JBAC-Strategy-Foundry/releases/tag/v0.1.0-dev

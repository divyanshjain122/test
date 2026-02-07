"""
Visual Demo - JSF-Core Trading System
Shows charts and plots of the complete trading system.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for Windows

from jsf.data import SyntheticDataLoader, PriceData
from jsf.signals import MomentumSignal, MeanReversionSignal, RSISignal
from jsf.strategies import MomentumStrategy, MeanReversionStrategy
from jsf.simulation import BacktestResult
from jsf.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_monthly_returns,
    plot_performance_summary
)

print("=" * 80)
print("JSF-CORE VISUAL DEMO - Trading System Visualization")
print("=" * 80)

# ============================================================================
# SECTION 1: Load Data & Generate Signals
# ============================================================================
print("\n[1] Loading data and generating signals...")

loader = SyntheticDataLoader(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    start_date='2023-01-01',
    end_date='2024-12-31',
    annual_return=0.12,
    annual_volatility=0.25,
    seed=42
)
price_df = loader.load()
price_data = PriceData(data=price_df)

# Generate multiple signals
momentum_signal = MomentumSignal(lookback=20)
mean_rev_signal = MeanReversionSignal(lookback=20)
rsi_signal = RSISignal(period=14, oversold=30, overbought=70)

signals_mom = momentum_signal.generate(price_data)
signals_mr = mean_rev_signal.generate(price_data)
signals_rsi = rsi_signal.generate(price_data)

print(f"   Loaded: {len(price_data.symbols)} symbols, {len(price_data.data)} periods")
print(f"   Generated 3 signal types")

# ============================================================================
# SECTION 2: Run Strategies & Create Results
# ============================================================================
print("\n[2] Running strategies...")

initial_capital = 100000.0

# Strategy 1: Momentum
strategy_mom = MomentumStrategy(
    lookback=20,
    entry_threshold=0.5,
    exit_threshold=-0.3,
    name="Momentum_Strategy"
)
portfolio_mom = strategy_mom.run(price_data, capital=initial_capital)

# Strategy 2: Mean Reversion
strategy_mr = MeanReversionStrategy(
    lookback=20,
    entry_threshold=-0.5,
    exit_threshold=0.3,
    name="MeanReversion_Strategy"
)
portfolio_mr = strategy_mr.run(price_data, capital=initial_capital)

print(f"   Momentum: {len(portfolio_mom.weights)} periods")
print(f"   Mean Reversion: {len(portfolio_mr.weights)} periods")

# ============================================================================
# SECTION 3: Calculate Performance & Create Backtest Results
# ============================================================================
print("\n[3] Calculating performance metrics...")

# Create equity curves from portfolio weights
initial_capital = 100000.0
prices_df = price_data.get_close_prices()

# Momentum equity curve
returns_mom = (prices_df.pct_change().fillna(0) * portfolio_mom.weights.shift(1).fillna(0)).sum(axis=1)
equity_mom = initial_capital * (1 + returns_mom).cumprod()

# Mean Reversion equity curve  
returns_mr = (prices_df.pct_change().fillna(0) * portfolio_mr.weights.shift(1).fillna(0)).sum(axis=1)
equity_mr = initial_capital * (1 + returns_mr).cumprod()

# Create BacktestResult objects for visualization
result_mom = BacktestResult(
    equity_curve=equity_mom,
    returns=returns_mom,
    positions=portfolio_mom.weights,  # Using weights as proxy for positions
    trades=pd.DataFrame()  # Empty trades df
)

result_mr = BacktestResult(
    equity_curve=equity_mr,
    returns=returns_mr,
    positions=portfolio_mr.weights,
    trades=pd.DataFrame()
)

print(f"   Momentum - Return: {result_mom.total_return:.2%}, Sharpe: {result_mom.sharpe_ratio:.2f}")
print(f"   Mean Reversion - Return: {result_mr.total_return:.2%}, Sharpe: {result_mr.sharpe_ratio:.2f}")

# ============================================================================
# SECTION 4: Generate Visualizations
# ============================================================================
print("\n[4] Generating visualizations...")
print("   (Close each plot window to see the next one)")

# Plot 1: Price Data
print("\n   [Plot 1/8] Price data for all symbols...")
fig1, ax1 = plt.subplots(figsize=(14, 6))
for symbol in price_data.symbols:
    normalized = prices_df[symbol] / prices_df[symbol].iloc[0] * 100
    ax1.plot(prices_df.index, normalized, label=symbol, linewidth=2)
ax1.set_title("Normalized Price Data (Base=100)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Normalized Price", fontsize=12)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Signals Comparison
print("   [Plot 2/8] Signal values across time...")
fig2, (ax21, ax22, ax23) = plt.subplots(3, 1, figsize=(14, 10))

# Momentum signals
signals_mom.mean(axis=1).plot(ax=ax21, label='Momentum', color='blue', linewidth=2)
ax21.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax21.set_title("Momentum Signal (Average)", fontsize=12, fontweight='bold')
ax21.set_ylabel("Signal Strength")
ax21.legend()
ax21.grid(True, alpha=0.3)

# Mean Reversion signals
signals_mr.mean(axis=1).plot(ax=ax22, label='Mean Reversion', color='green', linewidth=2)
ax22.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax22.set_title("Mean Reversion Signal (Average)", fontsize=12, fontweight='bold')
ax22.set_ylabel("Signal Strength")
ax22.legend()
ax22.grid(True, alpha=0.3)

# RSI signals
signals_rsi.mean(axis=1).plot(ax=ax23, label='RSI', color='red', linewidth=2)
ax23.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax23.set_title("RSI Signal (Average)", fontsize=12, fontweight='bold')
ax23.set_xlabel("Date")
ax23.set_ylabel("Signal Strength")
ax23.legend()
ax23.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 3: Strategy Comparison
print("   [Plot 3/8] Strategy equity curves comparison...")
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(equity_mom.index, equity_mom.values, label='Momentum Strategy', 
         linewidth=2, color='blue')
ax3.plot(equity_mr.index, equity_mr.values, label='Mean Reversion Strategy',
         linewidth=2, color='green')
ax3.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax3.set_title("Strategy Comparison - Equity Curves", fontsize=14, fontweight='bold')
ax3.set_xlabel("Date", fontsize=12)
ax3.set_ylabel("Portfolio Value ($)", fontsize=12)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 4: Momentum Strategy - Equity Curve (detailed)
print("   [Plot 4/8] Momentum strategy - detailed equity curve...")
plot_equity_curve(
    result_mom,
    title="Momentum Strategy - Detailed Performance",
    show=True,
    save_path="momentum_equity.png"
)

# Plot 5: Mean Reversion Strategy - Equity Curve (detailed)
print("   [Plot 5/8] Mean Reversion strategy - detailed equity curve...")
plot_equity_curve(
    result_mr,
    title="Mean Reversion Strategy - Detailed Performance",
    show=True,
    save_path="mean_reversion_equity.png"
)

# Plot 6: Momentum - Drawdown Analysis
print("   [Plot 6/8] Momentum strategy - drawdown analysis...")
plot_drawdown(
    result_mom,
    title="Momentum Strategy - Drawdown",
    show=True,
    save_path="momentum_drawdown.png"
)

# Plot 7: Returns Distribution Comparison
print("   [Plot 7/8] Returns distribution comparison...")
fig7, (ax71, ax72) = plt.subplots(1, 2, figsize=(14, 6))

# Momentum returns
returns_mom.hist(bins=50, ax=ax71, color='blue', alpha=0.7, edgecolor='black')
ax71.axvline(returns_mom.mean(), color='red', linestyle='--', linewidth=2,
             label=f'Mean: {returns_mom.mean():.3%}')
ax71.set_title("Momentum Strategy - Returns Distribution", fontsize=12, fontweight='bold')
ax71.set_xlabel("Daily Returns")
ax71.set_ylabel("Frequency")
ax71.legend()
ax71.grid(True, alpha=0.3)

# Mean Reversion returns
returns_mr.hist(bins=50, ax=ax72, color='green', alpha=0.7, edgecolor='black')
ax72.axvline(returns_mr.mean(), color='red', linestyle='--', linewidth=2,
             label=f'Mean: {returns_mr.mean():.3%}')
ax72.set_title("Mean Reversion Strategy - Returns Distribution", fontsize=12, fontweight='bold')
ax72.set_xlabel("Daily Returns")
ax72.set_ylabel("Frequency")
ax72.legend()
ax72.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 8: Performance Summary Dashboard
print("   [Plot 8/8] Complete performance dashboard...")
plot_performance_summary(
    result_mom,
    title="Momentum Strategy - Complete Analysis Dashboard",
    show=True,
    save_path="momentum_summary.png"
)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VISUAL DEMO COMPLETE")
print("=" * 80)
print("\nGenerated 8 visualization plots:")
print("  1. Normalized price data for all symbols")
print("  2. Signal values (Momentum, Mean Reversion, RSI)")
print("  3. Strategy equity curves comparison")
print("  4. Momentum strategy detailed performance")
print("  5. Mean Reversion strategy detailed performance")
print("  6. Drawdown analysis")
print("  7. Returns distribution comparison")
print("  8. Complete performance dashboard")
print("\nSaved files:")
print("  - momentum_equity.png")
print("  - mean_reversion_equity.png")
print("  - momentum_drawdown.png")
print("  - momentum_summary.png")
print("\n" + "=" * 80)
print("SYSTEM VISUALIZATION COMPLETE - Charts show full trading pipeline!")
print("=" * 80)

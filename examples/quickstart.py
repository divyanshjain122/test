"""
Quick Start Example for JSF-Core

This example will be functional once Phase 17 is complete.
For now, it demonstrates the intended API.
"""

from jsf import run_experiment, ExperimentConfig

# TODO: This will work after Phase 17
# Currently serves as API design documentation

def main():
    """Run a simple momentum strategy backtest."""
    
    # Define experiment configuration
    config = ExperimentConfig(
        # Universe and time period
        universe="SP500",
        start_date="2015-01-01",
        end_date="2023-01-01",
        
        # Strategy selection
        strategy="ts_momentum",
        
        # Strategy parameters
        parameters={
            "lookback": 60,
            "holding_period": 20,
        },
        
        # Portfolio settings
        initial_capital=100000,
        
        # Cost model
        slippage=0.001,  # 10 bps
        commission=0.0005,  # 5 bps
        
        # Risk management
        max_leverage=1.0,
    )
    
    # Run the experiment
    print("Running backtest...")
    result = run_experiment(config)
    
    # Display results
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    
    print(f"\nPerformance Metrics:")
    print(f"  Sharpe Ratio:      {result.metrics['sharpe_ratio']:.2f}")
    print(f"  Annual Return:     {result.metrics['annual_return']:.2%}")
    print(f"  Annual Volatility: {result.metrics['annual_volatility']:.2%}")
    print(f"  Max Drawdown:      {result.metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio:      {result.metrics['calmar_ratio']:.2f}")
    
    # Generate visualizations
    print("\nGenerating plots...")
    result.plot_pnl(save_path="pnl_curve.png")
    result.plot_drawdown(save_path="drawdown.png")
    result.plot_rolling_sharpe(save_path="rolling_sharpe.png")
    
    # Save full report
    print("\nSaving HTML report...")
    result.save_report("experiment_report.html")
    
    print("\nDone! Check the generated files.")


if __name__ == "__main__":
    # This will work once the core modules are implemented
    print("JSF-Core Quick Start Example")
    print("NOTE: This example requires Phase 17 to be complete.")
    print("Current status: Phase 1/20 - Foundation Complete")
    print("\nThe API design is ready, implementation in progress...")

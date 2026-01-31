"""
Real Data Integration Examples (Phase 13)
==========================================

This module demonstrates how to use the real data integration features
of JSF-Core, including Yahoo Finance data downloading and enhanced CSV loading.

Examples include:
1. Basic Yahoo Finance data download
2. Multi-symbol portfolio data
3. Enhanced CSV loading with quality checks
4. Data quality validation and reporting
5. Combining real data with backtesting

Author: JBAC Team
Phase: 13 - Real Data Integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# Example 1: Basic Yahoo Finance Data Download
# ============================================================================

def example_1_basic_yahoo_download():
    """
    Demonstrate basic Yahoo Finance data downloading.
    
    This example shows how to download historical price data for a single
    stock using the YahooFinanceLoader.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Yahoo Finance Data Download")
    print("=" * 70)
    
    try:
        from jsf.data import YahooFinanceLoader, load_yahoo_data
        
        # Method 1: Using the convenience function
        print("\n1.1 Using convenience function (load_yahoo_data):")
        print("-" * 50)
        
        # Download 1 year of AAPL data
        data = load_yahoo_data(
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2024-01-01',
        )
        
        print(f"Downloaded {len(data)} days of data for AAPL")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nFirst 5 rows:")
        print(data.head())
        
        # Method 2: Using the class directly
        print("\n\n1.2 Using YahooFinanceLoader class:")
        print("-" * 50)
        
        loader = YahooFinanceLoader(cache_dir='./data_cache')
        
        # Download SPY ETF data
        spy_data = loader.load_data(
            symbols='SPY',
            start_date='2023-06-01',
            end_date='2023-12-31',
        )
        
        print(f"Downloaded SPY data with shape: {spy_data.shape}")
        print(f"\nStatistics:")
        print(spy_data['Close'].describe())
        
        # Calculate returns
        spy_data['Returns'] = spy_data['Close'].pct_change()
        annual_return = spy_data['Returns'].mean() * 252
        annual_vol = spy_data['Returns'].std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        print(f"\nPerformance Metrics (SPY 2023 H2):")
        print(f"  Annualized Return: {annual_return:.2%}")
        print(f"  Annualized Volatility: {annual_vol:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        
        return data
        
    except ImportError as e:
        print(f"Note: yfinance not installed. Install with: pip install yfinance")
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Note: Could not download data (network/API issue): {e}")
        return None


# ============================================================================
# Example 2: Multi-Symbol Portfolio Data
# ============================================================================

def example_2_multi_symbol_portfolio():
    """
    Demonstrate downloading data for multiple symbols for portfolio analysis.
    
    This example shows how to download and prepare data for a diversified
    portfolio of stocks and ETFs.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multi-Symbol Portfolio Data")
    print("=" * 70)
    
    try:
        from jsf.data import YahooFinanceLoader
        from jsf.data.sources.yahoo import POPULAR_TECH_STOCKS, POPULAR_ETFs
        
        print("\n2.1 Available symbol lists:")
        print("-" * 50)
        print(f"Popular Tech Stocks: {POPULAR_TECH_STOCKS}")
        print(f"Popular ETFs: {POPULAR_ETFs}")
        
        # Create a diversified portfolio
        portfolio_symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
        
        print(f"\n\n2.2 Downloading portfolio data:")
        print("-" * 50)
        print(f"Symbols: {portfolio_symbols}")
        
        loader = YahooFinanceLoader()
        
        # Download 6 months of data for multiple symbols
        portfolio_data = loader.load_data(
            symbols=portfolio_symbols,
            start_date='2023-07-01',
            end_date='2023-12-31',
        )
        
        print(f"\nDownloaded data shape: {portfolio_data.shape}")
        print(f"Columns: {list(portfolio_data.columns)}")
        
        # Extract close prices for each symbol
        print("\n\n2.3 Portfolio analysis:")
        print("-" * 50)
        
        # Filter for close prices
        close_cols = [col for col in portfolio_data.columns if 'Close' in str(col)]
        
        if isinstance(portfolio_data.columns, pd.MultiIndex):
            # Multi-symbol case: columns are multi-index
            closes = portfolio_data.xs('Close', axis=1, level=0)
        else:
            # Single symbol case
            closes = portfolio_data[['Close']]
        
        print(f"Close prices shape: {closes.shape}")
        
        # Calculate returns
        returns = closes.pct_change().dropna()
        
        # Correlation matrix
        print("\nCorrelation Matrix:")
        print(returns.corr().round(3))
        
        # Calculate portfolio metrics (equal weight)
        weights = np.array([1/len(portfolio_symbols)] * len(portfolio_symbols))
        portfolio_returns = (returns * weights).sum(axis=1)
        
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        print(f"\nEqual-Weight Portfolio Performance:")
        print(f"  Annualized Return: {ann_return:.2%}")
        print(f"  Annualized Volatility: {ann_vol:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        
        return portfolio_data
        
    except ImportError:
        print("Note: yfinance not installed. Install with: pip install yfinance")
        return None
    except Exception as e:
        print(f"Note: Could not download data: {e}")
        return None


# ============================================================================
# Example 3: Enhanced CSV Loading
# ============================================================================

def example_3_enhanced_csv_loading():
    """
    Demonstrate the enhanced CSV loader with data quality checks.
    
    This example creates sample CSV data and shows how to load it
    with automatic quality validation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Enhanced CSV Loading with Quality Checks")
    print("=" * 70)
    
    from jsf.data import EnhancedCSVLoader, load_csv_data
    
    # Create sample CSV data
    print("\n3.1 Creating sample CSV file:")
    print("-" * 50)
    
    # Create sample data with some quality issues
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    np.random.seed(42)
    
    price = 100.0
    prices = [price]
    for _ in range(251):
        change = np.random.randn() * 0.02
        price = price * (1 + change)
        prices.append(price)
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.randn() * 0.005) for p in prices],
        'High': [p * (1 + abs(np.random.randn() * 0.01)) for p in prices],
        'Low': [p * (1 - abs(np.random.randn() * 0.01)) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, size=252),
    })
    
    # Introduce some data quality issues for demonstration
    sample_data.loc[10, 'Close'] = np.nan  # Missing value
    sample_data.loc[20, 'Volume'] = 0  # Zero volume
    sample_data.loc[100, 'Close'] = prices[100] * 3  # Outlier
    
    # Save to CSV
    csv_path = './sample_data.csv'
    sample_data.to_csv(csv_path, index=False)
    print(f"Created sample CSV at: {csv_path}")
    print(f"Data shape: {sample_data.shape}")
    
    # Load with enhanced loader
    print("\n\n3.2 Loading with EnhancedCSVLoader:")
    print("-" * 50)
    
    loader = EnhancedCSVLoader(
        date_column='Date',
        date_format='%Y-%m-%d',
        required_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
        check_nulls=True,
        check_outliers=True,
        outlier_threshold=3.0,  # 3 standard deviations
    )
    
    # Load and get quality report
    data, quality_report = loader.load_with_report(csv_path)
    
    print(f"Loaded data shape: {data.shape}")
    print(f"\nData Quality Report:")
    print(quality_report)
    
    # Handle missing data
    print("\n\n3.3 Handling data quality issues:")
    print("-" * 50)
    
    # Fill missing values
    data_clean = data.copy()
    data_clean['Close'] = data_clean['Close'].fillna(method='ffill')
    
    print(f"Missing values after cleaning: {data_clean['Close'].isna().sum()}")
    
    return data_clean


# ============================================================================
# Example 4: Data Quality Validation
# ============================================================================

def example_4_data_quality_validation():
    """
    Demonstrate comprehensive data quality validation.
    
    This example shows how to validate data quality and generate
    detailed reports for data governance.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Data Quality Validation")
    print("=" * 70)
    
    from jsf.data.sources.csv_enhanced import DataQualityReport
    
    # Create test data with various issues
    print("\n4.1 Creating test data with quality issues:")
    print("-" * 50)
    
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    np.random.seed(123)
    
    test_data = pd.DataFrame({
        'Date': dates,
        'Close': np.random.randn(100) * 10 + 100,
        'Volume': np.random.randint(1000, 10000, size=100),
    })
    
    # Introduce issues
    test_data.loc[5:7, 'Close'] = np.nan  # Multiple nulls
    test_data.loc[50, 'Close'] = 500  # Outlier (5 std devs away)
    test_data = pd.concat([test_data, test_data.iloc[[20]]]).sort_values('Date')  # Duplicate
    
    print(f"Test data created with {len(test_data)} rows")
    
    # Create quality report manually
    print("\n\n4.2 Quality validation checks:")
    print("-" * 50)
    
    report = DataQualityReport()
    
    # Check nulls
    null_counts = test_data.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            report.warnings.append(f"Column '{col}' has {count} null values")
            report.null_counts[col] = int(count)
    
    # Check duplicates
    dup_count = test_data.duplicated(subset=['Date']).sum()
    if dup_count > 0:
        report.warnings.append(f"Found {dup_count} duplicate dates")
        report.duplicate_rows = int(dup_count)
    
    # Check outliers
    for col in ['Close', 'Volume']:
        mean = test_data[col].mean()
        std = test_data[col].std()
        if std > 0:
            z_scores = abs((test_data[col] - mean) / std)
            outlier_count = (z_scores > 3).sum()
            if outlier_count > 0:
                report.warnings.append(
                    f"Column '{col}' has {outlier_count} outliers (>3 std)"
                )
                report.outliers[col] = int(outlier_count)
    
    report.rows_processed = len(test_data)
    report.rows_valid = len(test_data) - report.duplicate_rows - sum(report.null_counts.values())
    report.is_valid = len(report.warnings) == 0
    
    print(f"Quality Report:")
    print(report)
    
    # Validation summary
    print("\n\n4.3 Validation summary:")
    print("-" * 50)
    print(f"  Total rows: {report.rows_processed}")
    print(f"  Valid rows: {report.rows_valid}")
    print(f"  Null values: {report.null_counts}")
    print(f"  Outliers: {report.outliers}")
    print(f"  Duplicates: {report.duplicate_rows}")
    print(f"  Overall valid: {report.is_valid}")
    
    return report


# ============================================================================
# Example 5: Integration with Backtesting
# ============================================================================

def example_5_backtest_with_real_data():
    """
    Demonstrate using real data in a backtest.
    
    This example combines Yahoo Finance data with the JSF backtesting
    framework to run a complete strategy backtest.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Backtesting with Real Data")
    print("=" * 70)
    
    try:
        from jsf.data import load_yahoo_data
        from jsf.strategy import TrendStrategy
        from jsf.backtest import Backtest
        
        # Download real market data
        print("\n5.1 Downloading market data:")
        print("-" * 50)
        
        data = load_yahoo_data(
            symbols='SPY',
            start_date='2022-01-01',
            end_date='2023-12-31',
        )
        
        print(f"Downloaded {len(data)} days of SPY data")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Create a simple trend-following strategy
        print("\n\n5.2 Creating trend strategy:")
        print("-" * 50)
        
        strategy = TrendStrategy(
            short_window=20,
            long_window=50,
        )
        
        print(f"Strategy: {strategy}")
        
        # Run backtest
        print("\n\n5.3 Running backtest:")
        print("-" * 50)
        
        backtest = Backtest(
            strategy=strategy,
            data=data,
            initial_capital=100000,
            commission=0.001,  # 10 bps
        )
        
        results = backtest.run()
        
        print(f"\nBacktest Results (SPY 2022-2023):")
        print(f"  Initial Capital: $100,000")
        print(f"  Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"  Total Return: {results.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"  Number of Trades: {results.get('num_trades', 0)}")
        
        return results
        
    except ImportError as e:
        print(f"Note: Some modules not available: {e}")
        print("This example requires the full JSF framework and yfinance.")
        return None
    except Exception as e:
        print(f"Note: Could not complete backtest: {e}")
        return None


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("JSF-CORE REAL DATA INTEGRATION EXAMPLES (Phase 13)")
    print("=" * 70)
    print("""
This module demonstrates the real data integration capabilities of JSF-Core:

1. Yahoo Finance Data Download - Single symbol historical data
2. Multi-Symbol Portfolio Data - Downloading and analyzing multiple assets
3. Enhanced CSV Loading - Loading CSV files with quality validation
4. Data Quality Validation - Comprehensive data quality checks
5. Backtesting with Real Data - End-to-end backtest using market data

Prerequisites:
- pip install yfinance (for Yahoo Finance examples)
- pip install pandas numpy
""")
    
    # Run examples
    example_1_basic_yahoo_download()
    example_2_multi_symbol_portfolio()
    example_3_enhanced_csv_loading()
    example_4_data_quality_validation()
    example_5_backtest_with_real_data()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Example usage patterns for jsf.data module."""

import pandas as pd
from jsf.data import (
    CSVLoader,
    SyntheticDataLoader,
    InMemoryLoader,
    load_data,
    create_universe,
    handle_missing_data,
    resample_data,
    calculate_returns,
    normalize_prices,
)
from jsf.config import UniverseType


def example_1_synthetic_data() -> None:
    """Example 1: Generate synthetic price data."""
    print("\n=== Example 1: Synthetic Data Generation ===\n")
    
    # Generate synthetic data for a small universe
    price_data = load_data(
        source="synthetic",
        symbols=["AAPL", "GOOGL", "MSFT"],
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_price=100.0,
        annual_return=0.1,  # 10% annual return
        annual_volatility=0.2,  # 20% annual volatility
        seed=42,  # For reproducibility
    )
    
    print(f"Generated data for {len(price_data.symbols)} symbols")
    print(f"Date range: {price_data.start_date} to {price_data.end_date}")
    print(f"Total rows: {len(price_data.data)}")
    print("\nSummary:")
    for key, value in price_data.summary().items():
        print(f"  {key}: {value}")


def example_2_csv_data() -> None:
    """Example 2: Load data from CSV files."""
    print("\n=== Example 2: CSV Data Loading ===\n")
    
    # Note: This assumes you have a CSV file with proper format
    # Uncomment and modify the path when you have actual data
    
    # Option 1: Single CSV file with MultiIndex (date, symbol)
    # loader = CSVLoader(
    #     file_path="data/prices.csv",
    #     symbols=["AAPL", "GOOGL", "MSFT"],
    # )
    # price_data = loader.load_as_price_data()
    
    # Option 2: Directory of CSV files (one per symbol)
    # loader = CSVLoader(
    #     file_path="data/individual_stocks/",
    #     symbols=["AAPL", "GOOGL", "MSFT"],
    # )
    # price_data = loader.load_as_price_data()
    
    print("CSV loading requires actual data files.")
    print("See commented code for usage patterns.")


def example_3_universe_creation() -> None:
    """Example 3: Create and manipulate universes."""
    print("\n=== Example 3: Universe Creation ===\n")
    
    # Create universe from predefined list
    sp500 = create_universe("SP500")
    print(f"S&P 500 universe: {len(sp500)} symbols")
    
    dow30 = create_universe(UniverseType.DOW_30)
    print(f"DOW 30 universe: {len(dow30)} symbols")
    
    # Create custom universe
    tech_stocks = create_universe(["AAPL", "GOOGL", "MSFT", "AMZN", "META"])
    print(f"Tech stocks universe: {len(tech_stocks)} symbols")
    
    # Set operations
    sp500_and_dow = sp500.intersection(dow30)
    print(f"\nSymbols in both S&P 500 and DOW 30: {len(sp500_and_dow)}")
    
    combined = sp500.union(dow30)
    print(f"Combined universe: {len(combined)} symbols")
    
    # Sample from universe
    sampled = sp500.sample(10, seed=42)
    print(f"\nRandom sample of 10 from S&P 500: {sampled.symbols}")


def example_4_preprocessing() -> None:
    """Example 4: Preprocess price data."""
    print("\n=== Example 4: Data Preprocessing ===\n")
    
    # Generate some sample data
    price_data = load_data(
        source="synthetic",
        symbols=["AAPL", "GOOGL"],
        start_date="2020-01-01",
        end_date="2020-12-31",
        seed=42,
    )
    
    # Get close prices as DataFrame
    close_prices = price_data.get_close_prices()
    print(f"Original shape: {close_prices.shape}")
    print(f"Date range: {close_prices.index[0]} to {close_prices.index[-1]}")
    
    # Handle missing data (if any)
    clean_prices = handle_missing_data(close_prices, method="ffill")
    print(f"\nAfter cleaning: {clean_prices.isna().sum().sum()} missing values")
    
    # Normalize to base 100
    normalized = normalize_prices(clean_prices, base_value=100.0)
    print(f"\nNormalized prices (first row):")
    print(normalized.iloc[0])
    
    # Calculate returns
    returns = calculate_returns(clean_prices, method="simple", periods=1)
    print(f"\nReturns shape: {returns.shape}")
    print(f"Average daily return:")
    print(returns.mean())


def example_5_resample_data() -> None:
    """Example 5: Resample data to different frequencies."""
    print("\n=== Example 5: Data Resampling ===\n")
    
    # Generate daily data
    price_data = load_data(
        source="synthetic",
        symbols=["AAPL"],
        start_date="2020-01-01",
        end_date="2020-12-31",
        seed=42,
    )
    
    df = price_data.data
    print(f"Daily data: {len(df)} rows")
    
    # Resample to weekly
    weekly = resample_data(df, freq="W")
    print(f"\nWeekly data: {len(weekly)} rows")
    
    # Resample to monthly
    monthly = resample_data(df, freq="M")
    print(f"Monthly data: {len(monthly)} rows")
    
    # Resample to quarterly
    quarterly = resample_data(df, freq="Q")
    print(f"Quarterly data: {len(quarterly)} rows")


def example_6_price_data_container() -> None:
    """Example 6: Use PriceData container features."""
    print("\n=== Example 6: PriceData Container ===\n")
    
    # Load data
    price_data = load_data(
        source="synthetic",
        symbols=["AAPL", "GOOGL", "MSFT"],
        start_date="2020-01-01",
        end_date="2023-12-31",
        seed=42,
    )
    
    # Get data for specific symbol
    aapl_data = price_data.get_symbol_data("AAPL")
    print(f"AAPL data shape: {aapl_data.shape}")
    print(f"Columns: {list(aapl_data.columns)}")
    
    # Get close prices in wide format
    close_prices = price_data.get_close_prices()
    print(f"\nClose prices shape: {close_prices.shape}")
    print(f"Symbols: {list(close_prices.columns)}")
    
    # Calculate returns using container method
    returns = price_data.get_returns(periods=1)
    print(f"\nReturns shape: {returns.shape}")
    print(f"Mean returns:\n{returns.mean()}")
    
    # Get log returns
    log_returns = price_data.get_log_returns(periods=1)
    print(f"\nLog returns mean:\n{log_returns.mean()}")
    
    # Get summary statistics
    summary = price_data.summary()
    print(f"\nData summary:")
    for key, value in summary.items():
        if key != "symbols":  # Skip printing all symbols
            print(f"  {key}: {value}")


def example_7_in_memory_loader() -> None:
    """Example 7: Load data from existing DataFrame."""
    print("\n=== Example 7: In-Memory Data Loading ===\n")
    
    # Create a sample DataFrame with MultiIndex
    dates = pd.bdate_range("2020-01-01", periods=100)
    symbols = ["AAPL", "GOOGL"]
    
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    
    # Create sample OHLCV data
    import numpy as np
    np.random.seed(42)
    
    data = pd.DataFrame({
        "open": np.random.uniform(90, 110, len(index)),
        "high": np.random.uniform(95, 115, len(index)),
        "low": np.random.uniform(85, 105, len(index)),
        "close": np.random.uniform(90, 110, len(index)),
        "volume": np.random.randint(1_000_000, 10_000_000, len(index)),
    }, index=index)
    
    # Load using InMemoryLoader
    loader = InMemoryLoader(data)
    price_data = loader.load_as_price_data()
    
    print(f"Loaded {len(price_data.symbols)} symbols")
    print(f"Date range: {price_data.start_date} to {price_data.end_date}")
    print(f"Total rows: {len(price_data.data)}")


def main() -> None:
    """Run all examples."""
    example_1_synthetic_data()
    example_2_csv_data()
    example_3_universe_creation()
    example_4_preprocessing()
    example_5_resample_data()
    example_6_price_data_container()
    example_7_in_memory_loader()


if __name__ == "__main__":
    main()

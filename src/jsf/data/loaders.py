"""
Concrete data loader implementations.

Provides loaders for different data sources: CSV, Parquet, HDF5, and in-memory.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np

from .base import DataLoader, DataLoadError, PriceData
from jsf.utils import get_logger

logger = get_logger(__name__)


class CSVLoader(DataLoader):
    """
    Load OHLCV data from CSV files.
    
    Supports two formats:
    1. Single file with columns: date, symbol, open, high, low, close, volume
    2. Multiple files (one per symbol) with columns: date, open, high, low, close, volume
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
        date_column: str = "date",
        symbol_column: str = "symbol",
        parse_dates: bool = True,
    ):
        """
        Initialize CSV loader.
        
        Args:
            file_path: Path to CSV file or directory containing CSV files
            symbols: List of symbols to load
            start_date: Start date filter
            end_date: End date filter
            fields: Fields to load
            date_column: Name of date column
            symbol_column: Name of symbol column (for single-file format)
            parse_dates: Whether to parse dates automatically
        """
        super().__init__(symbols, start_date, end_date, fields)
        self.file_path = Path(file_path)
        self.date_column = date_column
        self.symbol_column = symbol_column
        self.parse_dates = parse_dates
    
    def load(self) -> pd.DataFrame:
        """
        Load data from CSV.
        
        Returns:
            DataFrame with MultiIndex (date, symbol)
        """
        try:
            if self.file_path.is_file():
                # Single file format
                df = self._load_single_file()
            elif self.file_path.is_dir():
                # Multiple files format
                df = self._load_multiple_files()
            else:
                raise DataLoadError(f"Path not found: {self.file_path}")
            
            # Validate and return
            df = self.validate_data(df)
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV data: {e}") from e
    
    def _load_single_file(self) -> pd.DataFrame:
        """Load data from single CSV file."""
        logger.info(f"Loading data from {self.file_path}")
        
        df = pd.read_csv(
            self.file_path,
            parse_dates=[self.date_column] if self.parse_dates else None,
        )
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Set MultiIndex
        df = df.set_index([self.date_column, self.symbol_column])
        
        # Filter
        df = self.filter_date_range(df)
        df = self.filter_symbols(df)
        
        # Sort index
        df = df.sort_index()
        
        logger.info(f"Loaded {len(df)} rows from single CSV file")
        return df
    
    def _load_multiple_files(self) -> pd.DataFrame:
        """Load data from multiple CSV files (one per symbol)."""
        logger.info(f"Loading data from directory {self.file_path}")
        
        # Find CSV files
        csv_files = list(self.file_path.glob("*.csv"))
        if not csv_files:
            raise DataLoadError(f"No CSV files found in {self.file_path}")
        
        # If symbols specified, filter files
        if self.symbols:
            csv_files = [
                f for f in csv_files 
                if f.stem in self.symbols or f.stem.upper() in self.symbols
            ]
        
        if not csv_files:
            raise DataLoadError(f"No matching CSV files found for symbols {self.symbols}")
        
        # Load each file
        dfs = []
        for csv_file in csv_files:
            symbol = csv_file.stem.upper()
            
            try:
                df = pd.read_csv(
                    csv_file,
                    parse_dates=[self.date_column] if self.parse_dates else None,
                )
                
                # Ensure date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
                    df[self.date_column] = pd.to_datetime(df[self.date_column])
                
                # Add symbol column
                df[self.symbol_column] = symbol
                
                # Set index
                df = df.set_index([self.date_column, self.symbol_column])
                
                dfs.append(df)
                logger.debug(f"Loaded {len(df)} rows for {symbol}")
                
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if not dfs:
            raise DataLoadError("No data successfully loaded")
        
        # Combine all dataframes
        df = pd.concat(dfs)
        
        # Filter
        df = self.filter_date_range(df)
        df = self.filter_symbols(df)
        
        # Sort index
        df = df.sort_index()
        
        logger.info(f"Loaded {len(df)} rows from {len(dfs)} CSV files")
        return df


class ParquetLoader(DataLoader):
    """
    Load OHLCV data from Parquet files.
    
    Parquet format is much faster and more efficient than CSV for large datasets.
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
        engine: str = "pyarrow",
    ):
        """
        Initialize Parquet loader.
        
        Args:
            file_path: Path to Parquet file
            symbols: List of symbols to load
            start_date: Start date filter
            end_date: End date filter
            fields: Fields to load
            engine: Parquet engine ('pyarrow' or 'fastparquet')
        """
        super().__init__(symbols, start_date, end_date, fields)
        self.file_path = Path(file_path)
        self.engine = engine
    
    def load(self) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Returns:
            DataFrame with MultiIndex (date, symbol)
        """
        try:
            logger.info(f"Loading data from {self.file_path}")
            
            # Load parquet
            df = pd.read_parquet(self.file_path, engine=self.engine)
            
            # Ensure proper index
            if not isinstance(df.index, pd.MultiIndex):
                raise DataLoadError("Parquet file must have MultiIndex (date, symbol)")
            
            # Filter
            df = self.filter_date_range(df)
            df = self.filter_symbols(df)
            
            # Sort index
            df = df.sort_index()
            
            # Validate
            df = self.validate_data(df)
            
            logger.info(f"Loaded {len(df)} rows from Parquet file")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load Parquet data: {e}") from e


class InMemoryLoader(DataLoader):
    """
    Wrap in-memory DataFrame as a data loader.
    
    Useful for testing or when data is already loaded.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ):
        """
        Initialize in-memory loader.
        
        Args:
            data: DataFrame with MultiIndex (date, symbol) or DatetimeIndex
            symbols: List of symbols to filter
            start_date: Start date filter
            end_date: End date filter
            fields: Fields to filter
        """
        super().__init__(symbols, start_date, end_date, fields)
        self._data = data.copy()
    
    def load(self) -> pd.DataFrame:
        """
        Return the in-memory data.
        
        Returns:
            DataFrame with MultiIndex (date, symbol)
        """
        df = self._data.copy()
        
        # Apply filters if specified
        if self.start_date or self.end_date:
            df = self.filter_date_range(df)
        
        if self.symbols:
            df = self.filter_symbols(df)
        
        # Validate
        df = self.validate_data(df)
        
        logger.info(f"Loaded {len(df)} rows from in-memory data")
        return df


class SyntheticDataLoader(DataLoader):
    """
    Generate synthetic OHLCV data for testing.
    
    Creates realistic price data using geometric Brownian motion.
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_price: float = 100.0,
        annual_return: float = 0.10,
        annual_volatility: float = 0.20,
        seed: Optional[int] = None,
    ):
        """
        Initialize synthetic data loader.
        
        Args:
            symbols: List of symbols to generate
            start_date: Start date
            end_date: End date
            initial_price: Starting price for all symbols
            annual_return: Expected annual return (drift)
            annual_volatility: Annual volatility
            seed: Random seed for reproducibility
        """
        super().__init__(symbols, start_date, end_date)
        self.initial_price = initial_price
        self.annual_return = annual_return
        self.annual_volatility = annual_volatility
        self.seed = seed
    
    def load(self) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data.
        
        Returns:
            DataFrame with MultiIndex (date, symbol)
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Generate date range (business days)
        dates = pd.bdate_range(start=self.start_date, end=self.end_date)
        n_days = len(dates)
        
        # Calculate daily parameters
        dt = 1 / 252  # Trading days per year
        daily_return = self.annual_return * dt
        daily_vol = self.annual_volatility * np.sqrt(dt)
        
        dfs = []
        
        for symbol in self.symbols:
            # Generate returns using geometric Brownian motion
            returns = np.random.normal(daily_return, daily_vol, n_days)
            
            # Generate close prices
            close = self.initial_price * np.exp(np.cumsum(returns))
            
            # Generate OHLC from close with some realistic variation
            intraday_range = np.random.uniform(0.005, 0.03, n_days)  # 0.5-3% range
            
            high = close * (1 + intraday_range * np.random.uniform(0.3, 1.0, n_days))
            low = close * (1 - intraday_range * np.random.uniform(0.3, 1.0, n_days))
            open_prices = low + (high - low) * np.random.uniform(0.3, 0.7, n_days)
            
            # Generate volume (correlated with price movement)
            base_volume = 1_000_000
            volume_multiplier = 1 + np.abs(returns) * 10
            volume = (base_volume * volume_multiplier * np.random.uniform(0.5, 1.5, n_days)).astype(int)
            
            # Create DataFrame for this symbol
            df = pd.DataFrame({
                "open": open_prices,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }, index=dates)
            
            df["symbol"] = symbol
            df = df.reset_index().rename(columns={"index": "date"})
            df = df.set_index(["date", "symbol"])
            
            dfs.append(df)
        
        # Combine all symbols
        df = pd.concat(dfs)
        df = df.sort_index()
        
        logger.info(f"Generated {len(df)} rows of synthetic data for {len(self.symbols)} symbols")
        return df


def load_data(
    source: str,
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs: Any,
) -> PriceData:
    """
    Convenience function to load data from various sources.
    
    Args:
        source: Data source path or type ('synthetic')
        symbols: List of symbols to load
        start_date: Start date
        end_date: End date
        **kwargs: Additional arguments for the loader
        
    Returns:
        PriceData container
        
    Examples:
        >>> data = load_data('data/prices.csv', symbols=['AAPL', 'GOOGL'])
        >>> data = load_data('data/prices.parquet', start_date='2020-01-01')
        >>> data = load_data('synthetic', symbols=['AAPL', 'MSFT'], start_date='2020-01-01', end_date='2023-12-31')
    """
    source_path = Path(source) if source != "synthetic" else None
    
    # Determine loader type
    if source == "synthetic":
        if not symbols:
            raise ValueError("symbols required for synthetic data")
        if not start_date or not end_date:
            raise ValueError("start_date and end_date required for synthetic data")
        
        loader = SyntheticDataLoader(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    
    elif source_path and source_path.suffix == ".parquet":
        loader = ParquetLoader(
            file_path=source_path,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    
    elif source_path and (source_path.suffix == ".csv" or source_path.is_dir()):
        loader = CSVLoader(
            file_path=source_path,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    
    else:
        raise ValueError(f"Unsupported data source: {source}")
    
    # Load and wrap in PriceData
    df = loader.load()
    return PriceData(df)

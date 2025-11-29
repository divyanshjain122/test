"""
Base classes for data loading.

Defines abstract interfaces for data loaders and common functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from jsf.utils import get_logger

logger = get_logger(__name__)


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    All data loaders must implement the load() method and return
    data in a standardized format.
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ):
        """
        Initialize data loader.
        
        Args:
            symbols: List of ticker symbols to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fields: List of fields to load (e.g., ['open', 'high', 'low', 'close', 'volume'])
        """
        self.symbols = symbols or []
        self.start_date = start_date
        self.end_date = end_date
        self.fields = fields or ["open", "high", "low", "close", "volume"]
        
        logger.info(
            f"Initialized {self.__class__.__name__} for {len(self.symbols)} symbols "
            f"from {start_date} to {end_date}"
        )
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load data and return as DataFrame.
        
        Returns:
            DataFrame with MultiIndex (date, symbol) and columns for each field.
            Index must be datetime, sorted chronologically.
        
        Raises:
            DataLoadError: If data cannot be loaded
        """
        pass
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate loaded data for common issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If data fails validation
        """
        if df.empty:
            raise ValueError("Loaded data is empty")
        
        # Check for required fields
        missing_fields = set(self.fields) - set(df.columns)
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Check for null values
        null_counts = df[self.fields].isnull().sum()
        if null_counts.any():
            logger.warning(f"Data contains null values:\n{null_counts[null_counts > 0]}")
        
        # Check for negative prices
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        for col in price_cols:
            if (df[col] < 0).any():
                raise ValueError(f"Column {col} contains negative values")
        
        # Check for negative volume
        if "volume" in df.columns and (df["volume"] < 0).any():
            raise ValueError("Volume column contains negative values")
        
        logger.info(f"Data validation passed: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to specified date range.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            Filtered DataFrame
        """
        if self.start_date:
            df = df[df.index.get_level_values(0) >= pd.Timestamp(self.start_date)]
        
        if self.end_date:
            df = df[df.index.get_level_values(0) <= pd.Timestamp(self.end_date)]
        
        return df
    
    def filter_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to specified symbols.
        
        Args:
            df: DataFrame with (date, symbol) MultiIndex
            
        Returns:
            Filtered DataFrame
        """
        if self.symbols:
            # Get symbol level (usually level 1)
            symbol_level = 1 if df.index.nlevels > 1 else 0
            available_symbols = df.index.get_level_values(symbol_level).unique()
            
            # Filter to requested symbols that exist in data
            valid_symbols = [s for s in self.symbols if s in available_symbols]
            if not valid_symbols:
                raise ValueError(f"None of the requested symbols found in data")
            
            missing_symbols = set(self.symbols) - set(valid_symbols)
            if missing_symbols:
                logger.warning(f"Symbols not found in data: {missing_symbols}")
            
            df = df[df.index.get_level_values(symbol_level).isin(valid_symbols)]
        
        return df


class PriceData:
    """
    Container for price data with utility methods.
    
    Standardizes access to OHLCV data across different loaders.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize price data container.
        
        Args:
            data: DataFrame with MultiIndex (date, symbol) or DatetimeIndex
        """
        self.data = data
        self._validate_structure()
    
    def _validate_structure(self) -> None:
        """Validate data structure."""
        if not isinstance(self.data.index, (pd.DatetimeIndex, pd.MultiIndex)):
            raise ValueError("Data must have DatetimeIndex or MultiIndex")
        
        if isinstance(self.data.index, pd.MultiIndex):
            if self.data.index.nlevels != 2:
                raise ValueError("MultiIndex must have exactly 2 levels (date, symbol)")
    
    @property
    def symbols(self) -> List[str]:
        """Get list of symbols in data."""
        if isinstance(self.data.index, pd.MultiIndex):
            return sorted(self.data.index.get_level_values(1).unique().tolist())
        return []
    
    @property
    def dates(self) -> pd.DatetimeIndex:
        """Get DatetimeIndex of dates."""
        if isinstance(self.data.index, pd.MultiIndex):
            return pd.DatetimeIndex(sorted(self.data.index.get_level_values(0).unique()))
        return self.data.index
    
    @property
    def start_date(self) -> pd.Timestamp:
        """Get first date in data."""
        return self.dates[0]
    
    @property
    def end_date(self) -> pd.Timestamp:
        """Get last date in data."""
        return self.dates[-1]
    
    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Get data for a specific symbol.
        
        Args:
            symbol: Symbol to extract
            
        Returns:
            DataFrame with DatetimeIndex for the symbol
        """
        if isinstance(self.data.index, pd.MultiIndex):
            return self.data.xs(symbol, level=1)
        return self.data
    
    def get_field(self, field: str) -> pd.DataFrame:
        """
        Get a specific field as DataFrame with symbols as columns.
        
        Args:
            field: Field name (e.g., 'close', 'volume')
            
        Returns:
            DataFrame with dates as index and symbols as columns
        """
        if field not in self.data.columns:
            raise ValueError(f"Field '{field}' not found in data")
        
        if isinstance(self.data.index, pd.MultiIndex):
            return self.data[field].unstack(level=1)
        return self.data[[field]]
    
    def get_close_prices(self) -> pd.DataFrame:
        """Get close prices as DataFrame."""
        return self.get_field("close")
    
    def get_returns(self, periods: int = 1) -> pd.DataFrame:
        """
        Calculate returns for all symbols.
        
        Args:
            periods: Number of periods for return calculation
            
        Returns:
            DataFrame of returns
        """
        close = self.get_close_prices()
        return close.pct_change(periods=periods)
    
    def get_log_returns(self, periods: int = 1) -> pd.DataFrame:
        """
        Calculate log returns for all symbols.
        
        Args:
            periods: Number of periods for return calculation
            
        Returns:
            DataFrame of log returns
        """
        close = self.get_close_prices()
        return np.log(close / close.shift(periods))
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the data.
        
        Returns:
            Dictionary with summary information
        """
        return {
            "n_symbols": len(self.symbols),
            "symbols": self.symbols,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "n_days": len(self.dates),
            "fields": list(self.data.columns),
            "shape": self.data.shape,
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024 / 1024,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        summary = self.summary()
        return (
            f"PriceData(\n"
            f"  symbols={summary['n_symbols']},\n"
            f"  period={summary['start_date']} to {summary['end_date']},\n"
            f"  days={summary['n_days']},\n"
            f"  fields={summary['fields']}\n"
            f")"
        )


class DataLoadError(Exception):
    """Exception raised when data loading fails."""
    pass

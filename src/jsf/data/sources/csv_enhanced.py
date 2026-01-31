"""
Enhanced CSV data loader with validation and quality checks.

This module provides an enhanced CSV loader with:
- Multiple format support (single file, directory, symbol-per-file)
- Data quality validation
- Missing data detection and handling
- Automatic date parsing
- Column mapping
- Data preview functionality
"""

import logging
from typing import List, Optional, Dict, Any, Union, Callable
from pathlib import Path
import pandas as pd
import numpy as np

from jsf.data.base import DataLoader, DataLoadError, PriceData

logger = logging.getLogger(__name__)


class DataQualityReport:
    """
    Report on data quality issues.
    
    Attributes:
        symbol: Symbol analyzed
        total_rows: Total number of rows
        missing_dates: Number of missing trading days
        null_values: Count of null values per column
        outliers: Count of detected outliers per column
        duplicates: Number of duplicate rows
        issues: List of identified issues
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.total_rows = 0
        self.date_range = (None, None)
        self.missing_dates = 0
        self.null_values: Dict[str, int] = {}
        self.outliers: Dict[str, int] = {}
        self.duplicates = 0
        self.negative_prices = 0
        self.issues: List[str] = []
    
    @property
    def is_clean(self) -> bool:
        """Check if data has no issues."""
        return len(self.issues) == 0
    
    def __str__(self) -> str:
        status = "✓ CLEAN" if self.is_clean else f"⚠ {len(self.issues)} ISSUES"
        lines = [
            f"Data Quality Report for {self.symbol}: {status}",
            f"  Total rows: {self.total_rows}",
            f"  Date range: {self.date_range[0]} to {self.date_range[1]}",
            f"  Missing dates: {self.missing_dates}",
            f"  Duplicates: {self.duplicates}",
        ]
        
        if self.null_values:
            null_str = ", ".join(f"{k}:{v}" for k, v in self.null_values.items() if v > 0)
            if null_str:
                lines.append(f"  Null values: {null_str}")
        
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(f"    - {issue}")
        
        return "\n".join(lines)


class EnhancedCSVLoader(DataLoader):
    """
    Enhanced CSV loader with validation and quality checks.
    
    Supports multiple CSV formats:
    1. Single file with symbol column
    2. Single file per symbol in a directory
    3. Custom column mapping
    
    Args:
        file_path: Path to CSV file or directory
        symbols: List of symbols to load
        start_date: Start date filter
        end_date: End date filter
        column_map: Dictionary mapping CSV columns to standard names
        date_format: Date format string for parsing
        validate: Whether to run data validation
        fill_missing: Strategy for missing data ('ffill', 'bfill', 'drop', None)
        remove_outliers: Whether to remove extreme outliers
        
    Example:
        >>> loader = EnhancedCSVLoader(
        ...     file_path='data/prices.csv',
        ...     symbols=['AAPL', 'GOOGL'],
        ...     column_map={'Date': 'date', 'Ticker': 'symbol', 'Close': 'close'}
        ... )
        >>> data = loader.load()
    """
    
    DEFAULT_COLUMN_MAP = {
        'Date': 'date',
        'date': 'date',
        'DATE': 'date',
        'Symbol': 'symbol',
        'symbol': 'symbol',
        'SYMBOL': 'symbol',
        'Ticker': 'symbol',
        'ticker': 'symbol',
        'Open': 'open',
        'open': 'open',
        'OPEN': 'open',
        'High': 'high',
        'high': 'high',
        'HIGH': 'high',
        'Low': 'low',
        'low': 'low',
        'LOW': 'low',
        'Close': 'close',
        'close': 'close',
        'CLOSE': 'close',
        'Adj Close': 'close',
        'adj_close': 'close',
        'Volume': 'volume',
        'volume': 'volume',
        'VOLUME': 'volume',
    }
    
    def __init__(
        self,
        file_path: Union[str, Path],
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
        column_map: Optional[Dict[str, str]] = None,
        date_format: Optional[str] = None,
        date_column: str = "date",
        symbol_column: str = "symbol",
        validate: bool = True,
        fill_missing: Optional[str] = "ffill",
        remove_outliers: bool = False,
        outlier_std: float = 5.0,
    ):
        """Initialize enhanced CSV loader."""
        super().__init__(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            fields=fields or ["open", "high", "low", "close", "volume"],
        )
        
        self.file_path = Path(file_path)
        self.column_map = {**self.DEFAULT_COLUMN_MAP, **(column_map or {})}
        self.date_format = date_format
        self.date_column = date_column
        self.symbol_column = symbol_column
        self.validate = validate
        self.fill_missing = fill_missing
        self.remove_outliers = remove_outliers
        self.outlier_std = outlier_std
        
        self.quality_reports: Dict[str, DataQualityReport] = {}
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to standard names."""
        # Apply column mapping
        rename_map = {}
        for col in df.columns:
            if col in self.column_map:
                rename_map[col] = self.column_map[col]
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug(f"Renamed columns: {rename_map}")
        
        return df
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column to datetime."""
        if self.date_column not in df.columns:
            # Check if index is already datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: self.date_column})
            else:
                raise DataLoadError(f"Date column '{self.date_column}' not found in data")
        
        try:
            if self.date_format:
                df[self.date_column] = pd.to_datetime(df[self.date_column], format=self.date_format)
            else:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
        except Exception as e:
            raise DataLoadError(f"Failed to parse dates: {e}")
        
        return df
    
    def _check_data_quality(self, df: pd.DataFrame, symbol: str) -> DataQualityReport:
        """Run data quality checks for a symbol."""
        report = DataQualityReport(symbol)
        report.total_rows = len(df)
        
        if len(df) == 0:
            report.issues.append("No data rows")
            return report
        
        # Date range
        dates = df.index.get_level_values(0) if isinstance(df.index, pd.MultiIndex) else df.index
        report.date_range = (dates.min(), dates.max())
        
        # Check for missing trading days
        if isinstance(dates, pd.DatetimeIndex):
            expected_dates = pd.bdate_range(dates.min(), dates.max())
            actual_dates = dates.unique()
            missing = len(expected_dates) - len(actual_dates)
            report.missing_dates = max(0, missing)
            
            if missing > len(expected_dates) * 0.1:  # More than 10% missing
                report.issues.append(f"Significant missing dates: {missing} days")
        
        # Check for null values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                report.null_values[col] = null_count
                if null_count > 0:
                    report.issues.append(f"Null values in {col}: {null_count}")
        
        # Check for duplicates
        report.duplicates = df.index.duplicated().sum()
        if report.duplicates > 0:
            report.issues.append(f"Duplicate rows: {report.duplicates}")
        
        # Check for negative prices
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        for col in price_cols:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                report.negative_prices += neg_count
                report.issues.append(f"Negative prices in {col}: {neg_count}")
        
        # Check for outliers (price changes > outlier_std standard deviations)
        if 'close' in df.columns and len(df) > 10:
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                outlier_threshold = returns.std() * self.outlier_std
                outliers = (returns.abs() > outlier_threshold).sum()
                report.outliers['close_returns'] = outliers
                if outliers > 0:
                    report.issues.append(f"Outlier returns detected: {outliers}")
        
        return report
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data according to strategy."""
        if self.fill_missing is None:
            return df
        
        if self.fill_missing == 'ffill':
            df = df.ffill()
        elif self.fill_missing == 'bfill':
            df = df.bfill()
        elif self.fill_missing == 'drop':
            df = df.dropna()
        else:
            logger.warning(f"Unknown fill strategy: {self.fill_missing}")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outlier rows based on returns."""
        if not self.remove_outliers or 'close' not in df.columns:
            return df
        
        returns = df['close'].pct_change()
        threshold = returns.std() * self.outlier_std
        
        # Keep rows where returns are within threshold
        mask = (returns.abs() <= threshold) | returns.isna()
        removed = (~mask).sum()
        
        if removed > 0:
            logger.info(f"Removed {removed} outlier rows")
        
        return df[mask]
    
    def _load_single_file(self) -> pd.DataFrame:
        """Load data from a single CSV file."""
        logger.info(f"Loading CSV file: {self.file_path}")
        
        # Read CSV
        df = pd.read_csv(self.file_path)
        
        # Standardize columns
        df = self._standardize_columns(df)
        
        # Parse dates
        df = self._parse_dates(df)
        
        # Check if symbol column exists
        if self.symbol_column not in df.columns:
            # Single symbol file - use filename as symbol
            symbol = self.file_path.stem.upper()
            df[self.symbol_column] = symbol
            logger.info(f"No symbol column found, using filename: {symbol}")
        
        # Filter symbols if specified
        if self.symbols:
            available = df[self.symbol_column].unique().tolist()
            valid_symbols = [s for s in self.symbols if s in available]
            
            if not valid_symbols:
                raise DataLoadError(
                    f"None of requested symbols found. "
                    f"Available: {available[:10]}{'...' if len(available) > 10 else ''}"
                )
            
            df = df[df[self.symbol_column].isin(valid_symbols)]
        
        # Set MultiIndex
        df = df.set_index([self.date_column, self.symbol_column])
        df = df.sort_index()
        
        return df
    
    def _load_directory(self) -> pd.DataFrame:
        """Load data from a directory of CSV files (one per symbol)."""
        logger.info(f"Loading CSV files from directory: {self.file_path}")
        
        csv_files = list(self.file_path.glob("*.csv"))
        
        if not csv_files:
            raise DataLoadError(f"No CSV files found in {self.file_path}")
        
        dfs = []
        
        for csv_file in csv_files:
            symbol = csv_file.stem.upper()
            
            # Skip if not in requested symbols
            if self.symbols and symbol not in self.symbols:
                continue
            
            try:
                df = pd.read_csv(csv_file)
                df = self._standardize_columns(df)
                df = self._parse_dates(df)
                df[self.symbol_column] = symbol
                dfs.append(df)
                logger.debug(f"Loaded {symbol}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if not dfs:
            raise DataLoadError("No data loaded from directory")
        
        # Combine all
        df = pd.concat(dfs, ignore_index=True)
        df = df.set_index([self.date_column, self.symbol_column])
        df = df.sort_index()
        
        return df
    
    def load(self) -> pd.DataFrame:
        """
        Load and validate data from CSV.
        
        Returns:
            DataFrame with MultiIndex (date, symbol) and OHLCV columns
            
        Raises:
            DataLoadError: If data cannot be loaded
        """
        try:
            # Load based on path type
            if self.file_path.is_file():
                df = self._load_single_file()
            elif self.file_path.is_dir():
                df = self._load_directory()
            else:
                raise DataLoadError(f"Path not found: {self.file_path}")
            
            # Select only OHLCV columns
            available_cols = [c for c in self.fields if c in df.columns]
            if not available_cols:
                raise DataLoadError(f"No OHLCV columns found. Available: {list(df.columns)}")
            
            df = df[available_cols]
            
            # Apply date filters
            df = self.filter_date_range(df)
            
            # Data quality checks
            if self.validate:
                symbols = df.index.get_level_values(1).unique()
                for symbol in symbols:
                    symbol_data = df.xs(symbol, level=1)
                    report = self._check_data_quality(symbol_data, symbol)
                    self.quality_reports[symbol] = report
                    
                    if not report.is_clean:
                        logger.warning(f"Quality issues for {symbol}: {len(report.issues)} issues")
            
            # Handle missing data
            df = self._handle_missing_data(df)
            
            # Remove outliers
            df = self._remove_outliers(df)
            
            # Final validation
            df = self.validate_data(df)
            
            logger.info(f"Loaded {len(df)} rows from CSV")
            return df
            
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV: {e}") from e
    
    def get_quality_report(self, symbol: Optional[str] = None) -> Union[DataQualityReport, Dict[str, DataQualityReport]]:
        """
        Get data quality report(s).
        
        Args:
            symbol: Specific symbol, or None for all
            
        Returns:
            Single report or dictionary of reports
        """
        if symbol:
            return self.quality_reports.get(symbol)
        return self.quality_reports
    
    def preview(self, n_rows: int = 5) -> pd.DataFrame:
        """
        Preview the first n rows of the CSV without full loading.
        
        Args:
            n_rows: Number of rows to preview
            
        Returns:
            DataFrame with first n rows
        """
        if self.file_path.is_file():
            return pd.read_csv(self.file_path, nrows=n_rows)
        return pd.DataFrame()


def load_csv_data(
    file_path: Union[str, Path],
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    column_map: Optional[Dict[str, str]] = None,
    validate: bool = True,
    fill_missing: Optional[str] = "ffill",
) -> PriceData:
    """
    Convenience function to load CSV data.
    
    Args:
        file_path: Path to CSV file or directory
        symbols: List of symbols to load
        start_date: Start date filter
        end_date: End date filter
        column_map: Column name mapping
        validate: Run data quality validation
        fill_missing: Missing data handling ('ffill', 'bfill', 'drop', None)
        
    Returns:
        PriceData container
        
    Example:
        >>> # Load from single file
        >>> data = load_csv_data('data/prices.csv', symbols=['AAPL', 'GOOGL'])
        >>> 
        >>> # Load from directory (one file per symbol)
        >>> data = load_csv_data('data/stocks/', start_date='2020-01-01')
        >>> 
        >>> # Load with custom column mapping
        >>> data = load_csv_data(
        ...     'data/custom.csv',
        ...     column_map={'date_col': 'date', 'price': 'close'}
        ... )
    """
    loader = EnhancedCSVLoader(
        file_path=file_path,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        column_map=column_map,
        validate=validate,
        fill_missing=fill_missing,
    )
    
    df = loader.load()
    return PriceData(df)

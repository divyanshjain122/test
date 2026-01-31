"""
Yahoo Finance data loader.

This module provides a loader for downloading historical market data
from Yahoo Finance using the yfinance library.

Features:
- Download OHLCV data for stocks, ETFs, indices
- Automatic date range handling
- Optional data caching to avoid repeated downloads
- Multi-symbol support with concurrent downloads
- Automatic adjustment for splits and dividends
"""

import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from jsf.data.base import DataLoader, DataLoadError, PriceData

logger = logging.getLogger(__name__)


# Check if yfinance is available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None


class YahooFinanceLoader(DataLoader):
    """
    Load OHLCV data from Yahoo Finance.
    
    Downloads historical market data for stocks, ETFs, indices, and other
    securities available on Yahoo Finance.
    
    Args:
        symbols: List of ticker symbols (e.g., ['AAPL', 'GOOGL', 'SPY'])
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        interval: Data interval ('1d', '1wk', '1mo')
        auto_adjust: Adjust OHLC prices for splits and dividends
        cache_dir: Directory to cache downloaded data
        cache_days: Days before cache expires (0 = no caching)
        
    Example:
        >>> loader = YahooFinanceLoader(
        ...     symbols=['AAPL', 'GOOGL', 'MSFT'],
        ...     start_date='2020-01-01',
        ...     end_date='2023-12-31'
        ... )
        >>> data = loader.load()
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_days: int = 1,
        fields: Optional[List[str]] = None,
    ):
        """Initialize Yahoo Finance loader."""
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is required for Yahoo Finance data. "
                "Install it with: pip install yfinance"
            )
        
        # Default end date to today
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        super().__init__(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            fields=fields or ["open", "high", "low", "close", "volume"]
        )
        
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_days = cache_days
        
        # Validate symbols
        self._validate_symbols()
        
        logger.info(
            f"Initialized YahooFinanceLoader for {len(self.symbols)} symbols: "
            f"{', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}"
        )
    
    def _validate_symbols(self) -> None:
        """Validate symbol format."""
        if not self.symbols:
            raise ValueError("At least one symbol is required")
        
        for symbol in self.symbols:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError(f"Invalid symbol: {symbol}")
    
    def _get_cache_path(self, symbol: str) -> Optional[Path]:
        """Get cache file path for a symbol."""
        if not self.cache_dir:
            return None
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{symbol}_{self.start_date}_{self.end_date}_{self.interval}.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid."""
        if not cache_path or not cache_path.exists():
            return False
        
        if self.cache_days <= 0:
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(days=self.cache_days)
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Try to load data from cache."""
        cache_path = self._get_cache_path(symbol)
        
        if cache_path and self._is_cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded {symbol} from cache")
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache for {symbol}: {e}")
        
        return None
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(symbol)
        
        if cache_path and self.cache_days > 0:
            try:
                df.to_parquet(cache_path)
                logger.debug(f"Saved {symbol} to cache")
            except Exception as e:
                logger.warning(f"Failed to save cache for {symbol}: {e}")
    
    def _download_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download data for a single symbol."""
        try:
            # Try cache first
            cached = self._load_from_cache(symbol)
            if cached is not None:
                return cached
            
            # Download from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                auto_adjust=self.auto_adjust,
                actions=False,
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure we have required columns
            if 'close' not in df.columns:
                logger.warning(f"No close price for {symbol}")
                return None
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Reset index to get date as column
            df = df.reset_index()
            df = df.rename(columns={'Date': 'date', 'index': 'date'})
            
            # Set MultiIndex (date, symbol)
            df['date'] = pd.to_datetime(df['date'])
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)  # Remove timezone
            df = df.set_index(['date', 'symbol'])
            
            # Select only OHLCV columns
            available_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[available_cols]
            
            # Cache the data
            self._save_to_cache(symbol, df)
            
            logger.info(f"Downloaded {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            return None
    
    def load(self) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.
        
        Returns:
            DataFrame with MultiIndex (date, symbol) and OHLCV columns
            
        Raises:
            DataLoadError: If no data could be loaded for any symbol
        """
        logger.info(f"Downloading data for {len(self.symbols)} symbols...")
        
        dfs = []
        failed_symbols = []
        
        for symbol in self.symbols:
            df = self._download_symbol(symbol)
            if df is not None and not df.empty:
                dfs.append(df)
            else:
                failed_symbols.append(symbol)
        
        if not dfs:
            raise DataLoadError(
                f"Failed to load data for any symbols. Tried: {self.symbols}"
            )
        
        if failed_symbols:
            logger.warning(f"Failed to load {len(failed_symbols)} symbols: {failed_symbols}")
        
        # Combine all symbols
        df = pd.concat(dfs)
        df = df.sort_index()
        
        # Apply date filters
        df = self.filter_date_range(df)
        
        # Validate
        df = self.validate_data(df)
        
        logger.info(
            f"Loaded {len(df)} total rows for {len(dfs)} symbols "
            f"({len(failed_symbols)} failed)"
        )
        
        return df
    
    def get_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get information about a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dictionary with symbol information
        """
        ticker = yf.Ticker(symbol)
        return ticker.info
    
    def get_dividends(self, symbol: str) -> pd.Series:
        """Get dividend history for a symbol."""
        ticker = yf.Ticker(symbol)
        return ticker.dividends
    
    def get_splits(self, symbol: str) -> pd.Series:
        """Get stock split history for a symbol."""
        ticker = yf.Ticker(symbol)
        return ticker.splits


def load_yahoo_data(
    symbols: Union[str, List[str]],
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    cache_dir: Optional[str] = None,
) -> PriceData:
    """
    Convenience function to load data from Yahoo Finance.
    
    Args:
        symbols: Single symbol or list of symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        interval: Data interval ('1d', '1wk', '1mo')
        auto_adjust: Adjust for splits and dividends
        cache_dir: Directory to cache data
        
    Returns:
        PriceData container with loaded data
        
    Example:
        >>> # Load single stock
        >>> data = load_yahoo_data('AAPL', start_date='2020-01-01')
        >>> 
        >>> # Load multiple stocks
        >>> data = load_yahoo_data(
        ...     ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        ...     start_date='2020-01-01',
        ...     end_date='2023-12-31'
        ... )
        >>> 
        >>> # Load with caching
        >>> data = load_yahoo_data(
        ...     ['SPY', 'QQQ', 'IWM'],
        ...     start_date='2020-01-01',
        ...     cache_dir='.cache/yahoo'
        ... )
    """
    # Handle single symbol
    if isinstance(symbols, str):
        symbols = [symbols]
    
    loader = YahooFinanceLoader(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        cache_dir=cache_dir,
    )
    
    df = loader.load()
    return PriceData(df)


# Popular ticker lists for convenience
POPULAR_TECH_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA']
POPULAR_ETFs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EFA', 'EEM']
POPULAR_INDICES = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']

# Market hours (US Eastern)
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

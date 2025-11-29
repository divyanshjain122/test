"""
Data preprocessing utilities.

Handles missing data, resampling, adjustments, and transformations.
"""

from typing import Optional, Literal
import pandas as pd
import numpy as np

from jsf.utils import get_logger

logger = get_logger(__name__)


def handle_missing_data(
    df: pd.DataFrame,
    method: Literal["ffill", "bfill", "drop", "interpolate"] = "ffill",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Handle missing data in DataFrame.
    
    Args:
        df: DataFrame with potential missing values
        method: Method to handle missing data:
            - 'ffill': Forward fill
            - 'bfill': Backward fill
            - 'drop': Drop rows with missing values
            - 'interpolate': Linear interpolation
        limit: Maximum number of consecutive NaNs to fill
        
    Returns:
        DataFrame with missing data handled
    """
    logger.debug(f"Handling missing data using method: {method}")
    
    if method == "ffill":
        return df.fillna(method="ffill", limit=limit)
    elif method == "bfill":
        return df.fillna(method="bfill", limit=limit)
    elif method == "drop":
        return df.dropna()
    elif method == "interpolate":
        return df.interpolate(method="linear", limit=limit, limit_direction="forward")
    else:
        raise ValueError(f"Unknown method: {method}")


def resample_data(
    df: pd.DataFrame,
    freq: str,
    agg_methods: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Resample OHLCV data to different frequency.
    
    Args:
        df: DataFrame with DatetimeIndex or MultiIndex
        freq: Target frequency ('D', 'W', 'M', etc.)
        agg_methods: Custom aggregation methods per column
        
    Returns:
        Resampled DataFrame
    """
    logger.info(f"Resampling data to frequency: {freq}")
    
    # Default OHLCV aggregation
    default_agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    
    agg_methods = agg_methods or default_agg
    
    # Handle MultiIndex (date, symbol)
    if isinstance(df.index, pd.MultiIndex):
        # Resample within each symbol group
        resampled = df.groupby(level=1).resample(freq, level=0).agg(agg_methods)
        # Reorder index to (date, symbol)
        resampled = resampled.swaplevel(0, 1).sort_index()
    else:
        resampled = df.resample(freq).agg(agg_methods)
    
    logger.info(f"Resampled from {len(df)} to {len(resampled)} rows")
    return resampled


def calculate_returns(
    prices: pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
    periods: int = 1,
) -> pd.DataFrame:
    """
    Calculate returns from prices.
    
    Args:
        prices: DataFrame of prices
        method: 'simple' for arithmetic returns, 'log' for log returns
        periods: Number of periods for return calculation
        
    Returns:
        DataFrame of returns
    """
    if method == "simple":
        returns = prices.pct_change(periods=periods)
    elif method == "log":
        returns = np.log(prices / prices.shift(periods))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return returns


def adjust_for_splits(
    df: pd.DataFrame,
    split_ratios: pd.Series,
) -> pd.DataFrame:
    """
    Adjust prices for stock splits.
    
    Args:
        df: DataFrame with OHLC data
        split_ratios: Series with split ratios (date -> ratio)
        
    Returns:
        Adjusted DataFrame
    """
    logger.info("Adjusting prices for splits")
    
    adjusted = df.copy()
    price_cols = ["open", "high", "low", "close"]
    
    for date, ratio in split_ratios.items():
        # Adjust all prices before the split date
        mask = adjusted.index.get_level_values(0) < date
        adjusted.loc[mask, price_cols] *= ratio
        
        # Adjust volume
        if "volume" in adjusted.columns:
            adjusted.loc[mask, "volume"] /= ratio
    
    return adjusted


def normalize_prices(
    prices: pd.DataFrame,
    base_value: float = 100.0,
) -> pd.DataFrame:
    """
    Normalize prices to start at a base value.
    
    Args:
        prices: DataFrame of prices
        base_value: Value to normalize to
        
    Returns:
        Normalized prices
    """
    first_prices = prices.iloc[0]
    return prices / first_prices * base_value


def calculate_rolling_stats(
    data: pd.DataFrame,
    window: int,
    stats: Optional[list] = None,
) -> pd.DataFrame:
    """
    Calculate rolling statistics.
    
    Args:
        data: DataFrame of data
        window: Rolling window size
        stats: List of statistics to calculate ['mean', 'std', 'min', 'max']
        
    Returns:
        DataFrame with rolling statistics
    """
    stats = stats or ["mean", "std"]
    
    results = {}
    for stat in stats:
        if stat == "mean":
            results[f"{data.columns[0]}_rolling_mean"] = data.rolling(window).mean()
        elif stat == "std":
            results[f"{data.columns[0]}_rolling_std"] = data.rolling(window).std()
        elif stat == "min":
            results[f"{data.columns[0]}_rolling_min"] = data.rolling(window).min()
        elif stat == "max":
            results[f"{data.columns[0]}_rolling_max"] = data.rolling(window).max()
    
    return pd.concat(results.values(), axis=1)


def align_data(
    *dfs: pd.DataFrame,
    how: Literal["inner", "outer", "left", "right"] = "inner",
) -> tuple:
    """
    Align multiple DataFrames to common index.
    
    Args:
        *dfs: DataFrames to align
        how: Join method
        
    Returns:
        Tuple of aligned DataFrames
    """
    if len(dfs) < 2:
        return dfs
    
    logger.debug(f"Aligning {len(dfs)} DataFrames using '{how}' join")
    
    # Get common index
    if how == "inner":
        common_idx = dfs[0].index
        for df in dfs[1:]:
            common_idx = common_idx.intersection(df.index)
    elif how == "outer":
        common_idx = dfs[0].index
        for df in dfs[1:]:
            common_idx = common_idx.union(df.index)
    else:
        # For left/right, use first/last DataFrame's index
        common_idx = dfs[0].index if how == "left" else dfs[-1].index
    
    # Reindex all DataFrames
    aligned = tuple(df.reindex(common_idx) for df in dfs)
    
    logger.debug(f"Aligned to {len(common_idx)} common rows")
    return aligned


def remove_outliers(
    data: pd.DataFrame,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Remove outliers from data.
    
    Args:
        data: DataFrame to process
        method: Outlier detection method:
            - 'iqr': Interquartile range method
            - 'zscore': Z-score method
        threshold: Threshold for outlier detection
            - For IQR: multiplier of IQR (default 3.0)
            - For Z-score: number of standard deviations (default 3.0)
        
    Returns:
        DataFrame with outliers removed
    """
    logger.info(f"Removing outliers using {method} method (threshold={threshold})")
    
    cleaned = data.copy()
    
    if method == "iqr":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        mask = (data >= lower_bound) & (data <= upper_bound)
        cleaned = data[mask.all(axis=1)]
    
    elif method == "zscore":
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = (z_scores < threshold).all(axis=1)
        cleaned = data[mask]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    n_removed = len(data) - len(cleaned)
    logger.info(f"Removed {n_removed} outlier rows ({n_removed/len(data)*100:.2f}%)")
    
    return cleaned


def winsorize(
    data: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """
    Winsorize data by clipping extreme values.
    
    Args:
        data: DataFrame to winsorize
        lower: Lower percentile to clip to
        upper: Upper percentile to clip to
        
    Returns:
        Winsorized DataFrame
    """
    lower_bound = data.quantile(lower)
    upper_bound = data.quantile(upper)
    
    return data.clip(lower=lower_bound, upper=upper_bound, axis=1)


def forward_fill_gaps(
    df: pd.DataFrame,
    max_gap: int = 5,
) -> pd.DataFrame:
    """
    Forward fill gaps up to a maximum size.
    
    Args:
        df: DataFrame with potential gaps
        max_gap: Maximum gap size to fill
        
    Returns:
        DataFrame with gaps filled
    """
    return df.fillna(method="ffill", limit=max_gap)


def ensure_business_days(
    df: pd.DataFrame,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Ensure data has all business days in range (fill missing days).
    
    Args:
        df: DataFrame with DatetimeIndex or MultiIndex
        fill_method: Method to fill missing business days
        
    Returns:
        DataFrame with all business days
    """
    if isinstance(df.index, pd.MultiIndex):
        # Handle MultiIndex
        dates = df.index.get_level_values(0).unique()
        symbols = df.index.get_level_values(1).unique()
        
        # Create full business day range
        full_dates = pd.bdate_range(start=dates.min(), end=dates.max())
        
        # Create full MultiIndex
        full_index = pd.MultiIndex.from_product(
            [full_dates, symbols],
            names=df.index.names,
        )
        
        # Reindex and fill
        df = df.reindex(full_index)
        if fill_method:
            df = df.fillna(method=fill_method)
    
    else:
        # Handle regular DatetimeIndex
        full_dates = pd.bdate_range(start=df.index.min(), end=df.index.max())
        df = df.reindex(full_dates)
        if fill_method:
            df = df.fillna(method=fill_method)
    
    return df


def calculate_volatility(
    returns: pd.DataFrame,
    window: int = 20,
    annualization_factor: float = 252,
) -> pd.DataFrame:
    """
    Calculate rolling volatility from returns.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window for volatility calculation
        annualization_factor: Factor to annualize volatility (252 for daily)
        
    Returns:
        DataFrame of annualized volatility
    """
    vol = returns.rolling(window=window).std() * np.sqrt(annualization_factor)
    return vol

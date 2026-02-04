"""ML preprocessing utilities.

Handles conversion between JSF data structures and scikit-learn compatible formats.
"""

from typing import Dict, List, Optional, Tuple, Union, Literal
from enum import Enum
from dataclasses import dataclass

import pandas as pd
import numpy as np

from jsf.utils.logging import get_logger
from jsf.data import PriceData

logger = get_logger(__name__)


class TargetType(Enum):
    """Target variable types for ML."""
    RETURNS = "returns"
    DIRECTION = "direction"
    BOTH = "both"


@dataclass
class MLDataset:
    """Container for ML-ready dataset."""
    X: pd.DataFrame  # Features (date x features) or (date*symbol x features)
    y_returns: Optional[pd.Series] = None  # Continuous target
    y_direction: Optional[pd.Series] = None  # Classification target
    dates: Optional[pd.DatetimeIndex] = None
    symbols: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.X)
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X.shape[1]


class MultiIndexConverter:
    """Convert between MultiIndex and flat DataFrame formats.
    
    JSF uses MultiIndex (date, symbol) format throughout.
    scikit-learn expects flat 2D arrays.
    This class handles the conversion.
    """
    
    def __init__(self):
        self._original_index = None
        self._dates = None
        self._symbols = None
    
    def to_flat(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert MultiIndex DataFrame to flat format.
        
        Args:
            df: DataFrame with MultiIndex (date, symbol) or wide format (date x symbol)
            
        Returns:
            Flat DataFrame where each row is one (date, symbol) observation
        """
        if isinstance(df.index, pd.MultiIndex):
            # Already in (date, symbol) format
            self._original_index = df.index
            self._dates = df.index.get_level_values(0).unique()
            self._symbols = df.index.get_level_values(1).unique().tolist()
            return df.reset_index(drop=False)
        else:
            # Wide format (date x symbol) - need to stack
            self._dates = df.index
            self._symbols = df.columns.tolist()
            stacked = df.stack()
            stacked.index.names = ['date', 'symbol']
            self._original_index = stacked.index
            return stacked.reset_index()
    
    def to_multiindex(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        dates: Optional[pd.DatetimeIndex] = None,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convert flat predictions back to MultiIndex format.
        
        Args:
            data: Flat array/series of predictions
            dates: Date index (uses stored if None)
            symbols: Symbol list (uses stored if None)
            
        Returns:
            DataFrame with MultiIndex (date, symbol)
        """
        dates = dates if dates is not None else self._dates
        symbols = symbols if symbols is not None else self._symbols
        
        if dates is None or symbols is None:
            raise ValueError("Must provide dates and symbols or call to_flat first")
        
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        # Create MultiIndex
        index = pd.MultiIndex.from_product(
            [dates, symbols],
            names=['date', 'symbol']
        )
        
        if len(data) == len(index):
            return pd.DataFrame(data.values, index=index, columns=['prediction'])
        else:
            # Data might be subset - try to match
            return pd.DataFrame(data.values, columns=['prediction'])
    
    def to_wide(
        self,
        data: Union[np.ndarray, pd.Series],
        dates: Optional[pd.DatetimeIndex] = None,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convert flat predictions to wide format (date x symbol).
        
        Args:
            data: Flat array of predictions
            dates: Date index
            symbols: Symbol list
            
        Returns:
            Wide DataFrame (date x symbol)
        """
        dates = dates if dates is not None else self._dates
        symbols = symbols if symbols is not None else self._symbols
        
        if dates is None or symbols is None:
            raise ValueError("Must provide dates and symbols")
        
        n_dates = len(dates)
        n_symbols = len(symbols)
        
        if isinstance(data, pd.Series):
            data = data.values
        
        # Reshape to (dates, symbols)
        if len(data) == n_dates * n_symbols:
            reshaped = data.reshape(n_dates, n_symbols)
            return pd.DataFrame(reshaped, index=dates, columns=symbols)
        else:
            raise ValueError(
                f"Data length {len(data)} doesn't match "
                f"dates*symbols = {n_dates * n_symbols}"
            )


def create_target_variable(
    price_data: PriceData,
    target_type: Union[str, TargetType] = TargetType.BOTH,
    forward_periods: int = 1,
    direction_threshold: float = 0.0,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Create target variables for ML training.
    
    Args:
        price_data: PriceData object with OHLCV data
        target_type: Type of target ('returns', 'direction', 'both')
        forward_periods: Number of periods forward for target
        direction_threshold: Threshold for direction (returns > threshold = 1)
        
    Returns:
        Tuple of (y_returns, y_direction) DataFrames
        Each is (date x symbol) format
    """
    if isinstance(target_type, str):
        target_type = TargetType(target_type)
    
    close = price_data.get_close_prices()
    
    # Calculate forward returns (shift by -forward_periods to get future)
    forward_returns = close.pct_change(periods=forward_periods).shift(-forward_periods)
    
    y_returns = None
    y_direction = None
    
    if target_type in [TargetType.RETURNS, TargetType.BOTH]:
        y_returns = forward_returns
        logger.debug(f"Created returns target with {forward_periods}-period forward returns")
    
    if target_type in [TargetType.DIRECTION, TargetType.BOTH]:
        # -1 for down, 0 for neutral, 1 for up
        y_direction = pd.DataFrame(
            np.where(
                forward_returns > direction_threshold, 1,
                np.where(forward_returns < -direction_threshold, -1, 0)
            ),
            index=forward_returns.index,
            columns=forward_returns.columns
        )
        logger.debug(f"Created direction target with threshold {direction_threshold}")
    
    return y_returns, y_direction


def prepare_ml_data(
    features: pd.DataFrame,
    y_returns: Optional[pd.DataFrame] = None,
    y_direction: Optional[pd.DataFrame] = None,
    dropna: bool = True,
) -> MLDataset:
    """Prepare data for ML training.
    
    Converts JSF format (date x symbol for features, date x symbol for targets)
    into flat format suitable for sklearn.
    
    Args:
        features: Feature DataFrame (date x features) or MultiIndex
        y_returns: Returns target (date x symbol)
        y_direction: Direction target (date x symbol)
        dropna: Whether to drop rows with NaN values
        
    Returns:
        MLDataset with X, y_returns, y_direction in flat format
    """
    # Stack features if in wide format
    if isinstance(features.index, pd.MultiIndex):
        X = features.copy()
        dates = features.index.get_level_values(0).unique()
        symbols = features.index.get_level_values(1).unique().tolist()
    else:
        # Assume features are already flat (date x features)
        X = features.copy()
        dates = features.index
        symbols = None
    
    feature_names = X.columns.tolist()
    
    # Stack targets to match features
    y_ret = None
    y_dir = None
    
    if y_returns is not None:
        if isinstance(y_returns.index, pd.MultiIndex):
            y_ret = y_returns.squeeze()
        else:
            # Stack wide format (date x symbol) to Series
            y_ret = y_returns.stack()
    
    if y_direction is not None:
        if isinstance(y_direction.index, pd.MultiIndex):
            y_dir = y_direction.squeeze()
        else:
            y_dir = y_direction.stack()
    
    # Align indices
    if y_ret is not None and len(X) != len(y_ret):
        common_idx = X.index.intersection(y_ret.index)
        X = X.loc[common_idx]
        y_ret = y_ret.loc[common_idx]
        if y_dir is not None:
            y_dir = y_dir.loc[common_idx]
    
    # Drop NaN values
    if dropna:
        # Find rows with any NaN in features
        valid_mask = ~X.isna().any(axis=1)
        
        if y_ret is not None:
            valid_mask &= ~y_ret.isna()
        if y_dir is not None:
            valid_mask &= ~y_dir.isna()
        
        X = X[valid_mask]
        if y_ret is not None:
            y_ret = y_ret[valid_mask]
        if y_dir is not None:
            y_dir = y_dir[valid_mask]
        
        logger.debug(f"Dropped {(~valid_mask).sum()} rows with NaN values")
    
    return MLDataset(
        X=X,
        y_returns=y_ret,
        y_direction=y_dir,
        dates=dates,
        symbols=symbols,
        feature_names=feature_names,
    )


def split_train_test(
    dataset: MLDataset,
    train_end_date: pd.Timestamp,
    test_start_date: Optional[pd.Timestamp] = None,
) -> Tuple[MLDataset, MLDataset]:
    """Split dataset into train and test by date.
    
    For time-series, we must split by date to avoid lookahead bias.
    
    Args:
        dataset: MLDataset to split
        train_end_date: Last date for training (exclusive)
        test_start_date: First date for testing (defaults to train_end_date)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if test_start_date is None:
        test_start_date = train_end_date
    
    # Get dates from index
    if isinstance(dataset.X.index, pd.MultiIndex):
        dates = dataset.X.index.get_level_values(0)
    else:
        dates = dataset.X.index
    
    train_mask = dates < train_end_date
    test_mask = dates >= test_start_date
    
    # Split features
    X_train = dataset.X[train_mask]
    X_test = dataset.X[test_mask]
    
    # Split targets
    y_ret_train = dataset.y_returns[train_mask] if dataset.y_returns is not None else None
    y_ret_test = dataset.y_returns[test_mask] if dataset.y_returns is not None else None
    y_dir_train = dataset.y_direction[train_mask] if dataset.y_direction is not None else None
    y_dir_test = dataset.y_direction[test_mask] if dataset.y_direction is not None else None
    
    train_dataset = MLDataset(
        X=X_train,
        y_returns=y_ret_train,
        y_direction=y_dir_train,
        dates=dataset.dates,
        symbols=dataset.symbols,
        feature_names=dataset.feature_names,
    )
    
    test_dataset = MLDataset(
        X=X_test,
        y_returns=y_ret_test,
        y_direction=y_dir_test,
        dates=dataset.dates,
        symbols=dataset.symbols,
        feature_names=dataset.feature_names,
    )
    
    logger.debug(
        f"Split data: train={len(X_train)} samples, test={len(X_test)} samples"
    )
    
    return train_dataset, test_dataset


def handle_missing_features(
    features: pd.DataFrame,
    method: Literal["drop", "ffill", "mean", "zero"] = "ffill",
    max_missing_pct: float = 0.5,
) -> pd.DataFrame:
    """Handle missing values in feature matrix.
    
    Args:
        features: Feature DataFrame
        method: How to handle missing values
            - 'drop': Drop rows with any NaN
            - 'ffill': Forward fill within each group
            - 'mean': Fill with column mean
            - 'zero': Fill with zero
        max_missing_pct: Max % missing before dropping column
        
    Returns:
        Features with missing values handled
    """
    # Drop columns with too many missing values
    missing_pct = features.isna().mean()
    cols_to_drop = missing_pct[missing_pct > max_missing_pct].index.tolist()
    
    if cols_to_drop:
        logger.warning(f"Dropping columns with >{max_missing_pct:.0%} missing: {cols_to_drop}")
        features = features.drop(columns=cols_to_drop)
    
    # Handle remaining missing values
    if method == "drop":
        features = features.dropna()
    elif method == "ffill":
        if isinstance(features.index, pd.MultiIndex):
            # Forward fill within each symbol
            features = features.groupby(level=1).ffill()
        else:
            features = features.ffill()
    elif method == "mean":
        features = features.fillna(features.mean())
    elif method == "zero":
        features = features.fillna(0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return features

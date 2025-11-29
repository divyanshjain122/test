"""
Universe definition and filtering.

Defines asset universes and provides filtering capabilities.
"""

from typing import List, Optional, Dict, Set, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from jsf.config import UniverseType
from jsf.utils import get_logger

logger = get_logger(__name__)


# Predefined universe constituents
# In production, these would be loaded from a database or API
UNIVERSE_CONSTITUENTS: Dict[UniverseType, List[str]] = {
    UniverseType.SP500: [
        # Top 50 SP500 companies (sample - would be full 500 in production)
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
        "V", "UNH", "JNJ", "WMT", "JPM", "MA", "PG", "XOM", "HD", "CVX",
        "MRK", "ABBV", "KO", "PEP", "AVGO", "COST", "LLY", "TMO", "CSCO",
        "ACN", "MCD", "DHR", "ABT", "VZ", "ADBE", "NKE", "TXN", "PM",
        "NEE", "CRM", "ORCL", "DIS", "CMCSA", "INTC", "BMY", "UPS", "T",
        "HON", "LOW", "WFC", "AMD", "QCOM",
    ],
    
    UniverseType.NASDAQ_100: [
        # Top 30 Nasdaq 100 companies (sample)
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
        "ASML", "COST", "PEP", "CSCO", "ADBE", "NFLX", "CMCSA", "TXN",
        "AMD", "QCOM", "INTC", "HON", "INTU", "AMGN", "AMAT", "ISRG",
        "BKNG", "ADP", "VRTX", "SBUX", "GILD", "MU",
    ],
    
    UniverseType.DOW_30: [
        "AAPL", "MSFT", "UNH", "GS", "HD", "MCD", "CAT", "V", "AMGN",
        "BA", "HON", "TRV", "JPM", "IBM", "AXP", "CVX", "WMT", "JNJ",
        "PG", "MMM", "DIS", "NKE", "CRM", "MRK", "KO", "CSCO", "DOW",
        "VZ", "INTC", "WBA",
    ],
}


@dataclass
class UniverseFilter:
    """Configuration for filtering a universe."""
    
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[float] = None
    min_market_cap: Optional[float] = None
    exclude_symbols: Optional[Set[str]] = None
    include_only: Optional[Set[str]] = None
    min_trading_days: Optional[int] = None


class Universe:
    """
    Represents a universe of tradable assets.
    
    Can be created from predefined universes or custom symbol lists.
    """
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        description: Optional[str] = None,
    ):
        """
        Initialize universe.
        
        Args:
            name: Universe name
            symbols: List of ticker symbols
            description: Optional description
        """
        self.name = name
        self.symbols = sorted(list(set(symbols)))  # Remove duplicates and sort
        self.description = description or f"Universe with {len(self.symbols)} symbols"
        
        logger.info(f"Created universe '{name}' with {len(self.symbols)} symbols")
    
    @classmethod
    def from_predefined(cls, universe_type: UniverseType) -> "Universe":
        """
        Create universe from predefined type.
        
        Args:
            universe_type: Predefined universe type
            
        Returns:
            Universe instance
        """
        if universe_type not in UNIVERSE_CONSTITUENTS:
            raise ValueError(f"Unknown universe type: {universe_type}")
        
        symbols = UNIVERSE_CONSTITUENTS[universe_type]
        return cls(
            name=universe_type.value,
            symbols=symbols,
            description=f"Predefined {universe_type.value} universe",
        )
    
    @classmethod
    def from_symbols(cls, symbols: List[str], name: str = "custom") -> "Universe":
        """
        Create universe from symbol list.
        
        Args:
            symbols: List of symbols
            name: Universe name
            
        Returns:
            Universe instance
        """
        return cls(name=name, symbols=symbols)
    
    def filter(
        self,
        data: pd.DataFrame,
        filters: Optional[UniverseFilter] = None,
    ) -> "Universe":
        """
        Filter universe based on data and criteria.
        
        Args:
            data: Price data to use for filtering
            filters: Filter configuration
            
        Returns:
            New filtered Universe
        """
        if filters is None:
            return self
        
        filtered_symbols = set(self.symbols)
        
        # Apply include_only filter first
        if filters.include_only:
            filtered_symbols &= filters.include_only
        
        # Exclude symbols
        if filters.exclude_symbols:
            filtered_symbols -= filters.exclude_symbols
        
        # Price and volume filters require data
        if isinstance(data.index, pd.MultiIndex):
            # Get latest prices and volumes for filtering
            latest_date = data.index.get_level_values(0).max()
            latest_data = data.xs(latest_date, level=0)
            
            # Price filters
            if filters.min_price is not None and "close" in latest_data.columns:
                mask = latest_data["close"] >= filters.min_price
                valid_symbols = set(latest_data[mask].index)
                filtered_symbols &= valid_symbols
            
            if filters.max_price is not None and "close" in latest_data.columns:
                mask = latest_data["close"] <= filters.max_price
                valid_symbols = set(latest_data[mask].index)
                filtered_symbols &= valid_symbols
            
            # Volume filter
            if filters.min_volume is not None and "volume" in latest_data.columns:
                mask = latest_data["volume"] >= filters.min_volume
                valid_symbols = set(latest_data[mask].index)
                filtered_symbols &= valid_symbols
            
            # Minimum trading days filter
            if filters.min_trading_days is not None:
                symbol_counts = data.index.get_level_values(1).value_counts()
                valid_symbols = set(
                    symbol_counts[symbol_counts >= filters.min_trading_days].index
                )
                filtered_symbols &= valid_symbols
        
        logger.info(
            f"Filtered universe from {len(self.symbols)} to {len(filtered_symbols)} symbols"
        )
        
        return Universe(
            name=f"{self.name}_filtered",
            symbols=sorted(list(filtered_symbols)),
            description=f"Filtered from {self.name}",
        )
    
    def intersection(self, other: "Universe") -> "Universe":
        """
        Get intersection of two universes.
        
        Args:
            other: Other universe
            
        Returns:
            New universe with common symbols
        """
        common_symbols = set(self.symbols) & set(other.symbols)
        return Universe(
            name=f"{self.name}_x_{other.name}",
            symbols=sorted(list(common_symbols)),
            description=f"Intersection of {self.name} and {other.name}",
        )
    
    def union(self, other: "Universe") -> "Universe":
        """
        Get union of two universes.
        
        Args:
            other: Other universe
            
        Returns:
            New universe with all symbols from both
        """
        all_symbols = set(self.symbols) | set(other.symbols)
        return Universe(
            name=f"{self.name}_+_{other.name}",
            symbols=sorted(list(all_symbols)),
            description=f"Union of {self.name} and {other.name}",
        )
    
    def difference(self, other: "Universe") -> "Universe":
        """
        Get difference of two universes.
        
        Args:
            other: Other universe
            
        Returns:
            New universe with symbols in self but not in other
        """
        diff_symbols = set(self.symbols) - set(other.symbols)
        return Universe(
            name=f"{self.name}_-_{other.name}",
            symbols=sorted(list(diff_symbols)),
            description=f"Difference of {self.name} and {other.name}",
        )
    
    def sample(self, n: int, seed: Optional[int] = None) -> "Universe":
        """
        Get random sample of symbols from universe.
        
        Args:
            n: Number of symbols to sample
            seed: Random seed for reproducibility
            
        Returns:
            New universe with sampled symbols
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        n = min(n, len(self.symbols))
        sampled = random.sample(self.symbols, n)
        
        return Universe(
            name=f"{self.name}_sample_{n}",
            symbols=sampled,
            description=f"Random sample of {n} symbols from {self.name}",
        )
    
    def __len__(self) -> int:
        """Get number of symbols."""
        return len(self.symbols)
    
    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is in universe."""
        return symbol in self.symbols
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Universe(name='{self.name}', n_symbols={len(self.symbols)})"
    
    def __str__(self) -> str:
        """String representation."""
        preview = ", ".join(self.symbols[:5])
        suffix = f", ... ({len(self.symbols) - 5} more)" if len(self.symbols) > 5 else ""
        return f"Universe '{self.name}': [{preview}{suffix}]"


def create_universe(
    universe_spec: str | List[str] | UniverseType,
    name: Optional[str] = None,
) -> Universe:
    """
    Convenience function to create a universe from various inputs.
    
    Args:
        universe_spec: Universe specification (predefined name, type, or symbol list)
        name: Optional custom name
        
    Returns:
        Universe instance
        
    Examples:
        >>> universe = create_universe("SP500")
        >>> universe = create_universe(UniverseType.NASDAQ_100)
        >>> universe = create_universe(["AAPL", "GOOGL", "MSFT"])
    """
    # Handle UniverseType enum
    if isinstance(universe_spec, UniverseType):
        return Universe.from_predefined(universe_spec)
    
    # Handle string (try as predefined universe)
    if isinstance(universe_spec, str):
        try:
            universe_type = UniverseType(universe_spec)
            return Universe.from_predefined(universe_type)
        except ValueError:
            # Treat as single symbol
            return Universe.from_symbols([universe_spec], name=name or "custom")
    
    # Handle list of symbols
    if isinstance(universe_spec, list):
        return Universe.from_symbols(universe_spec, name=name or "custom")
    
    raise ValueError(f"Invalid universe specification: {universe_spec}")

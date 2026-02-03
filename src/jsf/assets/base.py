"""Base classes for multi-asset support."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, Dict, Any, Union

import pandas as pd
import numpy as np

from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class AssetType(Enum):
    """Asset type enumeration."""
    
    EQUITY = "equity"
    FUTURE = "future"
    OPTION = "option"
    CRYPTO = "crypto"
    FOREX = "forex"
    ETF = "etf"
    INDEX = "index"
    BOND = "bond"


class TradingSession(Enum):
    """Trading session types."""
    
    REGULAR = "regular"      # Standard market hours
    EXTENDED = "extended"    # Pre/post market
    CONTINUOUS = "continuous"  # 24/7 (crypto)


@dataclass
class AssetSpec:
    """
    Asset specification containing static attributes.
    
    Attributes:
        symbol: Ticker symbol
        asset_type: Type of asset
        name: Full asset name
        currency: Quote currency (USD, EUR, etc.)
        exchange: Trading exchange
        lot_size: Minimum tradeable quantity
        tick_size: Minimum price movement
        margin_requirement: Initial margin percentage
        session: Trading session type
        metadata: Additional asset-specific data
    """
    symbol: str
    asset_type: AssetType
    name: str = ""
    currency: str = "USD"
    exchange: str = ""
    lot_size: float = 1.0
    tick_size: float = 0.01
    margin_requirement: float = 1.0  # 1.0 = 100% (no margin)
    session: TradingSession = TradingSession.REGULAR
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate asset spec."""
        if self.lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {self.lot_size}")
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
        if not 0 < self.margin_requirement <= 1.0:
            raise ValueError(f"margin_requirement must be in (0, 1], got {self.margin_requirement}")


class Asset(ABC):
    """
    Abstract base class for all asset types.
    
    Provides common interface for different instruments.
    """
    
    def __init__(
        self,
        symbol: str,
        currency: str = "USD",
        exchange: str = "",
        name: str = "",
        **kwargs: Any,
    ):
        """
        Initialize asset.
        
        Args:
            symbol: Ticker symbol
            currency: Quote currency
            exchange: Trading exchange
            name: Full name (defaults to symbol)
            **kwargs: Additional parameters
        """
        self.symbol = symbol
        self.currency = currency
        self.exchange = exchange
        self.name = name or symbol
        self._metadata = kwargs
        
        logger.debug(f"Created {self.__class__.__name__}: {symbol}")
    
    @property
    @abstractmethod
    def asset_type(self) -> AssetType:
        """Return the asset type."""
        pass
    
    @property
    def spec(self) -> AssetSpec:
        """Get asset specification."""
        return AssetSpec(
            symbol=self.symbol,
            asset_type=self.asset_type,
            name=self.name,
            currency=self.currency,
            exchange=self.exchange,
            lot_size=self.lot_size,
            tick_size=self.tick_size,
            margin_requirement=self.margin_requirement,
            session=self.trading_session,
            metadata=self._metadata,
        )
    
    @property
    def lot_size(self) -> float:
        """Minimum tradeable quantity."""
        return 1.0
    
    @property
    def tick_size(self) -> float:
        """Minimum price movement."""
        return 0.01
    
    @property
    def margin_requirement(self) -> float:
        """Initial margin requirement (1.0 = 100%)."""
        return 1.0
    
    @property
    def trading_session(self) -> TradingSession:
        """Trading session type."""
        return TradingSession.REGULAR
    
    def round_price(self, price: float) -> float:
        """
        Round price to valid tick size.
        
        Args:
            price: Raw price
            
        Returns:
            Price rounded to tick size
        """
        return round(price / self.tick_size) * self.tick_size
    
    def round_quantity(self, quantity: float) -> float:
        """
        Round quantity to valid lot size.
        
        Args:
            quantity: Raw quantity
            
        Returns:
            Quantity rounded to lot size
        """
        return round(quantity / self.lot_size) * self.lot_size
    
    def calculate_value(self, quantity: float, price: float) -> float:
        """
        Calculate position value.
        
        Args:
            quantity: Number of units
            price: Current price
            
        Returns:
            Position value in quote currency
        """
        return quantity * price * self.get_multiplier()
    
    def get_multiplier(self) -> float:
        """
        Get contract multiplier.
        
        Returns:
            Contract multiplier (1.0 for equities)
        """
        return 1.0
    
    def get_margin(self, quantity: float, price: float) -> float:
        """
        Calculate margin requirement.
        
        Args:
            quantity: Number of units
            price: Current price
            
        Returns:
            Margin required in quote currency
        """
        return self.calculate_value(abs(quantity), price) * self.margin_requirement
    
    def is_tradeable(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if asset is tradeable at given time.
        
        Args:
            timestamp: Time to check (defaults to now)
            
        Returns:
            True if tradeable
        """
        if self.trading_session == TradingSession.CONTINUOUS:
            return True
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check weekday (Mon-Fri for regular/extended)
        if timestamp.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # For regular session, check market hours (simplified: 9:30-16:00 ET)
        if self.trading_session == TradingSession.REGULAR:
            hour = timestamp.hour
            minute = timestamp.minute
            time_minutes = hour * 60 + minute
            
            # 9:30 AM = 570 minutes, 4:00 PM = 960 minutes
            return 570 <= time_minutes <= 960
        
        # Extended hours: 4:00 AM - 8:00 PM
        if self.trading_session == TradingSession.EXTENDED:
            hour = timestamp.hour
            return 4 <= hour <= 20
        
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(symbol='{self.symbol}')"
    
    def __eq__(self, other: Any) -> bool:
        """Check equality."""
        if not isinstance(other, Asset):
            return False
        return (
            self.symbol == other.symbol and
            self.asset_type == other.asset_type
        )
    
    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self.symbol, self.asset_type))


class Equity(Asset):
    """
    Equity (stock) asset.
    
    Standard equity shares traded on stock exchanges.
    
    Example:
        >>> aapl = Equity("AAPL", name="Apple Inc.", exchange="NASDAQ")
        >>> value = aapl.calculate_value(100, 175.50)
        >>> print(f"Position value: ${value:,.2f}")
    """
    
    def __init__(
        self,
        symbol: str,
        currency: str = "USD",
        exchange: str = "",
        name: str = "",
        fractional: bool = False,
        shortable: bool = True,
        hard_to_borrow: bool = False,
        dividend_yield: float = 0.0,
        **kwargs: Any,
    ):
        """
        Initialize equity.
        
        Args:
            symbol: Ticker symbol
            currency: Quote currency
            exchange: Stock exchange
            name: Company name
            fractional: Allow fractional shares
            shortable: Can be sold short
            hard_to_borrow: Difficult to borrow for shorting
            dividend_yield: Annual dividend yield
            **kwargs: Additional parameters
        """
        super().__init__(symbol, currency, exchange, name, **kwargs)
        
        self.fractional = fractional
        self.shortable = shortable
        self.hard_to_borrow = hard_to_borrow
        self.dividend_yield = dividend_yield
    
    @property
    def asset_type(self) -> AssetType:
        """Return asset type."""
        return AssetType.EQUITY
    
    @property
    def lot_size(self) -> float:
        """Minimum tradeable quantity."""
        return 0.001 if self.fractional else 1.0
    
    @property
    def tick_size(self) -> float:
        """Minimum price movement."""
        return 0.01
    
    def calculate_dividend(
        self,
        shares: float,
        price: float,
        holding_days: int = 365,
    ) -> float:
        """
        Estimate dividend income.
        
        Args:
            shares: Number of shares held
            price: Current share price
            holding_days: Days held
            
        Returns:
            Estimated dividend income
        """
        annual_dividend = shares * price * self.dividend_yield
        return annual_dividend * (holding_days / 365)


class ETF(Equity):
    """
    Exchange-Traded Fund.
    
    ETFs trade like stocks but track indices, commodities, or baskets.
    """
    
    def __init__(
        self,
        symbol: str,
        currency: str = "USD",
        exchange: str = "",
        name: str = "",
        expense_ratio: float = 0.0,
        leveraged: bool = False,
        leverage_factor: float = 1.0,
        inverse: bool = False,
        underlying: str = "",
        **kwargs: Any,
    ):
        """
        Initialize ETF.
        
        Args:
            symbol: Ticker symbol
            currency: Quote currency
            exchange: Trading exchange
            name: ETF name
            expense_ratio: Annual expense ratio
            leveraged: Is leveraged ETF
            leverage_factor: Leverage multiplier (2x, 3x)
            inverse: Is inverse/short ETF
            underlying: Underlying index/asset
            **kwargs: Additional parameters
        """
        super().__init__(symbol, currency, exchange, name, **kwargs)
        
        self.expense_ratio = expense_ratio
        self.leveraged = leveraged
        self.leverage_factor = leverage_factor
        self.inverse = inverse
        self.underlying = underlying
    
    @property
    def asset_type(self) -> AssetType:
        """Return asset type."""
        return AssetType.ETF
    
    def get_effective_exposure(self, shares: float, price: float) -> float:
        """
        Calculate effective market exposure.
        
        Args:
            shares: Number of shares
            price: Share price
            
        Returns:
            Effective exposure (accounting for leverage/inverse)
        """
        base_value = shares * price
        exposure = base_value * self.leverage_factor
        
        if self.inverse:
            exposure = -exposure
        
        return exposure


class Index(Asset):
    """
    Market Index (non-tradeable, reference only).
    
    Indices like S&P 500, NASDAQ-100 are used for benchmarking.
    """
    
    def __init__(
        self,
        symbol: str,
        name: str = "",
        components: Optional[list] = None,
        **kwargs: Any,
    ):
        """
        Initialize index.
        
        Args:
            symbol: Index symbol
            name: Index name
            components: List of component symbols
            **kwargs: Additional parameters
        """
        super().__init__(symbol, "USD", "", name, **kwargs)
        self.components = components or []
    
    @property
    def asset_type(self) -> AssetType:
        """Return asset type."""
        return AssetType.INDEX
    
    def is_tradeable(self, timestamp: Optional[datetime] = None) -> bool:
        """Indices are not directly tradeable."""
        return False
    
    def calculate_value(self, quantity: float, price: float) -> float:
        """Index value calculation (notional only)."""
        return quantity * price

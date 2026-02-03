"""Futures contract support with expiry handling and roll logic."""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Union

import pandas as pd
import numpy as np

from jsf.assets.base import Asset, AssetType, TradingSession
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class RollMethod(Enum):
    """Contract roll methods."""
    
    FIRST_NOTICE = "first_notice"  # Roll before first notice date
    LAST_TRADING = "last_trading"  # Roll before last trading day
    VOLUME = "volume"              # Roll when next contract has more volume
    CALENDAR = "calendar"          # Roll on fixed calendar days
    OPEN_INTEREST = "open_interest"  # Roll based on open interest


@dataclass
class ContractSpec:
    """
    Futures contract specification.
    
    Attributes:
        root_symbol: Base symbol (ES, NQ, CL, etc.)
        exchange: Trading exchange
        name: Full contract name
        multiplier: Contract multiplier
        tick_size: Minimum price movement
        tick_value: Value of one tick
        currency: Quote currency
        margin_initial: Initial margin requirement
        margin_maintenance: Maintenance margin
        trading_hours: Trading hours description
        settlement: Cash or physical settlement
        months: Contract months (H, M, U, Z for quarterlies)
    """
    root_symbol: str
    exchange: str
    name: str
    multiplier: float
    tick_size: float
    tick_value: float
    currency: str = "USD"
    margin_initial: float = 0.0
    margin_maintenance: float = 0.0
    trading_hours: str = ""
    settlement: str = "cash"
    months: str = "HMUZ"  # March, June, Sept, Dec


# Common futures specifications
FUTURES_SPECS: Dict[str, ContractSpec] = {
    "ES": ContractSpec(
        root_symbol="ES",
        exchange="CME",
        name="E-mini S&P 500",
        multiplier=50.0,
        tick_size=0.25,
        tick_value=12.50,
        margin_initial=12000.0,
        margin_maintenance=10800.0,
        trading_hours="Sun 6pm - Fri 5pm ET",
        settlement="cash",
        months="HMUZ",
    ),
    "NQ": ContractSpec(
        root_symbol="NQ",
        exchange="CME",
        name="E-mini NASDAQ-100",
        multiplier=20.0,
        tick_size=0.25,
        tick_value=5.00,
        margin_initial=16500.0,
        margin_maintenance=15000.0,
        trading_hours="Sun 6pm - Fri 5pm ET",
        settlement="cash",
        months="HMUZ",
    ),
    "CL": ContractSpec(
        root_symbol="CL",
        exchange="NYMEX",
        name="Crude Oil WTI",
        multiplier=1000.0,
        tick_size=0.01,
        tick_value=10.00,
        margin_initial=7000.0,
        margin_maintenance=6300.0,
        trading_hours="Sun 6pm - Fri 5pm ET",
        settlement="physical",
        months="FGHJKMNQUVXZ",  # All months
    ),
    "GC": ContractSpec(
        root_symbol="GC",
        exchange="COMEX",
        name="Gold",
        multiplier=100.0,
        tick_size=0.10,
        tick_value=10.00,
        margin_initial=10000.0,
        margin_maintenance=9000.0,
        trading_hours="Sun 6pm - Fri 5pm ET",
        settlement="physical",
        months="GJMQVZ",
    ),
    "ZB": ContractSpec(
        root_symbol="ZB",
        exchange="CBOT",
        name="30-Year US Treasury Bond",
        multiplier=1000.0,
        tick_size=0.03125,  # 1/32
        tick_value=31.25,
        margin_initial=4500.0,
        margin_maintenance=4000.0,
        trading_hours="Sun 6pm - Fri 5pm ET",
        settlement="physical",
        months="HMUZ",
    ),
    "6E": ContractSpec(
        root_symbol="6E",
        exchange="CME",
        name="Euro FX",
        multiplier=125000.0,
        tick_size=0.00005,
        tick_value=6.25,
        margin_initial=2500.0,
        margin_maintenance=2200.0,
        trading_hours="Sun 6pm - Fri 5pm ET",
        settlement="physical",
        months="HMUZ",
    ),
}


def get_contract_month_code(month: int) -> str:
    """
    Get futures month code from month number.
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Month code (F, G, H, J, K, M, N, Q, U, V, X, Z)
    """
    codes = "FGHJKMNQUVXZ"
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month: {month}")
    return codes[month - 1]


def get_month_from_code(code: str) -> int:
    """
    Get month number from futures month code.
    
    Args:
        code: Month code (F, G, H, etc.)
        
    Returns:
        Month number (1-12)
    """
    codes = "FGHJKMNQUVXZ"
    code = code.upper()
    if code not in codes:
        raise ValueError(f"Invalid month code: {code}")
    return codes.index(code) + 1


def parse_contract_symbol(symbol: str) -> tuple:
    """
    Parse futures contract symbol.
    
    Examples:
        ESH24 -> (ES, H, 2024)
        CLM2024 -> (CL, M, 2024)
        
    Args:
        symbol: Contract symbol
        
    Returns:
        Tuple of (root_symbol, month_code, year)
    """
    # Try common formats
    # Format 1: ESH24 (root + month + 2-digit year)
    # Format 2: ESH2024 (root + month + 4-digit year)
    
    symbol = symbol.upper()
    
    # Find month code position
    month_codes = "FGHJKMNQUVXZ"
    month_pos = -1
    month_code = ""
    
    for i, char in enumerate(symbol):
        if char in month_codes:
            month_pos = i
            month_code = char
            break
    
    if month_pos == -1:
        raise ValueError(f"Could not parse contract symbol: {symbol}")
    
    root_symbol = symbol[:month_pos]
    year_str = symbol[month_pos + 1:]
    
    # Parse year
    if len(year_str) == 2:
        year = 2000 + int(year_str)
    elif len(year_str) == 4:
        year = int(year_str)
    else:
        raise ValueError(f"Invalid year in contract symbol: {symbol}")
    
    return root_symbol, month_code, year


def build_contract_symbol(
    root_symbol: str,
    month: Union[int, str],
    year: int,
    format: str = "short",
) -> str:
    """
    Build futures contract symbol.
    
    Args:
        root_symbol: Base symbol (ES, NQ, etc.)
        month: Month number (1-12) or code (F, G, etc.)
        year: Full year (2024)
        format: "short" (ESH24) or "long" (ESH2024)
        
    Returns:
        Contract symbol
    """
    if isinstance(month, int):
        month_code = get_contract_month_code(month)
    else:
        month_code = month.upper()
    
    if format == "short":
        return f"{root_symbol}{month_code}{year % 100:02d}"
    else:
        return f"{root_symbol}{month_code}{year}"


@dataclass
class FutureContract:
    """
    Individual futures contract with specific expiry.
    
    Attributes:
        root_symbol: Base symbol
        month: Contract month (1-12)
        year: Contract year
        expiry: Expiration date
        first_notice: First notice date (for physical delivery)
        last_trading: Last trading day
        settlement_price: Final settlement price
    """
    root_symbol: str
    month: int
    year: int
    expiry: date
    first_notice: Optional[date] = None
    last_trading: Optional[date] = None
    settlement_price: Optional[float] = None
    
    @property
    def symbol(self) -> str:
        """Get contract symbol."""
        return build_contract_symbol(self.root_symbol, self.month, self.year)
    
    @property
    def month_code(self) -> str:
        """Get month code."""
        return get_contract_month_code(self.month)
    
    def days_to_expiry(self, as_of: Optional[date] = None) -> int:
        """
        Calculate days until expiry.
        
        Args:
            as_of: Reference date (defaults to today)
            
        Returns:
            Days until expiry (negative if expired)
        """
        if as_of is None:
            as_of = date.today()
        return (self.expiry - as_of).days
    
    def is_expired(self, as_of: Optional[date] = None) -> bool:
        """Check if contract is expired."""
        return self.days_to_expiry(as_of) <= 0
    
    def should_roll(
        self,
        as_of: Optional[date] = None,
        days_before: int = 5,
        method: RollMethod = RollMethod.LAST_TRADING,
    ) -> bool:
        """
        Check if contract should be rolled.
        
        Args:
            as_of: Reference date
            days_before: Days before event to roll
            method: Roll method
            
        Returns:
            True if should roll
        """
        if as_of is None:
            as_of = date.today()
        
        if method == RollMethod.FIRST_NOTICE and self.first_notice:
            return (self.first_notice - as_of).days <= days_before
        elif method == RollMethod.LAST_TRADING and self.last_trading:
            return (self.last_trading - as_of).days <= days_before
        else:
            return (self.expiry - as_of).days <= days_before
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FutureContract({self.symbol}, expiry={self.expiry})"


class Future(Asset):
    """
    Futures contract asset.
    
    Handles expiry, rolls, and margin calculations for futures.
    
    Example:
        >>> es = Future("ES", expiry="2024-03-15")
        >>> print(f"Days to expiry: {es.days_to_expiry()}")
        >>> margin = es.get_margin(1, 5000)
        >>> print(f"Margin required: ${margin:,.2f}")
    """
    
    def __init__(
        self,
        symbol: str,
        expiry: Optional[Union[str, date, datetime]] = None,
        multiplier: Optional[float] = None,
        tick_size: Optional[float] = None,
        margin: Optional[float] = None,
        first_notice: Optional[Union[str, date]] = None,
        currency: str = "USD",
        exchange: str = "",
        name: str = "",
        **kwargs: Any,
    ):
        """
        Initialize futures contract.
        
        Args:
            symbol: Contract symbol (ES, ESH24, etc.)
            expiry: Expiration date
            multiplier: Contract multiplier
            tick_size: Minimum price movement
            margin: Initial margin per contract
            first_notice: First notice date
            currency: Quote currency
            exchange: Trading exchange
            name: Contract name
            **kwargs: Additional parameters
        """
        # Parse symbol to get root
        try:
            root_symbol, month_code, year = parse_contract_symbol(symbol)
            self.root_symbol = root_symbol
            self._parsed_month = get_month_from_code(month_code)
            self._parsed_year = year
        except ValueError:
            # Just a root symbol (ES, NQ, etc.)
            self.root_symbol = symbol.upper()
            self._parsed_month = None
            self._parsed_year = None
        
        # Get spec if available
        self._spec = FUTURES_SPECS.get(self.root_symbol)
        
        # Set exchange and name from spec if not provided
        if not exchange and self._spec:
            exchange = self._spec.exchange
        if not name and self._spec:
            name = self._spec.name
        
        super().__init__(symbol.upper(), currency, exchange, name, **kwargs)
        
        # Parse expiry
        if expiry:
            if isinstance(expiry, str):
                expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
            elif isinstance(expiry, datetime):
                expiry = expiry.date()
        self._expiry = expiry
        
        # Parse first notice
        if first_notice:
            if isinstance(first_notice, str):
                first_notice = datetime.strptime(first_notice, "%Y-%m-%d").date()
            elif isinstance(first_notice, datetime):
                first_notice = first_notice.date()
        self._first_notice = first_notice
        
        # Set contract specs
        self._multiplier = multiplier or (self._spec.multiplier if self._spec else 1.0)
        self._tick_size = tick_size or (self._spec.tick_size if self._spec else 0.01)
        self._margin = margin or (self._spec.margin_initial if self._spec else 0.0)
        
        logger.info(f"Initialized Future: {self.symbol}, multiplier={self._multiplier}")
    
    @property
    def asset_type(self) -> AssetType:
        """Return asset type."""
        return AssetType.FUTURE
    
    @property
    def expiry(self) -> Optional[date]:
        """Get expiration date."""
        return self._expiry
    
    @property
    def first_notice(self) -> Optional[date]:
        """Get first notice date."""
        return self._first_notice
    
    @property
    def tick_size(self) -> float:
        """Minimum price movement."""
        return self._tick_size
    
    @property
    def tick_value(self) -> float:
        """Value of one tick."""
        return self._tick_size * self._multiplier
    
    @property
    def margin_requirement(self) -> float:
        """Margin as fraction (for compatibility)."""
        # Return a reasonable fraction if no margin set
        return 0.10  # 10% default margin for futures
    
    @property
    def trading_session(self) -> TradingSession:
        """Trading session type."""
        # Most futures trade nearly 24 hours
        return TradingSession.EXTENDED
    
    def get_multiplier(self) -> float:
        """Get contract multiplier."""
        return self._multiplier
    
    def get_margin(self, quantity: float, price: float) -> float:
        """
        Calculate margin requirement.
        
        Args:
            quantity: Number of contracts
            price: Contract price
            
        Returns:
            Margin required
        """
        if self._margin > 0:
            return abs(quantity) * self._margin
        else:
            # Fall back to notional * margin_requirement
            return self.calculate_value(abs(quantity), price) * self.margin_requirement
    
    def days_to_expiry(self, as_of: Optional[date] = None) -> Optional[int]:
        """
        Calculate days until expiry.
        
        Args:
            as_of: Reference date (defaults to today)
            
        Returns:
            Days until expiry, or None if no expiry set
        """
        if not self._expiry:
            return None
        if as_of is None:
            as_of = date.today()
        return (self._expiry - as_of).days
    
    def is_expired(self, as_of: Optional[date] = None) -> bool:
        """Check if contract is expired."""
        days = self.days_to_expiry(as_of)
        return days is not None and days <= 0
    
    def should_roll(
        self,
        as_of: Optional[date] = None,
        days_before: int = 5,
    ) -> bool:
        """
        Check if contract should be rolled.
        
        Args:
            as_of: Reference date
            days_before: Days before expiry to roll
            
        Returns:
            True if should roll
        """
        # Check first notice first (for physical delivery)
        if self._first_notice:
            if as_of is None:
                as_of = date.today()
            if (self._first_notice - as_of).days <= days_before:
                return True
        
        # Then check expiry
        days = self.days_to_expiry(as_of)
        return days is not None and days <= days_before
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
    ) -> float:
        """
        Calculate profit/loss.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Number of contracts (negative for short)
            
        Returns:
            P&L in currency
        """
        price_change = exit_price - entry_price
        return price_change * quantity * self._multiplier
    
    def get_contract(self) -> Optional[FutureContract]:
        """
        Get contract details.
        
        Returns:
            FutureContract if expiry is set
        """
        if not self._expiry or not self._parsed_month:
            return None
        
        return FutureContract(
            root_symbol=self.root_symbol,
            month=self._parsed_month,
            year=self._parsed_year,
            expiry=self._expiry,
            first_notice=self._first_notice,
        )


class FutureChain:
    """
    Chain of futures contracts for roll management.
    
    Manages multiple contracts across different expiries.
    
    Example:
        >>> chain = FutureChain("ES")
        >>> front = chain.front_contract()
        >>> next_contract = chain.get_next_contract()
    """
    
    def __init__(
        self,
        root_symbol: str,
        contracts: Optional[List[FutureContract]] = None,
        roll_days: int = 5,
        roll_method: RollMethod = RollMethod.LAST_TRADING,
    ):
        """
        Initialize futures chain.
        
        Args:
            root_symbol: Base symbol
            contracts: List of contracts (or auto-generate)
            roll_days: Days before event to roll
            roll_method: Roll method
        """
        self.root_symbol = root_symbol.upper()
        self.roll_days = roll_days
        self.roll_method = roll_method
        
        # Get spec
        self._spec = FUTURES_SPECS.get(self.root_symbol)
        
        # Store contracts sorted by expiry
        self._contracts: List[FutureContract] = []
        if contracts:
            self._contracts = sorted(contracts, key=lambda c: c.expiry)
    
    def add_contract(self, contract: FutureContract) -> None:
        """Add a contract to the chain."""
        self._contracts.append(contract)
        self._contracts.sort(key=lambda c: c.expiry)
    
    def generate_contracts(
        self,
        start_date: date,
        num_contracts: int = 4,
    ) -> List[FutureContract]:
        """
        Generate quarterly contracts.
        
        Args:
            start_date: Reference date
            num_contracts: Number of contracts to generate
            
        Returns:
            List of generated contracts
        """
        contracts = []
        months = self._spec.months if self._spec else "HMUZ"
        
        # Find next contract month
        current = start_date
        generated = 0
        
        while generated < num_contracts:
            month = current.month
            year = current.year
            
            # Find next valid month
            for m in range(month, 13):
                code = get_contract_month_code(m)
                if code in months:
                    # Third Friday of month as expiry (simplified)
                    first_day = date(year, m, 1)
                    # Find third Friday
                    days_until_friday = (4 - first_day.weekday()) % 7
                    third_friday = first_day + timedelta(days=days_until_friday + 14)
                    
                    if third_friday > start_date:
                        contract = FutureContract(
                            root_symbol=self.root_symbol,
                            month=m,
                            year=year,
                            expiry=third_friday,
                        )
                        contracts.append(contract)
                        generated += 1
                        if generated >= num_contracts:
                            break
            
            # Move to next year
            current = date(year + 1, 1, 1)
        
        self._contracts = contracts
        return contracts
    
    def front_contract(self, as_of: Optional[date] = None) -> Optional[FutureContract]:
        """
        Get front month contract.
        
        Args:
            as_of: Reference date
            
        Returns:
            Front month contract
        """
        if as_of is None:
            as_of = date.today()
        
        for contract in self._contracts:
            if not contract.is_expired(as_of):
                return contract
        
        return None
    
    def get_next_contract(self, as_of: Optional[date] = None) -> Optional[FutureContract]:
        """
        Get next contract after front month.
        
        Args:
            as_of: Reference date
            
        Returns:
            Next contract
        """
        if as_of is None:
            as_of = date.today()
        
        found_front = False
        for contract in self._contracts:
            if not contract.is_expired(as_of):
                if found_front:
                    return contract
                found_front = True
        
        return None
    
    def should_roll(self, as_of: Optional[date] = None) -> bool:
        """Check if front contract should be rolled."""
        front = self.front_contract(as_of)
        if not front:
            return False
        return front.should_roll(as_of, self.roll_days, self.roll_method)
    
    def get_active_contracts(
        self,
        as_of: Optional[date] = None,
        max_contracts: int = 4,
    ) -> List[FutureContract]:
        """
        Get list of active (non-expired) contracts.
        
        Args:
            as_of: Reference date
            max_contracts: Maximum contracts to return
            
        Returns:
            List of active contracts
        """
        if as_of is None:
            as_of = date.today()
        
        active = [c for c in self._contracts if not c.is_expired(as_of)]
        return active[:max_contracts]
    
    def __len__(self) -> int:
        """Number of contracts in chain."""
        return len(self._contracts)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FutureChain({self.root_symbol}, contracts={len(self._contracts)})"

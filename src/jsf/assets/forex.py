"""Forex (Foreign Exchange) asset support with pip calculations."""

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Union, Tuple

from jsf.assets.base import Asset, AssetType, TradingSession
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class LotSize(Enum):
    """Standard forex lot sizes."""
    
    STANDARD = 100000     # 1 standard lot = 100,000 units
    MINI = 10000          # 1 mini lot = 10,000 units
    MICRO = 1000          # 1 micro lot = 1,000 units
    NANO = 100            # 1 nano lot = 100 units
    
    @property
    def units(self) -> int:
        """Get number of units per lot."""
        return self.value


class CurrencyType(Enum):
    """Currency classification."""
    
    MAJOR = "major"           # USD, EUR, JPY, GBP, CHF, AUD, CAD, NZD
    MINOR = "minor"           # Crosses (non-USD pairs of majors)
    EXOTIC = "exotic"         # Major + emerging market currency
    EMERGING = "emerging"     # Emerging market currencies


@dataclass
class CurrencySpec:
    """
    Currency specification.
    
    Attributes:
        code: ISO 4217 currency code
        name: Full currency name
        symbol: Currency symbol
        decimals: Standard decimal places
        country: Primary country
    """
    code: str
    name: str
    symbol: str = ""
    decimals: int = 2
    country: str = ""


# Major currency specifications
CURRENCY_SPECS: Dict[str, CurrencySpec] = {
    "USD": CurrencySpec("USD", "US Dollar", "$", 2, "United States"),
    "EUR": CurrencySpec("EUR", "Euro", "€", 2, "Eurozone"),
    "JPY": CurrencySpec("JPY", "Japanese Yen", "¥", 0, "Japan"),
    "GBP": CurrencySpec("GBP", "British Pound", "£", 2, "United Kingdom"),
    "CHF": CurrencySpec("CHF", "Swiss Franc", "Fr", 2, "Switzerland"),
    "AUD": CurrencySpec("AUD", "Australian Dollar", "A$", 2, "Australia"),
    "CAD": CurrencySpec("CAD", "Canadian Dollar", "C$", 2, "Canada"),
    "NZD": CurrencySpec("NZD", "New Zealand Dollar", "NZ$", 2, "New Zealand"),
    "CNY": CurrencySpec("CNY", "Chinese Yuan", "¥", 2, "China"),
    "HKD": CurrencySpec("HKD", "Hong Kong Dollar", "HK$", 2, "Hong Kong"),
    "SGD": CurrencySpec("SGD", "Singapore Dollar", "S$", 2, "Singapore"),
    "SEK": CurrencySpec("SEK", "Swedish Krona", "kr", 2, "Sweden"),
    "NOK": CurrencySpec("NOK", "Norwegian Krone", "kr", 2, "Norway"),
    "DKK": CurrencySpec("DKK", "Danish Krone", "kr", 2, "Denmark"),
    "MXN": CurrencySpec("MXN", "Mexican Peso", "$", 2, "Mexico"),
    "ZAR": CurrencySpec("ZAR", "South African Rand", "R", 2, "South Africa"),
    "TRY": CurrencySpec("TRY", "Turkish Lira", "₺", 2, "Turkey"),
    "INR": CurrencySpec("INR", "Indian Rupee", "₹", 2, "India"),
    "BRL": CurrencySpec("BRL", "Brazilian Real", "R$", 2, "Brazil"),
    "KRW": CurrencySpec("KRW", "South Korean Won", "₩", 0, "South Korea"),
}


@dataclass
class ForexPairSpec:
    """
    Forex pair specification.
    
    Attributes:
        base: Base currency code
        quote: Quote currency code
        pip_position: Decimal position of pip (4 for most, 2 for JPY pairs)
        pip_value: Value of 1 pip per standard lot
        typical_spread: Typical spread in pips
        currency_type: Pair classification
    """
    base: str
    quote: str
    pip_position: int = 4
    pip_value: float = 10.0  # Per standard lot
    typical_spread: float = 1.0  # In pips
    currency_type: CurrencyType = CurrencyType.MAJOR
    
    @property
    def symbol(self) -> str:
        """Get pair symbol."""
        return f"{self.base}/{self.quote}"
    
    @property
    def pip_size(self) -> float:
        """Get pip size."""
        return 10 ** (-self.pip_position)


# Major forex pair specifications
FOREX_PAIR_SPECS: Dict[str, ForexPairSpec] = {
    "EURUSD": ForexPairSpec("EUR", "USD", 4, 10.0, 0.8, CurrencyType.MAJOR),
    "GBPUSD": ForexPairSpec("GBP", "USD", 4, 10.0, 1.2, CurrencyType.MAJOR),
    "USDJPY": ForexPairSpec("USD", "JPY", 2, 9.3, 0.9, CurrencyType.MAJOR),  # Pip value varies
    "USDCHF": ForexPairSpec("USD", "CHF", 4, 10.1, 1.5, CurrencyType.MAJOR),
    "AUDUSD": ForexPairSpec("AUD", "USD", 4, 10.0, 1.0, CurrencyType.MAJOR),
    "USDCAD": ForexPairSpec("USD", "CAD", 4, 7.6, 1.4, CurrencyType.MAJOR),
    "NZDUSD": ForexPairSpec("NZD", "USD", 4, 10.0, 1.5, CurrencyType.MAJOR),
    # Crosses (Minors)
    "EURGBP": ForexPairSpec("EUR", "GBP", 4, 12.7, 1.5, CurrencyType.MINOR),
    "EURJPY": ForexPairSpec("EUR", "JPY", 2, 9.3, 1.2, CurrencyType.MINOR),
    "GBPJPY": ForexPairSpec("GBP", "JPY", 2, 9.3, 2.5, CurrencyType.MINOR),
    "EURCHF": ForexPairSpec("EUR", "CHF", 4, 10.1, 1.8, CurrencyType.MINOR),
    "EURAUD": ForexPairSpec("EUR", "AUD", 4, 6.5, 2.0, CurrencyType.MINOR),
    "GBPCHF": ForexPairSpec("GBP", "CHF", 4, 10.1, 2.5, CurrencyType.MINOR),
    "AUDJPY": ForexPairSpec("AUD", "JPY", 2, 9.3, 1.5, CurrencyType.MINOR),
    "AUDNZD": ForexPairSpec("AUD", "NZD", 4, 6.2, 2.5, CurrencyType.MINOR),
    "NZDJPY": ForexPairSpec("NZD", "JPY", 2, 9.3, 2.0, CurrencyType.MINOR),
    "CADJPY": ForexPairSpec("CAD", "JPY", 2, 9.3, 1.8, CurrencyType.MINOR),
    # Exotics
    "USDZAR": ForexPairSpec("USD", "ZAR", 4, 0.55, 100, CurrencyType.EXOTIC),
    "USDMXN": ForexPairSpec("USD", "MXN", 4, 0.55, 50, CurrencyType.EXOTIC),
    "USDTRY": ForexPairSpec("USD", "TRY", 4, 0.35, 80, CurrencyType.EXOTIC),
    "USDBRL": ForexPairSpec("USD", "BRL", 4, 0.20, 60, CurrencyType.EXOTIC),
    "EURTRY": ForexPairSpec("EUR", "TRY", 4, 0.35, 100, CurrencyType.EXOTIC),
}


def pip_value(
    pair: str,
    lot_size: Union[LotSize, int] = LotSize.STANDARD,
    exchange_rate: Optional[float] = None,
    account_currency: str = "USD",
) -> float:
    """
    Calculate pip value for a forex pair.
    
    Args:
        pair: Currency pair (e.g., "EURUSD")
        lot_size: Lot size (standard, mini, micro, or units)
        exchange_rate: Current exchange rate (for non-USD account currencies)
        account_currency: Account base currency
        
    Returns:
        Pip value in account currency
    """
    # Normalize pair
    pair = pair.upper().replace("/", "")
    
    # Get spec
    spec = FOREX_PAIR_SPECS.get(pair)
    
    # Get units
    if isinstance(lot_size, LotSize):
        units = lot_size.value
    else:
        units = int(lot_size)
    
    # Standard pip value calculation
    if spec:
        pip_size = spec.pip_size
        quote_currency = spec.quote
    else:
        # Default to 4 decimal places
        pip_size = 0.0001
        quote_currency = pair[-3:] if len(pair) >= 6 else "USD"
    
    # Calculate pip value
    if quote_currency == account_currency:
        # Direct calculation
        return pip_size * units
    elif account_currency == "USD":
        # Need to convert from quote currency to USD
        if quote_currency == "JPY":
            # For JPY pairs, pip value = (pip size * units) / exchange_rate
            if exchange_rate:
                return (pip_size * units) / exchange_rate
            else:
                return (pip_size * units) / 110  # Approximate
        else:
            return pip_size * units
    else:
        # Convert through exchange rate
        if exchange_rate:
            return pip_size * units * exchange_rate
        else:
            return pip_size * units


def pips_to_price(pips: float, pair: str) -> float:
    """
    Convert pips to price change.
    
    Args:
        pips: Number of pips
        pair: Currency pair
        
    Returns:
        Price change
    """
    pair = pair.upper().replace("/", "")
    spec = FOREX_PAIR_SPECS.get(pair)
    
    if spec:
        return pips * spec.pip_size
    else:
        # Default to 4 decimal places (0.0001 per pip)
        return pips * 0.0001


def price_to_pips(price_change: float, pair: str) -> float:
    """
    Convert price change to pips.
    
    Args:
        price_change: Price change
        pair: Currency pair
        
    Returns:
        Number of pips
    """
    pair = pair.upper().replace("/", "")
    spec = FOREX_PAIR_SPECS.get(pair)
    
    if spec:
        return price_change / spec.pip_size
    else:
        return price_change / 0.0001


class ForexPair(Asset):
    """
    Forex currency pair asset.
    
    Features:
    - 24/5 trading (Sunday evening to Friday afternoon)
    - Pip-based calculations
    - Multiple lot size support
    - Leverage and margin handling
    
    Example:
        >>> eurusd = ForexPair("EUR/USD")
        >>> pip_val = eurusd.pip_value(LotSize.MINI)
        >>> print(f"Pip value: ${pip_val:.2f}")
        >>> pnl = eurusd.calculate_pnl_pips(entry=1.1000, exit=1.1050, lots=1)
    """
    
    def __init__(
        self,
        symbol: str,
        lot_size: LotSize = LotSize.STANDARD,
        leverage: float = 50.0,
        spread: Optional[float] = None,
        commission: float = 0.0,
        swap_long: float = 0.0,
        swap_short: float = 0.0,
        **kwargs: Any,
    ):
        """
        Initialize forex pair.
        
        Args:
            symbol: Currency pair (e.g., "EUR/USD", "EURUSD")
            lot_size: Default lot size
            leverage: Account leverage (e.g., 50 = 50:1)
            spread: Typical spread in pips
            commission: Commission per lot per side
            swap_long: Overnight swap for long positions
            swap_short: Overnight swap for short positions
            **kwargs: Additional parameters
        """
        # Parse symbol
        symbol = symbol.upper().replace("/", "")
        if len(symbol) != 6:
            raise ValueError(f"Invalid forex pair: {symbol}")
        
        self.base_currency = symbol[:3]
        self.quote_currency = symbol[-3:]
        
        # Get spec if available
        self._spec = FOREX_PAIR_SPECS.get(symbol)
        
        # Set name
        name = f"{self.base_currency}/{self.quote_currency}"
        
        super().__init__(name, self.quote_currency, "FOREX", name, **kwargs)
        
        self.default_lot_size = lot_size
        self.leverage = leverage
        self.commission = commission
        self.swap_long = swap_long
        self.swap_short = swap_short
        
        # Get pip info from spec or defaults
        if self._spec:
            self._pip_position = self._spec.pip_position
            self._typical_spread = spread or self._spec.typical_spread
            self._currency_type = self._spec.currency_type
        else:
            # Default to 4 decimal places for most pairs
            if "JPY" in symbol:
                self._pip_position = 2
            else:
                self._pip_position = 4
            self._typical_spread = spread or 2.0
            self._currency_type = CurrencyType.MAJOR
        
        logger.info(
            f"Initialized ForexPair: {self.symbol}, pip_pos={self._pip_position}, "
            f"leverage={self.leverage}:1"
        )
    
    @property
    def asset_type(self) -> AssetType:
        """Return asset type."""
        return AssetType.FOREX
    
    @property
    def pip_size(self) -> float:
        """Get pip size (minimum meaningful price movement)."""
        return 10 ** (-self._pip_position)
    
    @property
    def tick_size(self) -> float:
        """Minimum price movement (same as pip for most brokers)."""
        return self.pip_size
    
    @property
    def typical_spread(self) -> float:
        """Get typical spread in pips."""
        return self._typical_spread
    
    @property
    def spread_cost(self) -> float:
        """Get spread cost in price terms."""
        return self._typical_spread * self.pip_size
    
    @property
    def currency_type(self) -> CurrencyType:
        """Get currency pair type."""
        return self._currency_type
    
    @property
    def margin_requirement(self) -> float:
        """Margin requirement as fraction (1/leverage)."""
        return 1.0 / self.leverage if self.leverage > 0 else 1.0
    
    @property
    def trading_session(self) -> TradingSession:
        """Trading session type."""
        return TradingSession.EXTENDED  # 24/5
    
    def get_multiplier(self) -> float:
        """Get contract multiplier (lot size units)."""
        return float(self.default_lot_size.value)
    
    def pip_value(
        self,
        lot_size: Optional[Union[LotSize, int]] = None,
        exchange_rate: Optional[float] = None,
        account_currency: str = "USD",
    ) -> float:
        """
        Calculate pip value.
        
        Args:
            lot_size: Lot size (uses default if not specified)
            exchange_rate: Current exchange rate
            account_currency: Account currency
            
        Returns:
            Pip value in account currency
        """
        if lot_size is None:
            lot_size = self.default_lot_size
        
        pair_str = f"{self.base_currency}{self.quote_currency}"
        return pip_value(pair_str, lot_size, exchange_rate, account_currency)
    
    def pips_to_price(self, pips: float) -> float:
        """Convert pips to price change."""
        return pips * self.pip_size
    
    def price_to_pips(self, price_change: float) -> float:
        """Convert price change to pips."""
        return price_change / self.pip_size
    
    def calculate_pnl_pips(
        self,
        entry_price: float,
        exit_price: float,
        lots: float = 1.0,
        direction: int = 1,
    ) -> float:
        """
        Calculate P&L in pips.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            lots: Number of lots
            direction: 1 for long, -1 for short
            
        Returns:
            P&L in pips
        """
        price_diff = exit_price - entry_price
        pips = self.price_to_pips(price_diff)
        return pips * direction * lots
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        account_currency: str = "USD",
    ) -> float:
        """
        Calculate P&L in account currency.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size in lots
            account_currency: Account currency
            
        Returns:
            P&L in account currency
        """
        pips = self.calculate_pnl_pips(entry_price, exit_price, abs(quantity))
        pip_val = self.pip_value(self.default_lot_size, exit_price, account_currency)
        
        return pips * pip_val
    
    def calculate_margin(
        self,
        lots: float,
        price: float,
        account_currency: str = "USD",
    ) -> float:
        """
        Calculate margin requirement.
        
        Args:
            lots: Number of lots
            price: Current price
            account_currency: Account currency
            
        Returns:
            Margin required in account currency
        """
        units = lots * self.default_lot_size.value
        notional = units * price
        
        # Apply leverage
        margin = notional / self.leverage
        
        # Convert to account currency if needed
        if self.quote_currency != account_currency and account_currency == "USD":
            # This is a simplification - real conversion would need exchange rate
            if self.quote_currency in ["EUR", "GBP", "AUD", "NZD"]:
                margin *= price  # Approximate conversion
        
        return margin
    
    def calculate_position_size(
        self,
        risk_amount: float,
        stop_loss_pips: float,
        account_currency: str = "USD",
    ) -> float:
        """
        Calculate position size based on risk.
        
        Args:
            risk_amount: Amount willing to risk
            stop_loss_pips: Stop loss distance in pips
            account_currency: Account currency
            
        Returns:
            Position size in lots
        """
        if stop_loss_pips <= 0:
            return 0.0
        
        # Calculate pip value for one standard lot
        pip_val = self.pip_value(LotSize.STANDARD, account_currency=account_currency)
        
        # Position size = Risk / (Stop Loss in Pips * Pip Value)
        position_size = risk_amount / (stop_loss_pips * pip_val)
        
        return position_size
    
    def is_tradeable(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if pair is tradeable (forex market open).
        
        Args:
            timestamp: Time to check (defaults to now)
            
        Returns:
            True if market is open
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Forex market: Sunday 5pm ET to Friday 5pm ET (22:00 UTC Sun to 22:00 UTC Fri)
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        hour = timestamp.hour
        
        # Closed Saturday all day
        if weekday == 5:
            return False
        
        # Closed Sunday until 22:00 UTC
        if weekday == 6 and hour < 22:
            return False
        
        # Closed Friday after 22:00 UTC
        if weekday == 4 and hour >= 22:
            return False
        
        return True


def create_forex_pair(
    base: str,
    quote: str,
    **kwargs: Any,
) -> ForexPair:
    """
    Factory function to create a ForexPair.
    
    Args:
        base: Base currency code
        quote: Quote currency code
        **kwargs: Additional ForexPair parameters
        
    Returns:
        Configured ForexPair
    """
    symbol = f"{base.upper()}{quote.upper()}"
    return ForexPair(symbol, **kwargs)


def get_major_pairs() -> List[ForexPair]:
    """
    Get list of major forex pairs.
    
    Returns:
        List of major ForexPairs
    """
    major_symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    return [ForexPair(s) for s in major_symbols]


def get_minor_pairs() -> List[ForexPair]:
    """
    Get list of minor forex pairs (crosses).
    
    Returns:
        List of minor ForexPairs
    """
    minor_symbols = [
        "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "EURAUD",
        "GBPCHF", "AUDJPY", "AUDNZD", "NZDJPY", "CADJPY"
    ]
    return [ForexPair(s) for s in minor_symbols]


class ForexSession(Enum):
    """Major forex trading sessions."""
    
    SYDNEY = "sydney"      # 22:00 - 07:00 UTC
    TOKYO = "tokyo"        # 00:00 - 09:00 UTC
    LONDON = "london"      # 08:00 - 17:00 UTC
    NEW_YORK = "new_york"  # 13:00 - 22:00 UTC


def get_active_sessions(timestamp: Optional[datetime] = None) -> List[ForexSession]:
    """
    Get currently active forex sessions.
    
    Args:
        timestamp: Time to check (defaults to now UTC)
        
    Returns:
        List of active sessions
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    hour = timestamp.hour
    sessions = []
    
    # Sydney: 22:00 - 07:00 UTC
    if hour >= 22 or hour < 7:
        sessions.append(ForexSession.SYDNEY)
    
    # Tokyo: 00:00 - 09:00 UTC
    if 0 <= hour < 9:
        sessions.append(ForexSession.TOKYO)
    
    # London: 08:00 - 17:00 UTC
    if 8 <= hour < 17:
        sessions.append(ForexSession.LONDON)
    
    # New York: 13:00 - 22:00 UTC
    if 13 <= hour < 22:
        sessions.append(ForexSession.NEW_YORK)
    
    return sessions

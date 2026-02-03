"""Cryptocurrency asset support with 24/7 trading and exchange features."""

from dataclasses import dataclass, field
from datetime import datetime, date, time
from enum import Enum
from typing import Optional, Dict, List, Any, Union

from jsf.assets.base import Asset, AssetType, TradingSession
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class CryptoExchange(Enum):
    """Major cryptocurrency exchanges."""
    
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    FTX = "ftx"
    BITFINEX = "bitfinex"
    BITSTAMP = "bitstamp"
    HUOBI = "huobi"
    KUCOIN = "kucoin"
    OKEX = "okex"
    BYBIT = "bybit"
    GEMINI = "gemini"
    CRYPTO_COM = "crypto.com"


@dataclass
class CryptoSpec:
    """
    Cryptocurrency specification.
    
    Attributes:
        symbol: Crypto symbol (BTC, ETH, etc.)
        name: Full name
        decimals: Price decimal precision
        min_quantity: Minimum trade quantity
        maker_fee: Maker fee (decimal)
        taker_fee: Taker fee (decimal)
        is_stablecoin: Whether it's a stablecoin
    """
    symbol: str
    name: str
    decimals: int = 8
    min_quantity: float = 0.00001
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%
    is_stablecoin: bool = False


# Common cryptocurrency specifications
CRYPTO_SPECS: Dict[str, CryptoSpec] = {
    "BTC": CryptoSpec(
        symbol="BTC",
        name="Bitcoin",
        decimals=8,
        min_quantity=0.00001,
    ),
    "ETH": CryptoSpec(
        symbol="ETH",
        name="Ethereum",
        decimals=8,
        min_quantity=0.0001,
    ),
    "SOL": CryptoSpec(
        symbol="SOL",
        name="Solana",
        decimals=4,
        min_quantity=0.01,
    ),
    "BNB": CryptoSpec(
        symbol="BNB",
        name="Binance Coin",
        decimals=4,
        min_quantity=0.01,
    ),
    "XRP": CryptoSpec(
        symbol="XRP",
        name="Ripple",
        decimals=4,
        min_quantity=1,
    ),
    "ADA": CryptoSpec(
        symbol="ADA",
        name="Cardano",
        decimals=4,
        min_quantity=1,
    ),
    "DOGE": CryptoSpec(
        symbol="DOGE",
        name="Dogecoin",
        decimals=4,
        min_quantity=1,
    ),
    "MATIC": CryptoSpec(
        symbol="MATIC",
        name="Polygon",
        decimals=4,
        min_quantity=1,
    ),
    "DOT": CryptoSpec(
        symbol="DOT",
        name="Polkadot",
        decimals=4,
        min_quantity=0.1,
    ),
    "AVAX": CryptoSpec(
        symbol="AVAX",
        name="Avalanche",
        decimals=4,
        min_quantity=0.01,
    ),
    "LINK": CryptoSpec(
        symbol="LINK",
        name="Chainlink",
        decimals=4,
        min_quantity=0.1,
    ),
    "USDT": CryptoSpec(
        symbol="USDT",
        name="Tether USD",
        decimals=4,
        min_quantity=1,
        is_stablecoin=True,
    ),
    "USDC": CryptoSpec(
        symbol="USDC",
        name="USD Coin",
        decimals=4,
        min_quantity=1,
        is_stablecoin=True,
    ),
    "DAI": CryptoSpec(
        symbol="DAI",
        name="Dai Stablecoin",
        decimals=4,
        min_quantity=1,
        is_stablecoin=True,
    ),
}


@dataclass
class TradingPair:
    """
    Crypto trading pair information.
    
    Attributes:
        base: Base currency (what you're buying/selling)
        quote: Quote currency (what you pay with)
        min_order_size: Minimum order size in base currency
        max_order_size: Maximum order size in base currency
        tick_size: Minimum price movement
        step_size: Minimum quantity increment
        maker_fee: Maker fee (decimal)
        taker_fee: Taker fee (decimal)
    """
    base: str
    quote: str
    min_order_size: float = 0.0
    max_order_size: float = float("inf")
    tick_size: float = 0.01
    step_size: float = 0.00001
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    
    @property
    def symbol(self) -> str:
        """Get pair symbol."""
        return f"{self.base}/{self.quote}"
    
    @property
    def exchange_symbol(self) -> str:
        """Get exchange-format symbol."""
        return f"{self.base}{self.quote}"


class CryptoAsset(Asset):
    """
    Cryptocurrency asset with 24/7 trading support.
    
    Features:
    - Continuous 24/7 trading
    - Fractional quantity support
    - Exchange-specific fee handling
    - Stablecoin identification
    
    Example:
        >>> btc = CryptoAsset("BTC", quote_currency="USDT")
        >>> print(f"Symbol: {btc.symbol}")
        >>> print(f"Pair: {btc.trading_pair}")
        >>> quantity = btc.round_quantity(0.123456789)
    """
    
    def __init__(
        self,
        symbol: str,
        quote_currency: str = "USDT",
        exchange: Union[str, CryptoExchange] = CryptoExchange.BINANCE,
        min_quantity: Optional[float] = None,
        tick_size: Optional[float] = None,
        maker_fee: float = 0.001,
        taker_fee: float = 0.001,
        **kwargs: Any,
    ):
        """
        Initialize cryptocurrency asset.
        
        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            quote_currency: Quote currency (USDT, USD, BTC, etc.)
            exchange: Trading exchange
            min_quantity: Minimum trade quantity
            tick_size: Minimum price movement
            maker_fee: Maker fee (decimal)
            taker_fee: Taker fee (decimal)
            **kwargs: Additional parameters
        """
        # Get spec if available
        base_symbol = symbol.upper()
        self._spec = CRYPTO_SPECS.get(base_symbol)
        
        # Set name from spec
        name = self._spec.name if self._spec else base_symbol
        
        # Handle exchange enum
        if isinstance(exchange, CryptoExchange):
            exchange_str = exchange.value
        else:
            exchange_str = str(exchange)
        
        # Use quote currency as the currency
        super().__init__(base_symbol, quote_currency.upper(), exchange_str, name, **kwargs)
        
        self.quote_currency = quote_currency.upper()
        self.base_currency = base_symbol
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        # Set min quantity from spec or override
        if min_quantity is not None:
            self._min_quantity = min_quantity
        elif self._spec:
            self._min_quantity = self._spec.min_quantity
        else:
            self._min_quantity = 0.00001
        
        # Set tick size
        if tick_size is not None:
            self._tick_size = tick_size
        elif self._spec:
            # Derive tick size from decimals
            self._tick_size = 10 ** (-self._spec.decimals)
        else:
            self._tick_size = 0.01
        
        # Create trading pair
        self._pair = TradingPair(
            base=self.base_currency,
            quote=self.quote_currency,
            min_order_size=self._min_quantity,
            tick_size=self._tick_size,
            step_size=self._min_quantity,
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
        )
        
        logger.info(
            f"Initialized CryptoAsset: {self.symbol}/{self.quote_currency} "
            f"on {self.exchange}"
        )
    
    @property
    def asset_type(self) -> AssetType:
        """Return asset type."""
        return AssetType.CRYPTO
    
    @property
    def trading_pair(self) -> str:
        """Get trading pair string."""
        return f"{self.base_currency}/{self.quote_currency}"
    
    @property
    def is_stablecoin(self) -> bool:
        """Check if this is a stablecoin."""
        return self._spec.is_stablecoin if self._spec else False
    
    @property
    def tick_size(self) -> float:
        """Minimum price movement."""
        return self._tick_size
    
    @property
    def min_quantity(self) -> float:
        """Minimum trade quantity."""
        return self._min_quantity
    
    @property
    def lot_size(self) -> float:
        """Lot size (same as min_quantity for crypto)."""
        return self._min_quantity
    
    @property
    def margin_requirement(self) -> float:
        """Margin requirement (1.0 = full payment for spot)."""
        return 1.0  # Spot crypto requires full payment
    
    @property
    def trading_session(self) -> TradingSession:
        """Trading session type (always continuous for crypto)."""
        return TradingSession.CONTINUOUS
    
    def get_multiplier(self) -> float:
        """Contract multiplier (1.0 for spot crypto)."""
        return 1.0
    
    def is_tradeable(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if asset is tradeable (always True for crypto).
        
        Args:
            timestamp: Time to check (ignored for crypto)
            
        Returns:
            Always True (24/7 trading)
        """
        return True  # Crypto trades 24/7
    
    def round_quantity(self, quantity: float) -> float:
        """
        Round quantity to valid increment.
        
        Args:
            quantity: Raw quantity
            
        Returns:
            Rounded quantity (at least min_quantity)
        """
        if quantity <= 0:
            return 0.0
        
        # Round down to step size
        step = self._min_quantity
        rounded = (quantity // step) * step
        
        # Ensure at least min_quantity
        if rounded < self._min_quantity:
            return 0.0
        
        return rounded
    
    def calculate_fee(
        self,
        quantity: float,
        price: float,
        is_maker: bool = False,
    ) -> float:
        """
        Calculate trading fee.
        
        Args:
            quantity: Trade quantity
            price: Trade price
            is_maker: Whether this is a maker order
            
        Returns:
            Fee in quote currency
        """
        notional = abs(quantity) * price
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return notional * fee_rate
    
    def calculate_net_proceeds(
        self,
        quantity: float,
        price: float,
        is_maker: bool = False,
    ) -> float:
        """
        Calculate net proceeds after fees.
        
        Args:
            quantity: Trade quantity (positive for sell)
            price: Trade price
            is_maker: Whether this is a maker order
            
        Returns:
            Net proceeds in quote currency
        """
        gross = quantity * price
        fee = self.calculate_fee(quantity, price, is_maker)
        
        if quantity > 0:  # Selling
            return gross - fee
        else:  # Buying (quantity negative = outflow)
            return gross - fee  # More negative (cost + fee)
    
    def get_pair(self) -> TradingPair:
        """Get trading pair details."""
        return self._pair


class CryptoPortfolioAsset:
    """
    Represents a crypto holding in a portfolio.
    
    Tracks balance, average cost, and P&L.
    
    Example:
        >>> holding = CryptoPortfolioAsset(
        ...     asset=CryptoAsset("BTC"),
        ...     quantity=0.5,
        ...     avg_cost=45000,
        ... )
        >>> pnl = holding.unrealized_pnl(current_price=50000)
    """
    
    def __init__(
        self,
        asset: CryptoAsset,
        quantity: float = 0.0,
        avg_cost: float = 0.0,
    ):
        """
        Initialize portfolio holding.
        
        Args:
            asset: Crypto asset
            quantity: Current quantity held
            avg_cost: Average cost basis
        """
        self.asset = asset
        self.quantity = quantity
        self.avg_cost = avg_cost
        self._realized_pnl = 0.0
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return self.quantity * self.avg_cost
    
    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L
        """
        return self.quantity * (current_price - self.avg_cost)
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """
        Calculate unrealized P&L percentage.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L percentage
        """
        if self.avg_cost <= 0:
            return 0.0
        return (current_price - self.avg_cost) / self.avg_cost
    
    @property
    def realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self._realized_pnl
    
    def add_position(
        self,
        quantity: float,
        price: float,
        fee: float = 0.0,
    ) -> None:
        """
        Add to position (buy).
        
        Args:
            quantity: Quantity to add
            price: Purchase price
            fee: Trading fee
        """
        if quantity <= 0:
            return
        
        # Update average cost
        total_cost = self.quantity * self.avg_cost + quantity * price + fee
        self.quantity += quantity
        self.avg_cost = total_cost / self.quantity if self.quantity > 0 else 0
    
    def reduce_position(
        self,
        quantity: float,
        price: float,
        fee: float = 0.0,
    ) -> float:
        """
        Reduce position (sell).
        
        Args:
            quantity: Quantity to sell
            price: Sale price
            fee: Trading fee
            
        Returns:
            Realized P&L from this sale
        """
        if quantity <= 0 or self.quantity <= 0:
            return 0.0
        
        # Can't sell more than we have
        quantity = min(quantity, self.quantity)
        
        # Calculate realized P&L
        proceeds = quantity * price - fee
        cost = quantity * self.avg_cost
        realized = proceeds - cost
        
        self._realized_pnl += realized
        self.quantity -= quantity
        
        # Reset avg_cost if position closed
        if self.quantity <= 0:
            self.quantity = 0.0
            self.avg_cost = 0.0
        
        return realized
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CryptoPortfolioAsset({self.asset.symbol}, "
            f"qty={self.quantity:.8f}, avg_cost={self.avg_cost:.2f})"
        )


def get_crypto_asset(
    symbol: str,
    quote_currency: str = "USDT",
    exchange: Union[str, CryptoExchange] = CryptoExchange.BINANCE,
) -> CryptoAsset:
    """
    Factory function to create a CryptoAsset.
    
    Args:
        symbol: Crypto symbol (BTC, ETH, etc.)
        quote_currency: Quote currency
        exchange: Trading exchange
        
    Returns:
        Configured CryptoAsset
    """
    return CryptoAsset(
        symbol=symbol,
        quote_currency=quote_currency,
        exchange=exchange,
    )


def parse_crypto_pair(pair: str) -> tuple:
    """
    Parse crypto trading pair string.
    
    Examples:
        "BTC/USDT" -> ("BTC", "USDT")
        "ETHBTC" -> ("ETH", "BTC")
        
    Args:
        pair: Trading pair string
        
    Returns:
        Tuple of (base, quote)
    """
    # Try slash format first
    if "/" in pair:
        parts = pair.upper().split("/")
        return parts[0], parts[1]
    
    # Try common quote currencies
    pair = pair.upper()
    quote_currencies = ["USDT", "USDC", "USD", "BTC", "ETH", "BNB", "BUSD"]
    
    for quote in quote_currencies:
        if pair.endswith(quote) and len(pair) > len(quote):
            base = pair[:-len(quote)]
            return base, quote
    
    # Default: assume last 3-4 chars are quote
    if len(pair) > 4:
        return pair[:-4], pair[-4:]
    elif len(pair) > 3:
        return pair[:-3], pair[-3:]
    
    raise ValueError(f"Could not parse crypto pair: {pair}")

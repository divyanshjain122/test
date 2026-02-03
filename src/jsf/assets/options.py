"""Options support with Black-Scholes pricing and Greeks."""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, Dict, List, Any, Union, Tuple
import math

import numpy as np
from scipy import stats

from jsf.assets.base import Asset, AssetType, TradingSession
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class OptionType(Enum):
    """Option type (Call or Put)."""
    
    CALL = "call"
    PUT = "put"


class OptionStyle(Enum):
    """Option exercise style."""
    
    AMERICAN = "american"  # Can exercise any time before expiry
    EUROPEAN = "european"  # Can only exercise at expiry


@dataclass
class OptionGreeks:
    """
    Option Greeks - sensitivities of option price to various factors.
    
    Attributes:
        delta: Sensitivity to underlying price change (dOption/dSpot)
        gamma: Rate of change of delta (d²Option/dSpot²)
        theta: Time decay per day (dOption/dTime)
        vega: Sensitivity to volatility (dOption/dVol)
        rho: Sensitivity to interest rate (dOption/dRate)
    """
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptionGreeks(delta={self.delta:.4f}, gamma={self.gamma:.6f}, "
            f"theta={self.theta:.4f}, vega={self.vega:.4f}, rho={self.rho:.4f})"
        )


class BlackScholes:
    """
    Black-Scholes option pricing model.
    
    Provides analytical pricing for European options.
    
    Example:
        >>> bs = BlackScholes()
        >>> price = bs.price(
        ...     spot=100, strike=100, expiry=0.25,
        ...     vol=0.20, rate=0.05, option_type=OptionType.CALL
        ... )
        >>> print(f"Call price: ${price:.2f}")
    """
    
    @staticmethod
    def d1(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        dividend: float = 0.0,
    ) -> float:
        """
        Calculate d1 parameter.
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            dividend: Dividend yield (decimal)
            
        Returns:
            d1 value
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        
        return (
            (math.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * time_to_expiry)
            / (volatility * math.sqrt(time_to_expiry))
        )
    
    @staticmethod
    def d2(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        dividend: float = 0.0,
    ) -> float:
        """
        Calculate d2 parameter.
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            dividend: Dividend yield (decimal)
            
        Returns:
            d2 value
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(spot, strike, time_to_expiry, volatility, rate, dividend)
        return d1_val - volatility * math.sqrt(time_to_expiry)
    
    @staticmethod
    def price(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend: float = 0.0,
    ) -> float:
        """
        Calculate option price using Black-Scholes.
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            option_type: CALL or PUT
            dividend: Dividend yield (decimal)
            
        Returns:
            Option price
        """
        if time_to_expiry <= 0:
            # At expiry, return intrinsic value
            if option_type == OptionType.CALL:
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        d1_val = BlackScholes.d1(spot, strike, time_to_expiry, volatility, rate, dividend)
        d2_val = BlackScholes.d2(spot, strike, time_to_expiry, volatility, rate, dividend)
        
        discount = math.exp(-rate * time_to_expiry)
        dividend_discount = math.exp(-dividend * time_to_expiry)
        
        if option_type == OptionType.CALL:
            price = (
                spot * dividend_discount * stats.norm.cdf(d1_val)
                - strike * discount * stats.norm.cdf(d2_val)
            )
        else:  # PUT
            price = (
                strike * discount * stats.norm.cdf(-d2_val)
                - spot * dividend_discount * stats.norm.cdf(-d1_val)
            )
        
        return max(0, price)
    
    @staticmethod
    def delta(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend: float = 0.0,
    ) -> float:
        """
        Calculate option delta.
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            option_type: CALL or PUT
            dividend: Dividend yield (decimal)
            
        Returns:
            Delta value
        """
        if time_to_expiry <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0
        
        d1_val = BlackScholes.d1(spot, strike, time_to_expiry, volatility, rate, dividend)
        dividend_discount = math.exp(-dividend * time_to_expiry)
        
        if option_type == OptionType.CALL:
            return dividend_discount * stats.norm.cdf(d1_val)
        else:
            return -dividend_discount * stats.norm.cdf(-d1_val)
    
    @staticmethod
    def gamma(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        dividend: float = 0.0,
    ) -> float:
        """
        Calculate option gamma (same for calls and puts).
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            dividend: Dividend yield (decimal)
            
        Returns:
            Gamma value
        """
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(spot, strike, time_to_expiry, volatility, rate, dividend)
        dividend_discount = math.exp(-dividend * time_to_expiry)
        
        return (
            dividend_discount * stats.norm.pdf(d1_val)
            / (spot * volatility * math.sqrt(time_to_expiry))
        )
    
    @staticmethod
    def theta(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend: float = 0.0,
    ) -> float:
        """
        Calculate option theta (per day).
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            option_type: CALL or PUT
            dividend: Dividend yield (decimal)
            
        Returns:
            Theta value (per day)
        """
        if time_to_expiry <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(spot, strike, time_to_expiry, volatility, rate, dividend)
        d2_val = BlackScholes.d2(spot, strike, time_to_expiry, volatility, rate, dividend)
        
        discount = math.exp(-rate * time_to_expiry)
        dividend_discount = math.exp(-dividend * time_to_expiry)
        sqrt_t = math.sqrt(time_to_expiry)
        
        # Common term
        common = -(spot * dividend_discount * stats.norm.pdf(d1_val) * volatility) / (2 * sqrt_t)
        
        if option_type == OptionType.CALL:
            theta_annual = (
                common
                + dividend * spot * dividend_discount * stats.norm.cdf(d1_val)
                - rate * strike * discount * stats.norm.cdf(d2_val)
            )
        else:  # PUT
            theta_annual = (
                common
                - dividend * spot * dividend_discount * stats.norm.cdf(-d1_val)
                + rate * strike * discount * stats.norm.cdf(-d2_val)
            )
        
        # Convert to per day (assuming 365 days)
        return theta_annual / 365.0
    
    @staticmethod
    def vega(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        dividend: float = 0.0,
    ) -> float:
        """
        Calculate option vega (per 1% change in vol).
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            dividend: Dividend yield (decimal)
            
        Returns:
            Vega value (for 1% vol change)
        """
        if time_to_expiry <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(spot, strike, time_to_expiry, volatility, rate, dividend)
        dividend_discount = math.exp(-dividend * time_to_expiry)
        sqrt_t = math.sqrt(time_to_expiry)
        
        # Vega per 100% vol change
        vega_raw = spot * dividend_discount * stats.norm.pdf(d1_val) * sqrt_t
        
        # Return vega per 1% vol change
        return vega_raw / 100.0
    
    @staticmethod
    def rho(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend: float = 0.0,
    ) -> float:
        """
        Calculate option rho (per 1% change in rate).
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            option_type: CALL or PUT
            dividend: Dividend yield (decimal)
            
        Returns:
            Rho value (for 1% rate change)
        """
        if time_to_expiry <= 0:
            return 0.0
        
        d2_val = BlackScholes.d2(spot, strike, time_to_expiry, volatility, rate, dividend)
        discount = math.exp(-rate * time_to_expiry)
        
        if option_type == OptionType.CALL:
            rho_raw = strike * time_to_expiry * discount * stats.norm.cdf(d2_val)
        else:
            rho_raw = -strike * time_to_expiry * discount * stats.norm.cdf(-d2_val)
        
        # Return rho per 1% rate change
        return rho_raw / 100.0
    
    @staticmethod
    def greeks(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend: float = 0.0,
    ) -> OptionGreeks:
        """
        Calculate all Greeks.
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            option_type: CALL or PUT
            dividend: Dividend yield (decimal)
            
        Returns:
            OptionGreeks dataclass
        """
        return OptionGreeks(
            delta=BlackScholes.delta(spot, strike, time_to_expiry, volatility, rate, option_type, dividend),
            gamma=BlackScholes.gamma(spot, strike, time_to_expiry, volatility, rate, dividend),
            theta=BlackScholes.theta(spot, strike, time_to_expiry, volatility, rate, option_type, dividend),
            vega=BlackScholes.vega(spot, strike, time_to_expiry, volatility, rate, dividend),
            rho=BlackScholes.rho(spot, strike, time_to_expiry, volatility, rate, option_type, dividend),
        )
    
    @staticmethod
    def implied_volatility(
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend: float = 0.0,
        initial_guess: float = 0.25,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Market option price
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            rate: Risk-free rate (decimal)
            option_type: CALL or PUT
            dividend: Dividend yield (decimal)
            initial_guess: Initial volatility guess
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility (decimal)
        """
        if time_to_expiry <= 0:
            return 0.0
        
        vol = initial_guess
        
        for _ in range(max_iterations):
            price = BlackScholes.price(
                spot, strike, time_to_expiry, vol, rate, option_type, dividend
            )
            vega = BlackScholes.vega(
                spot, strike, time_to_expiry, vol, rate, dividend
            ) * 100  # Convert back to per 100% vol
            
            if abs(vega) < 1e-10:
                break
            
            diff = market_price - price
            if abs(diff) < tolerance:
                break
            
            vol = vol + diff / vega
            vol = max(0.001, min(vol, 5.0))  # Keep vol in reasonable range
        
        return vol


class Option(Asset):
    """
    Option contract asset.
    
    Supports pricing, Greeks calculation, and expiry handling.
    
    Example:
        >>> call = Option(
        ...     symbol="AAPL240315C00150000",
        ...     underlying="AAPL",
        ...     strike=150,
        ...     expiry="2024-03-15",
        ...     option_type=OptionType.CALL,
        ... )
        >>> price = call.theoretical_price(spot=155, vol=0.25, rate=0.05)
        >>> greeks = call.calculate_greeks(spot=155, vol=0.25, rate=0.05)
    """
    
    def __init__(
        self,
        symbol: str,
        underlying: str,
        strike: float,
        expiry: Union[str, date, datetime],
        option_type: OptionType = OptionType.CALL,
        style: OptionStyle = OptionStyle.AMERICAN,
        multiplier: float = 100.0,
        currency: str = "USD",
        exchange: str = "",
        **kwargs: Any,
    ):
        """
        Initialize option contract.
        
        Args:
            symbol: Option symbol (e.g., AAPL240315C00150000)
            underlying: Underlying asset symbol
            strike: Strike price
            expiry: Expiration date
            option_type: CALL or PUT
            style: AMERICAN or EUROPEAN
            multiplier: Contract multiplier (usually 100)
            currency: Quote currency
            exchange: Trading exchange
            **kwargs: Additional parameters
        """
        name = f"{underlying} {option_type.value.title()} ${strike}"
        super().__init__(symbol.upper(), currency, exchange, name, **kwargs)
        
        self.underlying = underlying.upper()
        self.strike = float(strike)
        self.option_type = option_type
        self.style = style
        self._multiplier = multiplier
        
        # Parse expiry
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
        elif isinstance(expiry, datetime):
            expiry = expiry.date()
        self._expiry = expiry
        
        logger.info(
            f"Initialized Option: {self.symbol}, {option_type.value} @ ${strike}, "
            f"expiry={self._expiry}"
        )
    
    @property
    def asset_type(self) -> AssetType:
        """Return asset type."""
        return AssetType.OPTION
    
    @property
    def expiry(self) -> date:
        """Get expiration date."""
        return self._expiry
    
    @property
    def is_call(self) -> bool:
        """Check if option is a call."""
        return self.option_type == OptionType.CALL
    
    @property
    def is_put(self) -> bool:
        """Check if option is a put."""
        return self.option_type == OptionType.PUT
    
    @property
    def tick_size(self) -> float:
        """Minimum price movement."""
        return 0.01  # Standard for equity options
    
    @property
    def margin_requirement(self) -> float:
        """Margin requirement (options require premium)."""
        return 1.0  # Must pay full premium for long options
    
    @property
    def trading_session(self) -> TradingSession:
        """Trading session type."""
        return TradingSession.REGULAR
    
    def get_multiplier(self) -> float:
        """Get contract multiplier."""
        return self._multiplier
    
    def days_to_expiry(self, as_of: Optional[date] = None) -> int:
        """
        Calculate days until expiry.
        
        Args:
            as_of: Reference date (defaults to today)
            
        Returns:
            Days until expiry
        """
        if as_of is None:
            as_of = date.today()
        return (self._expiry - as_of).days
    
    def time_to_expiry(self, as_of: Optional[date] = None) -> float:
        """
        Calculate time to expiry in years.
        
        Args:
            as_of: Reference date (defaults to today)
            
        Returns:
            Time to expiry in years
        """
        days = self.days_to_expiry(as_of)
        return max(0, days / 365.0)
    
    def is_expired(self, as_of: Optional[date] = None) -> bool:
        """Check if option is expired."""
        return self.days_to_expiry(as_of) <= 0
    
    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate intrinsic value.
        
        Args:
            spot: Current underlying price
            
        Returns:
            Intrinsic value
        """
        if self.is_call:
            return max(0, spot - self.strike)
        else:
            return max(0, self.strike - spot)
    
    def is_in_the_money(self, spot: float) -> bool:
        """Check if option is in the money."""
        return self.intrinsic_value(spot) > 0
    
    def is_at_the_money(self, spot: float, tolerance: float = 0.01) -> bool:
        """Check if option is at the money (within tolerance)."""
        return abs(spot - self.strike) / spot <= tolerance
    
    def moneyness(self, spot: float) -> float:
        """
        Calculate moneyness (spot/strike for calls, strike/spot for puts).
        
        Args:
            spot: Current underlying price
            
        Returns:
            Moneyness ratio
        """
        if self.is_call:
            return spot / self.strike
        else:
            return self.strike / spot
    
    def theoretical_price(
        self,
        spot: float,
        volatility: float,
        rate: float = 0.05,
        dividend: float = 0.0,
        as_of: Optional[date] = None,
    ) -> float:
        """
        Calculate theoretical option price using Black-Scholes.
        
        Args:
            spot: Current underlying price
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            dividend: Dividend yield (decimal)
            as_of: Reference date for time calculation
            
        Returns:
            Theoretical option price
        """
        t = self.time_to_expiry(as_of)
        return BlackScholes.price(
            spot, self.strike, t, volatility, rate, self.option_type, dividend
        )
    
    def calculate_greeks(
        self,
        spot: float,
        volatility: float,
        rate: float = 0.05,
        dividend: float = 0.0,
        as_of: Optional[date] = None,
    ) -> OptionGreeks:
        """
        Calculate option Greeks.
        
        Args:
            spot: Current underlying price
            volatility: Annualized volatility (decimal)
            rate: Risk-free rate (decimal)
            dividend: Dividend yield (decimal)
            as_of: Reference date for time calculation
            
        Returns:
            OptionGreeks dataclass
        """
        t = self.time_to_expiry(as_of)
        return BlackScholes.greeks(
            spot, self.strike, t, volatility, rate, self.option_type, dividend
        )
    
    def implied_volatility(
        self,
        market_price: float,
        spot: float,
        rate: float = 0.05,
        dividend: float = 0.0,
        as_of: Optional[date] = None,
    ) -> float:
        """
        Calculate implied volatility from market price.
        
        Args:
            market_price: Market option price
            spot: Current underlying price
            rate: Risk-free rate (decimal)
            dividend: Dividend yield (decimal)
            as_of: Reference date for time calculation
            
        Returns:
            Implied volatility (decimal)
        """
        t = self.time_to_expiry(as_of)
        return BlackScholes.implied_volatility(
            market_price, spot, self.strike, t, rate, self.option_type, dividend
        )
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
    ) -> float:
        """
        Calculate profit/loss.
        
        Args:
            entry_price: Entry option price
            exit_price: Exit option price
            quantity: Number of contracts (negative for short)
            
        Returns:
            P&L in currency
        """
        price_change = exit_price - entry_price
        return price_change * quantity * self._multiplier


class OptionChain:
    """
    Collection of options for a single underlying and expiry.
    
    Organizes calls and puts by strike price.
    
    Example:
        >>> chain = OptionChain("AAPL", expiry="2024-03-15")
        >>> chain.add_strike(150)
        >>> chain.add_strike(155)
        >>> atm_call = chain.get_atm_call(spot=152)
    """
    
    def __init__(
        self,
        underlying: str,
        expiry: Union[str, date, datetime],
        multiplier: float = 100.0,
    ):
        """
        Initialize option chain.
        
        Args:
            underlying: Underlying symbol
            expiry: Expiration date
            multiplier: Contract multiplier
        """
        self.underlying = underlying.upper()
        self.multiplier = multiplier
        
        # Parse expiry
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
        elif isinstance(expiry, datetime):
            expiry = expiry.date()
        self.expiry = expiry
        
        # Store options by strike
        self._calls: Dict[float, Option] = {}
        self._puts: Dict[float, Option] = {}
    
    def add_strike(self, strike: float) -> Tuple[Option, Option]:
        """
        Add call and put at a strike price.
        
        Args:
            strike: Strike price
            
        Returns:
            Tuple of (call, put) options
        """
        expiry_str = self.expiry.strftime("%y%m%d")
        
        # Create call
        call_symbol = f"{self.underlying}{expiry_str}C{int(strike * 1000):08d}"
        call = Option(
            symbol=call_symbol,
            underlying=self.underlying,
            strike=strike,
            expiry=self.expiry,
            option_type=OptionType.CALL,
            multiplier=self.multiplier,
        )
        self._calls[strike] = call
        
        # Create put
        put_symbol = f"{self.underlying}{expiry_str}P{int(strike * 1000):08d}"
        put = Option(
            symbol=put_symbol,
            underlying=self.underlying,
            strike=strike,
            expiry=self.expiry,
            option_type=OptionType.PUT,
            multiplier=self.multiplier,
        )
        self._puts[strike] = put
        
        return call, put
    
    @property
    def strikes(self) -> List[float]:
        """Get sorted list of strikes."""
        return sorted(set(self._calls.keys()) | set(self._puts.keys()))
    
    def get_call(self, strike: float) -> Optional[Option]:
        """Get call option at strike."""
        return self._calls.get(strike)
    
    def get_put(self, strike: float) -> Optional[Option]:
        """Get put option at strike."""
        return self._puts.get(strike)
    
    def get_atm_strike(self, spot: float) -> float:
        """
        Get at-the-money strike (closest to spot).
        
        Args:
            spot: Current underlying price
            
        Returns:
            ATM strike price
        """
        strikes = self.strikes
        if not strikes:
            return spot
        
        return min(strikes, key=lambda k: abs(k - spot))
    
    def get_atm_call(self, spot: float) -> Optional[Option]:
        """Get ATM call option."""
        atm_strike = self.get_atm_strike(spot)
        return self._calls.get(atm_strike)
    
    def get_atm_put(self, spot: float) -> Optional[Option]:
        """Get ATM put option."""
        atm_strike = self.get_atm_strike(spot)
        return self._puts.get(atm_strike)
    
    def get_straddle(self, strike: float) -> Tuple[Optional[Option], Optional[Option]]:
        """
        Get straddle (call + put at same strike).
        
        Args:
            strike: Strike price
            
        Returns:
            Tuple of (call, put)
        """
        return self._calls.get(strike), self._puts.get(strike)
    
    def get_strangle(
        self,
        spot: float,
        width: float,
    ) -> Tuple[Optional[Option], Optional[Option]]:
        """
        Get strangle (OTM call and put).
        
        Args:
            spot: Current underlying price
            width: Distance from ATM for each leg
            
        Returns:
            Tuple of (call, put)
        """
        call_strike = spot + width
        put_strike = spot - width
        
        # Find closest strikes
        strikes = self.strikes
        if not strikes:
            return None, None
        
        call_strike = min(strikes, key=lambda k: abs(k - call_strike) if k >= spot else float('inf'))
        put_strike = min(strikes, key=lambda k: abs(k - put_strike) if k <= spot else float('inf'))
        
        return self._calls.get(call_strike), self._puts.get(put_strike)
    
    def __len__(self) -> int:
        """Number of strikes in chain."""
        return len(self.strikes)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"OptionChain({self.underlying}, expiry={self.expiry}, strikes={len(self)})"

"""Tests for multi-asset support (Phase 18)."""

import pytest
from datetime import date, datetime, timedelta
import math

# Import all asset classes
from jsf.assets import (
    # Base
    Asset, AssetType, AssetSpec, TradingSession,
    Equity, ETF, Index,
    # Futures
    Future, FutureContract, FutureChain, RollMethod,
    ContractSpec, FUTURES_SPECS,
    get_contract_month_code, get_month_from_code,
    parse_contract_symbol, build_contract_symbol,
    # Options
    Option, OptionType, OptionStyle, OptionGreeks, OptionChain,
    BlackScholes,
    # Crypto
    CryptoAsset, CryptoExchange, CryptoSpec, TradingPair,
    CryptoPortfolioAsset, get_crypto_asset, parse_crypto_pair,
    CRYPTO_SPECS,
    # Forex
    ForexPair, LotSize, CurrencyType, ForexSession,
    pip_value, pips_to_price, price_to_pips,
    create_forex_pair, get_major_pairs, get_minor_pairs, get_active_sessions,
    FOREX_PAIR_SPECS, CURRENCY_SPECS,
)


# =============================================================================
# BASE ASSET TESTS
# =============================================================================

class TestAssetType:
    """Tests for AssetType enum."""
    
    def test_all_types_exist(self):
        """Test all expected asset types exist."""
        expected = ["EQUITY", "FUTURE", "OPTION", "CRYPTO", "FOREX", "ETF", "INDEX", "BOND"]
        for name in expected:
            assert hasattr(AssetType, name)
    
    def test_asset_type_values(self):
        """Test asset type values."""
        assert AssetType.EQUITY.value == "equity"
        assert AssetType.CRYPTO.value == "crypto"
        assert AssetType.FOREX.value == "forex"


class TestEquity:
    """Tests for Equity asset class."""
    
    def test_basic_creation(self):
        """Test basic equity creation."""
        aapl = Equity("AAPL", name="Apple Inc.")
        assert aapl.symbol == "AAPL"
        assert aapl.name == "Apple Inc."
        assert aapl.asset_type == AssetType.EQUITY
        assert aapl.currency == "USD"
    
    def test_equity_properties(self):
        """Test equity properties."""
        stock = Equity("MSFT", fractional=True, shortable=True)
        assert stock.fractional is True
        assert stock.shortable is True
        assert stock.hard_to_borrow is False
        assert stock.get_multiplier() == 1.0
    
    def test_calculate_value(self):
        """Test value calculation."""
        stock = Equity("GOOG")
        value = stock.calculate_value(100, 150.0)
        assert value == 15000.0
    
    def test_round_price(self):
        """Test price rounding."""
        stock = Equity("TEST")
        rounded = stock.round_price(123.456)
        assert abs(rounded - 123.46) < 0.001  # Default tick size 0.01
    
    def test_round_quantity(self):
        """Test quantity rounding."""
        stock = Equity("TEST", fractional=False)
        # Standard rounding (0.5 rounds up)
        assert stock.round_quantity(10.7) == 11  # Rounds to nearest whole
        assert stock.round_quantity(10.4) == 10  # Rounds to nearest whole
        
        fractional_stock = Equity("FRAC", fractional=True)
        rounded = fractional_stock.round_quantity(10.7)
        assert abs(rounded - 10.7) < 0.01  # Fractional allows fine-grained
    
    def test_dividend_yield(self):
        """Test dividend yield."""
        stock = Equity("DIV", dividend_yield=0.03)  # 3% yield
        assert stock.dividend_yield == 0.03
        
        # calculate_dividend(shares, price, holding_days)
        dividend = stock.calculate_dividend(100, 100, 365)  # 100 shares @ $100
        assert dividend == 300.0  # 3% of $10,000


class TestETF:
    """Tests for ETF asset class."""
    
    def test_basic_creation(self):
        """Test basic ETF creation."""
        spy = ETF("SPY", name="SPDR S&P 500")
        assert spy.symbol == "SPY"
        assert spy.asset_type == AssetType.ETF
    
    def test_leveraged_etf(self):
        """Test leveraged ETF."""
        tqqq = ETF("TQQQ", leveraged=True, leverage_factor=3.0, underlying="QQQ")
        assert tqqq.leveraged is True
        assert tqqq.leverage_factor == 3.0
        
        # Test effective exposure: get_effective_exposure(shares, price)
        exposure = tqqq.get_effective_exposure(100, 100)  # 100 shares @ $100
        assert exposure == 30000.0  # 3x leverage
    
    def test_inverse_etf(self):
        """Test inverse ETF."""
        sqqq = ETF("SQQQ", leveraged=True, inverse=True, leverage_factor=3.0)
        assert sqqq.inverse is True
        
        # Inverse should give negative exposure: get_effective_exposure(shares, price)
        exposure = sqqq.get_effective_exposure(100, 100)  # 100 shares @ $100
        assert exposure == -30000.0
    
    def test_expense_ratio(self):
        """Test expense ratio."""
        etf = ETF("TEST", expense_ratio=0.0003)  # 0.03%
        assert etf.expense_ratio == 0.0003


class TestIndex:
    """Tests for Index asset class."""
    
    def test_basic_creation(self):
        """Test index creation."""
        spx = Index("SPX", name="S&P 500 Index")
        assert spx.symbol == "SPX"
        assert spx.asset_type == AssetType.INDEX
    
    def test_index_not_tradeable(self):
        """Test that index is not directly tradeable."""
        index = Index("NDX")
        # is_tradeable() method returns False for indices
        assert index.is_tradeable() is False
    
    def test_index_components(self):
        """Test index components."""
        index = Index("TEST", components=["AAPL", "MSFT", "GOOG"])
        assert index.components == ["AAPL", "MSFT", "GOOG"]


# =============================================================================
# FUTURES TESTS
# =============================================================================

class TestFuturesUtilities:
    """Tests for futures utility functions."""
    
    def test_month_code_conversion(self):
        """Test month code conversion."""
        assert get_contract_month_code(1) == "F"   # January
        assert get_contract_month_code(3) == "H"   # March
        assert get_contract_month_code(6) == "M"   # June
        assert get_contract_month_code(9) == "U"   # September
        assert get_contract_month_code(12) == "Z"  # December
    
    def test_month_from_code(self):
        """Test month from code."""
        assert get_month_from_code("F") == 1
        assert get_month_from_code("H") == 3
        assert get_month_from_code("Z") == 12
    
    def test_parse_contract_symbol(self):
        """Test contract symbol parsing."""
        root, month, year = parse_contract_symbol("ESH24")
        assert root == "ES"
        assert month == "H"
        assert year == 2024
        
        root, month, year = parse_contract_symbol("CLM2024")
        assert root == "CL"
        assert month == "M"
        assert year == 2024
    
    def test_build_contract_symbol(self):
        """Test contract symbol building."""
        symbol = build_contract_symbol("ES", 3, 2024)
        assert symbol == "ESH24"
        
        symbol = build_contract_symbol("CL", "M", 2024, format="long")
        assert symbol == "CLM2024"


class TestFutureContract:
    """Tests for FutureContract class."""
    
    def test_basic_creation(self):
        """Test basic contract creation."""
        contract = FutureContract(
            root_symbol="ES",
            month=3,
            year=2024,
            expiry=date(2024, 3, 15),
        )
        assert contract.symbol == "ESH24"
        assert contract.month_code == "H"
    
    def test_days_to_expiry(self):
        """Test days to expiry calculation."""
        expiry = date.today() + timedelta(days=30)
        contract = FutureContract("ES", 3, 2024, expiry=expiry)
        
        assert contract.days_to_expiry() == 30
        assert contract.is_expired() is False
    
    def test_expired_contract(self):
        """Test expired contract detection."""
        expiry = date.today() - timedelta(days=5)
        contract = FutureContract("ES", 3, 2024, expiry=expiry)
        
        assert contract.is_expired() is True
        assert contract.days_to_expiry() == -5
    
    def test_should_roll(self):
        """Test roll detection."""
        expiry = date.today() + timedelta(days=3)
        contract = FutureContract("ES", 3, 2024, expiry=expiry)
        
        # Should roll with default 5 days
        assert contract.should_roll(days_before=5) is True
        assert contract.should_roll(days_before=2) is False


class TestFuture:
    """Tests for Future asset class."""
    
    def test_basic_creation(self):
        """Test basic future creation."""
        es = Future("ES", expiry="2024-03-15")
        assert es.symbol == "ES"
        assert es.asset_type == AssetType.FUTURE
        assert es.root_symbol == "ES"
    
    def test_es_specs(self):
        """Test E-mini S&P 500 specs."""
        es = Future("ESH24", expiry="2024-03-15")
        assert es.get_multiplier() == 50.0
        assert es.tick_size == 0.25
        assert es.tick_value == 12.50
    
    def test_margin_calculation(self):
        """Test margin calculation."""
        es = Future("ES", expiry="2024-03-15", margin=12000)
        margin = es.get_margin(2, 5000)  # 2 contracts
        assert margin == 24000.0
    
    def test_pnl_calculation(self):
        """Test P&L calculation."""
        es = Future("ES", expiry="2024-03-15")
        pnl = es.calculate_pnl(5000, 5050, 2)  # Buy 2 @ 5000, sell @ 5050
        assert pnl == 50 * 2 * 50  # 50 pts * 2 contracts * $50 multiplier
        assert pnl == 5000.0
    
    def test_days_to_expiry(self):
        """Test days to expiry."""
        expiry = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
        future = Future("ES", expiry=expiry)
        assert future.days_to_expiry() == 30


class TestFutureChain:
    """Tests for FutureChain class."""
    
    def test_basic_creation(self):
        """Test chain creation."""
        chain = FutureChain("ES")
        assert chain.root_symbol == "ES"
        assert len(chain) == 0
    
    def test_add_contract(self):
        """Test adding contracts."""
        chain = FutureChain("ES")
        contract = FutureContract("ES", 3, 2024, date(2024, 3, 15))
        chain.add_contract(contract)
        assert len(chain) == 1
    
    def test_front_contract(self):
        """Test front month selection."""
        chain = FutureChain("ES")
        
        # Add contracts in random order
        chain.add_contract(FutureContract("ES", 6, 2024, date.today() + timedelta(days=60)))
        chain.add_contract(FutureContract("ES", 3, 2024, date.today() + timedelta(days=30)))
        chain.add_contract(FutureContract("ES", 9, 2024, date.today() + timedelta(days=90)))
        
        front = chain.front_contract()
        assert front is not None
        assert front.month == 3  # March is front


class TestFuturesSpecs:
    """Tests for predefined futures specs."""
    
    def test_common_futures_exist(self):
        """Test common futures specs exist."""
        assert "ES" in FUTURES_SPECS
        assert "NQ" in FUTURES_SPECS
        assert "CL" in FUTURES_SPECS
        assert "GC" in FUTURES_SPECS
    
    def test_es_spec(self):
        """Test E-mini S&P 500 spec."""
        es_spec = FUTURES_SPECS["ES"]
        assert es_spec.multiplier == 50.0
        assert es_spec.tick_size == 0.25
        assert es_spec.exchange == "CME"


# =============================================================================
# OPTIONS TESTS
# =============================================================================

class TestBlackScholes:
    """Tests for Black-Scholes pricing."""
    
    def test_call_price(self):
        """Test call option pricing."""
        price = BlackScholes.price(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.20, rate=0.05, option_type=OptionType.CALL
        )
        # ATM call should be around $4-5 for 3 months, 20% vol
        assert 3 < price < 7
    
    def test_put_price(self):
        """Test put option pricing."""
        price = BlackScholes.price(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.20, rate=0.05, option_type=OptionType.PUT
        )
        # ATM put should be slightly less than call due to interest
        assert 3 < price < 6
    
    def test_put_call_parity(self):
        """Test put-call parity."""
        spot, strike, t, vol, r = 100, 100, 0.25, 0.20, 0.05
        
        call = BlackScholes.price(spot, strike, t, vol, r, OptionType.CALL)
        put = BlackScholes.price(spot, strike, t, vol, r, OptionType.PUT)
        
        # C - P = S - K*e^(-rT)
        lhs = call - put
        rhs = spot - strike * math.exp(-r * t)
        
        assert abs(lhs - rhs) < 0.01
    
    def test_itm_call(self):
        """Test in-the-money call."""
        price = BlackScholes.price(
            spot=110, strike=100, time_to_expiry=0.25,
            volatility=0.20, rate=0.05, option_type=OptionType.CALL
        )
        # ITM call should be worth at least intrinsic value
        assert price > 10  # Intrinsic = 110 - 100 = 10
    
    def test_otm_put(self):
        """Test out-of-the-money put."""
        price = BlackScholes.price(
            spot=110, strike=100, time_to_expiry=0.25,
            volatility=0.20, rate=0.05, option_type=OptionType.PUT
        )
        # OTM put should have small positive value
        assert 0 < price < 2
    
    def test_delta(self):
        """Test delta calculation."""
        delta = BlackScholes.delta(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.20, rate=0.05, option_type=OptionType.CALL
        )
        # ATM call delta should be around 0.5
        assert 0.45 < delta < 0.65
        
        put_delta = BlackScholes.delta(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.20, rate=0.05, option_type=OptionType.PUT
        )
        # Put delta should be negative
        assert -0.6 < put_delta < -0.4
    
    def test_gamma(self):
        """Test gamma calculation."""
        gamma = BlackScholes.gamma(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.20, rate=0.05
        )
        # Gamma should be positive and small
        assert 0 < gamma < 0.1
    
    def test_theta(self):
        """Test theta calculation."""
        theta = BlackScholes.theta(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.20, rate=0.05, option_type=OptionType.CALL
        )
        # Theta should be negative (time decay)
        assert theta < 0
    
    def test_vega(self):
        """Test vega calculation."""
        vega = BlackScholes.vega(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.20, rate=0.05
        )
        # Vega should be positive
        assert vega > 0
    
    def test_greeks_dataclass(self):
        """Test Greeks dataclass."""
        greeks = BlackScholes.greeks(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=0.20, rate=0.05, option_type=OptionType.CALL
        )
        
        assert isinstance(greeks, OptionGreeks)
        assert 0.4 < greeks.delta < 0.7
        assert greeks.gamma > 0
        assert greeks.theta < 0
        assert greeks.vega > 0
    
    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        # First, price an option with known vol
        known_vol = 0.25
        price = BlackScholes.price(
            spot=100, strike=100, time_to_expiry=0.25,
            volatility=known_vol, rate=0.05, option_type=OptionType.CALL
        )
        
        # Then recover the implied vol
        iv = BlackScholes.implied_volatility(
            market_price=price, spot=100, strike=100,
            time_to_expiry=0.25, rate=0.05, option_type=OptionType.CALL
        )
        
        assert abs(iv - known_vol) < 0.01


class TestOption:
    """Tests for Option asset class."""
    
    def test_basic_creation(self):
        """Test basic option creation."""
        call = Option(
            symbol="AAPL240315C00150000",
            underlying="AAPL",
            strike=150,
            expiry="2024-03-15",
            option_type=OptionType.CALL,
        )
        assert call.underlying == "AAPL"
        assert call.strike == 150.0
        assert call.is_call is True
        assert call.is_put is False
        assert call.asset_type == AssetType.OPTION
    
    def test_put_option(self):
        """Test put option."""
        put = Option(
            symbol="AAPL240315P00150000",
            underlying="AAPL",
            strike=150,
            expiry="2024-03-15",
            option_type=OptionType.PUT,
        )
        assert put.is_put is True
        assert put.is_call is False
    
    def test_intrinsic_value(self):
        """Test intrinsic value calculation."""
        call = Option(
            symbol="TEST", underlying="TEST",
            strike=100, expiry="2024-12-31",
            option_type=OptionType.CALL,
        )
        
        assert call.intrinsic_value(110) == 10  # ITM
        assert call.intrinsic_value(90) == 0   # OTM
        
        put = Option(
            symbol="TEST", underlying="TEST",
            strike=100, expiry="2024-12-31",
            option_type=OptionType.PUT,
        )
        
        assert put.intrinsic_value(90) == 10   # ITM
        assert put.intrinsic_value(110) == 0   # OTM
    
    def test_moneyness(self):
        """Test moneyness indicators."""
        call = Option(
            symbol="TEST", underlying="TEST",
            strike=100, expiry="2024-12-31",
            option_type=OptionType.CALL,
        )
        
        assert call.is_in_the_money(110) is True
        assert call.is_in_the_money(90) is False
        assert call.is_at_the_money(100.5, tolerance=0.01) is True
    
    def test_theoretical_price(self):
        """Test theoretical pricing."""
        # Create option expiring in ~90 days
        expiry = (date.today() + timedelta(days=90)).strftime("%Y-%m-%d")
        call = Option(
            symbol="TEST", underlying="TEST",
            strike=100, expiry=expiry,
            option_type=OptionType.CALL,
        )
        
        price = call.theoretical_price(spot=100, volatility=0.20, rate=0.05)
        assert price > 0
    
    def test_calculate_greeks(self):
        """Test Greeks calculation."""
        expiry = (date.today() + timedelta(days=90)).strftime("%Y-%m-%d")
        call = Option(
            symbol="TEST", underlying="TEST",
            strike=100, expiry=expiry,
            option_type=OptionType.CALL,
        )
        
        greeks = call.calculate_greeks(spot=100, volatility=0.20, rate=0.05)
        
        assert isinstance(greeks, OptionGreeks)
        assert greeks.delta > 0  # Call delta positive


class TestOptionChain:
    """Tests for OptionChain class."""
    
    def test_basic_creation(self):
        """Test chain creation."""
        chain = OptionChain("AAPL", expiry="2024-03-15")
        assert chain.underlying == "AAPL"
        assert len(chain) == 0
    
    def test_add_strike(self):
        """Test adding strikes."""
        chain = OptionChain("AAPL", expiry="2024-03-15")
        call, put = chain.add_strike(150)
        
        assert call.is_call is True
        assert put.is_put is True
        assert len(chain) == 1
        assert 150 in chain.strikes
    
    def test_atm_strike(self):
        """Test ATM strike selection."""
        chain = OptionChain("AAPL", expiry="2024-03-15")
        chain.add_strike(145)
        chain.add_strike(150)
        chain.add_strike(155)
        
        atm = chain.get_atm_strike(152)
        assert atm == 150
    
    def test_straddle(self):
        """Test straddle retrieval."""
        chain = OptionChain("AAPL", expiry="2024-03-15")
        chain.add_strike(150)
        
        call, put = chain.get_straddle(150)
        assert call is not None and call.is_call
        assert put is not None and put.is_put


# =============================================================================
# CRYPTO TESTS
# =============================================================================

class TestCryptoAsset:
    """Tests for CryptoAsset class."""
    
    def test_basic_creation(self):
        """Test basic crypto creation."""
        btc = CryptoAsset("BTC", quote_currency="USDT")
        assert btc.symbol == "BTC"
        assert btc.base_currency == "BTC"
        assert btc.quote_currency == "USDT"
        assert btc.asset_type == AssetType.CRYPTO
    
    def test_trading_pair(self):
        """Test trading pair string."""
        eth = CryptoAsset("ETH", quote_currency="USDT")
        assert eth.trading_pair == "ETH/USDT"
    
    def test_24_7_trading(self):
        """Test continuous trading."""
        btc = CryptoAsset("BTC")
        assert btc.trading_session == TradingSession.CONTINUOUS
        assert btc.is_tradeable() is True  # Always tradeable
    
    def test_round_quantity(self):
        """Test quantity rounding."""
        btc = CryptoAsset("BTC")
        # BTC min quantity is 0.00001
        assert btc.round_quantity(0.123456789) == 0.12345
        assert btc.round_quantity(0.000001) == 0  # Below min
    
    def test_fee_calculation(self):
        """Test fee calculation."""
        btc = CryptoAsset("BTC", maker_fee=0.001, taker_fee=0.001)
        fee = btc.calculate_fee(0.1, 50000, is_maker=False)
        assert fee == 0.1 * 50000 * 0.001  # 0.1% of $5000
        assert fee == 5.0
    
    def test_net_proceeds(self):
        """Test net proceeds calculation."""
        btc = CryptoAsset("BTC", taker_fee=0.001)
        # Sell 0.1 BTC @ $50000
        proceeds = btc.calculate_net_proceeds(0.1, 50000, is_maker=False)
        expected = 0.1 * 50000 - 5.0  # Gross - fee
        assert proceeds == expected
    
    def test_stablecoin_flag(self):
        """Test stablecoin detection."""
        usdt = CryptoAsset("USDT")
        btc = CryptoAsset("BTC")
        
        assert usdt.is_stablecoin is True
        assert btc.is_stablecoin is False
    
    def test_get_pair(self):
        """Test getting trading pair object."""
        btc = CryptoAsset("BTC", quote_currency="USDT")
        pair = btc.get_pair()
        
        assert isinstance(pair, TradingPair)
        assert pair.base == "BTC"
        assert pair.quote == "USDT"


class TestCryptoPortfolioAsset:
    """Tests for CryptoPortfolioAsset class."""
    
    def test_basic_creation(self):
        """Test portfolio asset creation."""
        btc = CryptoAsset("BTC")
        holding = CryptoPortfolioAsset(btc, quantity=0.5, avg_cost=45000)
        
        assert holding.quantity == 0.5
        assert holding.avg_cost == 45000
        assert holding.cost_basis == 22500.0
    
    def test_unrealized_pnl(self):
        """Test unrealized P&L."""
        btc = CryptoAsset("BTC")
        holding = CryptoPortfolioAsset(btc, quantity=0.5, avg_cost=45000)
        
        pnl = holding.unrealized_pnl(50000)
        assert pnl == 0.5 * (50000 - 45000)
        assert pnl == 2500.0
    
    def test_add_position(self):
        """Test adding to position."""
        btc = CryptoAsset("BTC")
        holding = CryptoPortfolioAsset(btc, quantity=0.5, avg_cost=45000)
        
        # Add 0.5 BTC @ $50000
        holding.add_position(0.5, 50000)
        
        assert holding.quantity == 1.0
        # New avg cost: (0.5*45000 + 0.5*50000) / 1.0 = 47500
        assert holding.avg_cost == 47500.0
    
    def test_reduce_position(self):
        """Test reducing position."""
        btc = CryptoAsset("BTC")
        holding = CryptoPortfolioAsset(btc, quantity=1.0, avg_cost=45000)
        
        # Sell 0.5 BTC @ $50000
        realized = holding.reduce_position(0.5, 50000)
        
        assert holding.quantity == 0.5
        # Realized = 0.5 * (50000 - 45000) = 2500
        assert realized == 2500.0
        assert holding.realized_pnl == 2500.0


class TestCryptoUtilities:
    """Tests for crypto utility functions."""
    
    def test_get_crypto_asset(self):
        """Test factory function."""
        btc = get_crypto_asset("BTC", quote_currency="USD")
        assert btc.base_currency == "BTC"
        assert btc.quote_currency == "USD"
    
    def test_parse_crypto_pair(self):
        """Test pair parsing."""
        base, quote = parse_crypto_pair("BTC/USDT")
        assert base == "BTC"
        assert quote == "USDT"
        
        base, quote = parse_crypto_pair("ETHBTC")
        assert base == "ETH"
        assert quote == "BTC"
    
    def test_crypto_specs_exist(self):
        """Test common crypto specs exist."""
        assert "BTC" in CRYPTO_SPECS
        assert "ETH" in CRYPTO_SPECS
        assert "USDT" in CRYPTO_SPECS


# =============================================================================
# FOREX TESTS
# =============================================================================

class TestForexPair:
    """Tests for ForexPair class."""
    
    def test_basic_creation(self):
        """Test basic forex pair creation."""
        eurusd = ForexPair("EUR/USD")
        assert eurusd.base_currency == "EUR"
        assert eurusd.quote_currency == "USD"
        assert eurusd.asset_type == AssetType.FOREX
    
    def test_symbol_parsing(self):
        """Test different symbol formats."""
        pair1 = ForexPair("EURUSD")
        pair2 = ForexPair("EUR/USD")
        
        assert pair1.base_currency == pair2.base_currency
        assert pair1.quote_currency == pair2.quote_currency
    
    def test_pip_size(self):
        """Test pip size calculation."""
        eurusd = ForexPair("EURUSD")
        assert eurusd.pip_size == 0.0001
        
        usdjpy = ForexPair("USDJPY")
        assert usdjpy.pip_size == 0.01
    
    def test_pip_value(self):
        """Test pip value calculation."""
        eurusd = ForexPair("EURUSD")
        pv = eurusd.pip_value(LotSize.STANDARD)
        assert pv == 10.0  # $10 per pip for EUR/USD standard lot
    
    def test_pips_conversion(self):
        """Test pips to/from price conversion."""
        eurusd = ForexPair("EURUSD")
        
        # 50 pips = 0.0050 price change
        price_change = eurusd.pips_to_price(50)
        assert price_change == 0.0050
        
        # 0.0050 = 50 pips
        pips = eurusd.price_to_pips(0.0050)
        assert pips == 50
    
    def test_pnl_calculation(self):
        """Test P&L calculation."""
        eurusd = ForexPair("EURUSD")
        
        # Long 1 standard lot, 50 pip move
        pips = eurusd.calculate_pnl_pips(1.1000, 1.1050, lots=1, direction=1)
        assert abs(pips - 50) < 0.001  # Allow floating point tolerance
    
    def test_margin_calculation(self):
        """Test margin calculation."""
        eurusd = ForexPair("EURUSD", leverage=50)
        margin = eurusd.calculate_margin(1, 1.1000)
        
        # 1 lot = 100k * 1.1 / 50 = $2200
        expected = 100000 * 1.1 / 50
        assert abs(margin - expected) < 1  # Allow small rounding
    
    def test_position_sizing(self):
        """Test position size calculation."""
        eurusd = ForexPair("EURUSD")
        
        # Risk $100, 20 pip stop
        lots = eurusd.calculate_position_size(100, 20)
        # Expected: 100 / (20 * 10) = 0.5 lots
        assert abs(lots - 0.5) < 0.01
    
    def test_spread(self):
        """Test spread properties."""
        eurusd = ForexPair("EURUSD")
        assert eurusd.typical_spread > 0
        assert eurusd.spread_cost == eurusd.typical_spread * eurusd.pip_size


class TestForexUtilities:
    """Tests for forex utility functions."""
    
    def test_pip_value_function(self):
        """Test pip_value utility function."""
        pv = pip_value("EURUSD", LotSize.MINI)
        assert pv == 1.0  # $1 per pip for mini lot
        
        pv = pip_value("EURUSD", LotSize.MICRO)
        assert pv == 0.1  # $0.10 per pip for micro lot
    
    def test_pips_to_price_function(self):
        """Test pips_to_price utility function."""
        change = pips_to_price(100, "EURUSD")
        assert change == 0.0100
        
        change = pips_to_price(100, "USDJPY")
        assert change == 1.0  # JPY pairs have 2 decimal pips
    
    def test_create_forex_pair(self):
        """Test factory function."""
        pair = create_forex_pair("EUR", "USD", leverage=100)
        assert pair.base_currency == "EUR"
        assert pair.quote_currency == "USD"
        assert pair.leverage == 100
    
    def test_get_major_pairs(self):
        """Test getting major pairs."""
        majors = get_major_pairs()
        assert len(majors) == 7
        
        symbols = [f"{p.base_currency}{p.quote_currency}" for p in majors]
        assert "EURUSD" in symbols
        assert "USDJPY" in symbols
    
    def test_get_minor_pairs(self):
        """Test getting minor pairs."""
        minors = get_minor_pairs()
        assert len(minors) == 10
    
    def test_forex_specs_exist(self):
        """Test forex specs exist."""
        assert "EURUSD" in FOREX_PAIR_SPECS
        assert "USDJPY" in FOREX_PAIR_SPECS
    
    def test_currency_specs_exist(self):
        """Test currency specs exist."""
        assert "USD" in CURRENCY_SPECS
        assert "EUR" in CURRENCY_SPECS
        assert "JPY" in CURRENCY_SPECS


class TestForexSessions:
    """Tests for forex session handling."""
    
    def test_lot_size_enum(self):
        """Test lot size values."""
        assert LotSize.STANDARD.value == 100000
        assert LotSize.MINI.value == 10000
        assert LotSize.MICRO.value == 1000
        assert LotSize.NANO.value == 100
    
    def test_currency_type_enum(self):
        """Test currency type enum."""
        assert CurrencyType.MAJOR.value == "major"
        assert CurrencyType.EXOTIC.value == "exotic"
    
    def test_forex_session_enum(self):
        """Test forex session enum."""
        assert ForexSession.LONDON.value == "london"
        assert ForexSession.NEW_YORK.value == "new_york"
    
    def test_get_active_sessions(self):
        """Test active session detection."""
        # Test during London hours (10:00 UTC)
        london_time = datetime(2024, 1, 15, 10, 0, 0)  # Monday 10am UTC
        sessions = get_active_sessions(london_time)
        assert ForexSession.LONDON in sessions
    
    def test_forex_trading_hours(self):
        """Test forex market open/closed detection."""
        eurusd = ForexPair("EURUSD")
        
        # Monday 10am UTC should be open
        monday_10am = datetime(2024, 1, 15, 10, 0, 0)
        assert eurusd.is_tradeable(monday_10am) is True
        
        # Saturday should be closed
        saturday = datetime(2024, 1, 13, 10, 0, 0)
        assert eurusd.is_tradeable(saturday) is False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMultiAssetIntegration:
    """Integration tests for multi-asset support."""
    
    def test_asset_type_identification(self):
        """Test correct asset type identification."""
        equity = Equity("AAPL")
        future = Future("ES")
        option = Option("TEST", "AAPL", 150, "2024-12-31", OptionType.CALL)
        crypto = CryptoAsset("BTC")
        forex = ForexPair("EURUSD")
        
        assert equity.asset_type == AssetType.EQUITY
        assert future.asset_type == AssetType.FUTURE
        assert option.asset_type == AssetType.OPTION
        assert crypto.asset_type == AssetType.CRYPTO
        assert forex.asset_type == AssetType.FOREX
    
    def test_value_calculation_consistency(self):
        """Test value calculation across asset types."""
        equity = Equity("AAPL")
        crypto = CryptoAsset("BTC")
        
        # Both should support calculate_value
        equity_value = equity.calculate_value(100, 150)
        crypto_value = crypto.calculate_value(0.5, 50000)
        
        assert equity_value == 15000
        assert crypto_value == 25000
    
    def test_trading_session_types(self):
        """Test trading session types vary by asset."""
        equity = Equity("AAPL")
        crypto = CryptoAsset("BTC")
        forex = ForexPair("EURUSD")
        future = Future("ES")
        
        assert equity.trading_session == TradingSession.REGULAR
        assert crypto.trading_session == TradingSession.CONTINUOUS
        assert forex.trading_session == TradingSession.EXTENDED
        assert future.trading_session == TradingSession.EXTENDED
    
    def test_tick_size_varies_by_asset(self):
        """Test tick size varies appropriately."""
        equity = Equity("AAPL")  # 0.01
        future = Future("ES")     # 0.25
        forex = ForexPair("EURUSD")  # 0.0001
        
        assert equity.tick_size == 0.01
        assert future.tick_size == 0.25
        assert forex.tick_size == 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

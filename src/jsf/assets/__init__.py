"""
Multi-asset support for different instrument types.

This module provides classes for handling various asset types:
- Equities (stocks)
- Futures (contracts with expiry)
- Options (calls/puts with Greeks)
- Crypto (24/7 digital assets)
- Forex (currency pairs)

Example:
    >>> from jsf.assets import Future, Option, CryptoAsset, ForexPair
    >>> 
    >>> # Futures with expiry handling
    >>> es_future = Future("ES", expiry="2024-03-15", multiplier=50)
    >>> 
    >>> # Options with Greeks
    >>> call = Option("AAPL", strike=150, expiry="2024-01-19", option_type=OptionType.CALL)
    >>> greeks = call.calculate_greeks(spot=155, volatility=0.25, rate=0.05)
    >>> 
    >>> # Crypto assets
    >>> btc = CryptoAsset("BTC", quote_currency="USDT")
    >>> 
    >>> # Forex pairs
    >>> eurusd = ForexPair("EUR/USD")
"""

from jsf.assets.base import (
    Asset,
    AssetType,
    AssetSpec,
    TradingSession,
    Equity,
    ETF,
    Index,
)
from jsf.assets.futures import (
    Future,
    FutureContract,
    FutureChain,
    RollMethod,
    ContractSpec,
    FUTURES_SPECS,
    get_contract_month_code,
    get_month_from_code,
    parse_contract_symbol,
    build_contract_symbol,
)
from jsf.assets.options import (
    Option,
    OptionType,
    OptionStyle,
    OptionGreeks,
    OptionChain,
    BlackScholes,
)
from jsf.assets.crypto import (
    CryptoAsset,
    CryptoExchange,
    CryptoSpec,
    TradingPair,
    CryptoPortfolioAsset,
    CRYPTO_SPECS,
    get_crypto_asset,
    parse_crypto_pair,
)
from jsf.assets.forex import (
    ForexPair,
    LotSize,
    CurrencyType,
    ForexSession,
    ForexPairSpec,
    FOREX_PAIR_SPECS,
    CURRENCY_SPECS,
    pip_value,
    pips_to_price,
    price_to_pips,
    create_forex_pair,
    get_major_pairs,
    get_minor_pairs,
    get_active_sessions,
)

__all__ = [
    # Base
    "Asset",
    "AssetType",
    "AssetSpec",
    "TradingSession",
    "Equity",
    "ETF",
    "Index",
    # Futures
    "Future",
    "FutureContract",
    "FutureChain",
    "RollMethod",
    "ContractSpec",
    "FUTURES_SPECS",
    "get_contract_month_code",
    "get_month_from_code",
    "parse_contract_symbol",
    "build_contract_symbol",
    # Options
    "Option",
    "OptionType",
    "OptionStyle",
    "OptionGreeks",
    "OptionChain",
    "BlackScholes",
    # Crypto
    "CryptoAsset",
    "CryptoExchange",
    "CryptoSpec",
    "TradingPair",
    "CryptoPortfolioAsset",
    "CRYPTO_SPECS",
    "get_crypto_asset",
    "parse_crypto_pair",
    # Forex
    "ForexPair",
    "LotSize",
    "CurrencyType",
    "ForexSession",
    "ForexPairSpec",
    "FOREX_PAIR_SPECS",
    "CURRENCY_SPECS",
    "pip_value",
    "pips_to_price",
    "price_to_pips",
    "create_forex_pair",
    "get_major_pairs",
    "get_minor_pairs",
    "get_active_sessions",
]

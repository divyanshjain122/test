# Trading Flow Detailed Report

This report explains how the project converts market data and sentiment data into signals, portfolio weights, backtest trades, and live/paper orders.

The main implementation paths covered here are:

- `demos/demo_real_data_finbert_backtest.py`: real Yahoo Finance prices plus local Kaggle financial headlines plus FinBERT sentiment.
- `demos/demo_full_real_data.py`: real Yahoo Finance prices plus NewsAPI/Yahoo RSS headlines plus FinBERT sentiment.
- `src/jsf/simulation/backtest.py`: the historical backtesting engine.
- `src/jsf/signals/technical.py` and `src/jsf/signals/statistical.py`: price-based signal generation.
- `src/jsf/portfolio/sizing.py` and `src/jsf/portfolio/constructors.py`: conversion from signals to weights.
- `src/jsf/live/engine.py`, `src/jsf/live/order_manager.py`, and `src/jsf/broker/paper.py`: live/paper order flow.

## Executive Summary

The demo backtests do not place real broker orders. They simulate trades by changing target portfolio weights over time.

In the FinBERT demos, a trade happens when a symbol's target portfolio weight changes between two dates. The target weights come from a combined signal:

```text
baseline_signal = 0.6 * momentum_signal + 0.4 * mean_reversion_signal
sentiment_enhanced_signal = 0.75 * baseline_signal + 0.25 * finbert_sentiment_signal
```

The combined signal is converted into long-only weights:

```text
1. Keep only positive signal values above threshold 0.05.
2. Normalize remaining positive values so each day's selected symbols sum to 1.0.
3. Shift weights by one day so today's signal is traded on the next date.
4. Run the backtest engine on those weights and close prices.
```

The backtest engine records a trade whenever the absolute difference between today's target weight and yesterday's target weight is greater than zero. It subtracts transaction cost plus slippage, applies returns from the previous day's held weights, and appends the new equity value to the equity curve.

## High-Level Backtest Flow

```text
Raw price/news inputs
        |
        v
PriceData close-price matrix
        |
        v
Momentum and mean-reversion signals
        |
        v
Optional FinBERT sentiment signal
        |
        v
Combined numeric signal per date and symbol
        |
        v
Long-only shifted target weights
        |
        v
Portfolio(weights=target_weights)
        |
        v
BacktestEngine.run(portfolio, price_data)
        |
        v
Equity curve, returns, positions, trades, metrics
```

## Data Model

### `PriceData`

File: `src/jsf/data/base.py`

`PriceData` is the common container used by signals and the backtest engine.

Important behavior:

- It accepts a `DataFrame` with either a `DatetimeIndex` or a two-level `(date, symbol)` `MultiIndex`.
- `symbols` returns the sorted list of symbols when the data has a `MultiIndex`.
- `dates` returns the sorted trading dates.
- `get_field("close")` unstacks the long-form `(date, symbol)` data into a date-by-symbol matrix.
- `get_close_prices()` is a convenience wrapper around `get_field("close")`.
- `get_returns(periods=1)` calculates percent returns from close prices.

The important code path is:

```python
def get_field(self, field: str) -> pd.DataFrame:
    if field not in self.data.columns:
        raise ValueError(f"Field '{field}' not found in data")

    if isinstance(self.data.index, pd.MultiIndex):
        return self.data[field].unstack(level=1)
    return self.data[[field]]

def get_close_prices(self) -> pd.DataFrame:
    return self.get_field("close")

def get_returns(self, periods: int = 1) -> pd.DataFrame:
    close = self.get_close_prices()
    return close.pct_change(periods=periods)
```

Line-by-line meaning:

- `if field not in self.data.columns`: rejects requests for missing fields such as `close` or `volume`.
- `isinstance(self.data.index, pd.MultiIndex)`: checks whether data is stored as `(date, symbol)` rows.
- `self.data[field].unstack(level=1)`: converts long-form data into a matrix where rows are dates and columns are symbols.
- `return self.data[[field]]`: handles single-index data by returning only the selected column.
- `get_close_prices()`: standardizes all signals on close prices.
- `pct_change(periods=periods)`: calculates asset returns used by the backtest and some signals.

## FinBERT Demo Flow

### Demo Inputs

File: `demos/demo_real_data_finbert_backtest.py`

The Kaggle demo uses:

- Yahoo Finance OHLCV prices through `YahooFinanceLoader`.
- Kaggle headline CSVs from a local archive.
- FinBERT to convert each headline into `positive`, `neutral`, or `negative` sentiment.
- A comparison between a price-only baseline and a price-plus-sentiment signal.

The similar `demos/demo_full_real_data.py` uses NewsAPI first and Yahoo Finance RSS as a fallback, but the signal and backtest mechanics are the same.

### Argument Parsing

The Kaggle demo begins by defining arguments:

```python
def parse_args():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, ...)
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET_FILE, choices=[...], ...)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS, ...)
    parser.add_argument("--start-date", help="Optional YYYY-MM-DD override for the backtest start date.")
    parser.add_argument("--end-date", help="Optional YYYY-MM-DD override for the backtest end date.")
    parser.add_argument("--max-headlines-per-symbol", type=int, default=500, ...)
    parser.add_argument("--price-warmup-days", type=int, default=60, ...)
    parser.add_argument("--post-news-days", type=int, default=365, ...)
    return parser.parse_args()
```

What each line/block does:

- `ArgumentParser(...)`: creates a command-line interface for choosing dataset, symbols, date range, and workload limits.
- `--dataset-dir`: tells the script where the uncompressed Kaggle archive is located.
- `--dataset-file`: restricts the allowed Kaggle files to the known schemas the loader can parse.
- `--symbols`: lets the user override the default stock universe.
- `--start-date` and `--end-date`: optionally filter the headline and backtest period.
- `--max-headlines-per-symbol`: caps FinBERT inference cost by keeping the latest N headlines per symbol.
- `--price-warmup-days`: loads extra price history before the backtest to calculate rolling signals correctly.
- `--post-news-days`: extends price history after the last headline so trades can be evaluated after news arrives.
- `return parser.parse_args()`: returns parsed settings to `main()`.

### Loading Kaggle News

```python
def load_kaggle_news(dataset_dir, dataset_file, symbols, start_date=None, end_date=None, max_per_symbol=500):
    dataset_path = Path(dataset_dir).expanduser() / dataset_file
    if not dataset_path.exists():
        raise FileNotFoundError(f"Kaggle dataset file not found: {dataset_path}")

    usecols = {...}[dataset_file]
    df = pd.read_csv(dataset_path, usecols=usecols)

    text_col = "headline" if "headline" in df.columns else "title"
    source_col = "publisher" if "publisher" in df.columns else None

    df = df.rename(columns={text_col: "text", "stock": "symbol"})
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"].isin([symbol.upper() for symbol in symbols])]
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""]
```

What each line/block does:

- `Path(dataset_dir).expanduser() / dataset_file`: creates the full CSV path and supports `~` in paths.
- `if not dataset_path.exists()`: fails early if the expected local file is missing.
- `usecols = {...}[dataset_file]`: selects only columns needed for the chosen Kaggle schema.
- `pd.read_csv(..., usecols=usecols)`: loads a smaller, cleaner DataFrame.
- `text_col = ...`: handles different file schemas where headline text may be named `headline` or `title`.
- `source_col = ...`: keeps publisher/source if available.
- `rename(...)`: standardizes data to the project schema: `date`, `symbol`, `text`, `source`.
- `astype(str).str.upper().str.strip()`: normalizes symbols for matching against the selected universe.
- `isin(...)`: removes headlines for symbols not being traded.
- `fillna("").astype(str).str.strip()`: normalizes text and removes missing text.
- `df[df["text"] != ""]`: drops empty headlines because FinBERT needs real text.

The date and source processing continues:

```python
parsed_dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
df["date"] = parsed_dates.dt.tz_convert(None).dt.normalize()
df = df.dropna(subset=["date"])

if start_date:
    df = df[df["date"] >= pd.to_datetime(start_date)]
if end_date:
    df = df[df["date"] <= pd.to_datetime(end_date)]

if source_col:
    df["source"] = df[source_col].fillna("Kaggle Daily Financial News").astype(str)
else:
    df["source"] = "Kaggle Daily Financial News"

news_df = df[["date", "symbol", "text", "source"]].drop_duplicates()
news_df = news_df.sort_values(["symbol", "date"])

if max_per_symbol and max_per_symbol > 0:
    news_df = news_df.groupby("symbol", group_keys=False).tail(max_per_symbol)

return news_df.sort_values(["date", "symbol"]).reset_index(drop=True)
```

What each line/block does:

- `pd.to_datetime(..., errors="coerce", utc=True)`: parses dates and turns bad dates into `NaT` instead of crashing.
- `tz_convert(None).dt.normalize()`: removes timezone and truncates timestamps to the date.
- `dropna(subset=["date"])`: removes rows whose dates could not be parsed.
- `if start_date` and `if end_date`: optionally filter headlines to the requested date range.
- `df["source"] = ...`: fills source information for output/debugging.
- `df[["date", "symbol", "text", "source"]]`: keeps the exact columns consumed by FinBERT logic.
- `drop_duplicates()`: prevents duplicated headlines from over-weighting sentiment.
- `sort_values(["symbol", "date"])`: prepares rows for per-symbol tail selection.
- `groupby(...).tail(max_per_symbol)`: keeps the most recent N headlines per symbol.
- `sort_values(["date", "symbol"]).reset_index(drop=True)`: returns a clean chronological DataFrame.

### Inferring Price Dates

```python
def infer_price_dates(news_df, start_date=None, end_date=None, warmup_days=60, post_news_days=365):
    if news_df.empty and (not start_date or not end_date):
        raise ValueError("Cannot infer dates because no Kaggle headlines matched the selected symbols")

    signal_start = pd.to_datetime(start_date) if start_date else news_df["date"].min()
    signal_end = pd.to_datetime(end_date) if end_date else news_df["date"].max()

    if pd.isna(signal_start) or pd.isna(signal_end) or signal_start > signal_end:
        raise ValueError("Invalid date range after filtering Kaggle headlines")

    price_start = signal_start - timedelta(days=max(warmup_days, 0))
    price_end = signal_end + timedelta(days=max(post_news_days, 1))
    yahoo_end = (price_end + timedelta(days=1)).strftime("%Y-%m-%d")
```

What each line/block does:

- Empty news plus missing explicit dates is rejected because the script cannot infer a backtest window.
- `signal_start`: first date where signals should be evaluated.
- `signal_end`: last date where news/sentiment directly exists.
- The invalid-date guard prevents inverted or missing ranges.
- `price_start`: pulls earlier prices so rolling momentum and mean reversion indicators have enough history.
- `price_end`: extends prices after news so the strategy can hold and evaluate positions after the final headline.
- `yahoo_end`: adds one extra day because Yahoo daily downloads commonly treat the end date as exclusive.

### Loading Prices

```python
price_df = YahooFinanceLoader(
    symbols=symbols,
    start_date=start_date,
    end_date=yahoo_end_date,
).load()

full_price_data = PriceData(price_df)
full_close_prices = full_price_data.get_close_prices()
price_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_price_data.csv")

backtest_price_df = slice_price_frame(price_df, news_start_date, display_end_date)
price_data = PriceData(backtest_price_df)
close_prices = price_data.get_close_prices()
```

What each line/block does:

- `YahooFinanceLoader(...).load()`: downloads historical OHLCV data for selected symbols.
- `PriceData(price_df)`: validates and wraps raw price data.
- `get_close_prices()`: creates the date-by-symbol close-price matrix used by signals and returns.
- `to_csv(...)`: saves the loaded price data for auditability.
- `slice_price_frame(...)`: removes warmup-only rows from the actual backtest period.
- The second `PriceData(...)`: wraps the backtest-only price range.
- The second `get_close_prices()`: provides the exact index/columns used by sentiment and backtest weights.

## Signal Generation

### Momentum Signal

File: `src/jsf/signals/technical.py`

```python
close_prices = price_data.get_close_prices()
momentum = close_prices.pct_change(periods=self.lookback)

if self.normalize:
    signal = np.tanh(momentum * 10)
else:
    signal = momentum

return signal
```

What each line/block does:

- `get_close_prices()`: gets prices in matrix form.
- `pct_change(periods=self.lookback)`: calculates the percentage price change over the lookback window.
- Positive values mean price rose over the lookback; negative values mean price fell.
- `np.tanh(momentum * 10)`: compresses large values into the range `[-1, 1]` while preserving sign.
- `return signal`: returns one numeric signal per date and symbol.

Trade implication:

- High positive momentum increases the chance the symbol receives a positive portfolio weight.
- Negative momentum is removed by the demo's long-only weight builder.

### Mean Reversion Signal

File: `src/jsf/signals/statistical.py`

```python
close_prices = price_data.get_close_prices()
rolling_mean = close_prices.rolling(window=self.lookback).mean()
rolling_std = close_prices.rolling(window=self.lookback).std()
z_score = (close_prices - rolling_mean) / rolling_std
signal = -np.tanh(z_score / self.entry_threshold)
return signal
```

What each line/block does:

- `get_close_prices()`: gets the close-price matrix.
- `rolling(...).mean()`: calculates the moving average for each symbol.
- `rolling(...).std()`: calculates moving volatility around that average.
- `z_score`: measures how far price is above or below its rolling mean.
- `-np.tanh(...)`: inverts the z-score so below-average prices become positive signals and above-average prices become negative signals.
- `return signal`: returns a normalized mean-reversion score per date and symbol.

Trade implication:

- If price is unusually low versus its recent average, mean reversion contributes a positive buy-like signal.
- If price is unusually high versus its recent average, mean reversion contributes a negative avoid/sell-like signal.

### Baseline Signal

File: `demos/demo_real_data_finbert_backtest.py`

```python
momentum = MomentumSignal(lookback=20).generate(full_price_data)
mean_reversion = MeanReversionSignal(lookback=10).generate(full_price_data)
baseline_signal = (0.6 * momentum + 0.4 * mean_reversion).replace([np.inf, -np.inf], 0.0)
baseline_signal = baseline_signal.reindex(index=close_prices.index, columns=close_prices.columns)
baseline_signal = baseline_signal.fillna(0.0)
```

What each line/block does:

- `MomentumSignal(lookback=20)`: creates a 20-period momentum indicator.
- `.generate(full_price_data)`: computes momentum on the full price range including warmup.
- `MeanReversionSignal(lookback=10)`: creates a 10-period mean-reversion indicator.
- `.generate(full_price_data)`: computes mean reversion on the same full price range.
- `0.6 * momentum + 0.4 * mean_reversion`: blends the two price signals, giving momentum more influence.
- `replace([np.inf, -np.inf], 0.0)`: removes invalid infinite values from division/rolling calculations.
- `reindex(...)`: aligns the signal exactly to the backtest date range and tradable symbols.
- `fillna(0.0)`: converts missing signal values to neutral.

Trade implication:

- A symbol is eligible for a long position only when this combined score is positive and later passes the threshold.

## FinBERT Sentiment Signal

### Headline Scoring

```python
from jsf.ml import FinBERT

finbert = FinBERT(use_mock=False)
results = finbert.predict(news_df["text"].tolist())

sentiment_df = news_df.copy()
sentiment_df["sentiment_label"] = [result.label.value for result in results]
sentiment_df["sentiment_score"] = [
    result.score
    if result.label.value == "positive"
    else -result.score
    if result.label.value == "negative"
    else 0.0
    for result in results
]
```

What each line/block does:

- `from jsf.ml import FinBERT`: imports the FinBERT wrapper.
- `FinBERT(use_mock=False)`: requests the real HuggingFace-backed model instead of a mock.
- `predict(news_df["text"].tolist())`: runs inference over headline text.
- `sentiment_df = news_df.copy()`: preserves original headline data and appends model output.
- `sentiment_label`: stores the raw class label: `positive`, `neutral`, or `negative`.
- `sentiment_score`: converts labels into signed numeric scores.
- Positive headlines become `+confidence`.
- Negative headlines become `-confidence`.
- Neutral headlines become `0.0`.

### Daily Sentiment Feature

```python
daily_sentiment = sentiment_df.groupby(["date", "symbol"])["sentiment_score"].mean().unstack()
daily_sentiment = daily_sentiment.reindex(index=close_prices.index, columns=close_prices.columns)
sentiment_signal = daily_sentiment.fillna(0.0).rolling(window=3, min_periods=1).mean()
sentiment_signal = sentiment_signal.where(sentiment_signal.abs() >= 0.15, 0.0).fillna(0.0)
```

What each line/block does:

- `groupby(["date", "symbol"])`: groups all headlines for the same symbol on the same date.
- `["sentiment_score"].mean()`: averages multiple headline scores into one daily score.
- `.unstack()`: converts the grouped result into a date-by-symbol matrix.
- `reindex(...)`: aligns sentiment with the exact trading calendar and symbols from price data.
- `fillna(0.0)`: assigns neutral sentiment when no headline exists on a trading date.
- `rolling(window=3, min_periods=1).mean()`: smooths sentiment over three trading rows.
- `where(abs >= 0.15, 0.0)`: removes weak sentiment values whose absolute score is below `0.15`.
- Final `fillna(0.0)`: removes any remaining missing values.

Trade implication:

- Strong positive sentiment increases the final combined signal.
- Strong negative sentiment decreases the final combined signal.
- Weak or missing sentiment contributes zero.

### Sentiment-Enhanced Signal

```python
sentiment_enhanced_signal = (0.75 * baseline_signal + 0.25 * sentiment_signal)
sentiment_enhanced_signal = sentiment_enhanced_signal.replace([np.inf, -np.inf], 0.0).fillna(0.0)
```

What each line/block does:

- `0.75 * baseline_signal`: keeps the technical signal as the dominant component.
- `0.25 * sentiment_signal`: adds sentiment as a secondary feature.
- `replace(...)`: removes infinite values.
- `fillna(0.0)`: converts missing values to neutral.

Trade implication:

- A positive technical setup can be strengthened by positive sentiment.
- A positive technical setup can be weakened or removed by negative sentiment.
- A weak positive technical setup can cross the trading threshold if sentiment is positive enough.

## Converting Signals Into Trades

### Long-Only Weight Builder

Files:

- `demos/demo_real_data_finbert_backtest.py`
- `demos/demo_full_real_data.py`

```python
def build_long_only_weights(signals, threshold=0.05):
    positive = signals.clip(lower=0.0).where(signals >= threshold, 0.0)
    weights = positive.div(positive.sum(axis=1), axis=0).fillna(0.0)
    return weights.shift(1).fillna(0.0)
```

What each line does:

- `def build_long_only_weights(...)`: defines the function that turns raw signal scores into target holdings.
- `threshold=0.05`: requires a signal of at least `0.05` to become tradable.
- `signals.clip(lower=0.0)`: removes negative values because the demo is long-only.
- `.where(signals >= threshold, 0.0)`: sets weak positive values below threshold to zero.
- `positive.sum(axis=1)`: calculates the total positive signal strength for each date.
- `positive.div(..., axis=0)`: normalizes each row so active positions sum to 100% exposure.
- `fillna(0.0)`: handles days where there are no active signals and division creates `NaN`.
- `weights.shift(1)`: trades one period after the signal is known, reducing lookahead bias.
- Final `fillna(0.0)`: makes the first shifted row zero because there was no previous signal.

Example:

```text
Raw signal today:
AAPL = 0.20, MSFT = 0.10, TSLA = -0.30, AMZN = 0.02

After long-only threshold:
AAPL = 0.20, MSFT = 0.10, TSLA = 0.00, AMZN = 0.00

After normalization:
AAPL = 0.6667, MSFT = 0.3333, TSLA = 0.0000, AMZN = 0.0000

After shift:
These target weights are used on the next trading date.
```

### Running The Signal Backtest

```python
def run_signal_backtest(name, signals, price_data):
    weights = build_long_only_weights(signals)
    portfolio = Portfolio(weights=weights, metadata={"name": name})
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            transaction_cost=0.001,
            slippage=0.0005,
        )
    )
    result = engine.run(portfolio, price_data)
    metrics = calculate_all_metrics(result.returns.fillna(0.0))
    metrics["total_return"] = result.total_return
    metrics["cagr"] = result.cagr
    metrics["volatility"] = result.volatility
    metrics["sharpe_ratio"] = result.sharpe_ratio
    metrics["max_drawdown"] = result.max_drawdown
    metrics["final_value"] = result.equity_curve.iloc[-1]
    metrics["total_trades"] = len(result.trades)
    return result, metrics
```

What each line/block does:

- `weights = build_long_only_weights(signals)`: converts signal scores into target allocation percentages.
- `Portfolio(weights=weights, metadata={"name": name})`: wraps target weights into the portfolio object expected by the engine.
- `BacktestEngine(...)`: creates the simulator.
- `initial_capital=INITIAL_CAPITAL`: starts with `100,000` demo currency units.
- `transaction_cost=0.001`: charges 10 basis points on turnover.
- `slippage=0.0005`: charges 5 basis points of additional execution drag.
- `engine.run(portfolio, price_data)`: simulates position changes and equity movement.
- `calculate_all_metrics(...)`: computes performance statistics from returns.
- The metric assignments overwrite/add headline metrics directly from `BacktestResult` so they match the cost-adjusted equity curve.
- `total_trades = len(result.trades)`: counts recorded position changes.
- `return result, metrics`: gives callers both detailed time series and summary stats.

## Backtest Engine Internals

File: `src/jsf/simulation/backtest.py`

### Configuration

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    margin_requirement: float = 1.0
    compound_returns: bool = True
    rebalance_on_signal: bool = True
```

What each line does:

- `@dataclass`: automatically creates initializer and representation methods.
- `initial_capital`: starting account value.
- `transaction_cost`: proportional cost charged when weights change.
- `slippage`: extra proportional cost representing worse execution than the observed close.
- `margin_requirement`: reserved for leverage/margin control; current demo uses no leverage.
- `compound_returns`: if true, returns compound on current equity.
- `rebalance_on_signal`: configuration flag for signal-driven rebalancing; the current `run()` uses the supplied weight matrix directly.

Validation:

```python
def __post_init__(self):
    if self.initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if self.transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative")
    if self.slippage < 0:
        raise ValueError("slippage must be non-negative")
```

This prevents nonsensical starting capital or negative cost assumptions.

### Main Simulation Loop

```python
weights = portfolio.weights
prices = price_data.get_close_prices()
common_dates = weights.index.intersection(prices.index)
weights = weights.loc[common_dates]
prices = prices.loc[common_dates]
asset_returns = prices.pct_change()

equity = self.config.initial_capital
equity_curve = []
portfolio_returns = []
trades_list = []
previous_weights = pd.Series(0, index=weights.columns)
```

What each line/block does:

- `portfolio.weights`: reads target allocation percentages for each date and symbol.
- `get_close_prices()`: gets the market prices used to calculate returns.
- `intersection(...)`: keeps only dates present in both weights and prices.
- `loc[common_dates]`: aligns both matrices to the same dates.
- `prices.pct_change()`: calculates daily asset returns.
- `equity`: initializes account value.
- `equity_curve`: stores account value over time.
- `portfolio_returns`: stores realized strategy return each period.
- `trades_list`: stores trade records generated by weight changes.
- `previous_weights`: starts with zero holdings in every symbol.

The per-date loop:

```python
for date in common_dates:
    target_weights = weights.loc[date]
    position_changes = (target_weights - previous_weights).abs()
    turnover = position_changes.sum() / 2

    if turnover > 0:
        cost = equity * turnover * (
            self.config.transaction_cost + self.config.slippage
        )
        equity -= cost

        for symbol in position_changes[position_changes > 0].index:
            trades_list.append({
                'date': date,
                'symbol': symbol,
                'change': position_changes[symbol],
                'cost': cost * (position_changes[symbol] / position_changes.sum()),
            })
```

What each line/block does:

- `for date in common_dates`: walks through the backtest chronologically.
- `target_weights = weights.loc[date]`: gets desired holdings for this date.
- `target_weights - previous_weights`: calculates how much each symbol's allocation changed.
- `.abs()`: treats buys and sells as positive turnover magnitude.
- `turnover = position_changes.sum() / 2`: estimates portfolio turnover. Dividing by two avoids double-counting when reallocating from one asset to another.
- `if turnover > 0`: only charges costs and records trades when the portfolio changes.
- `cost = equity * turnover * (...)`: computes cost as current equity multiplied by turnover and total cost rate.
- `equity -= cost`: immediately subtracts trading cost.
- `for symbol in position_changes[position_changes > 0].index`: creates one trade record per changed symbol.
- `'date'`: the date of the rebalance.
- `'symbol'`: the affected symbol.
- `'change'`: absolute target-weight change, not share quantity.
- `'cost'`: the symbol's proportional share of total rebalance cost.

Important detail:

The backtest `trades` output is a rebalance ledger. It does not store buy/sell side or share quantity. It records that the symbol's portfolio weight changed by a certain amount and assigns transaction cost to that change.

Return application:

```python
if date in asset_returns.index:
    period_returns = asset_returns.loc[date]
    portfolio_return = (previous_weights * period_returns).sum()

    if self.config.compound_returns:
        equity *= (1 + portfolio_return)
    else:
        equity += self.config.initial_capital * portfolio_return

    portfolio_returns.append(portfolio_return)
else:
    portfolio_returns.append(0.0)

equity_curve.append(equity)
previous_weights = target_weights.copy()
```

What each line/block does:

- `period_returns = asset_returns.loc[date]`: gets each symbol's return for the current date.
- `previous_weights * period_returns`: applies returns to positions held before the rebalance target becomes the new previous state.
- `.sum()`: aggregates asset-level returns into portfolio return.
- `compound_returns`: if true, multiplies current equity by `1 + return`.
- `else`: if false, applies return to original capital instead.
- `portfolio_returns.append(...)`: records the strategy return for this period.
- `equity_curve.append(equity)`: stores account value after cost and return.
- `previous_weights = target_weights.copy()`: today's targets become the held weights for the next period.

Result construction:

```python
equity_series = pd.Series(equity_curve, index=common_dates)
returns_series = pd.Series(portfolio_returns, index=common_dates)
trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

result = BacktestResult(
    equity_curve=equity_series,
    returns=returns_series,
    positions=weights,
    trades=trades_df,
    metadata={...}
)
```

What each line/block does:

- `equity_series`: creates a dated account-value time series.
- `returns_series`: creates a dated return time series.
- `trades_df`: stores all rebalance records or an empty DataFrame.
- `BacktestResult(...)`: packages outputs consumed by demos, dashboards, and metrics.
- `positions=weights`: stores target weights, not broker positions or share quantities.
- `metadata`: records simulation assumptions such as capital, cost, slippage, and periods.

## How A Backtest Trade Is Made

In the demo backtests, a trade is made by a target-weight change. No broker order is created.

Step-by-step:

1. A signal matrix is calculated for every date and symbol.
2. Negative and weak signals are set to zero.
3. Remaining positive signals are normalized into portfolio weights.
4. The weights are shifted one day.
5. The backtest engine compares today's target weights with yesterday's target weights.
6. Any non-zero absolute difference becomes a trade record.
7. Cost is deducted using `equity * turnover * (transaction_cost + slippage)`.
8. Portfolio return is calculated using previous weights and today's price returns.
9. Equity is updated and saved.

Example:

```text
Yesterday held weights:
AAPL = 0.00, MSFT = 0.00

Today target weights:
AAPL = 0.60, MSFT = 0.40

Position changes:
AAPL = 0.60, MSFT = 0.40

Turnover:
(0.60 + 0.40) / 2 = 0.50

Cost on 100,000 equity with 0.0015 total cost rate:
100,000 * 0.50 * 0.0015 = 75

Trade records:
AAPL change 0.60, assigned cost 45
MSFT change 0.40, assigned cost 30
```

## Strategy And Portfolio Constructor Path

Some code uses the reusable strategy abstraction instead of the demo-specific `build_long_only_weights()` helper.

### Strategy Base Class

File: `src/jsf/strategies/base.py`

```python
def run(self, price_data: PriceData, **kwargs: Any) -> Portfolio:
    signals = self.generate_signals(price_data, **kwargs)
    portfolio = self.construct_portfolio(signals, price_data, **kwargs)
    return portfolio
```

What each line does:

- `generate_signals(...)`: strategy-specific logic creates a date-by-symbol signal matrix.
- `construct_portfolio(...)`: converts signals into a `Portfolio` object using a portfolio constructor.
- `return portfolio`: returns weights to the backtest engine or caller.

### Momentum Strategy Template

File: `src/jsf/strategies/templates.py`

```python
signal = MomentumSignal(lookback=lookback)

if portfolio_constructor is None:
    sizer = EqualWeightSizer(long_only=long_only)
    portfolio_constructor = SimplePortfolioConstructor(
        position_sizer=sizer,
        name=f"{name}_constructor"
    )
```

What each line/block does:

- `MomentumSignal(...)`: creates the signal used by the strategy.
- `if portfolio_constructor is None`: chooses a default portfolio builder when the caller does not provide one.
- `EqualWeightSizer(long_only=long_only)`: gives equal weight to every active signal.
- `SimplePortfolioConstructor(...)`: packages sizing, constraints, and optional rebalancing into one constructor.

### Equal Weight Sizing

File: `src/jsf/portfolio/sizing.py`

```python
for idx in weights.index:
    row = weights.loc[idx]
    active = row != 0

    if self.long_only:
        active = active & (row > 0)

    if self.max_positions is not None and active.sum() > self.max_positions:
        top_positions = row.abs().nlargest(self.max_positions).index
        active = row.index.isin(top_positions)
        if self.long_only:
            active = active & (row > 0)

    if active.sum() > 0:
        equal_weight = 1.0 / active.sum()
        result.loc[idx, active] = equal_weight
    else:
        result.loc[idx] = 0.0
```

What each line/block does:

- Iterates over every date.
- Reads the signal row for that date.
- Marks every non-zero signal as active.
- In long-only mode, removes negative signals.
- If `max_positions` is set, keeps only the strongest absolute signals.
- If at least one symbol is active, assigns equal capital to each active symbol.
- If no symbol is active, assigns zero to all symbols.

Difference from the FinBERT demos:

- The reusable `EqualWeightSizer` treats all active signals equally.
- The demo helper uses signal-proportional weights after thresholding.
- The demo helper also shifts by one day; the generic sizer does not shift unless caller logic does so.

## Live/Paper Trading Flow

The live trading path creates actual `Order` objects and sends them to a broker abstraction. With `PaperBroker`, these are simulated fills; with a real broker implementation, they would be real orders.

### Live Engine Cycle

File: `src/jsf/live/engine.py`

```python
def _execute_cycle(self) -> None:
    prices = self.data_handler.get_prices()

    if self.config.require_prices_for_all:
        missing = [s for s in self.symbols if s not in prices]
        if missing:
            return

    if not self._check_risk_limits():
        return

    if self._strategy is not None:
        target_weights = self._strategy(self, prices)
        if target_weights:
            self._rebalance_to_weights(target_weights, prices)

    self._take_snapshot(prices)
    self._emit("on_cycle", {"prices": prices, "timestamp": datetime.now()})
```

What each line/block does:

- `get_prices()`: gets current market prices from the live data handler.
- `require_prices_for_all`: prevents trading if any configured symbol is missing a price.
- `_check_risk_limits()`: stops or pauses trading if risk limits are breached.
- `_strategy(self, prices)`: calls user-provided strategy logic and expects target weights back.
- `if target_weights`: skips rebalancing when the strategy returns nothing.
- `_rebalance_to_weights(...)`: converts target weights into buy/sell orders.
- `_take_snapshot(prices)`: records account, position, price, and state information.
- `_emit("on_cycle", ...)`: notifies callbacks that a cycle completed.

### Rebalancing To Orders

File: `src/jsf/live/engine.py`

```python
account = self.broker.get_account()
positions = self.get_positions()

current_weights = {}
for symbol, position in positions.items():
    if symbol in prices:
        position_value = position.quantity * prices[symbol]
        current_weights[symbol] = position_value / account.equity
```

What each line/block does:

- `get_account()`: reads current equity and cash.
- `get_positions()`: reads current broker positions.
- `current_weights = {}`: prepares a symbol-to-weight map.
- `position.quantity * prices[symbol]`: estimates current market value for each held symbol.
- `/ account.equity`: converts market value into portfolio weight.

Order calculation:

```python
for symbol, target_weight in target_weights.items():
    if symbol not in prices:
        continue

    target_weight = min(target_weight, self.config.max_position_size)
    target_weight = max(target_weight, -self.config.max_position_size)

    current_weight = current_weights.get(symbol, 0.0)
    weight_diff = target_weight - current_weight

    if abs(weight_diff) < 0.01:
        continue

    target_value = target_weight * account.equity
    current_value = current_weight * account.equity
    trade_value = target_value - current_value
```

What each line/block does:

- Iterates over every target symbol from the strategy.
- Skips symbols with no current price.
- Caps target weight to `max_position_size` on both long and short sides.
- Reads the current weight, defaulting to zero if no position exists.
- Calculates the difference between desired and current weights.
- Skips small changes below 1% to avoid noisy micro-trades.
- Converts target and current weights into dollar values.
- `trade_value`: positive means buy, negative means sell.

Order creation:

```python
if abs(trade_value) > self.config.max_order_value:
    trade_value = self.config.max_order_value * (1 if trade_value > 0 else -1)

price = prices[symbol]
quantity = abs(trade_value) / price

if quantity < 0.01:
    continue

side = OrderSide.BUY if trade_value > 0 else OrderSide.SELL
order_type = OrderType.MARKET if self.config.use_market_orders else OrderType.LIMIT

order = Order(
    symbol=symbol,
    side=side,
    quantity=round(quantity, 2),
    order_type=order_type,
    limit_price=price if order_type == OrderType.LIMIT else None,
    time_in_force=self.config.default_time_in_force,
)
```

What each line/block does:

- `max_order_value`: caps any single order's notional size.
- `price = prices[symbol]`: gets current execution reference price.
- `quantity = abs(trade_value) / price`: converts dollar notional into shares/contracts.
- `quantity < 0.01`: skips tiny orders.
- `OrderSide.BUY if trade_value > 0 else SELL`: picks side based on whether the target value is above or below current value.
- `OrderType.MARKET` or `LIMIT`: uses engine configuration to choose order type.
- `Order(...)`: creates the broker order object.
- `round(quantity, 2)`: rounds quantity to two decimals.
- `limit_price=price`: sets a limit price only for limit orders.
- `time_in_force`: applies the default order duration, usually `DAY`.

Submission:

```python
try:
    tracker = self.order_manager.submit_order(order)
    self._emit("on_trade", tracker)
except Exception as e:
    logger.error(f"Failed to submit order for {symbol}: {e}")
```

What each line/block does:

- `submit_order(order)`: sends the order to the `OrderManager`.
- `tracker`: tracks order state, fills, average fill price, and timestamps.
- `_emit("on_trade", tracker)`: notifies callbacks that a trade/order was submitted.
- `except`: logs failures without crashing the whole engine cycle.

## Order Manager Flow

File: `src/jsf/live/order_manager.py`

```python
tracker = OrderTracker(order=order)

with self._lock:
    if len(self._active_order_ids) >= self._max_pending_orders:
        raise OrderManagerError(...)

    self._orders[order.order_id] = tracker
    self._orders_by_symbol[order.symbol].append(order.order_id)
    self._active_order_ids.add(order.order_id)
```

What each line/block does:

- `OrderTracker(order=order)`: creates internal tracking before broker submission.
- `with self._lock`: protects shared order state from concurrent access.
- Pending-order limit check prevents too many active orders.
- `_orders[...] = tracker`: stores tracker by order ID.
- `_orders_by_symbol[...]`: allows symbol-level lookup.
- `_active_order_ids.add(...)`: marks the order as active.

Broker submission:

```python
result = self.broker.submit_order(order)

with self._lock:
    if tracker.state == OrderState.PENDING_SUBMIT:
        tracker.state = OrderState.SUBMITTED
    tracker.submitted_at = datetime.now()
    if result.order:
        tracker.broker_order_id = result.order.order_id
        if result.order.status == OrderStatus.FILLED:
            if tracker.state != OrderState.FILLED:
                tracker.filled_quantity = result.order.filled_quantity
                tracker.average_fill_price = result.order.avg_fill_price
                tracker.state = OrderState.FILLED
                tracker.filled_at = datetime.now()
                self._active_order_ids.discard(order.order_id)
```

What each line/block does:

- `broker.submit_order(order)`: hands the order to the broker implementation.
- The second lock updates tracker state safely.
- If no fill callback already changed state, the tracker becomes `SUBMITTED`.
- `submitted_at`: records when submission completed.
- `broker_order_id`: stores broker-side ID if available.
- If broker returns a filled order, the tracker records filled quantity, average price, fill time, and removes the order from active IDs.

## Paper Broker Execution

File: `src/jsf/broker/paper.py`

### Submitting An Order

```python
if not self._connected:
    return OrderResult(success=False, message="Broker not connected", error_code="NOT_CONNECTED")

if order.symbol not in self._prices:
    return OrderResult(success=False, message=f"No price set for {order.symbol}. Use set_price() first.", error_code="NO_PRICE")

if order.order_id is None:
    order.order_id = f"ord_{uuid.uuid4().hex[:12]}"
order.status = OrderStatus.SUBMITTED
order.updated_at = datetime.now()
self._orders[order.order_id] = order
```

What each line/block does:

- Rejects orders when the broker is not connected.
- Rejects orders when there is no known current price for the symbol.
- Creates an order ID if one is missing.
- Marks the order as submitted.
- Updates the timestamp.
- Stores the order in broker state.

Buying power and share checks:

```python
current_price = self._prices[order.symbol]
if order.side == OrderSide.BUY:
    estimated_cost = self._estimate_cost(order, current_price)
    if estimated_cost > self._cash:
        order.status = OrderStatus.REJECTED
        return OrderResult(success=False, order=order, message="Insufficient funds...", error_code="INSUFFICIENT_FUNDS")

if order.side == OrderSide.SELL:
    position = self._positions.get(order.symbol)
    available = position.quantity if position else 0
    if available < order.quantity:
        order.status = OrderStatus.REJECTED
        return OrderResult(success=False, order=order, message="Insufficient shares...", error_code="INSUFFICIENT_SHARES")
```

What each line/block does:

- `current_price`: fetches the current simulated market price.
- Buy orders estimate total cost including slippage and commission.
- If cost exceeds cash, the order is rejected.
- Sell orders check whether enough shares are currently held.
- If not enough shares are available, the order is rejected.

Fill model selection:

```python
order.status = OrderStatus.ACCEPTED
self._emit("on_order_update", order)

if self.fill_model == "immediate":
    return self._execute_order(order, current_price)
else:
    self._pending_orders.append(order.order_id)
    return OrderResult(success=True, order=order, message="Order accepted, pending execution at next bar")
```

What each line/block does:

- Accepted orders are broadcast to callbacks.
- With `immediate`, the broker fills the order at the current price.
- With `next_bar`, the broker queues the order until a later price update.

## Backtest Trades vs Live Orders

| Aspect | Backtest Demo | Live/Paper Engine |
| --- | --- | --- |
| Trade trigger | Target weight changes | Strategy target weight differs from broker position weight |
| Execution object | Row in `result.trades` | `Order` tracked by `OrderTracker` |
| Quantity | Weight-change magnitude | Share/contract quantity |
| Side | Not stored directly | Explicit `BUY` or `SELL` |
| Price | Only used for returns | Used for order sizing and paper fills |
| Cost | `equity * turnover * cost_rate` | Broker commission/slippage model |
| Fill | Assumed by weight transition | Accepted/rejected/filled order lifecycle |

## Saved Outputs

The FinBERT demos save reproducibility artifacts in `demos/`:

- `*_price_data.csv`: raw downloaded price data.
- `*_news_data.csv`: headline data used for sentiment.
- `*_finbert_sentiment.csv`: headline-level FinBERT labels and signed scores.
- `*_sentiment_feature.csv`: daily aligned sentiment signal.
- `*_sentiment_enhanced_signal.csv`: final technical-plus-sentiment signal.
- `*_baseline_equity.csv`: price-only baseline equity curve.
- `*_sentiment_equity.csv`: sentiment-enhanced equity curve.
- `*_baseline_positions.csv`: baseline target weights.
- `*_sentiment_positions.csv`: sentiment-enhanced target weights.
- `*_baseline_trades.csv`: baseline rebalance ledger.
- `*_sentiment_trades.csv`: sentiment-enhanced rebalance ledger.
- `*_comparison.csv`: side-by-side metrics.

## Practical Interpretation

The strategy is not saying “buy because FinBERT is positive” by itself. It says:

```text
Buy/hold a symbol only if the final combined score is positive enough relative to the other symbols.
```

The final score is mostly technical in the current demo because the sentiment-enhanced signal weights are `75%` technical and `25%` sentiment. FinBERT can still change trades by pushing a symbol above or below the `0.05` eligibility threshold, or by changing its normalized share of the active long-only portfolio.

The historical backtest's `trades` file should be read as “portfolio rebalance events,” not literal exchange order reports. Literal order reports are produced only in the live/paper path through `LiveTradingEngine`, `OrderManager`, and `Broker` implementations.

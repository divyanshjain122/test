# Result Log RAG Pipeline

Simple CSV-based RAG utility for explaining backtest result logs.

It reads a `resultLogs/...` run directory, detects material portfolio weight changes, retrieves nearby stock news and sentiment rows, and generates a trade summary with reasoning.

## Usage

Run with the built-in extractive fallback:

```bash
python3 rag_pipeline/summarize_result_log.py \
  --log-dir resultLogs/2026-05-09_15-03-11rf-200-8-regression-6525 \
  --strategy tech_sentiment \
  --no-llm
```

Run with an OpenAI-compatible external LLM:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4o-mini"

python3 rag_pipeline/summarize_result_log.py \
  --log-dir resultLogs/2026-05-09_15-03-11rf-200-8-regression-6525 \
  --strategy tech_sentiment \
  --question "Summarize trades made and explain likely reasoning from news sentiment"
```

For another OpenAI-compatible provider, set `OPENAI_BASE_URL`:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Strategies

Supported strategy names:

- `baseline_tech`
- `tech_sentiment`
- `ml_only`
- `ml_sentiment`

You can also pass a custom weight CSV stem if it follows the same format as `*_daily_holding_weights.csv`.

## Inputs

The script uses these files from the log directory when available:

- `date_wise_stock_news.csv`
- `date_wise_stock_news_summary.csv`
- `daily_sentiment_scores.csv`
- `{strategy}_daily_holding_weights.csv`
- `four_way_strategy_comparison.csv`
- `run_details.json`

## Outputs

The script writes these files into the same log directory:

- `rag_trade_summary.md`
- `rag_trade_summary.json`

## Useful Options

```bash
--lookback-days 3       # news window before trade date
--max-events 25         # max trade events sent to the LLM
--max-articles 5        # max headlines per trade event
--min-change-pct 5.0    # minimum weight change treated as a material trade
--no-llm                # skip external LLM
```

## Notes

The reasoning is evidence-bound to the logs. The script treats news as contextual evidence for explaining trades; it does not prove that the strategy directly used any particular article unless your strategy logs explicitly show that linkage.

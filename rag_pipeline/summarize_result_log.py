import argparse
import json
import os
import textwrap
from datetime import timedelta
from pathlib import Path
from urllib import request, error

import pandas as pd


WEIGHT_FILE_BY_STRATEGY = {
    "baseline_tech": "baseline_tech_daily_holding_weights.csv",
    "tech_sentiment": "tech_sentiment_daily_holding_weights.csv",
    "ml_only": "ml_only_daily_holding_weights.csv",
    "ml_sentiment": "ml_sentiment_daily_holding_weights.csv",
}


def load_csv(path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_run_context(log_dir, strategy):
    weight_file = WEIGHT_FILE_BY_STRATEGY.get(strategy, strategy)
    if not weight_file.endswith(".csv"):
        weight_file = f"{weight_file}_daily_holding_weights.csv"

    context = {
        "log_dir": log_dir,
        "strategy": strategy,
        "weights": load_csv(log_dir / weight_file),
        "news": load_csv(log_dir / "date_wise_stock_news.csv", required=False),
        "news_summary": load_csv(log_dir / "date_wise_stock_news_summary.csv", required=False),
        "sentiment": load_csv(log_dir / "daily_sentiment_scores.csv", required=False),
        "metrics": load_csv(log_dir / "four_way_strategy_comparison.csv", required=False),
        "run_details": {},
    }

    details_path = log_dir / "run_details.json"
    if details_path.exists():
        with details_path.open("r", encoding="utf-8") as file:
            context["run_details"] = json.load(file)

    return context


def normalize_dates(df, column="date"):
    if df.empty or column not in df.columns:
        return df
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce").dt.normalize()
    return df.dropna(subset=[column])


def pivot_weights(weights):
    weights = normalize_dates(weights)
    required = {"date", "symbol", "holding_weight_pct"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"Weights file missing columns: {sorted(missing)}")

    matrix = weights.pivot_table(
        index="date",
        columns="symbol",
        values="holding_weight_pct",
        aggfunc="last",
    )
    return matrix.sort_index().fillna(0.0)


def detect_trade_events(weights_matrix, min_change_pct):
    deltas = weights_matrix.diff().fillna(weights_matrix)
    events = []

    for date, row in deltas.iterrows():
        for symbol, change in row.items():
            if abs(change) < min_change_pct:
                continue

            previous_weight = weights_matrix.at[date, symbol] - change
            new_weight = weights_matrix.at[date, symbol]
            if previous_weight <= 0 and new_weight > 0:
                action = "new_position"
            elif previous_weight > 0 and new_weight <= 0:
                action = "exit"
            elif change > 0:
                action = "increase"
            else:
                action = "reduce"

            events.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "action": action,
                    "previous_weight_pct": round(float(previous_weight), 4),
                    "new_weight_pct": round(float(new_weight), 4),
                    "change_pct": round(float(change), 4),
                }
            )

    return sorted(events, key=lambda item: (item["date"], item["symbol"]))


def build_metric_summary(metrics, strategy):
    if metrics.empty:
        return {}

    metrics = metrics.copy()
    first_col = metrics.columns[0]
    metrics = metrics.rename(columns={first_col: "metric"})

    strategy_column = {
        "baseline_tech": "Baseline",
        "tech_sentiment": "Tech_Sentiment",
        "ml_only": "ML_Only",
        "ml_sentiment": "ML_Sentiment",
    }.get(strategy, strategy)

    if strategy_column not in metrics.columns:
        return {}

    selected = metrics.set_index("metric")[strategy_column].to_dict()
    keys = ["final_value", "net_total_return", "gross_total_return", "sharpe_ratio", "max_drawdown", "volatility", "win_rate"]
    return {key: selected.get(key) for key in keys if key in selected}


def retrieve_news_for_event(event, news, news_summary, sentiment, lookback_days, max_articles):
    symbol = event["symbol"]
    event_date = event["date"]
    start_date = event_date - timedelta(days=lookback_days)

    articles = pd.DataFrame()
    if not news.empty:
        news = normalize_dates(news)
        articles = news[
            (news["symbol"] == symbol)
            & (news["date"] >= start_date)
            & (news["date"] <= event_date)
        ].copy()
        if not articles.empty:
            articles = articles.drop_duplicates(subset=["date", "symbol", "source", "text"])
            if "sentiment_score" in articles.columns:
                articles["abs_sentiment"] = articles["sentiment_score"].abs()
            else:
                articles["abs_sentiment"] = 0.0
            articles["days_from_trade"] = (event_date - articles["date"]).dt.days
            articles = articles.sort_values(["days_from_trade", "abs_sentiment"], ascending=[True, False])
            articles = articles.head(max_articles)

    summary = pd.DataFrame()
    if not news_summary.empty:
        news_summary = normalize_dates(news_summary)
        summary = news_summary[
            (news_summary["symbol"] == symbol)
            & (news_summary["date"] >= start_date)
            & (news_summary["date"] <= event_date)
        ].copy()
        summary = summary.sort_values("date")

    sentiment_rows = pd.DataFrame()
    if not sentiment.empty:
        sentiment = normalize_dates(sentiment)
        sentiment_rows = sentiment[
            (sentiment["symbol"] == symbol)
            & (sentiment["date"] >= start_date)
            & (sentiment["date"] <= event_date)
        ].copy()
        sentiment_rows = sentiment_rows.sort_values("date")

    return {
        "articles": articles,
        "summary": summary,
        "sentiment": sentiment_rows,
    }


def format_event_context(event, retrieved):
    lines = [
        f"Trade date: {event['date'].date()} | Symbol: {event['symbol']} | Action: {event['action']}",
        f"Weight: {event['previous_weight_pct']:.2f}% -> {event['new_weight_pct']:.2f}% ({event['change_pct']:+.2f} pts)",
    ]

    summary = retrieved["summary"]
    if not summary.empty:
        lines.append("Daily news summary:")
        for _, row in summary.iterrows():
            sentiment = row.get("avg_sentiment_score", "")
            lines.append(f"- {row['date'].date()}: {int(row.get('news_count', 0))} articles, avg sentiment {sentiment}")

    signal = retrieved["sentiment"]
    if not signal.empty:
        lines.append("Sentiment signal:")
        for _, row in signal.iterrows():
            lines.append(f"- {row['date'].date()}: {row.get('sentiment_score', '')}")

    articles = retrieved["articles"]
    if not articles.empty:
        lines.append("Relevant headlines:")
        for _, row in articles.iterrows():
            source = row.get("source", "")
            score = row.get("sentiment_score", "")
            lines.append(f"- {row['date'].date()} | {source} | sentiment {score}: {row.get('text', '')}")

    return "\n".join(lines)


def build_prompt(question, run_details, metric_summary, event_contexts):
    run_text = json.dumps(run_details, indent=2, default=str) if run_details else "No run details available."
    metric_text = json.dumps(metric_summary, indent=2, default=str) if metric_summary else "No metric summary available."
    context_text = "\n\n".join(event_contexts) if event_contexts else "No material trade events were detected."

    return f"""
You are an investment research assistant. Use only the provided backtest logs and retrieved news context.
Do not claim that the strategy actually read these articles unless the logs prove it. Phrase news links as likely context or possible explanation.

User question:
{question}

Run details:
{run_text}

Selected strategy metrics:
{metric_text}

Retrieved trade and news context:
{context_text}

Return a concise markdown report with these sections:
1. Executive Summary
2. Trades Made
3. News/Sentiment Reasoning
4. Performance Context
5. Caveats
""".strip()


def call_openai_compatible_llm(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You summarize quantitative trading logs with careful, evidence-bound reasoning."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: {exc.code} {detail}") from exc

    return body["choices"][0]["message"]["content"]


def fallback_summary(strategy, metric_summary, events, event_contexts):
    action_counts = pd.Series([event["action"] for event in events]).value_counts().to_dict() if events else {}
    largest_events = sorted(events, key=lambda item: abs(item["change_pct"]), reverse=True)[:10]

    lines = [
        "# RAG Trade Summary",
        "",
        "LLM summary was not used because `OPENAI_API_KEY` is not set. This is an extractive fallback from the retrieved logs.",
        "",
        "## Executive Summary",
        f"Strategy: `{strategy}`.",
        f"Detected material trade events: {len(events)}.",
        f"Action counts: {json.dumps(action_counts)}.",
        "",
        "## Performance Context",
        json.dumps(metric_summary, indent=2, default=str),
        "",
        "## Largest Trades",
    ]

    for event in largest_events:
        lines.append(
            f"- {event['date'].date()} {event['symbol']} {event['action']}: "
            f"{event['previous_weight_pct']:.2f}% -> {event['new_weight_pct']:.2f}% "
            f"({event['change_pct']:+.2f} pts)"
        )

    lines.extend(["", "## Retrieved Evidence", ""])
    for context in event_contexts[:10]:
        lines.append("```text")
        lines.append(context)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def write_outputs(log_dir, report, payload):
    report_path = log_dir / "rag_trade_summary.md"
    json_path = log_dir / "rag_trade_summary.json"
    report_path.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return report_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize resultLogs trades with a simple CSV RAG pipeline.")
    parser.add_argument("--log-dir", required=True, type=Path, help="Path to a resultLogs run directory.")
    parser.add_argument(
        "--strategy",
        default="tech_sentiment",
        help="One of baseline_tech, tech_sentiment, ml_only, ml_sentiment, or a weight CSV stem/path name.",
    )
    parser.add_argument(
        "--question",
        default="Summarize trades made and explain likely reasoning using retrieved news and sentiment logs.",
    )
    parser.add_argument("--lookback-days", type=int, default=3, help="News lookback window before each trade date.")
    parser.add_argument("--max-events", type=int, default=25, help="Maximum trade events passed to the LLM.")
    parser.add_argument("--max-articles", type=int, default=5, help="Maximum article headlines retrieved per trade event.")
    parser.add_argument("--min-change-pct", type=float, default=5.0, help="Minimum weight change in percentage points to treat as a trade.")
    parser.add_argument("--no-llm", action="store_true", help="Skip external LLM and use extractive fallback summary.")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    context = load_run_context(log_dir, args.strategy)
    weights_matrix = pivot_weights(context["weights"])
    events = detect_trade_events(weights_matrix, args.min_change_pct)
    selected_events = sorted(events, key=lambda item: abs(item["change_pct"]), reverse=True)[: args.max_events]
    selected_events = sorted(selected_events, key=lambda item: item["date"])

    event_contexts = []
    for event in selected_events:
        retrieved = retrieve_news_for_event(
            event,
            context["news"],
            context["news_summary"],
            context["sentiment"],
            args.lookback_days,
            args.max_articles,
        )
        event_contexts.append(format_event_context(event, retrieved))

    metric_summary = build_metric_summary(context["metrics"], args.strategy)
    prompt = build_prompt(args.question, context["run_details"], metric_summary, event_contexts)

    report = None if args.no_llm else call_openai_compatible_llm(prompt)
    if report is None:
        report = fallback_summary(args.strategy, metric_summary, selected_events, event_contexts)

    payload = {
        "strategy": args.strategy,
        "question": args.question,
        "lookback_days": args.lookback_days,
        "min_change_pct": args.min_change_pct,
        "events_considered": selected_events,
        "metric_summary": metric_summary,
        "prompt_preview": textwrap.shorten(prompt, width=4000, placeholder="\n...[truncated]"),
    }
    report_path, json_path = write_outputs(log_dir, report, payload)
    print(f"Wrote markdown summary: {report_path}")
    print(f"Wrote JSON context: {json_path}")


if __name__ == "__main__":
    main()

# Data Cleaning, Preprocessing, and Feature Extraction Methodology

## 1. Overview

This report describes the data cleaning, preprocessing, and feature extraction methodology used in the portfolio construction framework. The system combines historical market data, financial news sentiment, technical indicators, and machine learning models to compare multiple portfolio construction approaches.

The major approaches evaluated are:

1. Baseline technical strategy
2. Technical strategy enhanced with FinBERT sentiment
3. Machine-learning-only strategy
4. Machine-learning strategy enhanced with FinBERT sentiment
5. Related Random Forest and XGBoost hyperparameter-tuning experiments

The methodology is designed to support financial time-series research by emphasizing temporal alignment, prevention of lookahead bias, valid feature construction, and realistic signal-to-portfolio conversion.

## 2. Market Data Cleaning and Preprocessing

### 2.1 Market Data Source

Historical market data is obtained from Yahoo Finance. The dataset contains standard OHLCV fields:

- Open price
- High price
- Low price
- Close price
- Volume

The analysis is performed on a fixed universe of large-cap stocks across multiple sectors, including technology, financial services, consumer-related stocks, energy, and industrial-related names. Using a fixed universe ensures that the comparison across strategies is performed on the same investable asset set.

### 2.2 Corporate Action Adjustment

The market data is downloaded in adjusted form. This means that prices are adjusted for corporate actions such as stock splits and dividends.

This adjustment is necessary because unadjusted price series may contain artificial jumps. For example, a stock split may appear as a sharp price decline even though shareholder value has not changed. Such artificial discontinuities would distort return calculations, volatility estimates, momentum indicators, and machine learning features.

### 2.3 Date Filtering and Warmup Period

The data-loading period begins before the actual backtest start date. This additional historical period is used as a warmup window.

The warmup period is required because many features depend on rolling historical windows, including:

- 20-day momentum
- 60-day momentum
- 120-day momentum
- Rolling volatility
- Rolling z-score mean reversion
- Moving-average crossover indicators
- Lagged features

Without sufficient warmup history, early backtest observations would contain missing or unstable feature values. The warmup period reduces initialization bias and allows rolling indicators to be computed reliably before the evaluation period begins.

### 2.4 Market Data Validation

The market data is validated before feature extraction. The validation process checks that:

- The dataset is not empty.
- Required OHLCV fields are available.
- Price fields do not contain negative values.
- Volume does not contain negative values.
- Missing values are identified and reported.

This step is important because invalid prices or volumes can produce misleading returns, volatility estimates, and trading signals.

### 2.5 Panel Data Structure

The cleaned market data is organized as a stock-date panel. Each observation corresponds to one stock on one trading date.

This structure is appropriate for cross-sectional portfolio research because it allows the framework to compare multiple securities on the same date and to train machine learning models on pooled asset-date observations.

## 3. News Data Cleaning and Preprocessing

### 3.1 News Dataset

The news dataset contains financial headlines or article titles associated with individual stocks. The main fields are:

- Stock symbol
- Article date
- Article title or headline
- Publisher or news source

The headline text is used as the input for sentiment analysis.

### 3.2 Column Standardization

Dataset-specific column names are converted into a common schema:

- Stock ticker becomes `symbol`.
- Article date becomes `date`.
- Article title becomes `text`.
- Publisher becomes `source`.

Column standardization is necessary because downstream processing expects consistent field names. It also improves reproducibility and makes the pipeline independent of the original CSV naming convention.

### 3.3 Stock Symbol Cleaning

Stock symbols are cleaned by:

- Converting tickers to uppercase
- Removing leading and trailing whitespace
- Filtering news records to include only the selected portfolio universe

This step ensures that news observations correctly align with market price observations. It also removes articles for securities that are not part of the experiment.

### 3.4 Date Parsing and Normalization

News dates are parsed into standardized datetime values. Timezone information is removed, and timestamps are normalized to daily frequency.

This is necessary because the portfolio construction and backtesting framework operates on daily market data. Daily normalization ensures that news sentiment can be aggregated by trading date and aligned with daily price observations.

### 3.5 Missing and Invalid Text Handling

Rows with missing dates or missing article text are removed. In the stricter preprocessing workflow, empty strings are also removed after trimming whitespace.

This is required because transformer-based language models require meaningful text input. Missing or blank text would either fail during inference or generate unreliable sentiment classifications.

### 3.6 Duplicate Handling

Duplicate articles are handled either by direct duplicate removal or through text hashing and caching. This prevents repeated headlines from being processed multiple times and from disproportionately affecting the daily sentiment score.

Duplicate control is important because syndicated financial news can appear multiple times across sources or datasets. If duplicates are not controlled, a repeated article could create an artificially strong sentiment signal.

## 4. FinBERT Sentiment Extraction

### 4.1 Model Selection

The framework uses FinBERT, a BERT-based transformer model specialized for financial sentiment classification. FinBERT is appropriate because financial language differs from general-domain language.

Examples of financial language complexity include:

- A reduction in liabilities may be positive.
- Rising volatility may indicate increased risk.
- Earnings beating expectations is generally positive.
- A revenue decline that is smaller than expected may be interpreted positively.

A general sentiment model may misclassify such statements, whereas FinBERT is trained for financial context.

### 4.2 Tokenization

Each headline is tokenized before model inference. The input is truncated to the maximum sequence length supported by the model architecture.

Tokenization converts raw text into numerical token identifiers that can be processed by the transformer model. Truncation ensures that all inputs fit within the model's maximum length constraint.

### 4.3 Sentiment Classification

FinBERT classifies each headline into one of three sentiment categories:

- Positive
- Negative
- Neutral

The model also produces a confidence score for the predicted class. This confidence score is used to determine the strength of the sentiment signal.

### 4.4 Numerical Sentiment Score Construction

The sentiment label is converted into a signed numerical value:

- Positive sentiment is assigned a positive score.
- Negative sentiment is assigned a negative score.
- Neutral sentiment is assigned zero.

The magnitude of the score is determined by the model's confidence. As a result, the sentiment score lies approximately between -1 and +1.

Interpretation:

- Scores close to +1 indicate strong positive sentiment.
- Scores close to -1 indicate strong negative sentiment.
- Scores near 0 indicate neutral or weak sentiment.

### 4.5 Sentiment Caching

A text hash is generated for each headline. The hash is used as a cache key so that previously processed headlines do not need to be passed through FinBERT again.

Caching provides two main benefits:

- It reduces computational cost because transformer inference is expensive.
- It improves reproducibility because the same headline receives the same stored sentiment value across repeated experiments.

### 4.6 Daily Sentiment Aggregation

Multiple articles for the same stock on the same date are aggregated by averaging their sentiment scores. This produces one daily sentiment score for each stock.

The resulting sentiment matrix has the form:

```text
Trading date x Stock symbol
```

Averaging assumes that all available headlines for a stock on a given day contribute equally to the net daily sentiment.

### 4.7 Trading Calendar Alignment

The daily sentiment matrix is aligned with the market price calendar. If a stock has no news on a trading day, the sentiment value is filled with zero.

This treats the absence of news as neutral sentiment rather than missing information. It also ensures that the sentiment signal has the same date-symbol structure as the price-based signals.

### 4.8 Sentiment Smoothing

A 3-day rolling average is applied to the daily sentiment signal.

This smoothing step is motivated by the fact that information from financial news may not be fully incorporated into prices immediately. Market participants may react over multiple days because of delayed attention, repeated media coverage, or gradual information diffusion.

### 4.9 Sentiment Thresholding

Weak sentiment values are removed by applying a minimum absolute threshold. Sentiment values below the threshold are set to zero.

This reduces noise from low-confidence or weakly directional classifications and keeps only materially positive or negative sentiment signals.

## 5. Baseline Technical Signal Construction

The baseline strategy uses a weighted combination of momentum and mean-reversion signals.

### 5.1 Momentum Signal

Momentum is calculated as the historical percentage price change over a fixed lookback window.

Conceptually:

```text
Momentum = Current price relative to price N days ago
```

Positive momentum indicates recent price appreciation, while negative momentum indicates recent underperformance.

A hyperbolic tangent transformation is applied to bound the signal. This prevents extreme price movements from dominating the portfolio allocation process.

### 5.2 Mean-Reversion Signal

The mean-reversion signal measures how far the current price is from its recent rolling average. It is based on:

- Rolling mean
- Rolling standard deviation
- Z-score of current price relative to the rolling distribution

The signal is inverted so that:

- Prices far below the rolling mean generate positive signals.
- Prices far above the rolling mean generate negative signals.

This reflects the hypothesis that short-term deviations from a recent average may partially reverse.

### 5.3 Hybrid Technical Signal

The baseline signal combines momentum and mean reversion using fixed weights:

- 60% momentum
- 40% mean reversion

This hybrid design balances two different market effects. Momentum captures continuation, while mean reversion captures reversal after overextension.

## 6. Machine Learning Feature Extraction

The machine learning strategy uses a broader set of engineered features derived from price behavior, volatility, trend strength, and cross-sectional rankings.

### 6.1 Feature Families

The main machine learning feature groups are:

- Momentum features
- Volatility features
- Trend features
- Mean-reversion features
- Additional price-derived features

These feature groups are designed to capture different dimensions of market behavior.

### 6.2 Momentum Features

Momentum features are calculated over multiple horizons, including short-term, medium-term, and longer-term windows.

The purpose of multi-horizon momentum is to allow the model to distinguish between short-lived movements and persistent trends.

### 6.3 Volatility Features

Volatility features measure rolling return variability and volatility regimes.

These features are important because expected returns and risk conditions are often regime-dependent. A signal that performs well in low-volatility markets may behave differently during high-volatility periods.

### 6.4 Trend Features

Trend features include trend-strength indicators and moving-average crossover indicators.

Trend strength is estimated using rolling linear regression on recent prices. A consistent positive slope indicates an upward trend, while a consistent negative slope indicates a downward trend.

Moving-average crossover features compare shorter-term and longer-term moving averages to identify directional shifts.

### 6.5 Mean-Reversion Features

Mean-reversion features use rolling z-scores to measure how far prices have moved away from recent averages.

These features help the model learn whether large deviations are followed by reversal or continuation.

### 6.6 Lagged Features

Lagged versions of base features are created using prior observations such as 1-day, 5-day, and 10-day lags.

Lagging is essential for preventing lookahead bias. A model predicting future returns must only use information that would have been available at the time of prediction.

### 6.7 Normalized Features

Cross-sectional z-score normalization is applied to feature values.

This standardizes each stock's feature value relative to the other stocks on the same date. It makes values comparable across assets and prevents raw scale differences from dominating model training.

### 6.8 Ranked Features

Cross-sectional percentile ranks are also computed.

Ranks indicate where each stock stands relative to the rest of the universe on a given date. This is useful because portfolio construction is often based on relative attractiveness rather than absolute signal magnitude.

### 6.9 Additional Price-Derived Features

The ML pipeline also includes direct market-derived features such as:

- 1-day return
- 5-day return
- 20-day return
- 20-day realized volatility
- 60-day realized volatility
- Daily high-low price range
- 20-day average range
- Volume ratio relative to average volume

These features provide additional information about recent return behavior, realized risk, intraday uncertainty, and trading activity.

## 7. Target Variable Construction

The supervised machine learning models are trained to predict forward returns.

Conceptually:

```text
Target at time t = Return from time t to time t+1
```

This formulation creates a supervised regression problem where features observed at a given date are used to predict future asset returns.

Forward-return targets are suitable for portfolio construction because predicted returns can be transformed into trading signals and then into portfolio weights.

## 8. Missing Value Handling in Machine Learning

Rows containing missing feature values or missing target values are removed before model training.

Missing values naturally arise from:

- Rolling-window calculations
- Lagged feature construction
- Unavailable high, low, or volume observations
- Forward-return target construction near the end of the sample

Removing incomplete observations ensures that the model is trained only on valid feature-target pairs and prevents unstable early-window values from contaminating the training process.

## 9. Walk-Forward Training Methodology

The ML strategy uses walk-forward training rather than random train-test splitting.

The process is:

1. Use an initial historical window for training.
2. Generate predictions only after sufficient history is available.
3. Retrain the model periodically.
4. At each prediction date, train only on observations from the past.

Walk-forward training is necessary for financial time-series research because random splits can leak future information into the training set. Preserving chronological order provides a more realistic estimate of out-of-sample performance.

## 10. Random Forest Model Approach

### 10.1 Model Type

The main ML approach uses a Random Forest regression model. Random Forest is an ensemble method that trains multiple decision trees and averages their predictions.

### 10.2 Suitability for Financial Features

Random Forest is suitable for this type of tabular financial data because it can:

- Capture nonlinear relationships
- Model interactions among indicators
- Reduce overfitting relative to a single decision tree
- Handle mixed feature types and scales after preprocessing
- Provide relatively stable predictions in noisy datasets

### 10.3 Output Transformation

Predicted returns are converted into bounded trading signals using a nonlinear compression function.

This prevents extreme model predictions from creating excessive portfolio exposure.

### 10.4 Long-Only Constraint

Negative ML signals are clipped to zero. Therefore, the strategy takes only long positions and does not short securities.

This constraint simplifies portfolio construction and reflects a common practical investment restriction.

## 11. XGBoost Experimental Approach

### 11.1 Model Type

The related experimental pipeline evaluates XGBoost regression. XGBoost is a gradient-boosted decision tree model.

Unlike Random Forest, which builds trees independently, XGBoost builds trees sequentially so that each new tree attempts to correct errors made by previous trees.

### 11.2 Hyperparameter Dimensions

The XGBoost experiments vary:

- Number of trees
- Maximum tree depth
- Learning rate
- Sentiment weight

### 11.3 Research Motivation

XGBoost is included because it is often effective for structured tabular datasets. It can capture nonlinear feature interactions and provides explicit controls for model complexity through tree depth, learning rate, subsampling, and number of estimators.

## 12. Signal Fusion Methodology

### 12.1 Technical and Sentiment Fusion

The technical sentiment-enhanced strategy combines the baseline technical signal with the FinBERT sentiment signal.

The technical signal receives the majority weight, while sentiment acts as a supplementary overlay. This design tests whether financial news sentiment adds incremental predictive value beyond price-based indicators.

### 12.2 Machine Learning and Sentiment Fusion

The ML sentiment-enhanced strategy combines the ML prediction signal with the FinBERT sentiment signal.

This is a late-fusion approach because sentiment is combined after ML predictions are generated rather than being inserted directly into the ML feature matrix.

### 12.3 Advantages and Limitations of Late Fusion

Late fusion has several advantages:

- It avoids sparse text features inside the ML training matrix.
- It reduces model complexity.
- It preserves interpretability of the ML-only model.
- It allows sentiment to be evaluated as an independent overlay.

However, late fusion also has a limitation: the ML model cannot learn interactions between sentiment and technical market states. For example, it cannot learn whether positive sentiment is more useful during high-momentum or low-volatility regimes.

## 13. Portfolio Weight Construction

### 13.1 Long-Only Filtering

Only positive signals are eligible for portfolio inclusion. Negative signals are set to zero.

This means the portfolio does not take short positions.

### 13.2 Signal Thresholding

Very small positive signals are removed using a minimum signal threshold.

This prevents weak and noisy signals from creating small, unstable positions.

### 13.3 Weight Normalization

Remaining positive signals are normalized so that portfolio weights sum to one.

Conceptually:

```text
Stock weight = Positive stock signal / Sum of all positive stock signals
```

This creates a fully invested long-only portfolio whenever at least one asset has a positive qualifying signal.

### 13.4 One-Day Weight Shift

Portfolio weights are shifted by one trading day before backtesting.

This is important because it prevents the strategy from using same-day signals to trade unrealistically before those signals would have been available. The shift improves the realism of the simulated trading process.

## 14. Hyperparameter Tuning of Models

### 14.1 Objective

Hyperparameter tuning was performed to evaluate whether model complexity and sentiment blending improve portfolio performance over the multi-year backtest period. The experiments compare alternative Random Forest and XGBoost configurations while keeping the asset universe, backtest dates, market data source, and portfolio construction logic consistent.

The main evaluation criteria were:

- Sharpe ratio
- Total return
- Final portfolio value
- Maximum drawdown, where available

Sharpe ratio was treated as the primary ranking metric because it adjusts return by realized volatility and is more informative than raw return alone for strategy comparison.

### 14.2 Random Forest Tuning Methodology

The Random Forest experiment used walk-forward model training with periodic retraining. For each hyperparameter configuration, the model generated return predictions from price-derived technical features, including momentum, volatility, trend, and mean-reversion features. The ML prediction signal was then blended with the FinBERT sentiment signal using late fusion:

```text
Final signal = (1 - sentiment weight) x ML signal + sentiment weight x sentiment signal
```

The tested Random Forest dimensions were:

- Number of estimators: 50, 100, 200
- Maximum tree depth: 3, 5, 8
- Sentiment weight: 0.2, 0.4

The backtest used an initial capital of $100,000 and covered the 2019-01-01 to 2023-12-31 evaluation period, with an additional warmup window for rolling features.

### 14.3 Random Forest Results

| N Estimators | Max Depth | Sentiment Weight | Sharpe | Total Return | Final Value |
|---:|---:|---:|---:|---:|---:|
| 200 | 8 | 0.2 | 1.04 | 186.23% | $286,231.68 |
| 200 | 8 | 0.4 | 1.04 | 186.23% | $286,231.68 |
| 100 | 8 | 0.2 | 1.04 | 185.58% | $285,578.61 |
| 100 | 8 | 0.4 | 1.04 | 185.58% | $285,578.61 |
| 50 | 5 | 0.0 | 1.03 | 162.31% | $262,312.22 |
| 50 | 5 | 0.4 | 1.03 | 188.89% | $288,894.44 |
| 50 | 5 | 0.2 | 1.03 | 188.89% | $288,894.44 |
| 100 | 3 | 0.4 | 1.03 | 193.37% | $293,367.12 |

The best Sharpe ratios were achieved by deeper Random Forest models, especially maximum depth 8 with 100 or 200 estimators. This suggests that allowing more tree depth helped the model capture nonlinear relationships in the engineered financial features. However, the difference between 100 and 200 estimators was small, indicating that most performance gains were already captured by the smaller ensemble.

An important observation is that several sentiment weights produced identical or near-identical Sharpe ratios. This implies that, in these configurations, the portfolio ranking and final allocation were dominated by the ML signal or that the sentiment overlay did not materially change the set of selected long positions. Sentiment still improved total return in some configurations, such as the 50-estimator depth-5 model, but the improvement was not consistently reflected in a higher Sharpe ratio.

### 14.4 XGBoost Tuning Methodology

The XGBoost experiment evaluated a gradient-boosted tree alternative using the same general walk-forward and portfolio construction framework. In addition to tree count, depth, and sentiment weight, XGBoost also included the learning rate as a tuning parameter.

The tested XGBoost dimensions included:

- Number of estimators: 100, 200
- Maximum tree depth: 3, 6
- Learning rate: 0.01, 0.10
- Sentiment weight: 0.2, 0.3

XGBoost builds trees sequentially, so the learning rate controls how strongly each additional tree contributes to the final prediction. Lower learning rates usually provide more conservative updates, while higher learning rates can adapt faster but may overfit noisy financial targets.

### 14.5 XGBoost Results

| N Estimators | Max Depth | Learning Rate | Sentiment Weight | Sharpe | Total Return | Drawdown |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 3 | 0.10 | 0.2 | 1.05 | 163.49% | -38.18% |
| 200 | 3 | 0.10 | 0.3 | 1.05 | 163.49% | -38.18% |
| 100 | 6 | 0.10 | 0.2 | 1.02 | 140.86% | -38.45% |
| 100 | 6 | 0.10 | 0.3 | 1.02 | 140.86% | -38.45% |
| 200 | 6 | 0.10 | 0.2 | 1.01 | 129.14% | -37.87% |
| 200 | 6 | 0.10 | 0.3 | 1.01 | 129.14% | -37.87% |
| 100 | 3 | 0.10 | 0.2 | 1.00 | 160.22% | -38.17% |
| 100 | 3 | 0.10 | 0.3 | 1.00 | 160.22% | -38.17% |
| 200 | 6 | 0.01 | 0.2 | 0.99 | 169.32% | -37.61% |
| 200 | 6 | 0.01 | 0.3 | 0.99 | 169.32% | -37.61% |
| 100 | 6 | 0.01 | 0.2 | 0.99 | 179.25% | -37.46% |
| 100 | 6 | 0.01 | 0.3 | 0.99 | 179.25% | -37.46% |
| 200 | 3 | 0.01 | 0.2 | 0.98 | 175.40% | -37.23% |
| 100 | 3 | 0.01 | 0.2 | 0.98 | 178.58% | -36.83% |

The strongest XGBoost Sharpe ratio was obtained with 200 estimators, maximum depth 3, and learning rate 0.10. This configuration produced a Sharpe ratio of 1.05, slightly higher than the best Random Forest Sharpe ratio. The result suggests that a moderately sized boosted model with shallow trees was better risk-adjusted than deeper boosted trees.

Deeper XGBoost models did not consistently improve performance. This is consistent with the noisy nature of short-horizon financial return prediction, where additional model complexity can fit unstable relationships that do not generalize well out of sample.

### 14.6 Conclusions from Hyperparameter Tuning

The hyperparameter experiments lead to the following conclusions:

1. Tree-based ML models produced strong multi-year performance, with top Sharpe ratios around 1.04 to 1.05.
2. Random Forest performance improved with deeper trees up to depth 8, but increasing estimators from 100 to 200 produced only marginal improvement.
3. XGBoost achieved the best reported Sharpe ratio using shallow trees with a higher learning rate, suggesting that controlled boosting was effective for this feature set.
4. Sentiment weights often produced identical results across nearby values, indicating that the FinBERT overlay did not always materially change final allocations.
5. Sentiment improved total return in some Random Forest configurations, but its risk-adjusted contribution was mixed.
6. Maximum drawdowns around 37% to 38% in the XGBoost experiment show that high returns came with meaningful downside risk.
7. The best model should therefore be selected using Sharpe ratio and drawdown together, not total return alone.

Overall, the tuning results support the use of tree-based supervised learning for portfolio signal generation, but they also show that model complexity and sentiment weighting must be controlled carefully. The best-performing configurations were not simply the largest models; rather, they were models that balanced nonlinear learning capacity with out-of-sample robustness.

## 15. Results and Performance Comparison

### 15.1 Average Performance Metrics (2019-2023)

| Metric | Baseline | Tech Sentiment | ML Only | ML + Sentiment |
|---|---:|---:|---:|---:|
| Avg Total Return | 18.84% | 25.15% | 17.72% | 19.24% |
| Avg CAGR | 29.13% | 36.13% | 28.96% | 29.22% |
| Avg Volatility ↓ | 24.62% | 23.69% | 25.38% | 24.73% |
| Avg Max Drawdown ↓ | -25.23% | -23.63% | -21.55% | -20.31% |
| Avg Sharpe Ratio | 1.18 | 1.30 | 1.24 | 1.32 |
| Avg Sortino Ratio | 1.77 | 1.99 | 1.75 | 1.88 |

### 15.2 Overall Interpretation

Tech Sentiment achieved the best average performance across:

1. Total return
2. CAGR
3. Volatility, with the lowest average volatility
4. Sortino ratio

ML + Sentiment achieved:

1. The best Sharpe ratio
2. The lowest average drawdown

ML Only provided relatively stable risk-adjusted performance but lagged behind the sentiment-enhanced approaches in average returns.

The Baseline strategy remained competitive in bullish years but underperformed overall compared with the enhanced strategies.

### 15.3 Year-Wise Performance Graphs

This section can include links to year-wise segregated line graphs comparing the different techniques across key performance metrics.

| Metric | Graph Link |
|---|---|
| Sortino Ratio | [sortinoRatio.png](sortinoRatio.png) |
| Sharpe Ratio | [sharpeRatio.png](sharpeRatio.png) |
| Maximum Drawdown | [maxDrawdown.png](maxDrawdown.png) |
| Volatility | [volatility.png](volatility.png) |
| Returns | [returns.png](returns.png) |
![growth of 100 dollar over the year](growth of 100 dollar invertment.png)

## 16. Summary of Techniques by Approach

| Approach | Data Cleaning | Preprocessing | Feature Extraction | Signal Construction |
|---|---|---|---|---|
| Baseline Technical | Adjusted OHLCV data, validation of prices and volume | Warmup window, daily alignment, missing signal fill | Momentum, mean reversion | Weighted technical signal |
| Technical + Sentiment | Market cleaning plus news symbol, date, and text cleaning | FinBERT scoring, daily aggregation, smoothing, thresholding | Technical signal plus sentiment score | Weighted fusion of technical and sentiment signals |
| ML Only | Adjusted OHLCV data, invalid/missing row removal | Walk-forward training, lagging, normalization, ranking | Momentum, volatility, trend, mean reversion, returns, range, volume | Random Forest return prediction |
| ML + Sentiment | Market and news cleaning | ML preprocessing plus FinBERT sentiment preprocessing | ML technical/statistical features plus external sentiment signal | Late fusion of ML and sentiment signals |
| XGBoost Experiment | Same as ML pipeline | Walk-forward training and hyperparameter testing | Same technical/statistical feature set | XGBoost prediction with optional sentiment fusion |

## 17. Methodological Strengths

The framework includes several academically sound design choices:

1. Adjusted market data is used to avoid artificial return distortions.
2. A warmup period is included for rolling-window feature construction.
3. Price and news data are aligned by date and symbol.
4. FinBERT is used as a financial-domain language model.
5. Sentiment is smoothed and thresholded to reduce noise.
6. Lagged features are created to reduce lookahead bias.
7. Walk-forward training is used instead of random time-series splitting.
8. Both rule-based and ML-based strategies are evaluated.
9. Sentiment-enhanced and non-sentiment strategies are compared.
10. Long-only constraints are applied for realistic portfolio construction.

## 18. Methodological Limitations

The following limitations should be acknowledged in an academic research paper:

1. Sentiment is added through late fusion rather than being learned directly inside the ML feature matrix.
2. Daily news aggregation ignores intraday publication time.
3. All headlines are equally weighted, regardless of publisher importance or article relevance.
4. Missing news is treated as neutral sentiment.
5. The 3-day sentiment smoothing window is manually chosen.
6. The sentiment threshold is manually specified rather than statistically optimized.
7. FinBERT confidence is treated as signal magnitude, although classification confidence may not be perfectly calibrated.
8. The asset universe is fixed, which may introduce survivorship or selection bias if not carefully justified.
9. Short-horizon return prediction is inherently noisy.
10. Real-world execution constraints may be more complex than the modeled transaction costs and slippage.

## 19. Academic Conclusion

The methodology implements a structured multi-modal portfolio construction framework combining price-based technical indicators, supervised machine learning, and transformer-based financial sentiment analysis. Data preprocessing is designed to ensure temporal consistency, remove invalid observations, reduce noise, and prevent lookahead bias.

Feature extraction captures several dimensions of market behavior, including momentum, reversal, volatility, trend strength, trading range, and volume activity. FinBERT sentiment adds a text-derived signal that reflects the market-relevant tone of financial news.

The comparison between baseline, sentiment-enhanced, ML-only, and ML-sentiment strategies allows the research to evaluate whether financial news sentiment provides incremental value beyond historical price-based information. The use of walk-forward training and lagged features makes the experimental design more appropriate for financial time-series research than conventional random-split machine learning workflows.

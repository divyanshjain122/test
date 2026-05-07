# Backtesting: Which Model is Actually Used?

## Critical Finding: TWO DIFFERENT MODELS

**The key insight:** The backtesting does NOT use the pre-trained model from STEP 5!

---

## Code Comparison

### STEP 5: Initial Model Training (Lines 162-193)

```python
# Lines 162-165: Create FIRST Random Forest model
print("\n[Step 5] Training Random Forest model...")

from jsf.ml import RandomForestModel

rf_model = RandomForestModel(
    n_estimators=100,              # ← 100 trees
    max_depth=5,                   # ← depth 5
    prediction_type='regression',
    random_state=42,
)

# Lines 168-169: Train on historical training set ONCE
rf_model.fit(X_train, y_train_reg)
print(f"  ✓ Model trained on {len(X_train)} samples")

# Lines 172-173: Test on test set to calculate IC
pred_dict = rf_model.predict(X_test)
predictions = np.array(pred_dict['returns']).flatten()

# Line 176: Calculate IC (Information Coefficient)
ic = np.corrcoef(predictions, y_test_values)[0, 1]
print(f"  ✓ Test IC (correlation): {ic:.4f}")

# ❌ AFTER THIS: rf_model is NEVER USED AGAIN!
```

**Summary of STEP 5 model:**
- ✅ Train once on historical data (80% of data)
- ✅ Evaluate on test set (20% of data)
- ✅ Calculate IC performance metric
- ❌ **NOT used in actual backtesting**

---

### STEP 7: Create Strategy Model (Lines 220-251)

```python
# Lines 225-231: Create SECOND Random Forest model (DIFFERENT!)
print("\n[Step 7] Running ML Strategy with walk-forward training...")

from jsf.ml import MLStrategy, FeatureExtractor

# ⚠️ IMPORTANT: This is a DIFFERENT model than STEP 5
model = RandomForestModel(
    n_estimators=50,              # ← Different: 50 trees (not 100!)
    max_depth=4,                  # ← Different: depth 4 (not 5!)
    prediction_type='regression',
)

# Lines 233-239: Create feature extractor
strategy_extractor = FeatureExtractor(
    feature_groups=['momentum', 'volatility'],  # ← Different: only 2 groups (not 4!)
    lag_periods=[1, 5],                         # ← Different: only [1,5] (not [1,5,10]!)
    normalize=False,                            # ← Different: not normalized!
    rank=False,                                 # ← Different: not ranked!
)

# Lines 241-247: Create strategy with the second model
strategy = MLStrategy(
    name='ml_demo',
    model=model,                           # ← THIS model is used in backtest
    feature_extractor=strategy_extractor,
    prediction_type='returns',
    retrain_frequency=63,                  # ← KEY: Retrain every 63 days!
    long_only=True,
)

# Lines 250-251: Adjust performance settings
strategy.config.warmup_period = 120
strategy.config.min_train_samples = 100
```

**Summary of STEP 7 model:**
- ✅ Model created (but NOT pre-trained)
- ✅ Encapsulated in MLStrategy
- ✅ Will be trained DURING backtest with walk-forward retraining
- ✅ **THIS is the model used in backtesting**

---

### STEP 8: Run Backtest (Lines 253-280)

```python
# Lines 257-258: Run strategy on price data
print("\n[Step 8] Backtesting ML strategy...")

from jsf.simulation import BacktestEngine, BacktestConfig

# ⚠️ THIS LINE IS CRITICAL:
portfolio = strategy.run(price_data)  # ← Runs STEP 7 model with walk-forward training

# Lines 261-264: Create backtester
backtest_config = BacktestConfig(
    initial_capital=100000,
    transaction_cost=0.001,
)
engine = BacktestEngine(backtest_config)

# Lines 267-268: Run backtest
results = engine.run(portfolio, price_data)
```

**What happens in `strategy.run(price_data)`:**
1. **NOT** using pre-trained `rf_model` from STEP 5
2. **Using** the `model` from STEP 7 (n_estimators=50, max_depth=4)
3. **Training dynamically** as it backtests (walk-forward style)
4. **Retraining every 63 days** with new data

---

## Model Comparison: STEP 5 vs STEP 7

### Side-by-Side Comparison

```
┌──────────────────────┬────────────────────┬────────────────────┐
│ Aspect               │ STEP 5 Model       │ STEP 7 Model       │
├──────────────────────┼────────────────────┼────────────────────┤
│ Variable Name        │ rf_model           │ model (in strategy)│
│ n_estimators         │ 100                │ 50                 │
│ max_depth            │ 5                  │ 4                  │
│ Feature Groups       │ 4 groups           │ 2 groups           │
│ Lag Periods          │ [1, 5, 10]         │ [1, 5]             │
│ Normalize            │ True               │ False              │
│ Rank                 │ True               │ False              │
│ Training Method      │ Fixed (once)       │ Walk-forward       │
│ training Data        │ 80% historical     │ Dynamic (expanding)│
│ Retraining           │ None               │ Every 63 days      │
│ Used in Backtest     │ ✗ NO               │ ✓ YES              │
│ IC Metric            │ 0.0456 (example)   │ Unknown (different │
│                      │                    │  features/params)  │
└──────────────────────┴────────────────────┴────────────────────┘
```

---

## What Actually Happens During Backtesting

### Walk-Forward Training Process

```
Timeline: Jan 2022 ──────────┬───────────────────────────────────── Dec 2024
                              │
                          Day 1 (Backtest starts)
                              │
Phase 1: Warmup Period (120 days = Jan-Apr 2022)
├─ Day 1-120:
│  ├─ Collect price data
│  ├─ Extract features
│  ├─ Do NOT trade yet
│  └─ Accumulate training data
│
Phase 2: First Training & Trading (Days 121-183)
├─ Day 120:
│  ├─ Train Model v1 on data from Jan-Apr
│  ├─ Generate signals
│  └─ Start trading
├─ Day 121-183:
│  ├─ Execute trades based on Model v1
│  ├─ Track P&L
│  └─ Accumulate new data
│
Phase 3: Retrain (Day 184)
├─ Day 184:
│  ├─ Retrain Model v2 on NEW data (Jan-Jun)
│  ├─ Discard old patterns
│  ├─ Generate NEW signals
│  └─ Continue trading with Model v2
├─ Day 185-247:
│  ├─ Execute trades based on Model v2
│  ├─ Track P&L
│  └─ Continue...
│
Phase 4: Retrain Again (Day 248)
├─ Day 248:
│  ├─ Retrain Model v3 on NEW data (Jan-Sep)
│  └─ Continue pattern...
│
Continue until end date (Dec 2024)
```

### Code Flow During Backtest

```python
# What happens INSIDE `strategy.run(price_data)`

FOR each_date in price_data.dates:
    
    # Check if warmup period is over
    if days_elapsed < 120:
        # Just accumulate data, don't trade
        continue
    
    # Check if it's time to retrain (every 63 days)
    if days_since_last_training >= 63:
        # STEP 1: Get recent training data
        recent_data = price_data[last_63_days:]
        
        # STEP 2: Extract features
        features = strategy_extractor.extract(recent_data)
        
        # STEP 3: Create targets (next 5-day returns)
        targets = create_target_variable(recent_data, forward_periods=5)
        
        # STEP 4: Train NEW model on this data
        model.fit(features, targets)  # ← NEW TRAINING, NOT using STEP 5 model
        
        # STEP 5: Reset retraining timer
        last_training_date = current_date
    
    # STEP 6: Make prediction for today
    today_features = strategy_extractor.extract(price_data[current_date])
    signal = model.predict(today_features)  # ← Using NEWLY TRAINED model
    
    # STEP 7: Generate trades based on signal
    if signal > 0.3:
        execute_buy_order(signal)
    elif signal < -0.3:
        execute_sell_order(signal)
    
    # STEP 8: Calculate P&L
    portfolio_value += daily_pnl
```

---

## Why Two Different Models?

### Reason 1: Different Purposes

```
STEP 5 Model:
Purpose: Evaluate model quality on historical data
Usage:   Calculate IC metric (0.0456)
Outcome: Shows how good the model is (for reference)

STEP 7 Model:
Purpose: Trade in real backtest scenario
Usage:   Walk-forward retraining every 63 days
Outcome: Simulates realistic trading performance
```

### Reason 2: Realistic Walk-Forward Simulation

```
❌ WRONG: Use STEP 5 model
   - Model trained on 2022-2023 data
   - Tested on 2024 data
   - But 2024 market might be completely different
   - Overfitting to past patterns
   - Unrealistic performance
   
✅ RIGHT: Walk-forward model (STEP 7)
   - Model retrains quarterly with newest data
   - Always uses most recent market patterns
   - Adapts to market regime changes
   - More realistic out-of-sample results
   - Avoids look-ahead bias
```

### Reason 3: Computational Efficiency

```
STEP 5 Model (n_estimators=100, max_depth=5):
├─ More complex
├─ Better IC: 0.0456
├─ But slower to train
└─ Overkill for live trading

STEP 7 Model (n_estimators=50, max_depth=4):
├─ Simpler
├─ Good enough for trading
├─ Faster to train (can retrain quarterly)
└─ Practical for real-time use
```

---

## Feature Differences Also Matter

### STEP 5 Features
```python
extractor = FeatureExtractor(
    feature_groups=['momentum', 'volatility', 'trend', 'mean_reversion'],
    lag_periods=[1, 5, 10],
    normalize=True,
    rank=True,
)
```

**Result:** 
```
4 feature groups × 3 lag periods = 12 base features
Plus: volatility, trend calculations
Total: ~25-30 features
```

### STEP 7 Features
```python
strategy_extractor = FeatureExtractor(
    feature_groups=['momentum', 'volatility'],      # Only 2 groups!
    lag_periods=[1, 5],                             # Only [1,5]!
    normalize=False,                                # No normalization
    rank=False,                                     # No ranking
)
```

**Result:**
```
2 feature groups × 2 lag periods = 4 base features
Much simpler!
Reason: Speed (can retrain every 63 days)
```

---

## Summary: Which Model in Backtest?

### Answer: DIFFERENT MODEL with Walk-Forward Training

```
STEP 5 Model:
  ├─ n_estimators = 100, max_depth = 5
  ├─ Features: 25-30
  ├─ Trained ONCE on 80% of data
  ├─ IC = 0.0456
  └─ ❌ NOT USED IN BACKTEST

BACKTEST (STEP 7 + 8) Model:
  ├─ n_estimators = 50, max_depth = 4
  ├─ Features: ~4 (much simpler)
  ├─ Trained FRESH at backtest start
  ├─ Retrained EVERY 63 DAYS with new data
  ├─ IC = Unknown (different features/params)
  └─ ✅ USED IN BACKTEST with walk-forward
```

### Key Points

1. **Pre-trained STEP 5 model is for validation only**
   - Shows how good ML is in theory
   - IC=0.0456 tells us model has predictive power
   - But NOT used for actual trading

2. **Backtest uses STEP 7 model with walk-forward training**
   - Simulates realistic trading scenario
   - Retrains every 63 days (quarterly)
   - Adapts to market changes
   - More reliable performance estimate

3. **Why walk-forward matters**
   ```
   Static model (STEP 5):
   Train: 2022-2023 full data
   Test: 2024 data
   Problem: 2024 might be completely different market
   
   Walk-forward model (STEP 7):
   Month 1 (Jan): Train on Jan, trade Feb
   Month 2 (Feb): Train on Jan-Feb, trade March
   Month 3 (Mar): Train on Jan-Feb-Mar, trade April
   ...
   Benefits: Always uses fresh data, adapts to regime changes
   ```

---

## Code References Quick Reference

**Lines 162-193: STEP 5 Model (NOT used in backtest)**
```python
rf_model = RandomForestModel(
    n_estimators=100,
    max_depth=5,
    ...
)
rf_model.fit(X_train, y_train_reg)
ic = np.corrcoef(predictions, y_test_values)[0, 1]  # IC=0.0456
# ❌ Never used again after this
```

**Lines 225-251: STEP 7 Model (USED in backtest)**
```python
model = RandomForestModel(
    n_estimators=50,              # Different
    max_depth=4,                  # Different
    ...
)
strategy = MLStrategy(
    model=model,                  # ← This model
    retrain_frequency=63,         # ← Walk-forward retrain
    ...
)
signals = strategy.generate_signals(price_data)
```

**Line 258: Backtest with STEP 7 Model**
```python
portfolio = strategy.run(price_data)  # ← Uses walk-forward model
```

**This is the key architectural insight of the demo!**

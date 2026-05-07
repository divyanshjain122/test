# STEP 5: ML Model Training - Detailed Analysis

**Location in demo:** Lines 314-365 of `demo_full_real_data.py`

---

## Table of Contents
1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Target Variable Creation](#target-variable-creation)
4. [Feature-Target Alignment](#feature-target-alignment)
5. [Train-Test Split](#train-test-split)
6. [XGBoost Model Training](#xgboost-model-training)
7. [LightGBM Model Training](#lightgbm-model-training)
8. [Model Evaluation with IC](#model-evaluation-with-ic)
9. [Results Interpretation](#results-interpretation)
10. [Complete Code Flow](#complete-code-flow)

---

## Overview

This step trains two gradient boosting models to predict next-day stock returns:

1. **XGBoost** (eXtreme Gradient Boosting)
2. **LightGBM** (Light Gradient Boosting Machine)

### Why Two Models?
- **Ensemble approach**: Two different algorithms capture different patterns
- **Robustness**: If one model overfits, the other may not
- **Comparison**: Evaluate which algorithm works better for your data
- **Redundancy**: One model can validate the other's findings

### What They Predict
- **Target**: Next day's stock return (%)
- **Input**: 100+ technical features from STEP 4
- **Task**: Regression (predicting continuous return values)

---

## Data Preparation

### Code Context
```python
# Lines 314-309: Import required modules
from jsf.ml import XGBoostModel, LightGBMModel, create_target_variable
```

### What These Modules Do

| Module | Purpose |
|--------|---------|
| `create_target_variable()` | Converts price data into prediction targets |
| `XGBoostModel` | Scikit-learn wrapper around XGBoost |
| `LightGBMModel` | Scikit-learn wrapper around LightGBM |

---

## Target Variable Creation

### Code (Lines 317-322)
```python
# Create targets from real data
y_returns, y_direction = create_target_variable(
    price_data,
    target_type='both',
    forward_periods=1,
)
```

### What Happens Here

#### Step 1: Calculate Forward Returns
```
For each date D and symbol S:
  Forward_Return[D,S] = (Close[D+1] - Close[D]) / Close[D]
```

**Example:**
```
Date       Close   Next_Close   Forward_Return
2024-01-15  1000      1010       +1.0%
2024-01-16  1010      1005       -0.5%
2024-01-17  1005      1020       +1.5%
```

#### Step 2: Output Format
```python
y_returns   # DataFrame: (N_dates × N_symbols) with return percentages
y_direction # DataFrame: (N_dates × N_symbols) with +1 (up), -1 (down), 0 (flat)
```

**Example output:**
```
               RELIANCE  HDFCBANK  BHARTIARTL  ...
2024-01-02       0.012     0.005      -0.002
2024-01-03      -0.003     0.015       0.008
2024-01-04       0.008    -0.001       0.003
...
```

### Key Points
- `forward_periods=1`: Look ahead 1 day only
- `target_type='both'`: Create both regression (returns) and classification (direction)
- We'll use **only returns** for regression, ignoring direction

---

## Feature-Target Alignment

### Why Alignment Matters
Features (X) and targets (y) must have:
- ✅ **Same dates**
- ✅ **Same symbols**
- ✅ **No NaN values**
- ✅ **Same row order**

Without alignment, model learns garbage → garbage predictions.

### Code (Lines 325-336)

```python
# Align data
y_ret_stacked = y_returns.stack()          # Convert (dates × symbols) to 1D Series
y_dir_stacked = y_direction.stack()        # Same for direction

common_idx = features.index.intersection(y_ret_stacked.index)
X = features.loc[common_idx].dropna()      # Keep only rows in both datasets
y_ret = y_ret_stacked.loc[X.index].dropna()  # Match y to X's index
y_dir = y_dir_stacked.loc[X.index].dropna()  # Match direction to X

valid_idx = y_ret.index.intersection(y_dir.index).intersection(X.index)
X = X.loc[valid_idx]    # Final aligned X
y_ret = y_ret.loc[valid_idx]  # Final aligned targets
```

### Step-by-Step Breakdown

#### Substep 1: Stacking
```
Input:  2D DataFrame (dates × symbols)
        
                RELIANCE  HDFCBANK  BHARTIARTL
        2024-01-02   0.012     0.005      -0.002
        2024-01-03  -0.003     0.015       0.008

Output: 1D Series (dates × symbols as MultiIndex)
        
        2024-01-02, RELIANCE      0.012
        2024-01-02, HDFCBANK      0.005
        2024-01-02, BHARTIARTL   -0.002
        2024-01-03, RELIANCE     -0.003
        2024-01-03, HDFCBANK      0.015
        ...
```

#### Substep 2: Find Common Index
```python
common_idx = features.index.intersection(y_ret_stacked.index)
```

This keeps only (date, symbol) pairs that exist in BOTH:
- Features from STEP 4
- Target returns calculated here

**Reason:** Some dates might be missing from feature data (gaps, holidays)

#### Substep 3: Remove NaN Values
```python
X = features.loc[common_idx].dropna()
```

Removes rows where ANY feature is NaN:
- Extremely high/low prices (outliers)
- Missing data gaps
- Feature calculation errors

#### Substep 4: Final Validation
```python
valid_idx = y_ret.index.intersection(y_dir.index).intersection(X.index)
```

Ensures X and y have identical indices:
- Same dates
- Same symbols
- Same order

### Output
```python
X.shape    # (N_valid_samples, N_features)
           # e.g., (12500, 108) = 12,500 date-symbol pairs, 108 features

y_ret.shape  # (N_valid_samples,)
             # e.g., (12500,) = 12,500 target returns

print(f"   [OK] Aligned real data: {len(X)} samples")
```

---

## Train-Test Split

### Why Split Data?

```
❌ WRONG: Train on ALL data, test on SAME data
         → Model memorizes data (overfitting)
         → Unrealistic performance metrics

✅ RIGHT: Train on 80%, test on unseen 20%
         → Model learns patterns
         → Test set evaluates generalization
```

### Code (Lines 339-340)

```python
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y_ret.iloc[:split_idx], y_ret.iloc[split_idx:]
```

### Example

```
Total samples: 12,500
Split point: 12,500 × 0.8 = 10,000

Training set:
  X_train: samples 0-9,999      (10,000 samples)
  y_train: targets 0-9,999      (10,000 samples)

Test set:
  X_test:  samples 10,000-12,499 (2,500 samples)
  y_test:  targets 10,000-12,499 (2,500 samples)

Timeline:
  Jan 2023 ──────────────── Sep 2023 | Oct 2023 ──────── Jan 2024
  Training period (80%)              | Testing period (20%)
```

### Chronological Order
- **Important**: Data is time-series
  - Training uses past data → learns patterns
  - Testing uses future data → true out-of-sample
  - Prevents look-ahead bias

---

## XGBoost Model Training

### What is XGBoost?

**XGBoost** = eXtreme Gradient Boosting

- **Type**: Ensemble of decision trees
- **Method**: Gradient boosting (builds trees sequentially)
- **How it works**: Each new tree corrects errors of previous trees

### Conceptual Process

```
Iteration 1: Build Tree 1
            Train_Error = 0.05
            
Iteration 2: Build Tree 2 (learns from Tree 1's mistakes)
            Combined_Error = 0.045
            
Iteration 3: Build Tree 3
            Combined_Error = 0.040
            
... (50 iterations total)

Final Prediction = Tree1 + Tree2 + ... + Tree50
```

### Code (Lines 343-350)

```python
# Create XGBoost model instance
xgb = XGBoostModel(
    n_estimators=50,              # Build 50 trees
    max_depth=3,                  # Each tree is 3 levels deep
    prediction_type='regression', # Predict continuous returns
    n_jobs=1,                     # Use 1 CPU core
)

# Train on training set
xgb.fit(X_train, y_returns=y_train)

# Make predictions on test set
xgb_pred = xgb.predict(X_test)
```

### Parameter Explanation

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_estimators` | 50 | Build 50 decision trees |
| `max_depth` | 3 | Each tree has max 3 levels |
| `prediction_type` | 'regression' | Predict numbers (returns) |
| `n_jobs` | 1 | Use 1 CPU core (avoid threading issues) |

### Why These Specific Values?

```
max_depth=3 (Small trees)
  ✅ Reduces overfitting
  ✅ Faster training
  ✅ Interpretable
  ❌ Less powerful

n_estimators=50 (Moderate number)
  ✅ Enough for learning
  ✅ Still fast
  ❌ May underfit for complex data
```

### What `.fit()` Does

```python
xgb.fit(X_train, y_returns=y_train)
```

**Internal steps:**
1. Initialize: Create 1st tree
2. Loop 50 times:
   - Predict on training data: ŷ = Tree1 + Tree2 + ... + TreeN
   - Calculate residuals: errors = y_train - ŷ
   - Build new tree to learn those errors
   - Update prediction: ŷ_new = ŷ_old + Tree_N+1

### What `.predict()` Does

```python
xgb_pred = xgb.predict(X_test)
```

**Returns:**
```python
xgb_pred = {
    'returns': array([0.0012, -0.0003, 0.0008, ...]),  # Predicted returns
    'raw_predictions': [...],                           # Internal scores
}
```

Length: same as X_test (2,500 samples)

---

## LightGBM Model Training

### What is LightGBM?

**LightGBM** = Light Gradient Boosting Machine

```
XGBoost  → Slower, more memory, older
           For < 10k samples

LightGBM → Faster, less memory, newer
           For large datasets (10k-1M samples)
           ✅ Better for this demo
```

### Code (Lines 352-359)

```python
# Create LightGBM model
lgb = LightGBMModel(
    n_estimators=50,
    max_depth=3,
    prediction_type='regression',
    n_jobs=1,
)

# Train
lgb.fit(X_train, y_returns=y_train)

# Predict
lgb_pred = lgb.predict(X_test)
```

### Differences from XGBoost

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| Tree building | Level-wise | Leaf-wise |
| Speed | Slower | Faster (2x) |
| Memory | More | Less |
| For 12.5k samples | Good | Better |
| Overfitting | Less risk | More risk |

**Tree building difference:**
```
Level-wise (XGBoost):
  Covers all branches at each level
  Balanced tree, safer
  
Leaf-wise (LightGBM):
  Grows deepest leaf with most error
  Less balanced, faster
```

---

## Model Evaluation with IC

### What is IC (Information Coefficient)?

**IC** = Correlation between predictions and actual returns

```
IC = correlation(predicted_returns, actual_returns)

Range: -1.0 to +1.0

  +1.0 → Perfect positive correlation (model is perfect)
   0.5 → Strong correlation (very good model)
   0.1 → Weak correlation (okay model)
   0.0 → No correlation (useless model)
  -0.5 → Inverted predictions (opposite signal)
```

### Code (Lines 362-365)

```python
# Calculate predictions vs actual returns
xgb_ic = np.corrcoef(xgb_pred['returns'], y_test)[0, 1]
lgb_ic = np.corrcoef(lgb_pred['returns'], y_test)[0, 1]

print(f"   [OK] XGBoost trained on {len(X_train)} real samples")
print(f"      → IC on test set: {xgb_ic:.4f}")
print(f"   [OK] LightGBM trained on {len(X_train)} real samples")
print(f"      → IC on test set: {lgb_ic:.4f}")
```

### How IC is Calculated

```python
np.corrcoef(xgb_pred['returns'], y_test)
```

Returns:
```
[[1.0    xgb_ic]
 [xgb_ic  1.0  ]]

[0, 1] → Get correlation coefficient
```

### Example Output

```
   [OK] XGBoost trained on 10000 real samples
      → IC on test set: 0.0523
   [OK] LightGBM trained on 10000 real samples
      → IC on test set: 0.0487
```

### Interpretation

- **IC = 0.052**: Weak but positive correlation
  - Better than random (IC = 0)
  - In real trading: 1 Sharpe point per 0.05 IC
  - Can be profitable with risk management

- **IC = 0.049**: Slightly weaker
  - Still profitable but less reliable
  - Suggests LightGBM captures fewer patterns

---

## Results Interpretation

### What Good IC Looks Like

```
IC ≥ 0.05 per day   → Excellent in ML/finance
  └─ Compounds to Sharpe ratio ~1.0 annually
  
IC 0.02-0.05       → Good, tradeable
  └─ Sharpe 0.5-1.0 annually
  
IC 0.01-0.02       → Okay, needs position sizing
  └─ Sharpe 0.2-0.5 annually
  
IC < 0.01          → Very weak, not tradeable
```

### Things to Watch

#### 1. Positive IC
```
IC > 0 ✅ Good: Model learned patterns
IC < 0 ❌ Bad: Model predicts opposite (inverted)
IC ≈ 0 ❌ Terrible: Random guessing
```

#### 2. Model Comparison
```
If XGBoost IC > LightGBM IC
  → XGBoost captures better patterns for THIS data
  
If LightGBM IC > XGBoost IC
  → LightGBM's leaf-wise approach works better
  
If ICs are similar (0.05 vs 0.049)
  → Both models learning same patterns (robust)
```

#### 3. Overfitting Check
```
Training accuracy >> Test accuracy
  → Model overfit trainng data
  → IC will be inflated
  
Training accuracy ≈ Test accuracy
  → Model generalizes well
  → IC is reliable
```

---

## Complete Code Flow

### Full Training Pipeline

```python
# =========================================================================
# STEP 5: REAL ML MODEL TRAINING
# =========================================================================
print("\n[Step 5] Training REAL ML models (XGBoost, LightGBM)...")

# 1. CREATE TARGETS
# ────────────────
from jsf.ml import XGBoostModel, LightGBMModel, create_target_variable

y_returns, y_direction = create_target_variable(
    price_data,
    target_type='both',
    forward_periods=1,
)
print(f"   Created targets: shape {y_returns.shape}")

# 2. ALIGN DATA
# ────────────
y_ret_stacked = y_returns.stack()
y_dir_stacked = y_direction.stack()

common_idx = features.index.intersection(y_ret_stacked.index)
X = features.loc[common_idx].dropna()
y_ret = y_ret_stacked.loc[X.index].dropna()
y_dir = y_dir_stacked.loc[X.index].dropna()

valid_idx = y_ret.index.intersection(y_dir.index).intersection(X.index)
X = X.loc[valid_idx]
y_ret = y_ret.loc[valid_idx]

print(f"   [OK] Aligned real data: {len(X)} samples")

# 3. SPLIT TRAIN/TEST
# ───────────────────
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y_ret.iloc[:split_idx], y_ret.iloc[split_idx:]

print(f"   Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# 4. TRAIN XGBOOST
# ────────────────
print("   [DEBUG] Creating XGBoost model...", flush=True)
xgb = XGBoostModel(
    n_estimators=50,
    max_depth=3,
    prediction_type='regression',
    n_jobs=1,
)

print("   [DEBUG] Fitting XGBoost model...", flush=True)
xgb.fit(X_train, y_returns=y_train)

print("   [DEBUG] Predicting with XGBoost model...", flush=True)
xgb_pred = xgb.predict(X_test)

# 5. TRAIN LIGHTGBM
# ─────────────────
print("   [DEBUG] Creating LightGBM model...", flush=True)
lgb = LightGBMModel(
    n_estimators=50,
    max_depth=3,
    prediction_type='regression',
    n_jobs=1,
)

print("   [DEBUG] Fitting LightGBM model...", flush=True)
lgb.fit(X_train, y_returns=y_train)

print("   [DEBUG] Predicting with LightGBM model...", flush=True)
lgb_pred = lgb.predict(X_test)

# 6. EVALUATE MODELS
# ──────────────────
xgb_ic = np.corrcoef(xgb_pred['returns'], y_test)[0, 1]
lgb_ic = np.corrcoef(lgb_pred['returns'], y_test)[0, 1]

print(f"   [OK] XGBoost trained on {len(X_train)} real samples")
print(f"      → IC on test set: {xgb_ic:.4f}")
print(f"   [OK] LightGBM trained on {len(X_train)} real samples")
print(f"      → IC on test set: {lgb_ic:.4f}")

# =========================================================================
# END STEP 5
# =========================================================================
```

---

## Summary Table

| Component | Input | Process | Output | Example |
|-----------|-------|---------|--------|---------|
| **Target** | price_data | Calculate forward returns | y_returns (12.5k samples) | [0.012, -0.003, 0.008] |
| **Alignment** | X, y | Match indices, drop NaN | X_align, y_align | 12.5k valid samples |
| **Split** | X_align | 80/20 chronological split | X_train, X_test, y_train, y_test | 10k train, 2.5k test |
| **XGBoost** | X_train, y_train | Fit 50 trees | xgb_pred (2.5k predictions) | [0.0089, -0.0012, ...] |
| **LightGBM** | X_train, y_train | Fit 50 trees | lgb_pred (2.5k predictions) | [0.0091, -0.0008, ...] |
| **Evaluation** | xgb_pred, y_test | Correlation | IC scores | XGBoost: 0.0523, LightGBM: 0.0487 |

---

## Key Takeaways

### ✅ What Happens in STEP 5

1. **Create targets**: Next-day returns (what to predict)
2. **Align data**: Match features and targets perfectly
3. **Split time-series**: 80% training, 20% testing
4. **Train XGBoost**: Build 50-tree ensemble
5. **Train LightGBM**: Build 50-tree ensemble (faster)
6. **Evaluate**: Calculate IC (correlation) on test set

### ⚠️ Common Pitfalls

```
❌ Training on test set
   → Inflated metrics

❌ Using future data in features
   → Look-ahead bias

❌ Not aligning X and y
   → Model learns wrong patterns

❌ Too many trees (n_estimators >> 100)
   → Overfitting, slower

❌ Too deep trees (max_depth >> 5)
   → Complex patterns, overfitting
```

### 📊 Success Indicators

✅ IC > 0.035: Definitely profitable
✅ IC 0.02-0.035: Likely profitable
✅ Both models have similar IC: Robust
✅ No error messages: Clean data pipeline

---

## Next Steps (STEP 6)

The trained models' predictions are used in **Trading Signal Generation**:
- Combine with technical signals (momentum, mean reversion)
- Generate buy/sell recommendations
- Execute trades in backtest

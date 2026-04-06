# Design Spec: Regime Detection in Crude Oil Markets

**Date**: 2026-04-06
**Course**: COMP0040 — Data-driven Modelling of Financial Markets
**Format**: 15-page PDF + Python code submission
**Language**: English

---

## 1. Research Overview

### Title
*Regime Detection in Crude Oil Markets: A Shannon Entropy and Kalman Filter Approach*

### Research Questions
- **RQ1**: Can a combination of Shannon Entropy and Kalman Filter effectively detect distinct market regimes in crude oil prices (2020-2026)?
- **RQ2**: Does incorporating detected regime information as a feature improve XGBoost oil price prediction performance?

### Approach Summary
A two-part study:
- **Part I (Regime Detection)**: Rolling Shannon Entropy captures market disorder from oil return distributions. A linear Gaussian state-space model with Kalman Filter estimates a latent "market stress" hidden state from Entropy, OVX, and absolute returns. The hidden state is thresholded into 3 regimes.
- **Part II (Regime-Enhanced Prediction)**: Regime labels and hidden state values are added as features to an XGBoost model. Baseline (no regime features) vs Enhanced (with regime features) performance is compared.

---

## 2. Paper Structure & Page Allocation

| Section | Pages | Content |
|---|---|---|
| 1. Introduction | 1.5 | Research motivation, oil market 2020-2026 context, two research questions, paper structure |
| 2. Data & Descriptive Statistics | 2 | Data sources, variable descriptions, descriptive statistics table, key time series plots |
| 3. Methodology | 3 | 3.1 Rolling Shannon Entropy; 3.2 State-space model & Kalman Filter; 3.3 Regime classification; 3.4 Regime-enhanced XGBoost framework |
| 4. Regime Detection Results | 4 | Entropy time series analysis, hidden state estimation, 3-regime identification, historical event alignment, regime characteristic comparison |
| 5. Regime-Enhanced Prediction | 3 | Baseline vs Enhanced XGBoost comparison, results tables and visualizations |
| 6. Conclusion | 1.5 | Findings summary, limitations, future directions |
| **Total** | **15** | |

No separate Literature Review or References sections required (covered in group assignment).

---

## 3. Data

### Datasets
All data covers **2020-01 to 2026-03**.

**Daily dataset** (1619 rows):
- `observation_date`: trading date
- `Brent_Price`: Brent crude oil price (USD/barrel)
- `WTI_Price`: WTI crude oil price (USD/barrel) — **primary target**
- `oil_return`: WTI log return
- `OVX_Close`: CBOE Oil Volatility Index
- `DTWEXBGS`: USD trade-weighted exchange rate index
- `EFFR`: Effective Federal Funds Rate
- `Gas_Price`: Henry Hub natural gas price
- `Gold_Price`: Gold spot price (USD/oz)

**Weekly dataset** (324 rows):
- `WCESTUS1`: US crude oil inventory (thousand barrels)
- `supply_shock`: week-over-week inventory change
- `supply_shock_pct`: percentage change

**Monthly dataset** (75 rows):
- `GPR`, `GPRT`, `GPRC`: Geopolitical Risk Index and sub-indices
- `GEPUCURRENT`: Economic Policy Uncertainty Index
- `CPILFESL`: Core CPI
- `INDPRO`: Industrial Production Index
- `industrial_production_growth`: IP growth rate

### Data Handling
- Primary analysis uses **daily dataset** (highest granularity).
- Weekly and monthly data are **forward-filled** to daily frequency for use as XGBoost features in Part II.
- No train/test split for Part I (full-sample regime detection).
- **80/20 chronological split** for Part II (XGBoost prediction).

### Key Historical Events for Validation
- **2020-03/04**: COVID-19 oil crash (WTI went negative)
- **2020-05 to 2021**: Recovery period
- **2022-02/03**: Russia-Ukraine war oil price spike
- **2023-2025**: Relative stabilization
- **2026-03**: Recent high volatility episode (WTI 94 → 83 → 98)

---

## 4. Methodology

### 4.1 Shannon Entropy — Rolling Window

**Input**: Daily `oil_return` series.

**Procedure**:
1. Rolling window: **60 trading days** (~3 months)
2. Within each window, discretize returns into **k = 20** equal-width bins spanning the window's min-max range
3. Compute frequency of each bin: $p_i = n_i / N$
4. Calculate Shannon Entropy: $H = -\sum_{i=1}^{k} p_i \log_2(p_i)$, with convention $0 \cdot \log_2(0) = 0$
5. Output: Entropy time series (first 59 values are NaN)

**Interpretation**:
- High entropy → returns spread across many bins → market disorder / high uncertainty
- Low entropy → returns concentrated in few bins → orderly / trending market

### 4.2 State-Space Model & Kalman Filter

**State equation** (hidden state evolution):

$$x_t = A \cdot x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$

- $x_t \in \mathbb{R}$: 1-dimensional hidden state ("market stress index")
- $A \in \mathbb{R}$: state transition coefficient (expected near 1, stress is persistent)
- $Q \in \mathbb{R}$: state noise variance

**Observation equation** (observable variables):

$$y_t = C \cdot x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

- $y_t \in \mathbb{R}^3$: observation vector containing:
  1. Rolling Shannon Entropy (from 4.1)
  2. `OVX_Close` (oil implied volatility)
  3. `|oil_return|` (absolute return, realized volatility proxy)
- $C \in \mathbb{R}^{3 \times 1}$: observation matrix
- $R \in \mathbb{R}^{3 \times 3}$: observation noise covariance

**Parameter estimation**: EM algorithm via `pykalman.KalmanFilter` to jointly estimate $A$, $C$, $Q$, $R$.

**Output**: Filtered hidden state series $\{\hat{x}_t\}$ for the full sample.

**Preprocessing**: All three observation variables are **standardized** (z-score) before feeding into the Kalman Filter so they are on comparable scales.

### 4.3 Regime Classification

Threshold the filtered hidden state $\{\hat{x}_t\}$ using percentiles:

- **Regime 0** (Low stress / Calm): $\hat{x}_t < P_{33}$
- **Regime 1** (Medium stress / Transition): $P_{33} \leq \hat{x}_t < P_{67}$
- **Regime 2** (High stress / Crisis): $\hat{x}_t \geq P_{67}$

where $P_{33}$ and $P_{67}$ are the 33rd and 67th percentiles of the hidden state series.

### 4.4 Regime-Enhanced XGBoost

**Baseline model features**: Brent_Price, DTWEXBGS, EFFR, Gas_Price, Gold_Price, OVX_Close, plus forward-filled weekly/monthly variables.

**Enhanced model features**: All baseline features + two new features:
- `regime_label`: categorical (0, 1, 2)
- `hidden_state`: continuous Kalman Filter hidden state value

**Target**: Next-day WTI_Price (i.e., use features at time $t$ to predict WTI_Price at $t+1$).

**Train/Test split**: 80% train / 20% test, chronological order (no shuffling).

**Evaluation metrics**: RMSE, MAE, MAPE.

**XGBoost hyperparameters**: Consistent with group project settings; no extensive tuning needed for this comparison.

---

## 5. Expected Outputs & Visualizations

### Part I — Regime Detection
1. **Figure: Oil price + Entropy time series** — dual-axis plot showing WTI price and rolling Shannon Entropy over time
2. **Figure: Kalman Filter hidden state** — hidden state time series with regime threshold lines (P33, P67) marked
3. **Figure: Regime-colored price chart** — WTI price plot with background shading by detected regime (green/yellow/red)
4. **Table: Regime characteristics** — for each regime, report mean/std of oil_return, OVX, Entropy, avg duration, number of periods
5. **Discussion**: Alignment of detected regimes with known events (COVID crash, Ukraine war, etc.)

### Part II — Regime-Enhanced Prediction
6. **Table: Model comparison** — Baseline XGBoost vs Enhanced XGBoost on RMSE, MAE, MAPE
7. **Figure: Prediction plot** — actual vs predicted WTI price for both models on test set
8. **Figure: Feature importance** — XGBoost feature importance comparison showing where regime features rank

---

## 6. Code Structure

```
claude/
  01_regime_detection.ipynb    # Data loading, preprocessing, Shannon Entropy,
                               # Kalman Filter, regime detection, visualizations,
                               # exports processed dataset with regime features
  02_xgboost_comparison.ipynb  # Loads processed dataset, Baseline vs Enhanced
                               # XGBoost comparison, evaluation metrics, plots
  dataset_daily.csv
  dataset_weekly.csv
  dataset_monthly.csv
```

The first notebook outputs a processed CSV (e.g., `dataset_with_regime.csv`) containing all original daily features plus `rolling_entropy`, `hidden_state`, and `regime_label`. The second notebook reads this CSV directly.

### Dependencies
- `numpy`, `pandas`: data handling
- `scipy.stats`: entropy calculation
- `pykalman`: Kalman Filter implementation
- `xgboost`: prediction model
- `matplotlib`, `seaborn`: visualization

---

## 7. Scope Boundaries

**In scope**:
- Shannon Entropy rolling window on oil returns
- Linear Gaussian Kalman Filter with 1D hidden state and 3 observation variables
- Percentile-based 3-regime classification
- XGBoost baseline vs enhanced comparison (single model)

**Out of scope**:
- Markov-switching models or nonlinear state-space models
- Deep hyperparameter tuning of XGBoost
- Multiple prediction models comparison (LSTM, Prophet — covered in group project)
- Literature review and references (covered in group project)
- Formal statistical tests for regime significance

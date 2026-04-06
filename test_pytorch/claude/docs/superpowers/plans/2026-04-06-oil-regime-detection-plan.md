# Oil Market Regime Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two Jupyter notebooks — one for regime detection (Shannon Entropy + Kalman Filter) and one for regime-enhanced XGBoost prediction — for a 15-page COMP0040 individual assignment.

**Architecture:** Notebook 1 loads three CSV datasets, computes rolling Shannon Entropy on oil returns, fits a linear Gaussian state-space model via Kalman Filter to extract a hidden "market stress" state, classifies 3 regimes by percentile thresholds, produces all Part I visualizations, and exports a processed CSV. Notebook 2 loads that CSV, trains baseline vs regime-enhanced XGBoost models (predicting log returns, converting to price), and compares them.

**Tech Stack:** Python 3, pandas, numpy, scipy.stats, pykalman, xgboost, scikit-learn, matplotlib, seaborn

---

## File Map

| File | Role |
|---|---|
| `claude/01_regime_detection.ipynb` | **Create.** Data loading, merging, Shannon Entropy, Kalman Filter, regime detection, all Part I figures/tables, export `dataset_with_regime.csv` |
| `claude/02_xgboost_comparison.ipynb` | **Create.** Load processed data, baseline XGBoost, enhanced XGBoost, evaluation metrics, Part II figures/tables |
| `claude/dataset_with_regime.csv` | **Generated output** from Notebook 1, consumed by Notebook 2 |
| `claude/dataset_daily.csv` | Existing input (1619 rows) |
| `claude/dataset_weekly.csv` | Existing input (324 rows) |
| `claude/dataset_monthly.csv` | Existing input (75 rows) |

---

## Task 1: Notebook 1 — Setup and Data Loading

**Files:**
- Create: `claude/01_regime_detection.ipynb`

- [ ] **Step 1: Create notebook with imports cell**

```python
# Cell 1 — Imports
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Install pykalman if needed
try:
    from pykalman import KalmanFilter
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'pykalman'])
    from pykalman import KalmanFilter

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')
```

- [ ] **Step 2: Add data loading cell**

```python
# Cell 2 — Load datasets
df_daily = pd.read_csv('dataset_daily.csv', parse_dates=['observation_date'])
df_weekly = pd.read_csv('dataset_weekly.csv', parse_dates=['observation_date'])
df_monthly = pd.read_csv('dataset_monthly.csv', parse_dates=['observation_date'])

print(f"Daily:   {df_daily.shape[0]} rows, {df_daily.shape[1]} cols, "
      f"{df_daily['observation_date'].min().date()} to {df_daily['observation_date'].max().date()}")
print(f"Weekly:  {df_weekly.shape[0]} rows, {df_weekly.shape[1]} cols")
print(f"Monthly: {df_monthly.shape[0]} rows, {df_monthly.shape[1]} cols")
df_daily.head()
```

- [ ] **Step 3: Add descriptive statistics cell**

```python
# Cell 3 — Descriptive statistics
desc_cols = ['Brent_Price', 'WTI_Price', 'oil_return', 'OVX_Close',
             'DTWEXBGS', 'EFFR', 'Gas_Price', 'Gold_Price']
desc = df_daily[desc_cols].describe().T
desc['skew'] = df_daily[desc_cols].skew()
desc['kurtosis'] = df_daily[desc_cols].kurtosis()
desc.round(4)
```

- [ ] **Step 4: Add WTI price time series plot cell**

```python
# Cell 4 — WTI price time series with key events
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_daily['observation_date'], df_daily['WTI_Price'], color='steelblue', linewidth=1)
ax.set_xlabel('Date')
ax.set_ylabel('WTI Price (USD/barrel)')
ax.set_title('WTI Crude Oil Price (2020–2026)')

# Annotate key events
events = {
    '2020-04-20': 'COVID Crash',
    '2022-03-08': 'Russia-Ukraine War',
    '2026-03-10': 'Recent Volatility'
}
for date_str, label in events.items():
    date = pd.Timestamp(date_str)
    if date in df_daily['observation_date'].values:
        price = df_daily.loc[df_daily['observation_date'] == date, 'WTI_Price'].values[0]
    else:
        idx = (df_daily['observation_date'] - date).abs().idxmin()
        price = df_daily.loc[idx, 'WTI_Price']
    ax.annotate(label, xy=(date, price), xytext=(0, 20),
                textcoords='offset points', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'),
                ha='center', color='red')
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Run all cells, verify data loads correctly**

Run the notebook top-to-bottom. Expected: daily dataset has 1619 rows with columns including `observation_date`, `WTI_Price`, `oil_return`, `OVX_Close`. The price plot should show the COVID crash dip around April 2020 and the Ukraine spike in early 2022.

- [ ] **Step 6: Commit**

```bash
git add claude/01_regime_detection.ipynb
git commit -m "feat: notebook 1 — data loading and descriptive statistics"
```

---

## Task 2: Notebook 1 — Shannon Entropy Calculation

**Files:**
- Modify: `claude/01_regime_detection.ipynb`

- [ ] **Step 1: Add rolling Shannon Entropy function cell**

```python
# Cell 5 — Rolling Shannon Entropy
def rolling_shannon_entropy(series, window=60, n_bins=20):
    """
    Compute rolling Shannon Entropy on a return series.

    For each window of `window` observations, discretize returns into
    `n_bins` equal-width bins and compute Shannon entropy (base 2).

    Parameters
    ----------
    series : pd.Series — return series
    window : int — rolling window size (default 60 trading days)
    n_bins : int — number of bins for discretization (default 20)

    Returns
    -------
    pd.Series — entropy values (first window-1 entries are NaN)
    """
    entropy_values = []
    for i in range(len(series)):
        if i < window - 1:
            entropy_values.append(np.nan)
            continue
        window_data = series.iloc[i - window + 1 : i + 1].values
        # Discretize into equal-width bins
        counts, _ = np.histogram(window_data, bins=n_bins)
        # Convert to probabilities
        probs = counts / counts.sum()
        # Remove zero probabilities (0 * log(0) = 0 by convention)
        probs = probs[probs > 0]
        # Shannon entropy in bits (base 2)
        H = -np.sum(probs * np.log2(probs))
        entropy_values.append(H)
    return pd.Series(entropy_values, index=series.index)
```

- [ ] **Step 2: Add entropy computation and dual-axis plot cell**

```python
# Cell 6 — Compute rolling entropy
df_daily['rolling_entropy'] = rolling_shannon_entropy(df_daily['oil_return'], window=60, n_bins=20)

print(f"Entropy range: {df_daily['rolling_entropy'].min():.4f} to {df_daily['rolling_entropy'].max():.4f}")
print(f"Entropy mean:  {df_daily['rolling_entropy'].mean():.4f}")
print(f"NaN count:     {df_daily['rolling_entropy'].isna().sum()}")
```

```python
# Cell 7 — Dual-axis plot: WTI Price + Shannon Entropy
fig, ax1 = plt.subplots(figsize=(14, 6))

color_price = 'steelblue'
ax1.set_xlabel('Date')
ax1.set_ylabel('WTI Price (USD/barrel)', color=color_price)
ax1.plot(df_daily['observation_date'], df_daily['WTI_Price'],
         color=color_price, linewidth=1, alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color_price)

ax2 = ax1.twinx()
color_entropy = 'darkorange'
ax2.set_ylabel('Rolling Shannon Entropy (60-day)', color=color_entropy)
ax2.plot(df_daily['observation_date'], df_daily['rolling_entropy'],
         color=color_entropy, linewidth=1, alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color_entropy)

fig.suptitle('WTI Crude Oil Price and Rolling Shannon Entropy', fontsize=14)
fig.tight_layout()
plt.show()
```

- [ ] **Step 3: Run cells, verify entropy time series**

Expected: Entropy values roughly between 2.0–4.3 (depends on data distribution). Entropy should spike around COVID crash (March–April 2020) and Ukraine war (Feb–March 2022) periods, reflecting elevated market disorder.

- [ ] **Step 4: Commit**

```bash
git add claude/01_regime_detection.ipynb
git commit -m "feat: notebook 1 — rolling Shannon Entropy calculation and plot"
```

---

## Task 3: Notebook 1 — Kalman Filter and Regime Detection

**Files:**
- Modify: `claude/01_regime_detection.ipynb`

- [ ] **Step 1: Add Kalman Filter state-space model cell**

```python
# Cell 8 — Prepare observation matrix for Kalman Filter
# Drop rows where rolling_entropy is NaN (first 59 rows)
df_kf = df_daily.dropna(subset=['rolling_entropy']).copy()

# Three observation variables, all standardized (z-score)
obs_cols = ['rolling_entropy', 'OVX_Close', 'abs_return']
df_kf['abs_return'] = df_kf['oil_return'].abs()

# Standardize
for col in obs_cols:
    mean_val = df_kf[col].mean()
    std_val = df_kf[col].std()
    df_kf[f'{col}_z'] = (df_kf[col] - mean_val) / std_val

obs_cols_z = [f'{col}_z' for col in obs_cols]
observations = df_kf[obs_cols_z].values

print(f"Observation matrix shape: {observations.shape}")
print(f"Date range: {df_kf['observation_date'].min().date()} to {df_kf['observation_date'].max().date()}")
```

- [ ] **Step 2: Add Kalman Filter fitting cell**

```python
# Cell 9 — Fit Kalman Filter using EM algorithm
# State-space model:
#   State eq:   x_t = A * x_{t-1} + w_t,  w_t ~ N(0, Q)
#   Obs eq:     y_t = C * x_t + v_t,       v_t ~ N(0, R)
#
# x_t: 1D hidden state ("market stress index")
# y_t: 3D observation (entropy_z, OVX_z, abs_return_z)

kf = KalmanFilter(
    n_dim_state=1,        # 1D hidden state
    n_dim_obs=3,          # 3 observation variables
    em_vars=['transition_matrices',        # A
             'observation_matrices',       # C
             'transition_covariance',      # Q
             'observation_covariance',     # R
             'initial_state_mean',
             'initial_state_covariance'],
)

# Fit via EM (10 iterations is typically sufficient)
kf = kf.em(observations, n_iter=10)

print("Estimated parameters:")
print(f"  A (transition matrix):  {kf.transition_matrices.flatten()}")
print(f"  C (observation matrix): {kf.observation_matrices.flatten()}")
print(f"  Q (state noise var):    {kf.transition_covariance.flatten()}")
print(f"  R diag (obs noise var): {np.diag(kf.observation_covariance)}")
```

- [ ] **Step 3: Add hidden state extraction and regime classification cell**

```python
# Cell 10 — Extract filtered hidden state
filtered_state_means, filtered_state_covs = kf.filter(observations)
hidden_state = filtered_state_means.flatten()

# Assign back to dataframe
df_kf['hidden_state'] = hidden_state

# Regime classification using 33rd and 67th percentiles
p33 = np.percentile(hidden_state, 33)
p67 = np.percentile(hidden_state, 67)

df_kf['regime_label'] = 0  # Low stress (Calm)
df_kf.loc[df_kf['hidden_state'] >= p33, 'regime_label'] = 1  # Medium stress (Transition)
df_kf.loc[df_kf['hidden_state'] >= p67, 'regime_label'] = 2  # High stress (Crisis)

print(f"Regime thresholds: P33 = {p33:.4f}, P67 = {p67:.4f}")
print(f"\nRegime distribution:")
print(df_kf['regime_label'].value_counts().sort_index())
```

- [ ] **Step 4: Add hidden state time series plot with threshold lines**

```python
# Cell 11 — Hidden state time series with regime thresholds
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_kf['observation_date'], df_kf['hidden_state'],
        color='navy', linewidth=1, label='Hidden State')
ax.axhline(y=p33, color='orange', linestyle='--', linewidth=1, label=f'P33 = {p33:.2f}')
ax.axhline(y=p67, color='red', linestyle='--', linewidth=1, label=f'P67 = {p67:.2f}')
ax.fill_between(df_kf['observation_date'], ax.get_ylim()[0], p33, alpha=0.1, color='green')
ax.fill_between(df_kf['observation_date'], p33, p67, alpha=0.1, color='orange')
ax.fill_between(df_kf['observation_date'], p67, ax.get_ylim()[1], alpha=0.1, color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Hidden State (Market Stress Index)')
ax.set_title('Kalman Filter Hidden State with Regime Thresholds')
ax.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Run cells, verify Kalman Filter converges and regimes are sensible**

Expected: The EM algorithm should converge. `A` should be close to 1 (state persistence). The hidden state time series should show elevated values during COVID (2020-03/04) and Ukraine war (2022-02/03). Each regime should have approximately 1/3 of the data points (by construction of percentile thresholds).

- [ ] **Step 6: Commit**

```bash
git add claude/01_regime_detection.ipynb
git commit -m "feat: notebook 1 — Kalman Filter state-space model and regime detection"
```

---

## Task 4: Notebook 1 — Regime Visualizations and Analysis

**Files:**
- Modify: `claude/01_regime_detection.ipynb`

- [ ] **Step 1: Add regime-colored price chart cell**

```python
# Cell 12 — Regime-colored WTI price chart
regime_colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
regime_labels = {0: 'Regime 0 (Calm)', 1: 'Regime 1 (Transition)', 2: 'Regime 2 (Crisis)'}

fig, ax = plt.subplots(figsize=(14, 6))

# Plot background shading by regime
for regime in [0, 1, 2]:
    mask = df_kf['regime_label'] == regime
    dates = df_kf.loc[mask, 'observation_date']
    for d in dates:
        ax.axvspan(d - pd.Timedelta(days=0.5), d + pd.Timedelta(days=0.5),
                   alpha=0.3, color=regime_colors[regime], linewidth=0)

# Plot price line on top
ax.plot(df_kf['observation_date'], df_kf['WTI_Price'],
        color='black', linewidth=1, zorder=5)

# Legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=regime_colors[r], alpha=0.4, label=regime_labels[r])
                  for r in [0, 1, 2]]
ax.legend(handles=legend_handles, loc='upper left')
ax.set_xlabel('Date')
ax.set_ylabel('WTI Price (USD/barrel)')
ax.set_title('WTI Price with Detected Market Regimes')
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Add regime characteristics table cell**

```python
# Cell 13 — Regime characteristics table
regime_stats = df_kf.groupby('regime_label').agg(
    count=('WTI_Price', 'size'),
    mean_return=('oil_return', 'mean'),
    std_return=('oil_return', 'std'),
    mean_OVX=('OVX_Close', 'mean'),
    std_OVX=('OVX_Close', 'std'),
    mean_entropy=('rolling_entropy', 'mean'),
    std_entropy=('rolling_entropy', 'std'),
    mean_WTI=('WTI_Price', 'mean'),
    std_WTI=('WTI_Price', 'std'),
).round(4)

# Calculate average consecutive duration per regime
def avg_consecutive_duration(series, regime):
    """Average number of consecutive days in a given regime."""
    is_regime = (series == regime).astype(int)
    # Find blocks: diff != 0 marks transitions
    blocks = (is_regime.diff().fillna(0) != 0).cumsum()
    # Only keep blocks that are in the regime
    regime_blocks = blocks[is_regime == 1]
    if len(regime_blocks) == 0:
        return 0
    durations = regime_blocks.value_counts()
    return durations.mean()

for r in [0, 1, 2]:
    regime_stats.loc[r, 'avg_duration_days'] = avg_consecutive_duration(
        df_kf['regime_label'], r)

regime_stats.index = ['Regime 0 (Calm)', 'Regime 1 (Transition)', 'Regime 2 (Crisis)']
regime_stats
```

- [ ] **Step 3: Add regime box plots cell**

```python
# Cell 14 — Box plots of key variables by regime
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax_i, (col, title) in enumerate([
    ('oil_return', 'Oil Return'),
    ('OVX_Close', 'OVX (Implied Volatility)'),
    ('rolling_entropy', 'Shannon Entropy')
]):
    data_to_plot = [df_kf.loc[df_kf['regime_label'] == r, col].dropna() for r in [0, 1, 2]]
    bp = axes[ax_i].boxplot(data_to_plot, labels=['Calm', 'Transition', 'Crisis'],
                            patch_artist=True)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[ax_i].set_title(title)

fig.suptitle('Distribution of Key Variables by Regime', fontsize=14)
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Run cells, verify visualizations**

Expected: The regime-colored price chart should show green (calm) during 2023–2025 stabilization, red (crisis) during COVID crash and Ukraine war, and orange (transition) in between. The characteristics table should show regime 2 having the highest return volatility and OVX, regime 0 the lowest.

- [ ] **Step 5: Commit**

```bash
git add claude/01_regime_detection.ipynb
git commit -m "feat: notebook 1 — regime visualizations and characteristics analysis"
```

---

## Task 5: Notebook 1 — Data Merging and Export

**Files:**
- Modify: `claude/01_regime_detection.ipynb`

- [ ] **Step 1: Add data merging cell (forward-fill weekly and monthly to daily)**

```python
# Cell 15 — Merge weekly and monthly data into daily via forward-fill
# Weekly: merge on date, forward-fill
df_merged = df_kf.copy()

# Merge weekly data
df_merged = pd.merge_asof(
    df_merged.sort_values('observation_date'),
    df_weekly.sort_values('observation_date'),
    on='observation_date',
    direction='backward'  # use most recent weekly value available
)

# Merge monthly data
df_merged = pd.merge_asof(
    df_merged.sort_values('observation_date'),
    df_monthly.sort_values('observation_date'),
    on='observation_date',
    direction='backward'  # use most recent monthly value available
)

print(f"Merged dataset shape: {df_merged.shape}")
print(f"Columns: {list(df_merged.columns)}")
print(f"\nNull counts in merged data:")
print(df_merged.isnull().sum()[df_merged.isnull().sum() > 0])
```

- [ ] **Step 2: Add target column creation and export cell**

```python
# Cell 16 — Create next-day WTI price target and export
# Target: next-day WTI price (shift -1)
df_merged['WTI_Price_next'] = df_merged['WTI_Price'].shift(-1)

# Drop last row (no next-day target) and any remaining NaN rows
df_export = df_merged.dropna().copy()

# Select columns to export
export_cols = [
    'observation_date',
    # Original daily features
    'Brent_Price', 'DTWEXBGS', 'EFFR', 'Gas_Price', 'Gold_Price',
    'OVX_Close', 'WTI_Price', 'oil_return',
    # Weekly features (forward-filled)
    'WCESTUS1', 'supply_shock', 'supply_shock_pct',
    # Monthly features (forward-filled)
    'GPR', 'GPRT', 'GPRC', 'GEPUCURRENT', 'CPILFESL',
    'INDPRO', 'industrial_production_growth',
    # Regime detection outputs
    'rolling_entropy', 'hidden_state', 'regime_label',
    # Target
    'WTI_Price_next'
]

df_export = df_export[export_cols]
df_export.to_csv('dataset_with_regime.csv', index=False)

print(f"Exported {df_export.shape[0]} rows, {df_export.shape[1]} columns")
print(f"Date range: {df_export['observation_date'].min()} to {df_export['observation_date'].max()}")
print(f"\nRegime distribution in exported data:")
print(df_export['regime_label'].value_counts().sort_index())
```

- [ ] **Step 3: Run cells, verify export**

Expected: The exported CSV should have ~1559 rows (1619 minus 59 NaN entropy rows minus 1 for next-day target shift). All columns should be present with no NaN values. The file `dataset_with_regime.csv` should appear in the `claude/` directory.

- [ ] **Step 4: Commit**

```bash
git add claude/01_regime_detection.ipynb claude/dataset_with_regime.csv
git commit -m "feat: notebook 1 — merge datasets, create target, export dataset_with_regime.csv"
```

---

## Task 6: Notebook 2 — Baseline XGBoost

**Files:**
- Create: `claude/02_xgboost_comparison.ipynb`

- [ ] **Step 1: Create notebook with imports and data loading cell**

```python
# Cell 1 — Imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')
```

```python
# Cell 2 — Load processed dataset
df = pd.read_csv('dataset_with_regime.csv', parse_dates=['observation_date'])
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()
```

- [ ] **Step 2: Add feature engineering and train/test split cell**

```python
# Cell 3 — Feature engineering (lag features, consistent with group project)
df['lag1'] = df['oil_return'].shift(1)
df['lag2'] = df['oil_return'].shift(2)
df['lag_price'] = df['WTI_Price'].shift(1)

# Target: log return (predict next-day return, then convert to price)
df['target_return'] = np.log(df['WTI_Price_next'] / df['WTI_Price'])

# Drop rows with NaN from lag creation
df = df.dropna().copy()

print(f"After lag features: {df.shape[0]} rows")
```

```python
# Cell 4 — Train/test split (80/20, chronological)
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"Train: {len(train_df)} rows ({train_df['observation_date'].min().date()} to {train_df['observation_date'].max().date()})")
print(f"Test:  {len(test_df)} rows ({test_df['observation_date'].min().date()} to {test_df['observation_date'].max().date()})")
```

- [ ] **Step 3: Add baseline XGBoost training cell**

```python
# Cell 5 — Define feature sets
# Baseline features: all features EXCEPT regime-related ones
baseline_features = [
    'Brent_Price', 'DTWEXBGS', 'EFFR', 'Gas_Price', 'Gold_Price', 'OVX_Close',
    'WCESTUS1', 'supply_shock', 'supply_shock_pct',
    'GPR', 'GPRT', 'GPRC', 'GEPUCURRENT', 'CPILFESL',
    'INDPRO', 'industrial_production_growth',
    'lag1', 'lag2', 'lag_price'
]

# Enhanced features: baseline + regime features
enhanced_features = baseline_features + ['rolling_entropy', 'hidden_state', 'regime_label']

target_col = 'target_return'
```

```python
# Cell 6 — Train baseline XGBoost
# Hyperparameters consistent with group project
xgb_params = dict(
    n_estimators=5000,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2,
    random_state=42,
    objective='reg:squarederror'
)

# Baseline model
model_baseline = xgb.XGBRegressor(**xgb_params)
model_baseline.fit(train_df[baseline_features], train_df[target_col])

# Predict
y_pred_baseline_train = model_baseline.predict(train_df[baseline_features])
y_pred_baseline_test = model_baseline.predict(test_df[baseline_features])

# Convert log return predictions to price predictions
price_pred_baseline = test_df['WTI_Price'].values * np.exp(y_pred_baseline_test)
price_true = test_df['WTI_Price_next'].values

# Evaluate
rmse_baseline = np.sqrt(mean_squared_error(price_true, price_pred_baseline))
mae_baseline = mean_absolute_error(price_true, price_pred_baseline)
mape_baseline = np.mean(np.abs((price_true - price_pred_baseline) / price_true)) * 100

print("=== Baseline XGBoost ===")
print(f"RMSE: {rmse_baseline:.4f}")
print(f"MAE:  {mae_baseline:.4f}")
print(f"MAPE: {mape_baseline:.4f}%")
```

- [ ] **Step 4: Run cells, verify baseline model trains successfully**

Expected: The baseline model should produce RMSE around 1–3 (depends on test period). Verify there are no NaN predictions and the metrics look reasonable.

- [ ] **Step 5: Commit**

```bash
git add claude/02_xgboost_comparison.ipynb
git commit -m "feat: notebook 2 — baseline XGBoost model"
```

---

## Task 7: Notebook 2 — Enhanced XGBoost and Comparison

**Files:**
- Modify: `claude/02_xgboost_comparison.ipynb`

- [ ] **Step 1: Add enhanced XGBoost training cell**

```python
# Cell 7 — Train enhanced XGBoost (with regime features)
model_enhanced = xgb.XGBRegressor(**xgb_params)
model_enhanced.fit(train_df[enhanced_features], train_df[target_col])

# Predict
y_pred_enhanced_test = model_enhanced.predict(test_df[enhanced_features])

# Convert to price
price_pred_enhanced = test_df['WTI_Price'].values * np.exp(y_pred_enhanced_test)

# Evaluate
rmse_enhanced = np.sqrt(mean_squared_error(price_true, price_pred_enhanced))
mae_enhanced = mean_absolute_error(price_true, price_pred_enhanced)
mape_enhanced = np.mean(np.abs((price_true - price_pred_enhanced) / price_true)) * 100

print("=== Enhanced XGBoost (with Regime Features) ===")
print(f"RMSE: {rmse_enhanced:.4f}")
print(f"MAE:  {mae_enhanced:.4f}")
print(f"MAPE: {mape_enhanced:.4f}%")
```

- [ ] **Step 2: Add comparison table cell**

```python
# Cell 8 — Model comparison table
comparison = pd.DataFrame({
    'Model': ['Baseline XGBoost', 'Enhanced XGBoost (+ Regime)'],
    'RMSE': [rmse_baseline, rmse_enhanced],
    'MAE': [mae_baseline, mae_enhanced],
    'MAPE (%)': [mape_baseline, mape_enhanced]
}).round(4)

# Add improvement row
comparison.loc[2] = [
    'Improvement (%)',
    f"{(rmse_baseline - rmse_enhanced) / rmse_baseline * 100:.2f}%",
    f"{(mae_baseline - mae_enhanced) / mae_baseline * 100:.2f}%",
    f"{(mape_baseline - mape_enhanced) / mape_baseline * 100:.2f}%"
]

comparison
```

- [ ] **Step 3: Add prediction plot cell**

```python
# Cell 9 — Prediction plot: actual vs predicted for both models
test_dates = test_df['observation_date'].values

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_dates, price_true, color='black', linewidth=1.5, label='Actual', alpha=0.9)
ax.plot(test_dates, price_pred_baseline, color='steelblue', linewidth=1,
        label='Baseline XGBoost', alpha=0.7, linestyle='--')
ax.plot(test_dates, price_pred_enhanced, color='darkorange', linewidth=1,
        label='Enhanced XGBoost', alpha=0.7, linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('WTI Price (USD/barrel)')
ax.set_title('WTI Price Prediction: Baseline vs Regime-Enhanced XGBoost')
ax.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Add feature importance comparison cell**

```python
# Cell 10 — Feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Baseline feature importance
imp_baseline = pd.Series(model_baseline.feature_importances_,
                          index=baseline_features).sort_values()
imp_baseline.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Baseline XGBoost')
axes[0].set_xlabel('Feature Importance')

# Enhanced feature importance
imp_enhanced = pd.Series(model_enhanced.feature_importances_,
                          index=enhanced_features).sort_values()
# Highlight regime features
colors = ['darkorange' if f in ['rolling_entropy', 'hidden_state', 'regime_label']
          else 'steelblue' for f in imp_enhanced.index]
imp_enhanced.plot(kind='barh', ax=axes[1], color=colors)
axes[1].set_title('Enhanced XGBoost (regime features in orange)')
axes[1].set_xlabel('Feature Importance')

fig.suptitle('Feature Importance Comparison', fontsize=14)
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Add summary markdown cell**

```markdown
# Cell 11 (Markdown)
## Summary

**Part II Findings:**
- The Enhanced XGBoost model incorporates three regime-derived features: `rolling_entropy`, `hidden_state`, and `regime_label`.
- Comparison metrics (RMSE, MAE, MAPE) are reported in the table above.
- The feature importance plot shows how regime features rank relative to traditional market variables.
```

- [ ] **Step 6: Run all cells, verify everything works end-to-end**

Run Notebook 2 top-to-bottom. Verify: data loads from `dataset_with_regime.csv`, both models train without errors, comparison table displays, prediction plot shows both models, feature importance plot highlights regime features.

- [ ] **Step 7: Commit**

```bash
git add claude/02_xgboost_comparison.ipynb
git commit -m "feat: notebook 2 — enhanced XGBoost comparison and all Part II visualizations"
```

---

## Task 8: Final Polish and Validation

**Files:**
- Modify: `claude/01_regime_detection.ipynb`
- Modify: `claude/02_xgboost_comparison.ipynb`

- [ ] **Step 1: Add markdown section headers throughout Notebook 1**

Ensure the following markdown cells exist at appropriate locations in Notebook 1:

```markdown
# 1. Data Loading and Descriptive Statistics
```
(before Cell 2)

```markdown
# 2. Shannon Entropy — Rolling Window
```
(before Cell 5)

```markdown
# 3. Kalman Filter — State-Space Model and Regime Detection
```
(before Cell 8)

```markdown
# 4. Regime Analysis and Visualization
```
(before Cell 12)

```markdown
# 5. Data Merging and Export
```
(before Cell 15)

- [ ] **Step 2: Add markdown section headers throughout Notebook 2**

Ensure the following markdown cells exist at appropriate locations in Notebook 2:

```markdown
# 1. Data Loading and Feature Engineering
```
(before Cell 2)

```markdown
# 2. Baseline XGBoost Model
```
(before Cell 5)

```markdown
# 3. Enhanced XGBoost Model (with Regime Features)
```
(before Cell 7)

```markdown
# 4. Model Comparison
```
(before Cell 8)

- [ ] **Step 3: Run both notebooks top-to-bottom in sequence**

1. Run Notebook 1 fully — verify `dataset_with_regime.csv` is generated
2. Run Notebook 2 fully — verify all outputs render

Expected: No errors. All figures display. All tables display. The exported CSV exists and is loaded correctly by Notebook 2.

- [ ] **Step 4: Final commit**

```bash
git add claude/01_regime_detection.ipynb claude/02_xgboost_comparison.ipynb
git commit -m "polish: add section headers and final cleanup to both notebooks"
```

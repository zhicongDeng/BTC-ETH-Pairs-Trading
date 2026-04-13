"""Fix cell 19 so df_kf includes all columns from df, not just obs_cols."""
import json

nb = json.load(open('01_regime_detection.ipynb', 'r', encoding='utf-8'))

lines_19 = [
    "# Prepare KF observations: 4 standardised variables\n",
    "# Use all columns from df (not just obs_cols) so that downstream cells\n",
    "# can access OVX_Close, WTI_Price, oil_return, GPRD etc.\n",
    "obs_list = ['rolling_entropy', 'log_OVX', 'rolling_std_21', 'log_GPRD_r21']\n",
    "valid_mask = df[obs_list].notna().all(axis=1)\n",
    "df_kf = df.loc[valid_mask].copy().reset_index(drop=True)\n",
    "\n",
    "for col in obs_list:\n",
    "    mean_val = df_kf[col].mean()\n",
    "    std_val  = df_kf[col].std()\n",
    "    df_kf[f'{col}_z'] = (df_kf[col] - mean_val) / std_val\n",
    "\n",
    "obs_z_cols = [f'{col}_z' for col in obs_list]\n",
    "observations = df_kf[obs_z_cols].values\n",
    "\n",
    'print(f"Observation matrix shape: {observations.shape}")\n',
    'print(f"Date range: {df_kf[\'observation_date\'].min().date()} to {df_kf[\'observation_date\'].max().date()}")\n',
    'print(f"df_kf columns: {len(df_kf.columns)}")\n',
]

nb['cells'][19]['source'] = lines_19

json.dump(nb, open('01_regime_detection.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print("Cell 19 fixed: df_kf now uses all columns from df")

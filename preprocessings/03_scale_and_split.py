# =============================================================================
# RanDe-W11 | Step 3: Feature Scaling and Cross-Validation Setup
# =============================================================================
# Applies StandardScaler and RobustScaler to the feature matrix, and
# initialises the unified 5-fold stratified cross-validation protocol
# used across all experiments reported in the paper.
#
# StandardScaler : used for classical ML and deep learning models
# RobustScaler   : used for radar/behavioral fingerprint visualisations
#                  (median-centered, resistant to outlier feature values)
#
# ⚠ IMPORTANT — Scaler Fitting Protocol:
#   Both scalers are fit on the FULL dataset in this script for inspection
#   and verification purposes only. In the experimental pipeline, scaling
#   is applied INSIDE each CV fold: the scaler is fit on the train split
#   only, then used to transform the test split — ensuring no test-set
#   statistics leak into training across folds.
#
# Input : RanDe_W11_dataset_final.csv
# Output: console report (no file written — scalers are fit per-fold in
#         the main experimental pipeline)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold

# --- Load ---
df = pd.read_csv('RanDe_W11_dataset_final.csv')

LABEL_COLS = ['ground_truth_family', 'ground_truth_binary']
FEAT_COLS  = [c for c in df.columns if c not in LABEL_COLS]

X   = df[FEAT_COLS].values.astype(float)
y   = df['ground_truth_binary'].values
y_f = df['ground_truth_family'].values

print(f"Loaded  : RanDe_W11_dataset_final.csv")
print(f"Shape   : {df.shape}  |  Features: {len(FEAT_COLS)}")
print(f"Classes : binary={int(y.sum())} ransomware / {int((y==0).sum())} benign  "
      f"|  families={len(np.unique(y_f))}\n")

# --- StandardScaler (full-dataset fit — inspection only) ---
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# NOTE: Scaler is fit on the full dataset here for inspection/verification
# purposes only. In the experimental pipeline, scaling is applied INSIDE
# each CV fold (fit on train split, transform test split) to prevent
# data leakage across folds.

print(f"StandardScaler applied (full-dataset fit — inspection only).")
print(f"  Mean range  : [{X_sc.mean(axis=0).min():.4f}, {X_sc.mean(axis=0).max():.4f}]")
print(f"  Std range   : [{X_sc.std(axis=0).min():.4f},  {X_sc.std(axis=0).max():.4f}]\n")

# --- RobustScaler (full-dataset fit — inspection only) ---
rob_scaler = RobustScaler()
X_rb       = rob_scaler.fit_transform(X)

# NOTE: RobustScaler uses median and IQR rather than mean and std, making
# it resistant to outlier feature values. Used specifically for behavioral
# radar chart visualisations (domain fingerprints) in the paper.

print(f"RobustScaler applied (IQR-based, outlier-resistant — inspection only).")
print(f"  Median range: [{np.median(X_rb, axis=0).min():.4f}, "
      f"{np.median(X_rb, axis=0).max():.4f}]\n")

# --- Stratified K-Fold ---
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_sizes = [(len(tr), len(te)) for tr, te in SKF.split(X_sc, y)]

print(f"Cross-validation protocol: 5-fold stratified CV  |  random_state=42")
print(f"{'Fold':<6} {'Train':>8} {'Test':>8}")
print("-" * 26)
for i, (tr, te) in enumerate(fold_sizes, 1):
    print(f"  {i:<4} {tr:>8} {te:>8}")
print(f"\n  Unified protocol applied to: binary classification, "
      f"multiclass classification,\n"
      f"  learning curves, and group ablation experiments.\n")

print("Setup verified. Proceed to experimental pipeline.")
print("Reminder: fit scalers inside each fold in the full pipeline.")

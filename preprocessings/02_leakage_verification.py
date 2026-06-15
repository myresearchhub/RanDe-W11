# =============================================================================
# RanDe-W11 | Step 2: Data Leakage Verification
# =============================================================================
# Verifies that no label-derived columns are present in the feature matrix X.
#
# Two-layer check:
#   Layer 1 — LABEL_COLS: authoritative exclusion list applied to X
#   Layer 2 — Keyword scan: independent verification across all column names
#
# The keyword list covers direct label columns, Any.run metadata fields used
# for sample selection (verdict, threat), and behavioral tags that could
# conflate sandbox judgment with ground truth (malicious, suspicious).
#
# Any column flagged by the keyword scan that is NOT in LABEL_COLS is printed
# as a WARNING for the researcher to consciously confirm before proceeding.
#
# Input : RanDe_W11_dataset_final.csv
# Output: console report (no file written)
# =============================================================================

import pandas as pd

# --- Load ---
df = pd.read_csv('RanDe_W11_dataset_final.csv')

print(f"Loaded  : RanDe_W11_dataset_final.csv  |  Shape: {df.shape}")

# --- Layer 1: Authoritative label exclusion ---
LABEL_COLS = ['ground_truth_family', 'ground_truth_binary']
FEAT_COLS  = [c for c in df.columns if c not in LABEL_COLS]

X = df[FEAT_COLS].values
y = df['ground_truth_binary'].values

print(f"\nFeature matrix X : {X.shape}")
print(f"Label vector y   : {y.shape}")
print(f"Label cols excluded from X : {LABEL_COLS}")

# --- Layer 2: Keyword scan ---
# Covers:
#   'truth', 'ground' — ground_truth_* label columns
#   'binary', 'label', 'class', 'target' — generic label naming patterns
#   'family'  — ground_truth_family, or any residual family name column
#   'verdict' — verdict_score (Any.run selection criterion, not a feature)
#   'threat'  — threat_level (Any.run metadata, not a behavioral feature)
#   'malicious', 'suspicious' — sandbox judgment tags; flagged for
#                               researcher confirmation (not auto-excluded,
#                               as proc_anomalous_* are valid behavioral features)
LEAKAGE_KEYWORDS = [
    'truth',
    'binary',
    'label',
    'class',
    'target',
    'family',
    'verdict',
    'threat',
    'ground',
    'malicious',
    'suspicious',
]

leakage_flagged = [c for c in df.columns
                   if any(kw in c.lower() for kw in LEAKAGE_KEYWORDS)]
extra_leakage   = [c for c in leakage_flagged if c not in LABEL_COLS]

print(f"\n[Leakage check] Keyword scan flagged : {leakage_flagged}")

if extra_leakage:
    print(f"\n  *** WARNING — additional potential leakage columns detected:")
    for col in extra_leakage:
        print(f"      - {col}")
    print(f"  Review each column before use.")
    print(f"  Note: behavioral tags (e.g. proc_anomalous_*) are valid features")
    print(f"  if their values are sandbox observations, not label derivatives. ***")
else:
    print(f"  [OK] No extra leakage columns detected beyond LABEL_COLS.")
    print(f"  Dataset is clean for ML use.\n")

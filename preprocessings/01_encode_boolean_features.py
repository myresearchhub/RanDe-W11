# =============================================================================
# RanDe-W11 | Step 1: Boolean Feature Encoding
# =============================================================================
# Converts string boolean values ('TRUE'/'FALSE') produced by the Any.run
# sandbox parser into numeric binary integers (1/0) required for ML pipelines.
#
# Input : anyrun_sandbox_export_raw.csv   (raw merged Any.run sandbox export)
# Output: RanDe_W11_dataset_final.csv     (numerically encoded, ML-ready)
# =============================================================================

import pandas as pd

# --- Load ---
df = pd.read_csv('anyrun_sandbox_export_raw.csv')

print(f"Loaded  : anyrun_sandbox_export_raw.csv  |  Shape: {df.shape}")

# --- Audit before encoding ---
true_count  = (df == 'TRUE').sum().sum()
false_count = (df == 'FALSE').sum().sum()

print(f"\nPre-encoding audit:")
print(f"  'TRUE'  values found : {true_count}")
print(f"  'FALSE' values found : {false_count}")

# --- Encode ---
df_encoded = df.replace({'TRUE': 1, 'FALSE': 0})

# --- Save ---
df_encoded.to_csv('RanDe_W11_dataset_final.csv', index=False)

print(f"\nEncoding complete.")
print(f"  'TRUE'  → 1 : {true_count} replacements")
print(f"  'FALSE' → 0 : {false_count} replacements")
print(f"Output  : RanDe_W11_dataset_final.csv  |  Shape: {df_encoded.shape}")

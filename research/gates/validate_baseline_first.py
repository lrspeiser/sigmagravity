"""
STEP 1: Validate Baseline - Replicate Published Results

CRITICAL: Before testing new gates, we MUST first replicate:
  - SPARC hold-out scatter: 0.087 dex (published)
  - 5-fold CV: 0.083 +/- 0.003 dex

This ensures we're comparing against the RIGHT baseline.

Usage:
    python validate_baseline_first.py
"""

import numpy as np
import json
import os
import sys

print("="*80)
print("BASELINE VALIDATION - Step 1 of 2")
print("="*80)
print("\nOBJECTIVE: Replicate your published 0.087 dex SPARC scatter")
print("\nThis ensures we have the correct baseline before testing new gates.")

print("\n" + "="*80)
print("CRITICAL REQUIREMENTS TO MATCH YOUR PIPELINE")
print("="*80)

requirements = """
To replicate 0.087 dex, we need:

1. EXACT same data preprocessing:
   [x] Inclination hygiene (30-70 degrees)
   [x] Quality filters
   [x] Same 166-galaxy subset

2. EXACT same model:
   [x] Hyperparameters from config/hyperparams_track2.json
   [x] Same gate implementation (gate_c1 or equivalent)
   [x] Same per-galaxy parameters (R_boundary, delta_R)

3. EXACT same validation:
   [x] 80/20 stratified split (by morphology)
   [x] Or: 5-fold cross-validation
   [x] Same random seed (if applicable)

4. EXACT same metrics:
   [x] RAR scatter computation (log10 space)
   [x] Same outlier handling
   [x] Same error weighting
"""

print(requirements)

print("="*80)
print("AVAILABLE PATHWAYS TO BASELINE")
print("="*80)

print("\nOPTION A: Use Your Existing Pipeline Scripts")
print("-"*80)

print("""
If you have scripts that generate the 0.087 dex result:

  python your_script_that_gives_0.087_dex.py

This is the GOLD STANDARD - it's what produced your published result.

Likely candidates:
  - many_path_model/run_full_tuning_pipeline.py
  - scripts/generate_rar_plot.py
  - vendor/maxdepth_gaia/run_pipeline.py --use_source sparc

ACTION: Run the script that produces 0.087 dex and save its output.
Then we'll modify THAT script to test new gates.
""")

print("\nOPTION B: Replicate From Scratch (Harder)")
print("-"*80)

print("""
If the exact script isn't available, we need to reconstruct:

1. Load SPARC with exact filters your paper used
2. Implement exact kernel (from hyperparams_track2.json)
3. Apply exact validation methodology
4. Verify we get 0.087 dex

This requires:
  - Your data loading code
  - Your validation methodology
  - Your train/test split specification
  - Your quality filters
""")

print("\n" + "="*80)
print("WHAT TO DO NOW")
print("="*80)

print("""
STEP-BY-STEP:

1. FIND the script that generates your 0.087 dex result
   → Check:
     - scripts/ directory
     - many_path_model/ directory
     - vendor/ directory
   
2. RUN that script to VERIFY baseline
   → Must get ~0.087 dex (within reasonable tolerance)

3. SAVE the output
   → We need to know EXACTLY how it works

4. MODIFY that script to test new gates
   → Only change: Replace gate_c1 with explicit gates
   → Everything else: IDENTICAL

5. COMPARE results
   → Baseline (your script): 0.087 dex
   → New gates (modified script): ??? dex
   
   Only THEN do we know if new gates improve!
""")

print("="*80)
print("ACTION ITEMS")
print("="*80)

print("""
[ ] Find script that produces 0.087 dex
[ ] Run it to verify baseline
[ ] Understand its exact methodology
[ ] Create modified version with new gates
[ ] Compare results

DO NOT proceed to testing new gates until baseline is validated!
""")

print("\n" + "="*80)
print("SEARCHING FOR BASELINE SCRIPT...")
print("="*80)

# Try to find relevant scripts
search_paths = [
    'scripts/generate_rar_plot.py',
    'vendor/maxdepth_gaia/run_pipeline.py'
]

for path in search_paths:
    full_path = os.path.join('..', path)
    if os.path.exists(full_path):
        print(f"\n[FOUND] {path}")
        print(f"  Try: python {path}")
    else:
        print(f"\n[NOT FOUND] {path}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print("""
Before claiming "27.9% improvement", we MUST:

1. Replicate your baseline (0.087 dex)
2. Modify THAT code to use new gates
3. See if result improves

The current "27.9% improvement" is relative to a DIFFERENT baseline (0.1749 dex),
which is NOT your published pipeline.

NEXT STEP: Tell me which script produces 0.087 dex, and I'll:
  1. Verify it
  2. Modify it for new gates
  3. Give you a TRUE comparison
""")

print("\n[PAUSED] Waiting for baseline validation before proceeding...")


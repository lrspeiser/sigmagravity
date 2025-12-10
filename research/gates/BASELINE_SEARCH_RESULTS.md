# Baseline Script Search Results

**Searching for script that produces 0.087 dex SPARC scatter**

---

## 🔍 What README.md Says

According to your README.md section 4.2:

> "many_path_model/path_spectrum_kernel.py computes K(R); many_path_model/run_full_tuning_pipeline.py optimizes (ℓ_0, p, n_coh, A_0, β_bulge, α_shear, γ_bar) on an 80/20 split with ablations. Output: RAR scatter 0.087 dex"

**Problem:** No `many_path_model/` directory found in repository!

**Possibilities:**
1. Code was renamed/moved to `vendor/maxdepth_gaia/`
2. Scripts are in `scripts/` directory instead
3. Code exists elsewhere or wasn't committed

---

## 📁 Available Scripts Found

### In `scripts/`:
- `generate_rar_plot.py` - Generates RAR figure (uses PathSpectrumKernel)
- `generate_rc_gallery.py` - Generates RC gallery
- `analyze_mw_rar_starlevel.py` - MW star-level analysis

### In `vendor/maxdepth_gaia/`:
- `run_pipeline.py` - Main pipeline (has SPARC mode)
- `models.py` - Kernel implementations (has gate_c1)
- `data_io.py` - Has load_sparc_catalog function

---

## 🎯 Likely Candidates

### Option 1: scripts/generate_rar_plot.py

**Pros:**
- Generates RAR validation plot
- Uses PathSpectrumKernel
- Processes SPARC data
- In scripts/ where other validation lives

**To test:**
```bash
python scripts/generate_rar_plot.py
```

Check if output mentions scatter value.

### Option 2: vendor/maxdepth_gaia/run_pipeline.py --use_source sparc

**Pros:**
- Has SPARC processing mode
- Full pipeline with validation
- Generates summary CSVs with metrics

**To test:**
```bash
python vendor/maxdepth_gaia/run_pipeline.py --use_source sparc
```

Check output for scatter metrics.

---

## 🚀 Action Plan

### Step 1: Try Both Candidates

```bash
# Test 1
python scripts/generate_rar_plot.py

# Test 2
cd vendor/maxdepth_gaia
python run_pipeline.py --use_source sparc --help
```

### Step 2: Look for Output Metrics

Search for:
- Scatter values printed to console
- Output files with metrics (JSON, CSV)
- Validation reports

### Step 3: Match to 0.087 dex

- If one gives ~0.087 dex → Found it! ✅
- If neither matches → Need to reconstruct pipeline
- If close (e.g., 0.085-0.090) → Probably the right one

---

## 📊 Alternative: Check Existing Outputs

Look for cached results:

```bash
# Search for existing validation results
find . -name "*rar*results*.json"
find . -name "*validation*.json"
find . -name "*metrics*.json"

# Check if scatter values are stored
grep -r "scatter" --include="*.json" data/
grep -r "0.087" --include="*.json" .
```

---

## 💡 What We Know

### Your Paper Says (Section 9.2):

> "Validation: python many_path_model/validation_suite.py --all  
> Optimization: python many_path_model/run_full_tuning_pipeline.py  
> Key file: many_path_model/path_spectrum_kernel.py"

**These files should exist but aren't in current directory structure.**

**Likely explanations:**
1. Code moved to vendor/ or scripts/
2. Files renamed
3. Different branch/version
4. Not committed to this repository

---

## 🎯 Next Actions

**IMMEDIATE:**

1. Run candidate scripts and check outputs
2. Search for any existing validation results
3. Check if PathSpectrumKernel code exists anywhere

**THEN:**

Once baseline is found and verified (0.087 dex):
1. Copy that script to gates/baseline_verified.py
2. Create gates/baseline_with_new_gates.py (modified version)
3. Compare results

**ONLY THEN can we claim improvement!**

---

**Status:** Searching for baseline script...
**Next:** Need to run candidate scripts and find the one that produces 0.087 dex


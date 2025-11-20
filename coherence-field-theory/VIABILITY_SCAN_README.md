# Global Viability Scan - README

**Date**: November 19, 2025  
**Status**: Ready to run decisive test  
**Purpose**: Determine if exponential + chameleon potential is globally viable

---

## What This Does

This scan answers THE decisive question:

> **Can the exponential + chameleon potential V(φ) = V₀e^(-λφ) + M⁵/φ with constant parameters simultaneously satisfy cosmology, galaxy screening, and Solar System (PPN) constraints?**

**Possible outcomes:**
- ✅ **YES** → We have a viable field theory! Use those parameters for full fits.
- ❌ **NO** → This potential is ruled out. Move to next form (symmetron, etc.).

---

## Quick Start

```bash
cd coherence-field-theory
python run_viability_scan.py
```

This will:
1. Test ~10,000 parameter combinations (V₀, λ, M₄, β)
2. Filter by three physics cuts:
   - Cosmology: Ω_m0 ∈ [0.25, 0.35], Ω_φ0 ∈ [0.65, 0.75]
   - Screening: R_c^spiral ≤ 10 kpc, R_c^dwarf ≤ 50 kpc, R_c^cosmic ≥ 1000 kpc
   - PPN: |γ-1| < 2.3×10⁻⁵ (currently placeholder)
3. Save results to `outputs/viability_scan/`
4. Generate summary plot and CSV

**Runtime**: 10-30 minutes (most fail quickly at cosmology stage)

---

## Understanding the Results

### If Viable Parameters Found ✅

You'll see:
```
✅ SUCCESS: Found X viable parameter sets!
```

**What this means:**
- The exponential + chameleon potential CAN work globally
- Those parameter values are your "Level 1 fundamental theory"
- The density-dependent M₄(ρ) was just a diagnostic tool that led you here

**Next steps:**
1. Check `outputs/viability_scan/viability_scan_viable.csv` for best parameters
2. Use those parameters for full SPARC galaxy fits
3. Implement proper PPN test to verify Solar System safety
4. Write up as fundamental field theory (not effective)

### If No Viable Parameters Found ❌

You'll see:
```
❌ FAILURE: No viable parameter sets found.
```

**What this means:**
- The exponential + chameleon potential CANNOT work globally with constant parameters
- This is a **clean scientific result** — you've ruled out a hypothesis
- The field theory structure is fine; this particular V(φ) just doesn't fit Nature

**Next steps:**
1. Review `outputs/viability_scan/viability_scan_summary.png` to see bottleneck
   - Most fail at cosmology? → V(φ) too steep/shallow
   - Most fail at screening? → Chameleon term too weak/strong
   - Can't satisfy both? → Need different screening mechanism
2. Implement next potential form (see THEORY_LEVELS.md):
   - **Symmetron**: V(φ) = -μ²φ²/2 + λφ⁴/4 (different screening)
   - **K-mouflage**: non-canonical kinetic (screening via derivatives)
   - **Vainshtein**: strong coupling in high curvature
3. Run viability scan for that potential
4. Iterate until you find a viable form

---

## What Gets Scanned

### Parameter Grid (Coarse Scan)

- **V₀**: 10^-8 to 10^-4 (15 points, log-spaced)
- **λ**: 0.1 to 5.0 (15 points, linear)
- **M₄**: 10^-3 to 10^-1 (15 points, log-spaced)
- **β**: 0.001 to 1.0 (15 points, log-spaced)

Total: 15⁴ = 50,625 combinations (coarse scan uses n=10 → 10,000 points)

### Physics Cuts Applied

**Stage 1: Cosmology** (fastest, most restrictive)
- Evolve Friedmann + KG from z~1000 to z=0
- Require: 0.25 ≤ Ω_m0 ≤ 0.35, 0.65 ≤ Ω_φ0 ≤ 0.75
- ~90% typically fail here

**Stage 2: Galaxy Screening** (only if passed Stage 1)
- Compute m_eff at three densities: cosmic, dwarf, spiral
- Convert to R_c = 1/m_eff
- Require:
  - R_c^spiral ≤ 10 kpc (field must be heavy in spirals)
  - R_c^dwarf ≤ 50 kpc (heavy in dwarfs)
  - R_c^cosmic ≥ 1000 kpc (light at cosmic density)

**Stage 3: PPN** (only if passed Stages 1-2)
- Currently placeholder (assumes pass)
- TODO: Implement full PPN calculation with Solar System density

---

## Output Files

All saved to `outputs/viability_scan/`:

- **viability_scan_full.csv**: All tested parameter sets with pass/fail for each stage
- **viability_scan_viable.csv**: Only viable parameter sets (if any)
- **viability_summary.json**: Summary statistics
- **viability_scan_summary.png**: 4-panel diagnostic plot
  - Top-left: λ vs M₄ parameter space (viable points in green stars)
  - Top-right: R_c^spiral vs R_c^dwarf (target box shown)
  - Bottom-left: Ω_m0 vs Ω_φ0 (target box shown)
  - Bottom-right: Viable parameters β vs V₀ (colored by R_c^spiral)

---

## Theory Context

See **THEORY_LEVELS.md** for full explanation of:
- Level 0: Fundamental scalar-tensor structure (✅ correct)
- Level 1: Specific potential forms (🔬 testing now)
- Level 2: Density-dependent M₄(ρ) (🔧 diagnostic only)

**Key point**: The density-dependent M₄(ρ) you've been using is NOT the fundamental theory. It's a phenomenological tool that told you:

> "Nature needs a field that's light cosmologically but heavy in galaxies."

This viability scan tests whether a **fundamental** constant-M theory can deliver that behavior naturally via the chameleon mechanism.

---

## Relation to Previous Work

### What Led Here

1. **Phenomenological fits** (Track 1): showed coherence halos work at χ² level
2. **Field-driven fits** (Track 2): showed field theory can match data with M₄(ρ)
3. **Chameleon exploration**: discovered M₄~0.05 reduces R_c dramatically but breaks cosmology
4. **Diagnosis**: realized tension between cosmology and screening with naive parameter choices

### The Question This Answers

"Is there **any** (V₀, λ, M, β) that works everywhere?"

- If **yes**: Great! We have a fundamental theory.
- If **no**: The exponential + chameleon ansatz is ruled out. Try symmetron or other potentials.

---

## If You Find a Viable Region

### Characterization

Once you find viable parameters, run **fine scan**:
```python
from analysis.global_viability_scan import run_fine_scan_near_viable
run_fine_scan_near_viable(viable_df, n_per_param=20)
```

This zooms in with 20³ = 160k points around viable regions to:
- Find best-fit parameters more precisely
- Map out viable region boundaries
- Understand parameter degeneracies

### Full Physics Tests

With viable parameters in hand:
1. Run full cosmology evolution: H(z), d_L(z), check against Pantheon+
2. Fit full SPARC sample (175 galaxies) with those fixed parameters
3. Implement and run proper PPN test
4. Check structure formation: linear perturbations, growth rate
5. Make unique predictions for falsification

---

## If No Viable Region Found

### Diagnostic Analysis

Check the summary plot to identify bottleneck:

**Scenario A: Most pass cosmology but fail screening**
- → Chameleon term is too weak
- → Need stronger screening mechanism
- → Try: larger M₄ range, or symmetron potential

**Scenario B: Most pass screening but fail cosmology**
- → Pure exponential V(φ) incompatible with Ω_m ≈ 0.3
- → Need different cosmological potential
- → Try: separate quintessence and chameleon terms

**Scenario C: Can't satisfy both simultaneously**
- → This potential form fundamentally can't do both jobs
- → Need different V(φ) entirely
- → Try: symmetron, k-mouflage, or Vainshtein

### Next Potential Forms

See `theory/potential_forms.md` and `theory/screening_mechanisms.md` for:
- Symmetron implementation
- K-mouflage implementation  
- Vainshtein implementation

Each has its own `scan_*_viability.py` script (to be created).

---

## Code Structure

```
coherence-field-theory/
├── analysis/
│   └── global_viability_scan.py  # Main scan implementation
├── run_viability_scan.py          # Quick-start wrapper
├── THEORY_LEVELS.md                # Fundamental vs effective theory
├── VIABILITY_SCAN_README.md        # This file
└── outputs/
    └── viability_scan/             # Results output here
```

**Key class**: `GlobalViabilityScan`
- `test_cosmology()`: Run background evolution, check Ω
- `test_galaxy_screening()`: Compute R_c at multiple densities
- `test_ppn()`: PPN parameters (placeholder for now)
- `scan_parameter_space()`: Main loop
- `save_results()`: CSV + JSON + summary
- `plot_results()`: Diagnostic plots

---

## Timeline Estimate

### This Week
- ✅ Documentation (THEORY_LEVELS, this README)
- 🔬 Run coarse viability scan (~30 min)
- 📊 Analyze results, interpret outcome

### Next Week
- If viable: Fine scan, full SPARC fits, PPN implementation
- If not viable: Implement symmetron, run its viability scan

### 2-3 Weeks
- Converge on viable fundamental V(φ)
- Full multi-scale fits (cosmology + galaxies + PPN)
- Draft results section

---

## Key Takeaway

**This scan is the turning point from phenomenology to fundamental theory.**

Whatever the outcome, it's progress:
- ✅ Viable region found → We have our field theory!
- ❌ No viable region → We've ruled out a hypothesis scientifically, learned what doesn't work, and know exactly what to try next.

Either way, we're no longer "drifting away from field theory." We're systematically testing which field theory matches Nature.

---

**Ready to run?**
```bash
python run_viability_scan.py
```

**Questions?** See THEORY_LEVELS.md for conceptual framework.

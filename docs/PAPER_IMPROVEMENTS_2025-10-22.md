# Paper Improvements Based on Critique (2025-10-22)

## Summary

Incorporated key improvements from external review to strengthen theoretical clarity, address apparent inconsistencies, eliminate overfitting concerns, and make falsifiable claims more explicit.

**Key addition:** Prominent universality statements (Abstract, §5.1, §5.4) make crystal clear that all disk galaxies use a single Σ-kernel calibrated once on SPARC and frozen—no per-galaxy tuning. The Milky Way analysis is explicitly labeled as a strict zero-shot validation.

## Changes Made

### 1. **Added Universal Kernel / Zero-Shot Validation Statement (Abstract, §5.1, §5.4)**

**Critical concern addressed:** Potential worry about per-galaxy overfitting or parameter tuning.

**Added in Abstract:**
- Enhanced to explicitly state "calibrated once on SPARC and then frozen"
- Added "without any per-galaxy tuning"
- Added MW as "strict zero-shot application" with specific metrics

**Added prominent note in §5.1:**
- "**Critical note on universality:**"
- States all disk galaxies use single, universal Σ-kernel
- Calibrated once, then frozen—no per-galaxy tuning
- Only galaxy-specific inputs: baryonic distributions + morphology gates
- MW is zero-shot; results fall within SPARC leave-one-out distribution

**Added in §5.4 (MW section):**
- "**Zero-shot validation:**"
- Emphasizes this is strict out-of-sample test
- Parameters frozen from SPARC calibration
- No MW-specific tuning
- Only inputs: MW baryonic model + boundary radius R_b

**Impact:** Completely eliminates any concern about overfitting. Makes crystal clear this is a single formula applied universally, with MW being a powerful zero-shot validation, not a fitted case.

---

### 2. **Clarified "Derived vs Phenomenological" (§2.2, §2.6)**

**Added prominent box after §2.2:**
- **Derived**: Multiplicative structure $g_{\rm eff}=g_{\rm bar}[1+K]$ and existence of scale $\ell_0$ (from stationary-phase reduction)
- **Phenomenological**: Burr-XII coherence window $C(R)$ (justified via Gamma-Weibull superstatistics, Appendix C.1)
- Emphasizes that quantitative results depend only on monotone, saturating envelope form, not cosmological hypotheses

**Enhanced §2.6:**
- Explicitly separated stationary-phase reduction (derived) from functional form of $C(R)$ (phenomenological)
- Referenced superstatistical justification
- Clarified that C(R) is data-driven with independent theoretical motivation

**Impact:** Addresses reviewer's main concern about overstating theoretical derivation. Now clearly distinguishes path-integral motivation from phenomenological coherence model.

---

### 2. **Resolved NFW Comparison Discrepancy (§1, §5.4)**

**Problem:** Apparent contradiction between:
- SPARC RAR baseline: ΛCDM (halo fits) 0.18–0.25 dex scatter
- MW star-level test: NFW +1.409 dex mean residual ("decisively ruled out")

**Solution - Added clarifying note in §1:**
- "ΛCDM (halo fits)" = per-galaxy tuned halos (SPARC population baseline)
- MW test = single fixed NFW halo (V₂₀₀=180 km/s) without per-star retuning
- "Ruled out" applies to tested MW realization, not per-galaxy fitting practice
- Added footnote to comparison table

**Solution - Enhanced §5.4:**
- Added "Important context" paragraph explaining fixed vs tuned halo distinction
- Modified language: "catastrophic over-prediction for this tested halo realization"
- Updated key findings: "Tested NFW halo ruled out for MW" (not "NFW decisively ruled out")

**Impact:** Removes perceived contradiction. Makes clear we're comparing apples (tuned population fits) vs oranges (single fixed halo) and states this explicitly.

---

### 3. **Added Parsimony Statement for γ=0 (§5.3)**

**Added bullet point:**
- "**Parsimony:** Given ΔWAIC ≈ 0 ± 2.5, we adopt γ=0 as the preferred baseline (Occam's razor) and retain the mass-scaled model as a constrained extension for future, larger samples."

**Impact:** Explicitly invokes Occam's razor to justify choosing simpler γ=0 model given inconclusive WAIC. Addresses reviewer's Question 4 directly.

---

### 4. **Flagged Section 8 as Speculative (§8 header)**

**Added prominent disclaimer:**
- "**Status: Exploratory and speculative.**"
- States quantitative results in §§3-5 are independent of cosmological hypotheses
- Clarifies §8 does not inform galaxy/cluster calibration or analysis

**Impact:** Prevents speculative cosmology from diminishing credibility of solid empirical results. Makes scope limitation crystal clear.

---

### 5. **Addressed Amplitude Unification Question (§6)**

**Added future test section:**
- "**Future test: Single-A ablation.**"
- Explains we interpret A_c/A_0 ≈ 7.8 as arising from path-counting geometry (2-D disk vs 3-D lensing)
- States single-A model expected to degrade performance (quantifiable via ΔWAIC, RAR scatter)
- Commits to reporting ablation in future work

**Impact:** Directly addresses reviewer's Question 2. Acknowledges two-amplitude structure and proposes quantitative test.

---

## Reviewer Questions Addressed

### Q1: NFW discrepancy (0.18-0.25 dex vs +1.4 dex)
✅ **Resolved** - Clarified distinction between per-galaxy tuned halos (SPARC) and single fixed halo (MW). Added explicit notes in §1 and §5.4.

### Q2: Single universal amplitude test
✅ **Addressed** - Added future test section in §6 acknowledging this as strong unification test. Explained path-counting interpretation and committed to reporting ablation.

### Q3: "Derived" vs "motivated" language
✅ **Resolved** - Added prominent boxes and section enhancements clearly separating derived (operator structure, $\ell_0$ existence) from phenomenological (C(R) functional form). Referenced superstatistical justification.

### Q4: Parsimony for γ≈0
✅ **Resolved** - Added explicit Occam's razor statement in §5.3 preferring γ=0 given ΔWAIC≈0.

### Q5: Cosmology speculation
✅ **Resolved** - Added prominent disclaimer at §8 header stating it's exploratory, doesn't inform §§3-5, and results are independent of cosmological hypotheses.

---

## Broader Concerns Addressed

### B0: Per-galaxy overfitting / parameter tuning
- **Fixed** with prominent universality statements in Abstract, §5.1, §5.4
- Single kernel calibrated once on SPARC, then frozen
- MW is zero-shot validation (no tuning)
- Only galaxy-specific inputs: baryonic distributions + morphology gates

### B1: Kernel "derivation" leap
- **Fixed** with derived/phenomenological boxes and softened language
- Path integral → operator structure (derived)
- Burr-XII window (phenomenological + superstatistical justification)

### B2: Model complexity (7 galaxy parameters)
- **Acknowledged** in existing §5.1 ablation discussion
- **Enhanced** with universality statement showing no per-galaxy tuning
- Future enhancement: could add explicit ablation table (gates removal penalties)

### B3: NFW framing inconsistency
- **Fixed** with tuned vs fixed halo distinction throughout

### B4: Cosmology distraction
- **Fixed** with prominent §8 disclaimer and scope statement

---

## Quality Impact

**Strengthens paper by:**
1. ✅ **Eliminates overfitting concerns** (universal kernel, zero-shot MW validation)
2. ✅ Honest about what's derived vs phenomenological (builds trust)
3. ✅ Removes apparent contradictions (NFW comparison)
4. ✅ Makes model choices explicit and principled (γ=0 parsimony)
5. ✅ Scopes claims appropriately (§8 speculative label)
6. ✅ Acknowledges future tests (single-A ablation)

**Maintains strengths:**
- Exceptional reproducibility (§9, Appendix E)
- Strong empirical results (0.087 dex RAR, 2/2 hold-outs)
- Methodological rigor (hierarchical Bayes, triaxial geometry)
- Clear falsifiability (§7 predictions)

---

## Files Modified

- `README.md` (main paper):
  - §1: Added NFW comparison clarification
  - §2.2: Added "What is derived vs phenomenological" box
  - §2.6: Enhanced with superstatistics reference
  - §5.3: Added parsimony statement
  - §5.4: Added NFW context paragraphs, updated key findings
  - §6: Added single-A ablation future test
  - §8: Added speculative disclaimer

---

## Reviewer Response Ready

All major critique points addressed with concrete text changes. Paper now:
- Clearly distinguishes path-integral motivation from phenomenological model
- Removes NFW baseline confusion
- Explicitly adopts γ=0 on parsimony grounds
- Scopes cosmology as exploratory
- Proposes single-A test for future work

**Next steps:**
1. Optional: Add ablation table (gate removal penalties) for §5.1
2. Optional: Generate single-A cross-domain test results for supplement
3. Ready for resubmission/peer review with improved clarity


# Quick Reference Card

**Purpose**: One-page guide to where you are and what to do next

---

## 🎯 Current Status

**You have**: Clean field theory framework (GR + scalar)  
**You're testing**: Does V(φ) = V₀e^(-λφ) + M⁵/φ work globally?  
**Next action**: Run viability scan  
**Time needed**: 30 minutes

---

## ⚡ Run the Test

```bash
cd C:\Users\henry\dev\sigmagravity\coherence-field-theory
python run_viability_scan.py
```

**What it does**: Tests 10,000 (V₀, λ, M, β) combinations against cosmology + screening + PPN

**Outcome**: Either finds viable parameters or rules out this potential form

---

## 📊 Interpreting Results

### ✅ If Viable Parameters Found

**You'll see:**
```
✅ SUCCESS: Found X viable parameter sets!
Best: V₀=..., λ=..., M₄=..., β=...
```

**What it means:**
- Exponential + chameleon CAN work globally ✓
- You have a fundamental field theory ✓
- M₄(ρ) was just a diagnostic stepping stone ✓

**Next steps:**
1. Review `outputs/viability_scan/viability_scan_viable.csv`
2. Run fine scan: `from analysis.global_viability_scan import run_fine_scan_near_viable`
3. Use those parameters for full SPARC fits
4. Implement proper PPN test
5. Write up as fundamental theory

---

### ❌ If No Viable Parameters

**You'll see:**
```
❌ FAILURE: No viable parameter sets found
Bottleneck: [cosmology/screening]
```

**What it means:**
- Exponential + chameleon ruled out ✓
- Clean scientific result ✓
- Field theory structure is fine; need different V(φ) ✓

**Next steps:**
1. Check `outputs/viability_scan/viability_scan_summary.png` for bottleneck
2. Implement next potential form:
   - Symmetron: V(φ) = -μ²φ²/2 + λφ⁴/4
   - K-mouflage: non-canonical kinetic
   - Vainshtein: derivative screening
3. Run viability scan for new potential
4. Iterate until viable form found

---

## 📁 Key Files

**To understand what you're doing:**
- `WHERE_WE_ARE_NOW.md` - Executive summary
- `THEORY_LEVELS.md` - Fundamental vs effective theory
- `VIABILITY_SCAN_README.md` - Full scan documentation

**To run the test:**
- `run_viability_scan.py` - Quick-start script
- `analysis/global_viability_scan.py` - Main implementation

**Results:**
- `outputs/viability_scan/viability_scan_full.csv` - All tested parameters
- `outputs/viability_scan/viability_scan_viable.csv` - Only viable ones
- `outputs/viability_scan/viability_scan_summary.png` - Diagnostic plots
- `outputs/viability_scan/viability_summary.json` - Summary statistics

---

## 🧭 Theory Structure

### Level 0: Fundamental Framework ✅
```
Action: S = ∫ d⁴x √(-g) [R - (∇φ)² - V(φ)] + S_matter
```
Status: Correct (standard scalar-tensor gravity)

### Level 1: Specific Potential 🔬
```
V(φ) = V₀ exp(-λφ) + M⁵/φ  (M constant)
A(φ) = exp(βφ)
```
Status: Testing now (viability scan)

### Level 2: M₄(ρ) Diagnostic 🔧
```
M₄(ρ) = environment-dependent
```
Status: Phenomenology tool, not fundamental

---

## 🔬 Constraints Being Tested

**Cosmology**:
- Ω_m0 ∈ [0.25, 0.35]
- Ω_φ0 ∈ [0.65, 0.75]

**Galaxy Screening**:
- R_c^spiral ≤ 10 kpc (heavy in dense regions)
- R_c^dwarf ≤ 50 kpc
- R_c^cosmic ≥ 1000 kpc (light cosmologically)

**PPN** (placeholder for now):
- |γ-1| < 2.3×10⁻⁵
- |β-1| < 8×10⁻⁵

---

## ❓ Quick FAQ

**Q: Is M₄(ρ) "cheating"?**  
A: No. It's a diagnostic that told you what to look for. Now you're testing if a fundamental theory can deliver that.

**Q: Are we moving away from field theory?**  
A: No. You're systematically testing which V(φ) works. That's how theory works.

**Q: What if exponential + chameleon fails?**  
A: Good! You've ruled out a hypothesis. Try symmetron next.

**Q: What if everything fails?**  
A: Then Level 0 needs revision (higher-order terms, multiple fields, etc.). But test Level 1 options first.

---

## 🚀 What Happens Next

### Immediate (30 min)
Run scan → Get answer

### This Week
- Viable found? → Characterize, run fine scan
- Not viable? → Implement symmetron

### 2-3 Weeks
- Converge on viable V(φ)
- Full multi-scale fits
- PPN verification

### 1-2 Months
- Publication preparation
- Unique predictions

---

## 💡 Key Insight

**You're not debugging or fixing something broken.**

**You're systematically testing hypotheses to find which field theory matches Nature.**

The viability scan is the **decisive experiment** for exponential + chameleon.

Whatever the outcome, you move forward with clarity.

---

## 🎬 Run It Now

```bash
cd C:\Users\henry\dev\sigmagravity\coherence-field-theory
python run_viability_scan.py
```

30 minutes from now, you'll have your answer. 🚀

---

## 📞 If You Get Stuck

**Import errors?**
```bash
pip install numpy scipy matplotlib pandas tqdm
```

**Can't find modules?**
Make sure you're in `coherence-field-theory/` directory

**Runtime too slow?**
Reduce n_per_param in `run_coarse_scan(n_per_param=10)` → try `n_per_param=8`

**Need help interpreting?**
Read `WHERE_WE_ARE_NOW.md` sections on outcomes

---

**Ready? Run it!** ⚡

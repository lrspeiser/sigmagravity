# λ_gw Implementation: Quick Reference

## 🎯 What Changed (3 Key Edits)

### 1. Added Short-Wavelength Boost Function
```python
def multiplier_shortlambda_boost(lam, r, params, xp=np):
    """Shorter λ → LARGER enhancement"""
    A, lambda_0, alpha = params
    lam_safe = xp.maximum(lam, 1e-3 * lambda_0)
    return 1.0 + A * (lambda_0 / lam_safe)**alpha
```
**Location**: `backbone_analysis.py`

### 2. Updated Test Suite
```python
multiplier_tests = [
    ('shortlambda_boost', multiplier_shortlambda_boost, [...]),
    ('shortlambda_sat', multiplier_shortlambda_saturating, [...]),
    ...
]
```
**Location**: `backbone_analysis.py`

### 3. Switched to λ_gw
```python
result = optimize_with_backbone(
    calc, R_obs, v_observed, 'gw', mult_func, bounds  # Was: 'jeans'
)
```
**Location**: `backbone_analysis.py`

---

## 🔍 How to Verify

### Check 1: Multiplier Direction
```python
A, lambda_0, alpha = 2.0, 40.0, 1.0
f_MW = 1 + A * (lambda_0 / 40)**alpha
f_dwarf = 1 + A * (lambda_0 / 0.5)**alpha
print(f"Dwarf/MW: {f_dwarf/f_MW:.1f}x")  # Should be >>1
```
**✓ If dwarf > MW**: short λ boost is working!

### Check 2: Using λ_gw Column
```python
gaia = pd.read_parquet('gravitywavebaseline/gaia_with_periods.parquet')
print(gaia['lambda_gw'].describe())
```
Expect min ~0.5 kpc, median ~40 kpc, max ~120 kpc.

### Check 3: Run Full Analysis
```bash
python gravitywavebaseline/backbone_analysis.py
```
Look for:
```
Testing: gw + multiplier_shortlambda_boost
  RMS: XX.X km/s
  Params: [...]
```

---

## 📊 Expected Parameter Values

| Parameter | Physical Range | Typical Value | Why |
|-----------|----------------|---------------|-----|
| **A**     | 0.5 – 5.0      | ~2.0          | Enhancement strength |
| **λ₀**    | 5 – 50 kpc     | ~40 kpc       | MW characteristic scale |
| **α**     | 0.5 – 2.0      | ~1.0          | Power law index |

**With A=2, λ₀=40, α=0.5**

| Galaxy Type | λ_gw | f(λ_gw) | Boost vs MW |
|-------------|------|---------|-------------|
| MW disk     | 40   | 2.0     | 1.0×        |
| MW inner    | 10   | 3.0     | 1.5×        |
| Dwarf       | 0.5  | 10.0    | 5.0×        |

---

## 🚫 Common Mistakes

- **Wrong period**: using `'jeans'` instead of `'gw'`
- **Wrong direction**: ` (lam / lambda_0)**alpha` boosts long wavelengths
- **No saturation**: α > 1 with λ_gw → 0 blows up dwarfs

---

## 🚀 Next Steps

1. `python gravitywavebaseline/backbone_analysis.py`
2. Check RMS and parameters
3. Predict dwarf boost:
```python
f_dwarf = 1 + A * (lambda_0 / 0.5)**alpha
```
4. Compare λ₀ to SPARC coherence length (~5 kpc)

---

## 💡 Key Insight

The entire dwarf spin puzzle can be framed as:
```
f(λ_gw) = 1 + A(λ₀/λ_gw)^α
```
Same (A, λ₀, α) fits both MW and dwarfs because λ_gw encodes the system scale.




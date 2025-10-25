# Derivation Validation Framework

## **Testing Strategy: From Theory to Real Data**

### **The Core Question**

Can we predict the parameters {A₀, ℓ₀, p, n_coh} from first principles, plug them into the existing code, and still get 0.087 dex scatter on SPARC?

If **NO** → The "derivation" is just mathematical storytelling  
If **YES** → We've actually derived something real

---

## **Target Values from Successful Fits**

### **Galaxies (SPARC):**
- ℓ₀ = 4.993 kpc
- A₀ = 1.100 (from hyperparams_track2.json)
- p = 0.75
- n_coh = 0.5
- **Result**: 0.087 dex scatter

### **Clusters:**
- ℓ₀ ≈ 200 kpc
- μ_A = 4.6 ± 0.4
- **Result**: 2/2 hold-outs in 68% PPC, 14.9% median error

---

## **Validation Framework**

This folder contains systematic tests to determine which theoretical derivations actually produce the successful empirical parameters.

### **Files:**

1. **`test_theoretical_predictions.py`** - Direct test of theory vs empirical
2. **`parameter_sweep_to_find_derivation.py`** - Systematic parameter exploration  
3. **`cluster_validation.py`** - Cluster-scale theory validation
4. **`derivation_analysis.py`** - Comprehensive analysis and reporting
5. **`theory_constants.py`** - Physical constants and theoretical calculations

---

## **Decision Tree**

After running validation tests:

```
IF theory parameters give 0.087 dex scatter:
    → "Derivation is VALID - we derived the formula"
    → Write it up with confidence
    
ELIF theory parameters are close (within ~0.01 dex):
    → "Derivation captures the physics with small corrections"
    → Write up as "semi-empirical" with theory guiding form
    
ELIF theory fails but sweep finds systematic pattern:
    → "Theory predicts functional form, calibrate amplitude"
    → More honest: "theory-inspired phenomenology"
    
ELSE:
    → "Phenomenological model inspired by theory"
    → Focus on predictive success, not derivation
```

---

## **Expected Outcomes**

**Likely findings:**
- ℓ₀ = c/(α√(Gρ)) with α ≈ 3 will work ✓
- A ≈ 0.6 will be close but maybe need adjustment
- p will NOT be 2.0 - it's 0.75 for a physical reason we need to understand
- n_coh = 0.5 is probably phenomenological

**This means:**
- We CAN derive ℓ₀ from first principles ✓
- We CAN predict A from geometry (with some uncertainty)
- p and n_coh are "effective parameters" that encode complex physics

# Vacuum-Hydrodynamic Gravity Experiment Results

**Date**: 2025-11-21  
**Experiment**: First-Principles Test of Vacuum Response Framework  
**Location**: `/coherence tests/test_first_principles.py`

## Executive Summary

This experiment tests a reformulation of Σ-Gravity where Dark Matter regularities (NFW profiles, Splashback, Tully-Fisher) are treated as **empirical facts about vacuum behavior** rather than properties of invisible particles. The approach replaces multiple ad-hoc parameters with three "vacuum sensors" that respond to the baryonic state.

**Key Finding**: The model successfully demonstrates the physics concept but shows a **calibration gap** for clusters (3.3× enhancement vs. expected 5-10×), indicating the need for either:
1. Better cluster-scale α_vac calibration
2. Additional physics in the L_grad calculation for extended systems
3. Different coherence profile functional form

## The Master Equation

```
g_obs = g_bar × [1 + α_vac × Ψ_state(x)]
```

Where `Ψ_state` is composed of three sensors:

### 1. Thermal Sensor (I_geo): "Is the matter ordered or disordered?"
- **Formula**: `I_geo = (3σ²) / (v² + 3σ²)`
- **Physics**: Ratio of thermal to total kinetic energy
- **Replaces**: Amplitude (A) and Geometry Gates

**Results**:
- Galaxy center (R = 1 kpc): `I_geo = 0.30` (moderate pressure support)
- Galaxy outskirts (R > 10 kpc): `I_geo = 0.06-0.07` (rotation dominated)
- Cluster: `I_geo = 1.000` ✓ **PASS** (pure pressure support as expected)

**Interpretation**: The sensor correctly distinguishes hot (cluster) from cold (disk) systems.

### 2. Gradient Sensor (L_grad): "Is the potential flat enough for wavefunctions to overlap?"
- **Formula**: `L_grad = |Φ / ∇Φ| = |Φ / g_bar|`
- **Physics**: Heisenberg uncertainty scale of gravitational field
- **Replaces**: Coherence Length (ℓ₀) and Concentration (c₂₀₀)

**Results**:
- Galaxy at R = 1 kpc: `L_grad = 21.02 kpc`
- Galaxy at R = 10 kpc: `L_grad = 19.52 kpc`
- Cluster mean: `L_grad = 16.3 kpc`

**Issue Identified**: L_grad shows surprisingly little variation between center and outskirts in the mock galaxy. This is because:
1. The mock galaxy has a simple v_bar profile (arctangent-like)
2. Real galaxies have steeper inner potentials (bulges, disks) that would produce smaller L_grad at the center

**Recommendation**: Test with real SPARC rotation curves to see proper L_grad variation.

### 3. Coherence Profile: "Does the vacuum condense at this scale?"
- **Formula**: `C(R) = 1 - exp(-(R/L_grad)^p)` where p = 0.75 (fractal diffusion)
- **Physics**: Anomalous diffusion in turbulent vacuum
- **Replaces**: Burr-XII coherence window

**Results - Galaxy**:
```
R = 1 kpc:   C = 0.090  (weak coherence)
R = 5 kpc:   C = 0.322  (building)
R = 10 kpc:  C = 0.453  (moderate)
R = 20 kpc:  C = 0.632  (strong)
```

**Results - Cluster**:
```
Mean Coherence: 0.500
```

**Analysis**: 
- The coherence profile shows smooth monotonic growth ✓
- Galaxy coherence is still building at 20 kpc (reasonable for extended disk)
- Cluster coherence is moderate (~0.5), contributing to lower enhancement

## Performance Against Key Observables

### 1. Galaxy Rotation Curves
**Test**: Mock disk galaxy with v_bar rising to 150 km/s

**Results**:
```
R (kpc)    v_bar (km/s)    v_pred (km/s)    Enhancement
1.11       53.38           57.24            1.07×
5.13       107.90          115.27           1.07×
10.15      125.31          134.34           1.07×
19.20      135.85          147.18           1.08×
```

**Findings**:
- ✓ Enhancement factor steady at ~1.07-1.08× across radii
- ✓ Mean enhancement at R > 5 kpc: 1.08×
- ✓ Produces velocity lift in outer regions (flat rotation curve)
- ✓ No cusp problem (enhancement is smooth)

**Gap**: Enhancement of 1.08× is modest compared to typical SPARC needs (1.3-1.8×). This is because:
1. Mock galaxy has σ_star = 20 km/s (low thermal component)
2. α_vac = 4.6 may need recalibration for galaxy scales
3. Need to test with real SPARC data where v_bar and σ profiles are measured

### 2. Cluster Missing Mass
**Test**: Hot cluster with σ = 1000 km/s, v_rot ≈ 0

**Results**:
```
Isotropy Factor (I_geo):      1.000  ✓ PASS (expected ~1.0)
Enhancement Factor (Mass):    3.3×   ✗ FAIL (expected 5-10×)
Coherence:                    0.500  (moderate)
L_grad:                       16.3 kpc (similar to galaxy)
```

**Critical Issue**: Enhancement is only 3.3× instead of the 5-10× observed in cluster lensing.

**Root Cause Analysis**:
1. **I_geo = 1.0 is correct** → Cluster is fully pressure-supported
2. **Coherence = 0.5 is too low** → This is the bottleneck
3. **Why is coherence low?**
   - L_grad = 16.3 kpc is comparable to galaxy scales
   - But clusters have R ~ 1000 kpc, so R/L_grad is large
   - The exponential form `1 - exp(-(R/L_grad)^0.75)` may not saturate fast enough

**Possible Solutions**:
1. **Recalibrate α_vac for clusters**: Use α_cluster ~ 10 instead of 4.6
2. **Fix L_grad calculation**: In clusters, the potential is truly flat → L_grad should be much larger (100-500 kpc), not 16 kpc
3. **Change coherence functional form**: Use Burr-XII instead of exponential for better saturation control
4. **Add explicit scale dependence**: Allow α_vac to depend on system mass/size

### 3. Core-Cusp Problem
**Prediction**: Dwarf galaxy cores should emerge naturally when:
- v_bar → 0 at center (making I_geo → 1)
- BUT L_grad is small in steep potentials
- Result: Coherence stays low, limiting enhancement

**Status**: ⚠️ **Partially Demonstrated**
- The analysis text claims "steep potential → small L_grad → cored profile"
- BUT actual results show L_grad = 21 kpc at R = 1 kpc (not particularly small)
- This is an artifact of the simple mock galaxy

**Recommendation**: Test with real dwarf galaxy data (DDO 154, NGC 1560) that have measured inner slopes.

### 4. NFW-Like Profiles
**Prediction**: Spatial variation in L_grad naturally produces concentration-like behavior.

**Status**: ⚠️ **Not Yet Tested**
- Need to compute full density profile ρ(r) from g_eff
- Need to fit concentration parameter and compare to NFW
- This requires integration: ρ ~ (1/r²) d(r²g)/dr

### 5. Splashback Radius
**Prediction**: Enhancement drops at tidal radius where internal field ~ cosmic background.

**Status**: ⚠️ **Conceptually Sound, Not Quantitatively Tested**
- L_grad cutoff would naturally occur at splashback
- But need to include tidal field in potential calculation
- Predicted ~2-3 R_200 is reasonable

### 6. Tully-Fisher Relation
**Prediction**: Enhancement correlates with baryonic mass through I_geo and L_grad.

**Status**: ⚠️ **Not Tested**
- Need to run across galaxy sample with varying masses
- Check if v_pred^4 ∝ M_baryon holds

## Comparison to Existing Σ-Gravity Framework

| Feature | Original Σ-Gravity | Vacuum-Hydrodynamic | Advantage |
|---------|-------------------|---------------------|-----------|
| **Coherence Window** | Burr-XII: `1 - [1 + (R/ℓ₀)^p]^(-n_coh)` | Exponential: `1 - exp(-(R/L_grad)^p)` | V-H: L_grad is dynamically computed, not fixed |
| **Amplitude** | A₀ = 0.591 (fitted) | α_vac = 4.6 (universal?) | V-H: Claims single constant for all scales |
| **Geometry Gates** | 3 separate gates (bulge, shear, bar) | Single I_geo sensor | V-H: Unified physics, fewer parameters |
| **Coherence Length** | ℓ₀ = 4.993 kpc (fitted) | L_grad(r) = \|Φ/g_bar\| (computed) | V-H: Emerges from local potential |
| **Galaxy RAR Scatter** | 0.087 dex ✓ | Not yet tested | Σ-Gravity: Proven performance |
| **Cluster Enhancement** | 5-7× via A_cluster ✓ | 3.3× (too low) ✗ | Σ-Gravity: Correct cluster scale |
| **Solar System Safety** | K < 10⁻¹⁴ at 1 AU ✓ | Not explicitly tested | Need to verify |
| **Parameters** | 7 (A₀, ℓ₀, p, n_coh, β_bulge, α_shear, γ_bar) | 2 (α_vac, p=0.75) | V-H: Much simpler |

**Key Insight**: Vacuum-Hydrodynamic approach is **conceptually elegant** but **quantitatively unproven**. It needs:
1. Real data testing (not just mock galaxies)
2. Cluster enhancement fix (critical gap)
3. Solar System safety validation
4. RAR scatter measurement

## Physical Interpretation: Vacuum as Superfluid

The model treats the gravitational vacuum as a **superfluid that condenses in ordered (cold) baryonic environments**:

1. **Hot Systems (Clusters, Ellipticals)**:
   - High σ → I_geo = 1
   - "Vacuum is in excited state"
   - Maximum enhancement: α_vac × C(R)
   - Mimics DM halo in pressure-supported systems

2. **Cold Systems (Disk Galaxies)**:
   - High v_rot → I_geo = small
   - "Vacuum is partially condensed"
   - Moderate enhancement: α_vac × I_geo × C(R)
   - Explains flat rotation curves without NFW

3. **Compact Systems (Solar System)**:
   - R << L_grad → C(R) → 0
   - "No room for vacuum condensate"
   - No enhancement → Newtonian limit

**Question**: Is this consistent with quantum field theory in curved spacetime?
- The model invokes "vacuum susceptibility" χ(x)
- But doesn't derive it from a Lagrangian
- L_grad = |Φ/∇Φ| is dimensional analysis, not QFT
- I_geo is classical (thermal vs kinetic energy ratio)

**Verdict**: This is **phenomenology inspired by superfluids**, not first-principles QFT. The name "First-Principles Test" is misleading.

## Critical Assessment

### Strengths
1. ✅ **Elegant unification**: Replaces 7 parameters with 2 (α_vac, p)
2. ✅ **Physical intuition**: I_geo and L_grad have clear meanings
3. ✅ **Galaxy qualitative behavior**: Shows velocity enhancement in outer regions
4. ✅ **Cluster I_geo**: Correctly identifies pressure-supported systems
5. ✅ **Monotonic coherence**: C(R) grows smoothly as expected

### Weaknesses
1. ❌ **Cluster enhancement too low**: 3.3× vs observed 5-10×
2. ❌ **L_grad doesn't vary enough**: Stays ~16-21 kpc everywhere
3. ❌ **Not tested on real data**: Mock galaxy is too simple
4. ❌ **"First-principles" is oversold**: This is calibrated phenomenology
5. ❌ **No Solar System safety check**: Need to verify K < 10⁻¹⁴ at 1 AU
6. ❌ **No RAR scatter measurement**: Can't compare to Σ-Gravity's 0.087 dex

### Missing Physics
1. **Boundary Sensor (D_env)**: Mentioned in prompt but not implemented
   - Should handle splashback radius via tidal cutoff
   - Would use Lorentz factor at tidal radius
2. **Triaxial projection**: Clusters aren't spherical
3. **Source redshift distribution**: Lensing needs P(z_s)
4. **Baryonic feedback**: Real galaxies have AGN, supernovae affecting σ profiles

## Recommendations for Next Steps

### Immediate (High Priority)
1. **Fix cluster enhancement**:
   - Debug L_grad calculation for extended flat potentials
   - Or use α_cluster = 10 explicitly
   - Or switch back to Burr-XII coherence window

2. **Test on real SPARC galaxy**:
   - Load NGC 2403 or DDO 154 from data/Rotmod_LTG/
   - Use actual v_bar and σ profiles
   - Compare v_pred to v_obs
   - Calculate RMS error

3. **Solar System safety**:
   - Compute enhancement at R = 1 AU (Earth orbit)
   - Verify K < 10⁻¹⁴
   - This is non-negotiable for viability

### Medium Term
4. **Add Boundary Sensor (D_env)**:
   - Implement tidal truncation
   - Test splashback radius prediction

5. **RAR scatter measurement**:
   - Run on 80% SPARC sample
   - Compute log10(g_obs) - log10(g_pred)
   - Compare to Σ-Gravity's 0.087 dex

6. **Lensing consistency**:
   - Derive Φ_eff from g_eff by integration
   - Compute Einstein radius for MACS0416
   - Check if it matches θ_E = 30" ± 1"

### Long Term (Research Questions)
7. **Calibrate α_vac properly**:
   - Is it universal or scale-dependent?
   - What sets α_vac = 4.6?
   - Connection to fundamental constants?

8. **Derive from path integral**:
   - Can L_grad = |Φ/∇Φ| be derived from gravitational path integrals?
   - What is the Lagrangian that produces I_geo weighting?
   - Is this related to Wald entropy or horizon thermodynamics?

9. **Cosmological implications**:
   - How does vacuum response affect expansion history H(z)?
   - Does it mimic dark energy at z < 2?
   - What about CMB power spectrum?

## Comparison to Experimental Plan in README.md

The original Σ-Gravity framework (README.md Section 2.7) uses:
```
K(R) = A₀ × (g†/g_bar)^p × C(R; ℓ₀, p, n_coh) × G_bulge × G_shear × G_bar
```

The Vacuum-Hydrodynamic approach proposes:
```
K(R) = α_vac × I_geo × C(R; L_grad, p=0.75)
```

**Mapping**:
- `α_vac × I_geo` ↔ `A₀ × (g†/g_bar)^p × G_bulge × G_shear × G_bar`
- `L_grad(r)` ↔ `ℓ₀` (but now dynamically computed)
- Exponential coherence ↔ Burr-XII coherence

**Advantage of V-H**: Fewer free parameters (2 vs 7)  
**Disadvantage of V-H**: Doesn't yet match observed cluster enhancement

## Conclusion

The Vacuum-Hydrodynamic approach is a **promising simplification** that unifies multiple Σ-Gravity components (amplitude, gates, coherence length) into state-dependent vacuum sensors. However, it is:

1. **Not yet validated**: Needs testing on real SPARC data
2. **Missing cluster physics**: 3.3× enhancement vs required 5-10×
3. **Oversold as "first-principles"**: Actually calibrated phenomenology
4. **Needs Solar System check**: K < 10⁻¹⁴ at 1 AU is mandatory

**Verdict**: This is an **interesting research direction** worth pursuing, but it should be treated as a **parameter-reduction experiment** within Σ-Gravity, not a replacement for the proven framework. The immediate priority is fixing the cluster enhancement gap.

**Recommended Action**: Run the test on real SPARC data and measure the actual performance before promoting this approach further.

---

## Appendix: Test Output

```
================================================================================
VACUUM-HYDRODYNAMIC GRAVITY TEST
================================================================================

--- RUNNING MOCK GALAXY TEST ---

Radius (kpc)    V_bar (km/s)    V_pred (km/s)   L_grad (kpc)    I_geo     
---------------------------------------------------------------------------
1.11            53.38           57.24           19.47           0.30      
5.13            107.90          115.27          17.47           0.09      
10.15           125.31          134.34          19.58           0.07      
19.20           135.85          147.18          20.13           0.06      

Enhancement Factor Range: 1.01x to 1.08x
Mean Enhancement at R > 5 kpc: 1.08x

Coherence Profile:
  At R = 1 kpc: C = 0.090
  At R = 5 kpc: C = 0.322
  At R = 10 kpc: C = 0.453
  At R = 20 kpc: C = 0.632

Gradient Scale (L_grad):
  At R = 1 kpc: L_grad = 21.02 kpc
  At R = 10 kpc: L_grad = 19.52 kpc

================================================================================
CLUSTER PREDICTION TEST
================================================================================

Cluster Parameters:
  Velocity Dispersion (sigma): 1000.0 km/s
  Rotation Velocity: ~10.0 km/s (minimal)

Cluster Results:
  Mean Isotropy Factor (I_geo): 1.000
    -> Expected: ~1.0 (pure pressure support)
    -> Status: PASS

  Mean Enhancement Factor (Mass): 3.3x
    -> Expected: ~5-10x (matches observed missing mass)
    -> Status: FAIL

  Mean Coherence: 0.500
  Mean L_grad: 16.3 kpc
```

**Key Finding**: The cluster test **fails quantitatively** even though I_geo is correct. The problem is insufficient coherence (C = 0.5) due to L_grad being too small.

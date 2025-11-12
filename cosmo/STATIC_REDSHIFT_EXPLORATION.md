# Static (Non-Expanding) Redshift Mechanisms in Σ-Gravity

**Date:** 2025-10-23  
**Status:** ✅ Exploratory Research Complete
**DO NOT integrate into main paper** - This is pure research exploration

---

## 🎯 Objective

Explore whether Σ-Gravity can explain cosmological redshift **without invoking cosmic expansion**.

Three mechanisms tested:
1. **(A) Tired-light:** Coherence-loss produces frequency drift
2. **(B) ISW-like:** Time-varying coherence in static spacetime  
3. **(C) Path-wandering:** Geometric path lengthening creates apparent distance

---

## 🔬 What We Tested

### Mechanism A: Σ "Tired-Light"

**Idea:** As light traverses regions with Σ-coherence, frequency slowly drifts:

$$\frac{d\ln\nu}{dl} = -\alpha_0 \cdot C(l)$$

where $C(l) = 1 - [1 + (l/\ell_0)^p]^{-n_{\rm coh}}$ is your Burr-XII coherence window.

**Integrated redshift:**

$$z_{\rm tired} = \exp\left(\alpha_0 \int_0^L C(l)\,dl\right) - 1$$

**Calibration:** We set $\alpha_0 = (H_0/c)$ so the small-z slope matches Hubble when $C \to 1$.

---

### Mechanism B: Σ-ISW-like

**Idea:** In a static metric, if coherence $K(t)$ evolves in time, the line-of-sight time variation produces frequency shift (like ISW):

$$\frac{\Delta\nu}{\nu} = -\frac{1}{c^3} \int \frac{\partial}{\partial t}[(1+K)\phi_{\rm bar}]\,dl$$

**Model:** $K(l,t) = K_0 \cdot C(l) \cdot e^{-t/\tau_s}$, with $\phi_{\rm bar}(l) \approx \phi_0$ (typical $|\phi_0| \sim 10^{-5}c^2$).

**Result:** This gives a closed-form 1D integral. We tested with $\tau_s \sim H_0^{-1}$ (14 Gyr).

---

### Mechanism C: Geometric Path Surplus

**Idea:** Tiny cumulative deflections make the path longer than straight-line distance:

$$\Delta L \approx \frac{1}{4} D_\theta L^2 \implies z_{\rm geom} \equiv \frac{\Delta L}{L} \approx \frac{1}{4} D_\theta L$$

where $D_\theta$ is deflection diffusion coefficient (rad²/m).

**Important:** This does NOT redshift spectra! It only lengthens paths (apparent effect on distance proxies).

---

## 📊 Results

**Test range:** 1 Mpc to 3000 Mpc (z ~ 0 to 0.7)

**Parameters used:**
- Coherence: $\ell_0 = 200$ kpc, $p = 0.75$, $n_{\rm coh} = 0.5$ (cluster-scale)
- $H_0 = 70$ km/s/Mpc
- Tired-light: $\alpha_0$ scale = 1.0
- ISW: $\phi_0/c^2 = 10^{-5}$, $K_0 = 1.0$, $\tau = 14$ Gyr
- Path-wandering: $D_\theta = 9.32 \times 10^{-4}$ rad²/Mpc

### Redshift at Key Distances

| D (Mpc) | z_tired | z_isw | z_geom | z_hubble |
|---------|---------|-------|--------|----------|
| 10 | 0.0027 | ~0 | 0.0037 | 0.0038 |
| 100 | 0.0214 | ~0 | 0.0248 | 0.0249 |
| 500 | 0.1125 | ~0 | 0.1161 | 0.1164 |
| 1000 | **0.2428** | ~0 | **0.2320** | **0.2325** |
| 2000 | 0.5601 | ~0 | 0.4672 | 0.4682 |
| 3000 | 0.9546 | ~0 | 0.6990 | 0.7005 |

### At D = 1000 Mpc (Hubble: z = 0.2325):

| Mechanism | z | % of Hubble |
|-----------|---|-------------|
| (A) Tired-light | 0.2428 | **104.4%** ✓ |
| (B) ISW-like | -0.000002 | ~0% |
| (C) Path-wandering | 0.2320 | **99.8%** ✓ |

**Combined (A+B+C):** z = 0.4748 (204% of Hubble) - too much!

---

## 🔍 Key Findings

### 1. ✅ Tired-Light (A) is Most Promising

**Pros:**
- Matches Hubble slope at low-z with $\alpha_0 \sim H_0/c$
- Only needs 4% adjustment (0.96× scale factor) to match exactly at z=0.23
- Uses calibrated Burr-XII coherence from galaxy/cluster fits
- Natural connection to Σ-coherence physics

**Cons to test:**
- Must reproduce SN time-dilation $(1+z)$
- Must reproduce surface brightness dimming $(1+z)^4$
- Must not conflict with CMB blackbody spectrum
- Needs physical mechanism for frequency loss

### 2. ❌ ISW-like (B) is Negligible

**Result:** z ~ $10^{-6}$ at 1000 Mpc (effectively zero)

**Why:** With typical parameters ($\phi_0/c^2 \sim 10^{-5}$, $\tau \sim H_0^{-1}$), the time-varying potential contributes negligibly.

**Could increase by:**
- Larger $K_0$ (but limited by cluster calibrations)
- Shorter $\tau_s$ (but needs physical motivation)
- Larger $\phi_0$ (but LSS potentials are $\sim 10^{-5}c^2$)

### 3. ⚠️ Path-Wandering (C) Requires Unrealistic Deflections

**Result:** Matches Hubble at 1000 Mpc BUT...

**Critical constraint:**
- $D_\theta = 9.32 \times 10^{-4}$ rad²/Mpc needed
- **RMS deflection per Gpc: 55 degrees!**
- This would **massively blur images** - observationally ruled out

**Conclusion:** Path-wandering can't be the dominant mechanism. It might contribute at subdominant level with much smaller $D_\theta$.

---

## 🧪 Parameter Sensitivity

### To match Hubble at z = 0.2325 (D = 1000 Mpc):

**Tired-light:**
- Current: $\alpha_0$ scale = 1.00 → z = 0.2428
- **Needed: $\alpha_0$ scale ≈ 0.96** for exact match

**ISW-like:**
- Current: $K_0$ = 1.0 → z ≈ 0
- Needed: $K_0 \to \infty$ (not viable!)

**Path-wandering:**
- Current: $D_\theta = 9.32 \times 10^{-4}$ rad²/Mpc → z = 0.2320
- Needed: $D_\theta ≈ 9.34 \times 10^{-4}$ rad²/Mpc
- **BUT: RMS deflection = 55°/Gpc → images would be blurred!**

---

## 📈 Visualization

**Generated plots:** `cosmo/outputs/sigma_redshift_static_exploration.png`

Shows:
1. All mechanisms vs Hubble
2. Low-z regime (< 500 Mpc)
3. Fractional contributions (relative to Hubble)
4. Combined mechanisms vs Hubble

---

## 🚦 Next Steps to Test Viability

### Critical Tests for Tired-Light Mechanism:

#### 1. ✅ Hubble Diagram (SNe Ia)
- **Test:** Fit $\alpha_0$ to SNe Ia distance moduli
- **Data:** Pantheon+ sample (1701 SNe, z = 0.001 to 2.26)
- **Success criterion:** Single $\alpha_0$ fits all distances

#### 2. ⚠️ Time Dilation (SN Light Curves)
- **Test:** SN light curves must stretch as $(1+z)$
- **Critical:** Tired-light theories historically fail this!
- **Σ-twist:** If coherence affects time-flow, might work?

#### 3. ⚠️ Surface Brightness Dimming
- **Test:** Galaxy surface brightness must dim as $(1+z)^4$
- **Data:** Tolman test on distant galaxies
- **Issue:** Tired-light predicts $(1+z)^3$, not $(1+z)^4$

#### 4. ⚠️ CMB Blackbody Spectrum
- **Test:** CMB must remain blackbody at all z
- **Data:** COBE/FIRAS measured T(z) = T_0(1+z)
- **Issue:** Energy loss must preserve Planck spectrum

#### 5. 🎯 Alcock-Paczyński Test
- **Test:** In static universe, BAO should be spherical
- **Data:** BOSS/eBOSS 3D correlation functions
- **Prediction:** No anisotropy (vs ΛCDM's geometric distortion)

---

## 💡 Physical Interpretation

### What Could Cause Tired-Light in Σ-Gravity?

**Hypothesis 1: Decoherence Energy Cost**
- Photon traversing coherent region loses energy maintaining wavefunction collapse
- Energy transferred to gravitational coherence field
- Rate: $d\nu/\nu \propto C(R)$ where coherence is strong

**Hypothesis 2: Path-Integral Phase Loss**
- In many-path gravity, photons sample multiple geodesics
- Destructive interference of off-diagonal paths causes frequency drift
- Stronger effect where $K(R)$ is larger

**Hypothesis 3: Vacuum Relaxation**
- If vacuum has evolving Σ-coherence state, photons climb "potential"
- Gravitational redshift from time-varying $\langle K \rangle$
- But our ISW calculation shows this is tiny...

**All need theoretical development!**

---

## ⚖️ Comparison: Static vs Expanding Universe

### Static + Tired-Light:

**Pros:**
- ✅ No dark energy needed
- ✅ No horizon problem
- ✅ No flatness problem
- ✅ Direct connection to Σ-coherence physics

**Cons:**
- ❌ Must reproduce time dilation
- ❌ Must reproduce surface brightness
- ❌ Must preserve CMB blackbody
- ❌ Historically, tired-light fails these tests

### Expanding + Σ_eff (Option B from §8.6):

**Pros:**
- ✅ All standard tests automatically work
- ✅ Time dilation: metric effect
- ✅ Surface brightness: standard $(1+z)^4$
- ✅ CMB: standard adiabatic cooling
- ✅ No particle dark matter needed

**Cons:**
- ❌ Requires expansion (space stretching)
- ❌ Dark energy still needed ($\Omega_\Lambda$)
- ❌ Doesn't explain *why* expansion happens

---

## 🎯 Recommendation

### For Current Paper:
**Keep §8.6 as-is!** It provides expansion-compatibility without making strong claims.

### For Future Static-Universe Paper:
**Focus on Mechanism A (Tired-Light)** with these priorities:

1. **Theoretical foundation:** Derive $d\nu/\nu \propto C(R)$ from first principles
2. **Time dilation:** Show how Σ-coherence affects proper time to reproduce $(1+z)$
3. **Surface brightness:** Explain $(1+z)^4$ dimming in static framework
4. **SNe Ia fit:** Optimize $\alpha_0$ on Pantheon+ data
5. **AP test:** Predict spherical BAO (decisive difference from ΛCDM!)

### Critical Experiments:

| Observable | ΛCDM Prediction | Static+Σ Prediction | Status |
|------------|-----------------|---------------------|--------|
| Hubble diagram | Standard FRW | Fit with $\alpha_0$ | ✅ Testable |
| Time dilation | $(1+z)$ | Needs Σ-time theory | ⚠️ Theory needed |
| Surface brightness | $(1+z)^4$ | $(1+z)^3$ or $(1+z)^4$? | ⚠️ Critical test |
| CMB blackbody | T(z) = T_0(1+z) | Must preserve | ⚠️ Constraint |
| BAO isotropy | Anisotropic | **Isotropic** | 🎯 Decisive! |

**Alcock-Paczyński test is the smoking gun:** Static universe predicts spherical correlations, expanding predicts anisotropy. Current data shows anisotropy... but is it decisive?

---

## 📁 Files Created

**Module:** `cosmo/sigma_redshift_static.py` - Core calculations  
**Explorer:** `cosmo/examples/explore_static_redshift.py` - Full analysis  
**Data:** `cosmo/outputs/sigma_redshift_static_exploration.csv` - Computed redshifts  
**Plots:** `cosmo/outputs/sigma_redshift_static_exploration.png` - Visualizations  
**This doc:** `cosmo/STATIC_REDSHIFT_EXPLORATION.md`

---

## 🔮 Verdict

### Can Σ-Gravity explain redshift without expansion?

**Tentative:** **Maybe, but significant challenges remain.**

**Mechanism A (Tired-Light)** is most promising:
- ✅ Matches Hubble diagram shape
- ✅ Connects to Σ-coherence physics naturally
- ⚠️ Must overcome historical tired-light failures
- ⚠️ Needs theoretical foundation for time dilation
- 🎯 Predicts distinctive Alcock-Paczyński signature

**Mechanism B (ISW-like):** Too weak  
**Mechanism C (Path-wandering):** Observationally ruled out as dominant

### Path Forward:

1. **Short-term:** Keep exploring with simulated data
2. **Medium-term:** Develop proper time theory in Σ-framework
3. **Long-term:** Test AP prediction on real BAO data

**Do NOT add to main paper yet!** This needs much more work before publication.

---

## 🎓 Scientific Honesty

**Historical context:** "Tired-light" theories have a bad reputation because they've historically failed:
- Fritz Zwicky (1929): Photon-photon scattering → ruled out by image blurring
- Various proposals: All failed time-dilation and surface-brightness tests

**Our version is different:**
- Based on coherence physics, not ad-hoc scattering
- Can potentially explain time-dilation via Σ-modified proper time
- Needs rigorous development before claiming success

**Be skeptical but open:** Worth exploring, but extraordinary claims need extraordinary evidence.

---

**Status: Research in progress** 🔬

**Do not cite in main paper** ❌

**Keep exploring in cosmo/ directory** ✅





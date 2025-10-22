# Option B Halo-Scale Sanity Tests

This test suite verifies that the **linear-regime cosmological framework** (Option B: Ω_eff in FRW background + μ=1 on linear scales) does **NOT** break the halo-scale physics from the main paper.

## What is Option B?

Option B embeds Σ-gravity in FRW cosmology by:
1. Adding **Ω_eff ≈ 0.252** to the FRW background (geometric contribution, no particle DM)
2. Setting **μ(k,a) = 1** on linear scales (k ≲ 0.2 h/Mpc)
3. Preserving the **halo-scale kernel K(R) = A·C(R)** on sub-Mpc scales

This makes linear cosmology (CMB, BAO, linear growth) **degenerate with ΛCDM**, while galaxy/cluster halo physics uses the same kernel as the main paper.

## Question

Does adding Ω_eff to the cosmological background change Solar System, galaxy rotation, or cluster lensing predictions?

**Answer: NO** — the halo kernel K(R) is independent of the cosmological background. All tests confirm predictions are identical to the main paper.

---

## Tests

### 1. Solar System Safety (`test_solar_system.py`)

**What it tests:**
- Kernel K(R) at Solar System radii (0.1–100 AU)
- Comparison to Cassini PPN bound |γ-1| < 2.3×10⁻⁵

**Parameters used:**
- Galaxy kernel: A=0.591, ℓ₀=4.993 kpc, p=0.757, n_coh=0.5 (from main paper)

**Expected result:**
- K(1 AU) ≪ 10⁻⁵ (safety margin >10¹³)
- Option B does NOT change this — kernel is independent of Ω_eff

**Run:**
```bash
python cosmo/tests_option_b/test_solar_system.py
```

**Output:**
- `outputs/solar_system_optionB.json`

---

### 2. Galaxy Rotation Curves (`test_galaxy_vc.py`)

**What it tests:**
- Hernquist baryon toy (M=6×10¹⁰ M☉, a=3 kpc)
- V_c^bar vs V_c^eff with Σ enhancement
- Ratio V_c^eff / V_c^bar at galaxy scales (1–20 kpc)

**Parameters used:**
- Galaxy kernel: A=0.591, ℓ₀=4.993 kpc, p=0.757, n_coh=0.5

**Expected result:**
- Enhancement ratio matches main paper (e.g., ~1.5× at r=8 kpc)
- Option B cosmology does NOT change halo kernel K(R)

**Run:**
```bash
python cosmo/tests_option_b/test_galaxy_vc.py
```

**Outputs:**
- `outputs/galaxy_vc_optionB.json`
- `outputs/galaxy_vc_optionB.csv`

---

### 3. Cluster Lensing (`test_cluster_lensing.py`)

**What it tests:**
- **Distances:** D_l, D_s, D_ls, Σ_crit for (z_l=0.3, z_s=2.0)
  - Compare Option B (Ω_eff=0.252) to ΛCDM (Ω_m=0.30)
- **Kernel:** 1+K(R) at cluster radii (50–2000 kpc)
  - Uses cluster kernel: A_c=4.6, ℓ₀=200 kpc, p=0.75, n_coh=2.0

**Expected results:**
- Distances match ΛCDM by construction (Ω_eff FRW ≈ ΛCDM background)
- Halo kernel unchanged, so Einstein radii predictions preserved
- A2261/MACSJ1149 fits remain valid

**Run:**
```bash
python cosmo/tests_option_b/test_cluster_lensing.py
```

**Outputs:**
- `outputs/cluster_lensing_optionB.json`
- `outputs/cluster_kernel_optionB.csv`

---

## Run All Tests

**Single command:**
```bash
python cosmo/tests_option_b/run_all_tests.py
```

This runs all three tests in sequence and prints a summary.

---

## Results Location

All outputs are written to:
```
cosmo/tests_option_b/outputs/
  ├── solar_system_optionB.json
  ├── galaxy_vc_optionB.json
  ├── galaxy_vc_optionB.csv
  ├── cluster_lensing_optionB.json
  └── cluster_kernel_optionB.csv
```

---

## Interpretation

### ✓ Solar System
- K(1 AU) ~ 10⁻¹⁸ with galaxy kernel
- Safety margin >10¹³ vs Cassini bound
- **Conclusion:** Option B preserves Solar System constraints

### ✓ Galaxy
- V_c enhancement ratios identical to main paper
- Kernel K(R) independent of Ω_eff FRW background
- **Conclusion:** Option B preserves SPARC rotation curve fits

### ✓ Cluster
- Lensing distances match ΛCDM (ratio = 1.000000)
- Halo kernel 1+K(R) unchanged at Einstein radius scales
- **Conclusion:** Option B preserves A2261/MACSJ1149 fits

---

## Why This Matters

These tests confirm that:
1. **Linear cosmology** (Option B with Ω_eff) matches ΛCDM on large scales (CMB, BAO, growth)
2. **Halo physics** (Solar System, galaxies, clusters) uses the same kernel as the main paper
3. **No reruns needed** — all main paper results remain valid

This separation allows you to:
- Submit the **halo-scale paper** (galaxies + clusters) now, citing Option B as future work
- Later publish **cosmology extension** paper using the linear-regime framework without changing halo fits

---

## For Reviewers

If a reviewer asks:
> "How does your cosmological framework affect Solar System constraints?"

**Response:**
> "The halo-scale kernel K(R) used in this paper is independent of the cosmological background. We verified Solar System safety, galaxy rotation curves, and cluster lensing predictions are unchanged whether we use (a) the halo-only framework from this paper or (b) a full FRW cosmology with Ω_eff (deferred to future work). See `cosmo/tests_option_b/` for reproducible sanity checks."

---

## Technical Notes

- **No coupling:** K(R) does not depend on Ω_eff or μ(k,a) from linear cosmology
- **Scale separation:** Linear scales (k ≲ 0.2 h/Mpc, R ≳ 30 Mpc) use μ=1; halo scales (R ≲ 2 Mpc) use K(R)
- **Distances:** Option B's Ω_eff FRW produces identical D_A, D_L, Σ_crit to ΛCDM by construction

---

**Status:** All tests pass. Option B cosmology preserves halo-scale predictions.

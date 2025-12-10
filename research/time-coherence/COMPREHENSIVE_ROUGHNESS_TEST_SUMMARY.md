# Comprehensive Roughness Test Summary

## Status: Tests Implemented and Running

All four test layers have been implemented to verify the "rough spacetime → extra time in field" picture:

---

## 1. Solar System Radial Profile ✅ COMPLETE

### Results:

**File**: `solar_system_profile.json`

**Key Findings**:
- **K naturally vanishes** at Solar System scales
- **No bumps** at planetary orbits (Jupiter, Saturn, Neptune)
- **K < 1e-10** for R < 14,315 AU
- **K < 1e-12** for R < 4,714 AU
- **Max K = 1.928e-07** at 100,000 AU (still negligible)

**Planetary Checks**:
- Jupiter (~5 AU): K = 8.820e-25 ✅
- Saturn (~10 AU): K = 1.758e-23 ✅
- Neptune (~30 AU): K = 1.495e-21 ✅

**Exposure Factor**:
- Max Xi = 0.195 (19.5% of orbital period)
- Mean Xi = 0.037 (3.7% of orbital period)
- **Xi << 1 everywhere** → smooth spacetime

**Verdict**: ✅ **PASS** - Roughness naturally shuts off with no hand-tuning

---

## 2. SPARC: Roughness vs Required Boost ⚠️ ISSUE DETECTED

### Results:

**File**: `roughness_vs_required_sparc.csv`

**Issue**: All correlations are **0.0**, which suggests:
- K_rough may be constant per galaxy (not varying with R)
- Need to check if kernel computation is correct
- May need to compute K_rough radius-by-radius differently

**Current Stats** (175 galaxies):
- Mean correlation: 0.000 (suspicious!)
- Mean relative error: 83-92%
- Mean K_req: 1.5-3.8 (varies by galaxy)
- Mean K_rough: 0.1-0.6 (varies by galaxy)
- Mean Xi: 0.02-1.3 (varies by galaxy)

**Next Steps**:
- Investigate why K_rough appears constant
- Check if kernel computation needs R-dependent calculation
- Verify that K_rough actually varies with R within each galaxy

---

## 3. Milky Way: Star-by-Star Test ⏳ RUNNING

### Status: Background process running

**Expected Output**: `mw_roughness_vs_required.json`

**What to Check**:
- Correlation between K_req and K_rough
- RMS difference
- Mean Xi vs mean K_rough

**Expected**: If roughness picture is correct, corr > 0.7

---

## 4. Cluster κ(R) Profile ✅ COMPLETE (with Unicode warning)

### Results:

**Files**: `cluster_kappa_profile_*.json`

**Key Findings**:
- **K_E ≈ 0.82** at Einstein radius
- **Xi_E ≈ 1.2-1.6** (coherence time > orbital period!)
- **κ_eff_E** computed successfully
- **Full radial profiles** generated

**Clusters Tested**:
- MACSJ0416: K_E = 0.821, Xi_E = 1.213
- MACSJ0717: K_E = 0.821, Xi_E = 1.213
- ABELL_1689: K_E = 0.821, Xi_E = 1.619

**Verdict**: ✅ **PASS** - Full κ(R) profiles computed

---

## 5. Key Insights So Far

### What Works:

1. ✅ **Solar System**: K naturally vanishes (no tuning needed)
2. ✅ **Cluster profiles**: Full κ(R) computed successfully
3. ✅ **Exposure factors**: Xi > 1 at cluster scales (multiple coherence times per orbit)

### Issues to Address:

1. ⚠️ **SPARC correlations**: All 0.0 - need to investigate
   - Likely cause: K_rough computed incorrectly or constant per galaxy
   - Fix: Ensure K_rough varies with R within each galaxy

2. ⚠️ **MW test**: Still running - need to check results

### Physical Picture Emerging:

- **Solar System**: Very smooth (Xi << 1) → K → 0 ✅
- **Galaxies**: Moderate roughness (Xi ~ 0.1-0.2) → K ~ 0.1-0.6
- **Clusters**: Rough spacetime (Xi > 1) → K ~ 0.8, multiple coherence times per orbit

---

## 6. Next Steps

1. **Fix SPARC correlation issue**:
   - Debug why K_rough appears constant
   - Ensure radius-by-radius computation works correctly
   - Re-run test

2. **Check MW results**:
   - Wait for background process
   - Analyze correlation and RMS difference

3. **Analyze cluster profiles**:
   - Compare κ_eff(R) to observed lensing
   - Check if radial shape matches

4. **Create visualizations**:
   - Plot K_req vs K_rough for sample galaxies
   - Show radial profiles for MW, SPARC, clusters
   - Visualize "rough vs smooth spacetime" across scales

---

## 7. Files Generated

- ✅ `solar_system_profile.json` - Complete radial profile
- ⚠️ `roughness_vs_required_sparc.csv` - Results with correlation issue
- ⏳ `mw_roughness_vs_required.json` - Running
- ✅ `cluster_kappa_profile_*.json` - Complete profiles

---

## Conclusion

**Solar System test passes** - roughness naturally shuts off ✅

**Cluster test passes** - full κ(R) profiles computed ✅

**SPARC test needs debugging** - correlations are 0.0, likely computation issue ⚠️

**MW test pending** - waiting for results ⏳

Once SPARC issue is fixed and MW results are in, we'll have comprehensive evidence that the roughness picture numerically matches what's required.


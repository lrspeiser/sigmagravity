# SRC2/SC Diagnostics - Bulge Galaxy Analysis

## Diagnostic Output

The diagnostic output shows `f_coh` and `C_src` for the worst 5 bulge galaxies (f_bulge > 0.3) in SRC2 and SC modes.

### SRC2 Mode Results

| Galaxy | f_bulge | RMS (km/s) | mean(f_coh) | mean(C_src) |
|--------|---------|------------|-------------|-------------|
| NGC2841 | 0.323 | 85.25 | 0.6672 | 0.6397 |
| NGC7814 | 0.773 | 68.49 | 0.2386 | 0.2093 |
| UGC06787 | 0.764 | 67.09 | 0.2847 | 0.2594 |
| UGC06786 | 0.447 | 57.19 | 0.5232 | 0.4866 |
| UGC11914 | 0.483 | 49.04 | 0.4307 | 0.4186 |

### Key Observations

1. **Coherent-fraction gating is working**:
   - Higher bulge fraction → lower `f_coh` (as intended)
   - Example: NGC7814 (f_bulge=0.773) has f_coh=0.2386, while NGC2841 (f_bulge=0.323) has f_coh=0.6672

2. **C_src follows f_coh**:
   - `C_src = f_coh * C_sub`, so C_src is lower when f_coh is lower
   - This confirms the gating mechanism is functioning correctly

3. **But RMS is still very high**:
   - Worst galaxy (NGC2841): RMS = 85.25 km/s
   - Even with lower f_coh, predictions are still poor

## Interpretation

The diagnostics confirm that:
- ✅ The coherent-fraction gating is working as designed
- ✅ Bulge-dominated regions correctly get low coherence
- ❌ But this doesn't solve the bulge problem - RMS remains high

This suggests that:
1. **The problem is not just about coherence suppression** - even with correct gating, predictions are poor
2. **Bulge galaxies may need a fundamentally different treatment** - perhaps they need some enhancement, but through a different mechanism
3. **The issue might be in how the enhancement is applied** - maybe the formula `V_new^2 = V_bulge^2 + V_coh^2 * Sigma_coh` is not appropriate for bulge-dominated systems

## Next Steps

1. Compare with baseline C mode to see what f_coh and C would be without gating
2. Investigate if the problem is in the velocity combination formula
3. Consider if bulge galaxies need a different coherence model entirely




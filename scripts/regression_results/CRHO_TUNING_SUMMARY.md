# CRHO Parameter Tuning Summary

## Sweep Results

The full parameter sweep (35 combinations) showed **identical results** for all parameter values:
- SPARC RMS: 17.414 km/s
- Bulge RMS: 28.926 km/s  
- Disk RMS: 16.060 km/s

This suggests that the **density-dependent scaling is not activating** - either:
1. The density proxy values are below the threshold
2. The scaling factor is too small to have measurable effect
3. The density_factor calculation needs adjustment

## Baseline Comparison

- **Baseline (C mode)**: RMS=17.42, Bulge=28.93, Disk=16.06
- **CRHO (all parameters)**: RMS=17.41, Bulge=28.93, Disk=16.06

CRHO matches baseline exactly, indicating the modification is not having an effect.

## Next Steps

1. **Check actual density values** in SPARC galaxies to verify threshold
2. **Remove density-dependent scaling** temporarily to see full effect
3. **Increase CRHO_SCALE** significantly (try 1.0, 2.0) to test if effect exists
4. **Lower rho_threshold** dramatically (try 1e-20) to activate in more regions

## Hypothesis

The density proxy from `rho_proxy_from_vbar()` may be giving values that are:
- Too low (below threshold)
- Too uniform (not distinguishing bulge vs disk)
- Computed incorrectly for the spherical approximation

We may need to:
- Use component-based density (bulge vs disk separately)
- Adjust the density proxy calculation
- Use a different density indicator (e.g., acceleration-based)



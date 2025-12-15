# Flow Coherence: Data Limitation Analysis

## The Key Question

**Is the 6D data version not possible with SPARC because:**
1. SPARC data doesn't have 6D information, OR
2. It doesn't predict SPARC properly?

## Answer: SPARC Data Doesn't Have 6D Information

### SPARC Data (External Galaxies)
- **Available:** Rotation curves only
  - R (kpc) - galactocentric radius
  - V_obs (km/s) - observed circular speed
  - V_gas, V_disk, V_bulge (km/s) - baryonic component speeds
- **Missing:** Individual star positions/velocities
- **Limitation:** Only **integrated** measurements (line-of-sight velocities, circular speeds)
- **Flow topology approximation:** Axisymmetric (ω = V/R, s = |dV/dR|)

### Gaia Data (Milky Way)
- **Available:** 6D phase-space for individual stars
  - R, z, φ (kpc) - positions
  - v_R, v_φ, v_z (km/s) - velocity components
- **Enables:** Direct computation of flow topology
  - Local vorticity (from velocity gradients)
  - Local shear (from velocity gradients)
  - Local density (from neighbor spacing)
- **Flow topology:** Full 3D structure

## Current Implementation

We implemented flow coherence using **axisymmetric approximation** for SPARC:
- ω = V/R (angular velocity)
- s = |dV/dR| (velocity gradient)
- θ ≈ 0 (divergence, assumed small for rotation curves)

**Result:** Flow mode produces worse results (RMS=25.38 vs 17.42 baseline)

## Why It's Not Working

The axisymmetric approximation may not capture the full 3D flow structure that matters:
- **Disks:** High vorticity, low shear → should work
- **Bulges:** Multi-stream, pressure-supported → axisymmetric approximation fails
- **Real 3D structure:** Requires individual star positions/velocities

## Solution: Hybrid Approach

**Use flow topology only where we have 6D data:**

1. **Gaia/MW test:** Use flow coherence (C_flow) - we have 6D data
2. **SPARC test:** Use baseline coherence (C = v²/(v²+σ²)) - only rotation curves

This respects the data limitations while leveraging flow topology where it's measurable.

## Implementation Strategy

Add a flag to `predict_velocity()`:
- `use_flow_coherence: bool = False` - enable flow topology
- For SPARC: `use_flow_coherence=False` (default)
- For Gaia: `use_flow_coherence=True` (when 6D data available)

This way:
- SPARC uses proven baseline C
- Gaia/MW uses flow topology (where vorticity is #1 driver)
- Both tests can run independently

## Next Steps

1. Implement hybrid mode: flow for Gaia, baseline for SPARC
2. Test if flow topology improves Gaia/MW predictions
3. Keep baseline C for SPARC (where 6D data unavailable)



# h(g) High-Acceleration Tail Softening - Approach

## Concept

The enhancement formula is **Σ = 1 + A × C × h**, and analysis shows that **h moves Σ the most** (correlation 0.754, largest variation). 

**The bulge problem**: Bulge galaxies have high internal acceleration → low h → low enhancement → under-prediction.

**The solution**: Soften the high-acceleration tail of h(g) so that bulges (which live at high g) get more enhancement without affecting disk outskirts or clusters.

## Implementation

### Baseline h(g)

```
h(g) = √(g†/g) × g†/(g†+g)
     → h ~ (g/g†)^(-3/2) for g >> g†
```

### Modified h(g) (hi_power mode)

For x = g/g† > x₀, multiply baseline by (x/x₀)^(1.5 - p_hi):

```
h ~ x^(-p_hi) for x >> x₀
```

where:
- **p_hi < 1.5**: Softer tail (less suppression at high g)
- **x₀**: Knee where modification turns on (in units of g/g†)

## Key Parameters

- **H_MODEL**: 'baseline' or 'hi_power'
- **H_HI_P**: Target high-g exponent (default 1.5, try 0.7-1.0 for softer tail)
- **H_HI_X0**: Knee position (default 10.0, try 1.0-3.0 for earlier activation)

## CLI Usage

```bash
# Enable hi_power mode
python scripts/run_regression_experimental.py --core --h=hi_power --h-hi-p=0.8 --h-hi-x0=10

# Sweep parameters
python scripts/sweep_h_hi_power.py
```

## Initial Test Results

### Baseline (p_hi=1.5, x0=N/A)
- Overall RMS: 17.42 km/s
- Bulge RMS: 28.93 km/s
- Disk RMS: 16.06 km/s
- Solar System: |γ-1| = 1.77e-09

### p_hi=0.9, x0=10
- Overall RMS: 17.42 km/s (no change)
- Bulge RMS: 28.95 km/s (no change)
- **Reason**: Bulge accelerations are x ~ 3.8, below x0=10, so modification doesn't activate

### p_hi=0.8, x0=3
- Overall RMS: 17.47 km/s (+0.05 km/s, slightly worse)
- Bulge RMS: 29.13 km/s (+0.20 km/s, worse)
- Disk RMS: 16.10 km/s (+0.04 km/s, slightly worse)
- Solar System: |γ-1| = 9.95e-06 (still safe, < 2.3e-05)
- **Reason**: Modification activates (x0=3 < mean x=3.8), but makes things worse

### p_hi=0.9, x0=3
- Overall RMS: 17.47 km/s (same as p_hi=0.8)
- Bulge RMS: 29.13 km/s (same)
- **Reason**: Similar effect, still worse

## Key Finding

**Bulge accelerations are lower than expected**: Mean x = g/g† ≈ 3.8 for high-bulge galaxies, not >> 10.

This means:
- **x0=10 is too high** - modification doesn't activate for most bulge regions
- **x0=3 activates** but makes predictions worse (possibly too aggressive or wrong direction)

## Next Steps

1. **Test x0 < 3** (e.g., x0=1.0, 1.5) to see if earlier activation helps
2. **Test p_hi > 0.8** (e.g., p_hi=1.0, 1.2) for gentler modification
3. **Check if the problem is that we need p_hi > 1.5** (steeper, not softer) - unlikely but worth testing
4. **Run full sweep** to find optimal (p_hi, x0) combination

## Expected Behavior

- **Bulge RMS should decrease** (that's the goal)
- **Disk RMS should stay stable** (disk outskirts have low g, below x0)
- **Clusters should stay ~1.0** (clusters at 200 kpc have moderate g, near g†)
- **Solar System must stay < 2.3e-05** (Cassini bound)

## Files

- **Implementation**: `scripts/run_regression_experimental.py` (h_function, CLI parsing)
- **Sweep script**: `scripts/sweep_h_hi_power.py`
- **Results**: `scripts/regression_results/H_HI_POWER_SWEEP.json` and `.md` (after running sweep)



# Analysis: Can We Eliminate k?

## Surprising Results

| Model | Parameters | RMS (km/s) | vs Current |
|-------|------------|------------|------------|
| Current (k×σ/Ω) | k=0.24 | 24.46 | — |
| **No W(r) at all** | — | 24.73 | **+1.1%** |
| **Fixed global ξ** | ξ=0.16 kpc | 24.15 | **-1.3%** |
| ξ = α×R_d | α=0.10 | 24.69 | +0.9% |
| ξ = (2/3)R_d | α=0.67 | 30.22 | +23.6% |
| **Modified h(g) exponent** | n=0.40 | 23.00 | **-6.0%** |

## Key Insights

### 1. W(r) Barely Matters!

Removing W(r) entirely only degrades RMS by 1.1%. This means the coherence window is contributing almost nothing to the predictions.

**Why?** For typical galaxies:
- ξ ≈ 0.2-0.5 kpc (with k=0.24)
- Rotation curve radii: 1-30 kpc
- At r = 5 kpc with ξ = 0.3 kpc: W = 1 - (0.3/5.3)^0.5 = 0.76
- At r = 10 kpc: W = 0.83
- At r = 20 kpc: W = 0.88

W(r) is already near saturation (>0.75) for almost all data points!

### 2. A Fixed Global ξ Works Better Than k×σ/Ω

A single fixed ξ = 0.16 kpc for all galaxies gives **better** RMS than the "dynamical" formula. This strongly suggests k×σ/Ω is not capturing real physics.

### 3. The Historical (2/3)R_d Is Actually Worse

The historical baseline ξ = (2/3)R_d gives 23.6% worse RMS. This is because:
- R_d varies from ~1 to ~10 kpc across galaxies
- This makes ξ too large for many galaxies
- The optimal α is 0.10, not 0.67

### 4. Modifying h(g) Works Better

Changing the h(g) exponent from 0.5 to 0.4 (without W) gives 6% **improvement**. This suggests the spatial dependence should be in h(g), not W(r).

## What Does This Mean?

### The Coherence Window W(r) Is Redundant

The data suggests W(r) is not doing useful work. The enhancement is driven almost entirely by:
- A(G) - the amplitude
- h(g) - the acceleration dependence

### Why Did We Think k×σ/Ω Helped?

The 16% improvement from k×σ/Ω over (2/3)R_d may be because:
1. k×σ/Ω gives **smaller** ξ values (median ~0.3 kpc vs ~2 kpc for (2/3)R_d)
2. Smaller ξ → W closer to 1 → W effectively removed
3. The "improvement" is really just removing the harmful (2/3)R_d prescription

### The Real Comparison

| Model | ξ value | W at r=5 kpc | RMS |
|-------|---------|--------------|-----|
| k×σ/Ω | ~0.3 kpc | 0.76 | 24.46 |
| Fixed ξ | 0.16 kpc | 0.83 | 24.15 |
| (2/3)R_d | ~2 kpc | 0.44 | 30.22 |
| No W | — | 1.00 | 24.73 |

The "dynamical" k×σ/Ω is really just a way to get small ξ, which makes W≈1.

## Recommendation: Simplify the Model

### Option A: Remove W(r) Entirely

The simplest model:
```
Σ = 1 + A(G) × h(g_N)
```

This has:
- No k parameter
- No ξ parameter  
- Only 1.1% worse than current
- Fewer assumptions

### Option B: Modify h(g) to Include Spatial Dependence

If we want spatial dependence, put it in h(g):
```
h(g) = (g†/g)^n × g†/(g†+g)  with n ≈ 0.4
```

This gives 6% **better** RMS than current model.

### Option C: Keep W(r) with Fixed Small ξ

If we want to keep the W(r) structure:
```
W(r) = 1 - (ξ/(ξ+r))^0.5  with ξ = 0.2 kpc (fixed)
```

This removes k, σ, and Ω from the formula while maintaining the window structure.

## Conclusion

**The parameter k is not needed because W(r) itself is not needed.**

The coherence window was designed to model how enhancement "builds up" with radius, but the data shows this effect is negligible. The enhancement is dominated by h(g), which depends on acceleration, not radius.

The "dynamical" coherence scale k×σ/Ω was an improvement over (2/3)R_d only because it gave smaller ξ values, effectively removing the W(r) contribution.

**Recommended action:** Consider removing W(r) from the model, or at minimum, acknowledging that it contributes <2% to predictions.


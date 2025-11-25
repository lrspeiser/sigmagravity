# Σ-Gravity LIGO Data Analysis Guide

## Quick Start

### 1. Install Required Packages

Open your terminal and run:

```bash
pip install gwosc gwpy pesummary h5py numpy matplotlib scipy --break-system-packages
```

### 2. Run the Analysis

```bash
cd /path/to/your/folder
python sigma_gravity_ligo_analysis.py
```

---

## Data Sources

### Primary Source: GWOSC (Gravitational Wave Open Science Center)

**Website:** https://gwosc.org/

**What's available:**
- **Strain time series** (h(t)) - the raw detector output
- **Parameter estimation posteriors** - inferred masses, distances, spins
- **Data quality information** - flags for good/bad data segments

### Available Catalogs (as of 2024-2025)

| Catalog | Observing Run | Events | Zenodo Link |
|---------|---------------|--------|-------------|
| GWTC-1 | O1 + O2 | 11 events | [zenodo.org/records/6513631](https://zenodo.org/records/6513631) |
| GWTC-2.1 | O3a | 44 confident | [zenodo.org/records/5117703](https://zenodo.org/records/5117703) |
| GWTC-3 | O3b | 35 confident | [zenodo.org/records/8177023](https://zenodo.org/records/8177023) |
| GWTC-4.0 | O4a | 80+ candidates | [zenodo.org/records/16053484](https://zenodo.org/records/16053484) |

### Key Events for Your Analysis

| Event | Total Mass (M☉) | Distance (Mpc) | Notes |
|-------|-----------------|----------------|-------|
| GW150914 | ~65 | ~430 | First detection, clean signal |
| GW190521 | ~150 | ~5300 | Most massive BBH in O3 |
| GW231123 | ~225 | ~6500 | Most massive ever (O4a) |
| GW190814 | ~26 | ~240 | Unusual mass ratio |

---

## Understanding the Data

### Strain Data (h(t))

The "strain" is the fractional change in arm length: h = ΔL/L

- **Format:** HDF5 files with arrays of strain values
- **Sample rates:** 4096 Hz or 16384 Hz
- **Typical values:** h ~ 10^-21 (incredibly small!)

### Parameter Estimation Posteriors

These are probability distributions for the source parameters, derived by fitting GR waveforms to the data.

**Key parameters for your Σ-Gravity analysis:**

| Parameter | Name in Files | Description |
|-----------|---------------|-------------|
| `mass_1_source` | Primary mass | Heavier BH mass (source frame) |
| `mass_2_source` | Secondary mass | Lighter BH mass (source frame) |
| `total_mass_source` | Total mass | m1 + m2 (source frame) |
| `chirp_mass_source` | Chirp mass | (m1*m2)^(3/5)/(m1+m2)^(1/5) |
| `luminosity_distance` | Distance | In Mpc |
| `redshift` | Redshift | Cosmological redshift |
| `chi_eff` | Effective spin | Aligned spin parameter |

---

## Connecting to Your Σ-Gravity Theory

### The Core Calculation

Your coherence enhancement factor:

```
C(r) = g_eff(r) / g_bar(r) = M_eff(<r) / M_bar(<r)
```

**From LIGO data:**
- `M_eff` = GR-inferred mass (what we measure)
- This comes directly from the `total_mass_source` posterior

**From your theory:**
- `M_bar` = "baryonic" mass (what you predict WITHOUT coherence)
- You need to estimate this independently!

### The Challenge for LIGO Events

For galaxies, you can estimate M_bar from:
- Luminous matter (stars, gas)
- Your SPARC fits

For LIGO black holes, estimating M_bar is harder:
- Stellar evolution models predict progenitor masses
- Population synthesis gives mass distributions
- But there's no independent "baryonic" measurement

**Possible approaches:**
1. Use stellar evolution upper limits as M_bar
2. Compare LIGO masses to population predictions
3. Look for systematic offsets across many events

### Per-Period Gain

If your model is C = (1 + ε)^N where N = r/λ_coh:

```
ε = C^(1/N) - 1
```

For LIGO events at ~500 Mpc with λ_coh ~ 2.2 kpc:
- N ~ 500,000 kpc / 2.2 kpc ~ 230,000 periods
- Even tiny ε would accumulate significantly!

---

## Modifying the Code

### Where to Change M_bar Estimate

In `sigma_gravity_ligo_analysis.py`, find this section around line 320:

```python
    # =========================================================================
    # MODIFY THIS SECTION FOR YOUR ANALYSIS:
    # =========================================================================
    
    # Event to analyze
    event_name = "GW150914"
    
    # Your M_bar estimate (what mass you predict WITHOUT coherence)
    # This is the key theoretical input from your Σ-Gravity model
    M_bar_estimate = 60.0  # M_sun - ADJUST THIS BASED ON YOUR THEORY
    
    # Your coherence wavelength from Σ-Gravity (you derived ~2.2 kpc)
    lambda_coh = 2.2  # kpc
    
    # =========================================================================
```

### Adding New Events

To analyze a different event, change `event_name`:

```python
    event_name = "GW190521"  # Change to any event from the catalogs
```

### Batch Analysis of Multiple Events

Uncomment and modify the batch analysis section at the end of the file:

```python
    events_to_analyze = {
        "GW150914": 60.0,   # Event: M_bar estimate
        "GW190521": 130.0,  
        "GW190814": 25.0,
    }
    batch_results = batch_analyze_events(events_to_analyze, lambda_coh_kpc=2.2)
```

---

## Troubleshooting

### "Module not found" errors

Make sure all packages are installed:
```bash
pip install gwosc gwpy pesummary h5py numpy matplotlib scipy --break-system-packages
```

### "Event not found" errors

Check the exact event name:
```python
from gwosc.datasets import find_datasets
events = find_datasets(type='events', catalog='GWTC-1-confident')
print(events)
```

### Network/download issues

The scripts fetch data from GWOSC servers. If you have firewall issues:
1. Try downloading files manually from https://gwosc.org/
2. Use the `fetch_pe_posteriors_direct()` function with local files

### PE samples don't match paper values

Parameter estimation is model-dependent. Different waveform approximants give slightly different results. The "mixed" samples combine multiple analyses.

---

## Next Steps for Your Research

1. **Start simple:** Analyze GW150914 first (cleanest signal)

2. **Estimate M_bar systematically:** 
   - Use stellar evolution models
   - Or fit your Σ-Gravity predictions

3. **Look for patterns:**
   - Does C correlate with distance?
   - Does C correlate with mass?
   - Is there a systematic offset?

4. **Compare to galaxy results:**
   - Your SPARC analysis gave specific coherence parameters
   - Do LIGO events require consistent parameters?

5. **Consider model-dependent effects:**
   - GR waveforms assume GR
   - Modified gravity would change the waveform templates
   - This is a deeper analysis requiring waveform recomputation

---

## References

- GWOSC Documentation: https://gwosc.org/about/
- GWpy Tutorial: https://gwpy.github.io/docs/latest/
- GWTC-4.0 Paper: https://dcc.ligo.org/LIGO-P2400386/public
- PESummary Documentation: https://lscsoft.docs.ligo.org/pesummary/

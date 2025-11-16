# Coherence test runs (2025-11-16)

## Data usage
- Real SPARC rotation curves (`gravitywavebaseline/sparc_results/*.json`).
- Real velocity dispersions from `data/sparc/sparc_combined.csv`.
- Derived orbital wavelength: lambda_orb = 2*pi*R from each SPARC radius.
- Baryonic accelerations inferred as g_bar = V_GR^2 / R. No toy or synthetic
  inputs were added.

## Method
`run_sparc_coherence_fits.py` fits each coherence-microphysics model to the
SPARC multipliers f(R) = (V_obs/V_GR)^2 for three benchmark galaxies
(NGC2403, NGC3198, NGC5055). Parameter estimation uses SciPy
`least_squares` with mild bounds to keep the Burr-XII-like shapes
reasonable.

## Key results
- **NGC2403 (sigma_v=20.2 km/s)**: Metric resonance gives the lowest RMS
  multiplier error (0.44) and velocity RMS 10.5 km/s. Other mechanisms are
  ~35-55% worse in RMS(mult).
- **NGC3198 (sigma_v=22.3 km/s)**: Vacuum condensation is slightly best
  (RMS(mult)=0.39, RMS(v)=18.9 km/s), closely followed by entanglement and
  graviton pairing.
- **NGC5055 (sigma_v=26.9 km/s)**: Graviton pairing has the smallest RMS
  multiplier (0.44) and velocity (60.7 km/s) but still struggles with this
  high-mass system.

## Parameter behavior
- Several fits push the amplitude A and coherence lengths ell0/xi0 toward
  the upper bounds, signaling degeneracies that need tighter priors tied to
  the measured Î£-Gravity kernel.
- Metric resonance prefers very large lambda_m0 (~80 kpc) for the SPARC
  discs except for NGC5055, where it collapses toward the lower limit,
  indicating sensitivity to the assumed orbital wavelength proxy.
- Vacuum condensation fits gravitate toward high sigma_c and alpha, making
  the transition very gentle; more galaxies are needed to pin down the
  critical dispersion.

## Next steps
1. Expand the galaxy set (full SPARC or per-mass bins) so the optimizer
   cannot hide behind the bounds.
2. Replace the simple lambda_orb = 2*pi*R proxy with the actual spectral
   lambda output from the gravity-wave baseline pipeline.
3. Fit the Burr-XII hyperparameters (ell0, p, n_coh) directly from the
   existing Î£-Gravity kernel snapshots so each mechanism inherits the
   empirically validated window.

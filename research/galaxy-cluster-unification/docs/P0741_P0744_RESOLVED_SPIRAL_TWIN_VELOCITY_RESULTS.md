# P0741-P0744: resolved spiral twins and real velocity fields

These stages answer two separate questions that a scientific simulator must
never collapse into one score:

1. Does a generated galaxy preserve the observed baryonic structure closely
   enough that the same formula makes nearly the same prediction on both?
2. Does that prediction match the galaxy's observed motion?

## Data and leakage boundary

The development sample is NGC2403, NGC3198, NGC5055, and NGC7793. P0741 fused
SINGS and AllWISE stellar light with THINGS H I maps into common face-on
baryonic grids. No velocity or dispersion array and no gravity parameter was
used to build those grids.

P0742 then showed that the earlier radial/Fourier generator was not adequate
for resolved spirals. Its best tier retained radial formula predictions fairly
well, but introduced artificial rings, failed the 2D morphology gates, and
changed line-of-sight formula predictions too much. The failed result is
retained because it identifies a real generator limitation.

P0743 replaced that representation with a local two-dimensional Haar basis.
The smallest passing package stores 256 coefficients for gas and 256 for stars
per galaxy. It conserves each component's mass and contains no gravity
parameter or observed velocity. On the four development galaxies it achieved:

- total-map median/worst normalized error: 0.114/0.157;
- minimum total-map pixel correlation: 0.987;
- fixed-formula radial prediction transport median/worst RMSE: 1.22/2.09 km/s;
- fixed-formula line-of-sight prediction transport median/worst RMSE:
  4.38/6.70 km/s.

The 256-coefficient choice was made on the development maps and is not a blind
validation result.

## First direct velocity result

P0744 froze the generator and both formula fixtures before opening the four
development velocity and dispersion fields. It then scored 76,182 H I pixels.
Only the systemic velocity and a binary rotation handedness were inferred as
observation-coordinate nuisances. Distance, inclination, position angle,
mass-to-light ratio, pressure support, bars, warps, streaming motion, MOND
parameters, and dark halos were not fitted to the residuals.

| Galaxy | Fixed MOND RMSE | Error / declared uncertainty | Newtonian RMSE | Twin-to-source MOND transport |
|---|---:|---:|---:|---:|
| NGC2403 | 25.93 km/s | 2.39 | 44.44 km/s | 5.75 km/s |
| NGC3198 | 10.74 km/s | 0.69 | 49.70 km/s | 3.41 km/s |
| NGC5055 | 26.39 km/s | 1.59 | 37.57 km/s | 5.10 km/s |
| NGC7793 | 34.00 km/s | 2.78 | 37.69 km/s | 1.25 km/s |

The table uses registered-baryon formula scores and reports the fake-twin
transport metric separately. Fixed simple MOND is consistent with the raw 2D
field uncertainty for NGC3198, close for NGC5055, and misses NGC2403 and
NGC7793. Baryon-only Newtonian gravity misses all four. The fake twins preserve
the formulas' predictions much better than either formula necessarily predicts
the observations.

This is precisely the distinction the simulator UI should display: source
fidelity, formula transport, and observational accuracy are three different
tests. The remaining validation and holdout velocity arrays are still sealed.


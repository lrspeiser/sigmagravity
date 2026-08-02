# P0690 full source-routing empirical screen

Frozen before scores: 2026-08-02  
Verdict: full routing **fails**; topology improves but amplitude is far too large

## Tested operator

P0690 evaluates the complete P0689 relocation:

\[
S_{\rm route}=S_0+A_+{\rho_b\over\int\rho_b dV}
-A_+{W_t\over\int W_t dV}.
\]

It uses no new constant, per-object gravity value, or fitted photon amplitude.
The protocol jointly freezes the spent galaxy shallow-field check, six-cluster
spherical transfer, registered RX J2129 photon field, and raw image topology.

## Frozen result

| Domain | Metric | Result | Gate | Verdict |
|---|---|---:|---:|---|
| galaxies | maximum local-generator fractional change | `0.1265` | `<=0.001` | fail |
| galaxies | reported base-limit RMSE / fixed RAR | `1.000` | `<=1.001` | numerical comparator only |
| clusters | all-five radial log RMS | `0.871 dex` | `<=0.20` | fail |
| clusters | reliable-three radial log RMS | `1.089 dex` | `<=0.20` | fail |
| clusters | fixed-RAR gap closed | `-49.4%` | `>=75%` | fail |
| 3D field | median physical deflection | `23.81 arcsec` | `>=1` | pass but excessive |
| raw lens | training / heldout roots | `14/15`, `4/7` | `15/15`, `7/7` | fail |
| topology | missing / exact / observable-surplus families | `1 / 4 / 2` | conjunctive gates | fail |
| topology | parity-diverse families | `7/7` | `7/7` | pass |
| topology | critical-curve families | `7/7` | `7/7` | pass |

The candidate overpredicts every cluster. Mean log10 prediction/target ranges
from `+0.30` for MACS0329 to `+1.65` for RXJ1347. Both fitted map-center
nuisances hit their `-3 arcsec` bounds.

The galaxy shallow-field substitution is also rejected. Although the displayed
score is exactly fixed RAR by construction, the locked local generator differs
by as much as 12.65% on the spent points. The frozen bound correctly prevents
that comparator from being presented as a source-routing galaxy result. Real
2D galaxy field solves remain required.

## What improved

Full routing is not merely random over-bending. Relative to P0686:

- parity diversity improves from `5/7` to `7/7`;
- critical curves remain `7/7`;
- family 1 changes from one root to exact three-image multiplicity; and
- only one family is missing images instead of three.

Source placement therefore affects the intended topology, but routing 100% of
the positive generator grossly over-amplifies the inner field.

## Multipole-gated generator

Do not fit a routing fraction to RX J2129. Calculate it from the baryonic
quadrupole. For the mass-weighted covariance tensor `C`, define

\[
q_b=\sqrt{3\over2}{\|C-\mathrm{tr}(C)I/3\|_F\over\mathrm{tr}(C)}.
\]

This has exact geometric limits: `q_b=0` for a sphere and `q_b=1` for a
line-like source. The spent RX J2129 baryonic map gives `q_b=0.1188629`.

The generated source equation is

\[
S_{\rm mix}=(1-q_b)S_{\rm local}+q_bS_{\rm route}.
\]

A spherical cluster profile retains the successful P0684 local radial law;
nonspherical real maps activate only their measured amount of source
relocation. This is a genuinely geometric multipole term, not a fitted blend.
It must be frozen in a new protocol before its field or topology is calculated.

## Reproduction

```powershell
python scripts/run_p0690_source_routing_empirical_screen.py
python -m pytest tests/test_source_routing_spherical.py tests/test_source_routing_qumond.py tests/test_spatial_qumond_3d.py tests/test_potential_channel_qumond.py -q
```

Artifacts are in `results/p0690_source_routing_empirical_screen/`.

## Claim boundary

P0690 is entirely spent development evidence. The topology improvement does
not validate the source-routing interpretation, and the zero-slip closure is
not a covariant lens theory. P0633 and P0640 remain sealed.

# P0564-P0565 morphology-sign gate

## Outcome

The four-cluster sign conflict is associated with a measurable baryonic
morphology difference, and that difference predicted the **direction** of RX
J2129's source-plane response. It did **not** produce a stable exact
image-position improvement.

This is the first small transfer success in this sequence, but it is not a
validated gravity gate: two thresholds were derived post-hoc from one negative
and three positive clusters, RX J2129 is globally spent in earlier work, and
its exact result changes sign between optimizer ensembles.

## P0564: what distinguishes MACS0429

P0564 measured a frozen set of star/gas morphology descriptors at 30, 60, and
120 arcsec. It used registered F160W starlight and smoothed square-root Chandra
brightness; no lens score entered the maps.

The two clearest physically related differences are:

| Descriptor | MACS0429 | Other three clusters |
|---|---:|---:|
| Star-gas pixel correlation inside 30″ | **0.505** | 0.080–0.151 |
| Star-gas quadrupole misalignment at 120″ | **59.7°** | 4.6–25.1° |
| Quadrupole cos(2Δ) alignment at 120″ | **−0.492** | 0.640–0.987 |
| Star-gas centroid offset / 30″ | **0.0447** | 0.0770–0.0844 |

The descriptive picture is a comparatively coherent, centered inner core
whose gas and stellar quadrupoles twist apart on larger scales. Because the
response labels were already known, this only nominated an observable.

## Frozen RX J2129 sign rule

Before computing RX J2129's morphology or tensor response, P0565 froze:

$$
t = \begin{cases}
-0.3, & C_{30} > 0.3278835\ \text{and}\
\cos(2\Delta_{120}) < 0.0744005,\\
+0.3, & \text{otherwise}.
\end{cases}
$$

Each threshold is the midpoint of the P0564 gap. The rule has two learned
thresholds, one universal magnitude, and no RX J2129-fitted gravity parameter.

RX J2129 measured:

- inner correlation `C30 = 0.3933`;
- outer quadrupole misalignment `48.85°`;
- outer alignment `cos(2Δ120) = -0.1340`.

Both triggers fired, prospectively predicting **negative** coupling.

## Directional transfer

Two independent 12-start zero-geometry ensembles give near-zero source-plane
slopes of `+0.09360` and `+0.09392` arcsec per unit `t`. A positive slope means
negative `t` reduces the held-out source separation. Thus the frozen sign
prediction succeeds in both ensembles.

The full conditioning-robust response continues toward the `t=-6` boundary,
where the source-plane diagnostic improves by about 33%. That boundary optimum
is not used as a formula: P0561 already showed that large couplings frequently
destroy exact roots in other clusters.

## Exact transfer

At the frozen `t=-0.3`, every RX J2129 held-out root remains present:

| Ensemble | Zero exact RMS | Gated exact RMS | Change |
|---|---:|---:|---:|
| seed 1 | 2.8608″ | 2.8952″ | **1.20% worse** |
| seed 2 | 2.9010″ | 2.8872″ | **0.48% better** |

The candidate passes the directional and root gates but fails the requirement
to improve exact RMS in both optimizer ensembles. It is not validated or
promoted.

## What this teaches us

1. **Multiscale baryonic morphology may predict response direction.** The
   coherent-core/twisted-outskirts rule transferred once to RX J2129.
2. **Direction is not enough.** A correct local derivative does not guarantee
   a better refitted inverse-lens prediction.
3. **The exact effect is below nuisance-basin stability.** A swing from −1.20%
   to +0.48% is too unstable for a physics claim.
4. **The promising observable is relational and multiscale.** Absolute gas
   mass failed earlier; star–gas alignment changing with radius is more
   discriminating.
5. **The next requirement is more clusters, not another threshold tweak.** The
   rule must be frozen unchanged and tested on additional systems with raw
   images, HST, Chandra, and ACCEPT profiles.

## Reproduce

```powershell
python scripts/run_p0564_baryon_morphology_sign_audit.py
python scripts/run_p0565_rxj2129_morphology_gate_transfer.py
python -m pytest tests/test_p0564_p0565_morphology_gate_results.py -q
```

Machine-readable outputs are under
`results/p0564_baryon_morphology_sign_audit/` and
`results/p0565_rxj2129_morphology_gate_transfer/`.

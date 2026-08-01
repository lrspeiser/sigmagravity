# Next falsification for baryon-sourced gravity arcs

## Purpose

The project can already backtrack projected baryonic origins to apparent
lensing-excess locations. What it cannot yet do is show that a field traveled
along those paths. The next experiment should therefore predict observables
that depend on the potential and on source distance, not just reproduce a
two-dimensional convergence shape.

## Frozen workflow

1. Choose a cluster not used in P0554--P0609 and freeze it before inspecting
   lens residuals.
2. Build registered component maps for member and diffuse stellar mass,
   BCG/ICL, and temperature/emissivity-corrected X-ray or SZ gas surface mass.
3. Construct the conventional lens-model excess only in the development
   sample. Infer a scale-free source-to-arrival kernel there.
4. Express the kernel using baryonic observables only: potential depth, tidal
   eigenvalues, component misalignment, concentration, and boundary extent.
5. Freeze one amplitude and all gates. Do not use the new cluster's lens map to
   draw its routes.
6. Forward-predict raw multiple-image positions, time-delay ratios, and weak
   shear or magnification with one curl-free potential.

## Required gates

| Observable | Advance threshold | Why it matters |
|---|---:|---|
| Exact image roots | 100% | Topology must exist before RMS is meaningful |
| Strong-image RMS | <2 arcsec initially; target <0.5 | Prevents morphology-only success |
| Time-delay ratios | Within published covariance | Tests potential depth, not only its gradient |
| Outer weak shear/magnification | Positive held-out likelihood gain | Stops central geometry from absorbing the route |
| Source-redshift scaling | One frozen law across families | Tests single-plane versus extended response |
| Galaxy rotation | No worse than fixed RAR/MOND benchmark | Preserves cross-scale relevance |
| Solar proxies | Cassini and perihelion gates pass | Requires screening/local GR recovery |
| Parameter count | One universal amplitude, no cluster gravity retuning | Keeps the comparison scientifically meaningful |

## Decision rule

Promote a field formula only if the same baryon-only kernel passes all raw
image roots and improves at least two independent lens observables on a cluster
that did not select it. If it improves only a normalized mass map, one cluster,
or one angular mode, keep it as a conditional morphology observation rather
than a theory candidate.

The current leading conditional observation is dual component misalignment:
MACS0429 responds while the other transferred clusters do not, and it is also
the only measured system in which gas disagrees strongly with both discrete
members and smooth starlight. Freeze the posthoc P0610 candidate before opening
the new cluster:

$$
A_{\rm dual}=\sqrt{\max(0,1-c_{gm})\max(0,1-c_{gs})},\qquad
H={A_{\rm dual}^4\over A_{\rm dual}^4+0.3^4},\qquad
s_{\rm eff}=0.0025H.
$$

This gate must be calculated from baryonic maps alone. Do not adjust the 0.3
threshold, fourth power, sign, or base strength after seeing the new lensing
residuals. P0610 is explicitly a posthoc candidate generator dominated by one
cluster; only the fresh test can assess it.

## P0611 disposition

That exact gate has now failed its chronologically prospective A383/MS2137
transfer. A383's activation is effectively zero, while MS2137 activates at
0.595 but misses one training root and changes the held-out-only RMS by just
0.026%. Do not retune this gate. The next genuinely unused test should instead
freeze a destination rule based on tidal saddles or member-to-member paths and
retain the raw-root, time-delay, and weak-lensing requirements above.

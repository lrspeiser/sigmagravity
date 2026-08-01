# Galaxy-locked metric-slip test

## Outcome

A universal metric slip is useful but insufficient.  It preserves the fixed-RAR
galaxy result, passes the declared screened Solar-System check, and improves raw
cluster-lensing predictions relative to zero slip and fixed simple MOND.  It
does not approach the compact-halo control on unseen cluster images, and it
cannot repair the independently poor cluster-central matter dynamics.

The selected complete-root value is

$$
s=5,
$$

which makes the non-Newtonian contribution bend light 3.5 times as strongly as
it accelerates matter.  This is a large physical split, not a small correction.

## Equation tested

Write the galaxy-locked matter acceleration as

$$
g_{\rm dyn}=g_{\rm bar}+g_\Sigma,
$$

where $g_\Sigma$ is the extra acceleration required by the fixed RAR.  Split the
two weak-field metric potentials as

$$
\Phi=\Phi_N+\phi,
\qquad
\Psi=\Phi_N+(1+s)\phi.
$$

Slow matter responds to $\Phi$, while light responds to the Weyl combination
$(\Phi+\Psi)/2$.  Therefore

$$
g_{\rm lens}
=g_{\rm bar}+\left(1+{s\over2}\right)g_\Sigma,
$$

and

$$
\eta\equiv{\Psi\over\Phi}
=1+s{g_{\rm dyn}-g_{\rm bar}\over g_{\rm dyn}}.
$$

When the extra field is screened, $g_\Sigma\rightarrow0$, both matter and light
return to the GR/Newtonian result regardless of $s$.  This is why the formula
can be large in clusters without changing the declared Solar-System numerical
check.  A full PPN derivation from an action has not been supplied, so this is a
phenomenological limit check rather than a completed Cassini proof.

## Frozen testing order

1. Lock the matter law without reading lensing data.  Four smooth
   curvature-running alternatives were fit only to 2,066 inner SPARC points and
   tested on 968 untouched outer points from 131 galaxies.
2. Advance only a galaxy law within 10% of fixed RAR on the outer points.  Only
   fixed RAR advanced: 10.348 km/s outer RMSE.  The curvature candidates scored
   82.809--88.377 km/s.
3. Scan one shared $s$ from -1 to 8 in steps of 0.5.  MACS0329 and MACS0429
   selected $s$; MACS1115 and MACS1931 were untouched cross-cluster validation
   systems.  Each system retained six geometric nuisance parameters, but no
   cluster received its own slip.
4. Require every image root to converge before a grid point can be selected.
   Stress-test the answer on RXJ1347 and RXJ2129 and change the RAR integration
   cutoff from 3,000 kpc to 1,000,000 kpc.

An initial implementation allowed incomplete training roots to enter the
selector.  The run exposed that defect.  Requiring complete roots, as the
frozen scoring rule intends, changes the selected value from the invalid
$s=3$ point to $s=5$.  All reported results below are from the corrected run.

## Main numerical results

| Test | Equal-system radial RMS | Interpretation |
|---|---:|---|
| Unseen clusters, zero slip | 25.673 arcsec | Galaxy matter law alone |
| Unseen clusters, selected $s=5$ | 18.432 arcsec | 28.2% improvement, but still poor |
| Unseen clusters, fixed simple MOND | 25.636 arcsec | Essentially the zero-slip result |
| Unseen clusters, prior $p=2$ curvature law | 18.617 arcsec | Statistically similar to slip |
| Unseen clusters, compact dark halo | 9.989 arcsec | 45.8% lower error than slip |
| Selected slip, far-tail control | 18.527 arcsec | Only 0.51% cutoff change |

The selected model is 1.845 times the compact-halo error, outside the frozen
1.25 ratio gate.  Its absolute 18.432-arcsec error is also far outside the
0.75-arcsec target.

The unseen clusters do not fail equally:

| System | Zero slip | $s=5$ | Compact halo |
|---|---:|---:|---:|
| MACS1115 | 29.568 arcsec | 24.635 arcsec | 14.057 arcsec |
| MACS1931 | 21.070 arcsec | 8.520 arcsec | 1.401 arcsec |

Individual training preferences span $s=4.0$, 5.5, 8.0, and 7.5 across the
four core systems.  A per-cluster normalization could exploit this, but it
would abandon the universal-setting objective.  Leave-one-cluster-out scores
remain poor: 20.415, 14.639, 24.632, and 8.520 arcsec.

The two stress systems also disagree.  RXJ2129 reaches 2.829 arcsec on seven
held-out images, but RXJ1347 loses one of six roots and pushes four geometric
parameters to boundaries.  The scalar slip is not a stable description of
cluster geometry.

## What the radial and matter checks add

On the secondary Tian 20-system radial lensing product, $s=5$ improves the
equal-system error from 0.509 to 0.161 dex and changes the median predicted to
observed ratio from 0.316 to 0.749.  This says that a photon/matter amplitude
split addresses a real part of the cluster-lensing deficit.

It does not solve the raw observable.  A single scalar multiplier changes the
strength of every deflection but cannot place separate galaxy members, gas
clumps, offsets, or asymmetric external structure.  The raw image residuals
therefore remain much larger than the radial-profile result suggests.

The fixed-RAR matter law also predicts only 55.3% of the observed acceleration
at the median of 44 cluster-central BCG points (0.299 dex RMSE).  Since $s$
changes only $\Psi$, it cannot improve those non-photon data.  The theory needs
an environmental matter response as well as a light/matter split.

## What is ruled out and what is not

This run rejects the tested **one-parameter scalar slip plus spherical
pseudo-elliptical baryon model** as a universal galaxy/cluster theory.  It does
not rule out every metric-slip theory.  In particular, the local raw-lens model
does not contain measured member-galaxy and gas maps, and no complete
same-object cluster dynamics-plus-lensing likelihood is currently local.

The result argues against adding another universal amplitude.  The next
minimal equation should let the baryonic environment determine direction as
well as strength.  One testable extension is

$$
g_{{\rm lens},i}
=g_{{\rm dyn},i}+{1\over2}S_{ij}g_{\Sigma,j},
$$

with

$$
S_{ij}=s_0\delta_{ij}
+s_2\left(\widehat T_{ij}-{1\over3}{\rm tr}(\widehat T)\delta_{ij}\right),
$$

where $\widehat T_{ij}$ is a dimensionless, normalized tidal tensor calculated
from the observed baryonic distribution.  $s_0$ controls the photon/matter
amplitude already tested; the one new universal constant $s_2$ redirects the
extra deflection in multi-galaxy and gas-rich environments.  Both must be
shared across every system.  This proposal is worth testing only after member
galaxy and gas maps are included, because otherwise $s_2$ would absorb missing
baryonic structure rather than reveal new gravity.

## Reproducible artifacts

- `configs/metric_slip_galaxy_matter_protocol.json`
- `scripts/run_metric_slip_galaxy_matter.py`
- `results/metric_slip_galaxy_matter/report.json`
- `configs/metric_slip_raw_lensing_protocol.json`
- `src/voidscreen/metric_slip.py`
- `scripts/run_metric_slip_raw_lensing.py`
- `results/metric_slip_raw_lensing/report.json`
- `results/metric_slip_raw_lensing/grid_scores.csv`
- `results/metric_slip_raw_lensing/predictions.csv`
- `results/metric_slip_raw_lensing/metric_slip.png`

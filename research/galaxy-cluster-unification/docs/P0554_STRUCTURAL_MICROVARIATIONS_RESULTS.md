# P0554 structural microvariation results

## Outcome

Seventy-three formulas tested small, parent-preserving changes to how the P0554
terms are combined. Every formula was scored on 131 SPARC galaxies, 20 CLASH
systems, five raw strong-lensing clusters, and Solar proxies, with no gravity
or lens-geometry fit.

The largest structural controls are the addition laws: how the new dynamical
channel joins baryonic gravity and how its lensing channel joins baryonic GR.
They are substantially more influential than changing the potential/path
cross-term or moving baryonic extent into the transition radius. They do not
provide a universal direction:

- the dynamics addition shape prefers one sign for galaxies and the opposite
  sign for CLASH; and
- the lensing addition shape prefers opposite signs in RX J2129 and the other
  four raw clusters.

The only material structural deformation with the same fixed-geometry
direction in multiple non-Solar domains is a slightly softer acceleration
screen. It is a useful operator to retain for exact-refit testing, not a
selected formula.

## Parent-preserving operators

The generalized addition operator is

$$
\mathcal A_k(z)=
\left[1+(2^k-1)z^k\right]^{1/k}.
$$

It satisfies $\mathcal A_1(z)=1+z$ and $\mathcal A_k(1)=2$. It was inserted
separately into dynamics and lensing:

$$
\frac{g_{\rm dyn}}{g_b}=\mathcal A_{k_d}(qF),
\qquad
\frac{g_{\rm lens}}{g_b}
=\mathcal A_{k_\gamma}
\left[m_\gamma\left(\frac{g_{\rm dyn}}{g_b}-1\right)\right].
$$

The midpoint-preserving screen deformation is

$$
S_k(y)=
\left[1+(2^k-1)y^{nk}\right]^{-1/k},
\qquad y=\frac{g_b}{a_0s}.
$$

It retains $S(1)=1/2$, the low-field limit, and the high-field $y^{-n}$ power.
The residence law was changed from a harmonic soft minimum to a generalized
soft minimum. Potential depth received the analogous generalized-addition
shape.

The original potential/path product was written as

$$
I=P+R-1+k_\times(P-1)(R-1),
$$

where $k_\times=1$ is exactly $PR$. Three additional operators tested whether
baryonic extent or potential depth should move the accumulation radius, and
whether the enclosed-mass slope should contribute directly.

All value-one and value-zero parent settings execute the original code paths.
The reproduced parent scores are exactly 12.570912 km/s for SPARC, 0.199077
dex for CLASH, -1.729829 mas/century for Mercury, and 17/18 raw roots.

## Impact ranking

| Domain | Strongest stable structural lever | Median normalized slope | Better direction |
|---|---|---:|---|
| SPARC | dynamics addition softness $k_d$ | 0.0987 | increase |
| CLASH | lensing addition softness $k_\gamma$ | 0.1334 | decrease |
| RX J2129 | screen softness $k_s$ | 0.5897 | decrease |
| other four raw clusters | lensing addition softness $k_\gamma$ | 0.0879 | increase |
| Mercury | dynamics addition softness $k_d$ | 5.3102 | increase |

The large raw slopes for some other structures are based on only one or two
complete central pairs and therefore are not ranked as stable.

## Dynamics addition law

Increasing $k_d$ suppresses a small extra channel more strongly while keeping
the midpoint fixed. It improves SPARC near the parent and strongly suppresses
the Solar tail, but reduces the cluster enhancement that CLASH asks for.

| $k_d$ | SPARC RMSE | CLASH RMSE | Mercury |
|---:|---:|---:|---:|
| 0.95 | 12.966 km/s | 0.1928 dex | -6.141 mas/century, fails |
| 1.00 | 12.571 km/s | 0.1991 dex | -1.730 mas/century |
| 1.05 | 12.349 km/s | 0.2058 dex | -0.486 mas/century |
| 1.10 | 12.290 km/s | 0.2126 dex | -0.136 mas/century |

This is a clean structural tradeoff. A smaller $k_d$ supplies more cluster
field but violates Mercury by $k_d=0.95$; a larger value improves galaxies and
Solar safety but worsens cluster amplitude. The effect is not a candidate for
a common scalar solution.

## Lensing addition law

Changing $k_\gamma$ leaves galaxy dynamics and Mercury unchanged. Lower values
improve the derived CLASH score: $k_\gamma=0.9$ gives 0.1876 dex and 0.8 gives
0.1838 dex. But raw clusters disagree on direction. RX J2129 favors decreasing
$k_\gamma$, while the other four clusters stably favor increasing it. A tiny
decrease from 1.00 to 0.98 also changes total image multiplicity from 17 to 18.

This is further evidence against treating light-only amplification as a
universal scalar correction. It moves caustics differently in different
cluster geometries.

## Screen shape

Screen softness is the only structural coordinate with a material common
direction across multiple non-Solar domains. Decreasing it from 1.00 to 0.98
slightly improves SPARC (12.571 to 12.562 km/s), CLASH (0.19908 to 0.19865 dex),
and the finite RX J2129 comparison while remaining Solar safe. A larger
decrease to 0.9 recovers all 18 fixed-geometry roots and remains inside the
analytic Mercury margin at -2.030 mas/century.

The four-cluster RMS direction is inconsistent across perturbation sizes, so
this is not yet universal. It is the clearest structural candidate to test
with ordinary geometry refits.

## Other structural lessons

- **Residence saturation shape matters mainly for galaxies.** Increasing its
  softness improves SPARC to 12.407 km/s at 1.2 but worsens CLASH to 0.2021
  dex. Its Solar effect is negligible.
- **Potential-to-radius coupling is a strong CLASH/RX lever.** Increasing it
  improves galaxies, CLASH, and the smallest complete RX comparisons, but the
  other four clusters prefer the opposite direction and roots quickly become
  incomplete.
- **Potential addition shape affects CLASH but destabilizes RX roots.** It is
  another amplitude/topology tradeoff rather than a common direction.
- **The potential/path cross-term is nearly irrelevant to scalar scores.** Its
  median normalized slopes are only 0.00018 for SPARC and 0.00191 for CLASH,
  yet it still changes a raw root at a larger step. Raw sensitivity here is a
  caustic effect, not evidence for the cross-term.
- **Moving extent into the scale radius is weaker than amplitude leakage.** It
  has only 0.0079 and 0.0112 normalized slopes for SPARC and CLASH. It changes
  topology only after half of the declared move.
- **Enclosed-mass growth is a moderate cluster lever.** It has a stable 0.0438
  CLASH slope but inconsistent raw directions and no Solar effect.

## Topology result

The 73 formulas span 14--18 of 18 held-out image roots around the 17-root
parent. Every one of the nine structures changes root count somewhere in its
declared interval. Four do so at the smallest step:

- dynamics addition softness by 2%;
- lensing addition softness by 2%;
- potential-to-radius coupling by 0.01; and
- potential-addition softness by 2%.

This reinforces the multi-scale coefficient result: a useful field law must
predict two-dimensional caustic topology, not only radial acceleration. Root
recovery alone is not an accuracy result.

## Universal conclusions from this stage

1. The algebra used to add a new channel is more consequential than the old
   apogee or potential/path cross-term.
2. A nonlinear dynamics addition law creates a hard galaxy/cluster/Solar
   tradeoff near P0554.
3. A nonlinear lens-only addition law creates a cluster-to-cluster sign
   conflict.
4. Screen shape is the only tested structure with a partially shared direction
   and should be tested under exact geometry refits.
5. Potential and profile-shape structures affect radial cluster amplitude, but
   their raw-lensing effects are dominated by caustic bifurcations.
6. No tested algebraic deformation supplies a universal improvement.

## Limits

These operators are phenomenological and are not derived from a covariant
action. All systems are spent exploratory data, CLASH targets are derived from
conventional GR/NFW profiles, raw ordinary geometries are fixed, and the Solar
checks are analytic proxies. The test ranks structural influence; it does not
identify a new physical law.

## Reproduction

```powershell
python scripts/run_p0554_structural_microvariations.py
python -m pytest tests/test_arc_invariants.py tests/test_p0554_structural_microvariations.py -q
```

Machine-readable outputs are in `results/p0554_structural_microvariations/`.

## Exact-refit continuation

Eight structural controls were subsequently replayed with six ordinary lens-
geometry nuisances refit in every cluster. All fixed-geometry root recoveries
disappeared: every formula returned to 17/18 roots. Lower lensing-addition
softness retained the broadest continuous improvement on complete lensing
systems, while the dynamics-addition galaxy/cluster tradeoff remained. See
[`P0554_STRUCTURAL_EXACT_REFIT_RESULTS.md`](P0554_STRUCTURAL_EXACT_REFIT_RESULTS.md).

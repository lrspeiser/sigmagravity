# Sigma v9B local first-gradient mechanism closure

## Verdict

The regular, static, shift-symmetric local completion

$$
\Delta\mathcal L=F(Y,Z,U)
$$

is **closed as the galaxy--cluster unification lane** under the assumptions
listed below.  This includes rational variants of the v9A Gram term, angle
interpolations such as $YZ-\chi U^2$, and arbitrary smooth changes of their
coefficients.  In spherical symmetry they all reduce to a universal local
acceleration relation.

The already-spent development products require different enhancements at
essentially identical baryonic accelerations:

- all 72 cluster points lie inside the 968-point SPARC outer-$g_{\rm bar}$
  range;
- the median nearest cross-domain separation is `0.001448 dex` in
  $g_{\rm bar}$;
- the median required cluster-minus-galaxy enhancement is `0.50934 dex`, a
  factor `3.231`;
- all 72 gaps are positive and 70/72 exceed `0.2 dex`.

No coefficient or angular function inside $F(Y,Z,U)$ can distinguish states
that the theory itself makes identical.  A different branch or integration
charge per object would be a hidden halo state, not a universal law.

This is not a raw-lensing falsification.  The cluster accelerations are the
existing NFW-deprojected CLASH development target, and the point match ignores
its covariance.  The exact result is the conditional spherical theorem; the
data audit shows that the current development target violates its consequence.

## Scope and assumptions

Let

$$
S_\mu=q_\mu{}^\nu\nabla_\nu\phi,
\qquad
J_\mu=A^\nu\nabla_\nu A_\mu,
$$

$$
Y=S_\mu S^\mu,
\qquad
Z=J_\mu J^\mu,
\qquad
U=S_\mu J^\mu.
$$

The closure applies when all of the following are true:

1. the configuration is static and spherical;
2. the extra quasistatic action is a local, shift-symmetric function of only
   $Y$, $Z$, and $U$;
3. the constitutive map is regular and single-valued on the physical branch;
4. universal boundary conditions forbid an object-specific scalar/vector
   charge;
5. matter and photons use one physical metric.

It does **not** include explicit potential values, density, pressure, curvature,
Hessians, matter kinematics, finite memory, nonlocal kernels, or additional
carrier fields.  Those are possible escape directions, not loopholes inside
the theorem.

## Spherical flux theorem

On a static spherical branch,

$$
S_i=S(r)\hat r_i,
\qquad
J_i=J(r)\hat r_i,
$$

so

$$
Y=S^2,
\qquad
Z=J^2,
\qquad
U=SJ.
$$

The Euler equations for the shift-symmetric potentials are divergences of
constitutive fluxes.  Schematically,

$$
\nabla_i P_a^i=c_a\,4\pi G\rho_b,
$$

where $a$ labels the coupled scalar/metric-aether equations and $c_a$ are fixed
action coefficients.  Spherical integration gives

$$
4\pi r^2P_a^r
=c_a\,4\pi G M_b(<r),
$$

hence

$$
\boxed{
P_a(J,S)=c_a{G M_b(<r)\over r^2}=c_a g_{\rm bar}.
}
$$

If the joint constitutive map $P_a(J,S)$ has a unique regular inverse, then

$$
(J,S)=\mathcal C(g_{\rm bar}).
$$

The physical acceleration, which is a fixed combination of the one-metric
weak potentials, must therefore satisfy

$$
\boxed{
{g_{\rm phys}\over g_{\rm bar}}=E(g_{\rm bar}).
}
$$

This is a universal RAR.  Changing $F$ changes the curve $E$, but it cannot
give two different answers at the same $g_{\rm bar}$.

The conclusion is familiar in AQUAL-like spherical reductions; the point here
is that adding a second local first-gradient magnitude and its angle does not
escape it.  The one-metric AeST base is prior art from
[Skordis and Zlosnik](https://arxiv.org/abs/2007.00082), and the observed galaxy
RAR is documented by
[McGaugh, Lelli, and Schombert](https://arxiv.org/abs/1609.05917).  No novelty
claim is made for the spherical integration itself.

## Equal-force systems expose the missing variable

Consider two spherical evaluation points:

$$
(M_1,r_1)=(1,1),
\qquad
(M_2,r_2)=(100,10).
$$

Their surface fields are identical:

$$
{M_2/r_2^2\over M_1/r_1^2}=1.
$$

Every regular $F(Y,Z,U)$ theory therefore assigns the same local constitutive
state to both.  But two natural environment variables differ:

$$
{M_2/r_2\over M_1/r_1}=10
$$

for potential depth, and

$$
{M_2/r_2^3\over M_1/r_1^3}=0.1
$$

for tidal curvature or mean density.  This is precisely why a potential,
Hessian, density, or finite-environment state can distinguish systems that a
first-gradient law cannot.

## Spent-development data audit

The audit uses no new observational product:

- SPARC: the 968 already-held-out outer points from the completed independent
  nuisance-refit report, using only the fixed-RAR/invariant rows;
- clusters: the 72 existing `domain=cluster` points from the completed
  phenomenology sweep.

For SPARC,

$$
g_{\rm obs}={v_{\rm obs}^2\over r}
$$

is calculated from the saved nuisance-adjusted velocity and radius.  Required
enhancement is

$$
\Delta=\log_{10}{g_{\rm obs}\over g_{\rm bar}}.
$$

Each cluster point is compared with the nearest SPARC $g_{\rm bar}$ and with
the median enhancement of its ten nearest SPARC neighbors.

### Acceleration overlap

| Quantity | SPARC outer points | Cluster points |
|---|---:|---:|
| Count | 968 | 72 |
| Minimum $\log_{10}g_{\rm bar}$ | -12.0684 | -10.8760 |
| Median | -11.0320 | -10.4375 |
| Maximum | -9.5133 | -9.6680 |

The entire cluster interval `[-10.876, -9.668]` is contained inside the SPARC
outer interval.  Nearest-match distances are:

| Cross-domain $|\Delta\log_{10}g_{\rm bar}|$ | Value |
|---|---:|
| Median | 0.001448 dex |
| 95th percentile | 0.014296 dex |
| Maximum | 0.024823 dex |

### Required enhancement conflict

| Cluster minus nearest-SPARC enhancement | Value |
|---|---:|
| Minimum | 0.11069 dex |
| Median | 0.50934 dex |
| 95th percentile | 0.67658 dex |
| Maximum | 0.85843 dex |
| Median factor | 3.231 |
| Positive fraction | 72/72 |
| Above 0.2 dex | 70/72 |

Using the median of ten SPARC neighbors gives essentially the same result:
the median gap is `0.50650 dex`, a factor `3.210`.  This is not an isolated
nearest-neighbor accident.

The declared closure gate required at least 95% range inclusion, median
nearest separation at most 0.01 dex, median ten-neighbor gap at least 0.3 dex,
and at least 90% of nearest gaps above 0.2 dex.  All four conditions pass.

## Why branch multiplicity is not a rescue

A multivalued constitutive law could assign one branch to a disk and another
to a cluster at the same $g_{\rm bar}$.  But the action contains no disk or
cluster label.  The theory must derive the branch from universal initial and
boundary conditions.

If each object is allowed an independent integration charge, history, or
boundary value, that state is operationally a halo profile: it carries the
missing information without being predicted by the baryons.  It violates the
goal's zero-object-specific-gravity-parameter and source-uniqueness gates.

A universal causal memory law may legitimately select different states, but
then the memory field and its boundary prescription are part of a new nonlocal
carrier theory.  They are outside $F(Y,Z,U)$ and must pass their own health and
uniqueness audits.

## Connection to earlier empirical results

This closure explains several previous outcomes:

- RAR/MOND is excellent across disk galaxies because a universal function of
  $g_{\rm bar}$ is approximately what their data require.
- Fixed galaxy-scale MOND misses cluster amplitude because cluster points at
  the same $g_{\rm bar}$ require roughly three times more enhancement.
- Local acceleration gates damaged galaxies when made strong enough for
  clusters.
- Potential depth can separate much of the amplitude, because it is not fixed
  by the local surface force; nevertheless, the P0599 potential multiplier
  failed raw image positions because it was nearly a radial rescaling.
- The empirical density/coherence bridge works numerically because it inserts
  variables explicitly excluded from this local first-gradient theorem.

The lesson is not to add a better angular polynomial.  The missing state must
encode finite environment and two-dimensional orientation.

## Decision and successor requirement

Close further variants of

$$
F(Y,Z,U),
\qquad
F(Y,Z,YZ-\chi U^2),
$$

as standalone unification mechanisms.  Do not fit another coefficient or
transition exponent in this lane.

The successor must contain a uniquely baryon-forced variable that:

1. is nonzero for spherical systems and distinguishes equal-$g_{\rm bar}$
   spheres through density, curvature, potential, or finite environment;
2. also has a traceless/tensor response capable of predicting shear
   orientation and image topology;
3. has bounded first-order dynamics or an exact arbitrary-background
   degeneracy identity;
4. carries no freely assigned object charge or halo-like initial profile;
5. preserves one metric, at most five universal constants, Solar limits, and
   luminal tensor propagation.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v9b_local_state_closure.py
python -m pytest -q tests/test_sigma_v9b_local_state_closure.py
```

The protocol is retrospective by construction and is stored at
[`../configs/sigma_v9b_local_state_closure.json`](../configs/sigma_v9b_local_state_closure.json).
The machine-readable report, including the exact input hashes, is
[`../results/sigma_v9b_local_state_closure/report.json`](../results/sigma_v9b_local_state_closure/report.json).

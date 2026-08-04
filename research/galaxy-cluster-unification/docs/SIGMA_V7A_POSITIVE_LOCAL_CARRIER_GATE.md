# Sigma v7A positive local carrier gate

## Decision

The unscreened Sigma v7A carrier is retired before observational data.  Its
positive spin-2 spectrum is mathematically healthier than the multiplier-
localized v6D memory, but the same positivity makes its long-range residue
non-negative.  A carrier long-ranged enough to affect a galaxy or cluster is
therefore unsuppressed in the Solar System.  The Solar high-field gate limits
its residue to `7.5e-6`, giving less than `0.00075%` extra lensing, while a useful
cluster channel was conservatively required to provide at least `50%`.

The positive spin-2 carrier remains eligible only with a derived nonlinear
screening mechanism.  The next candidate is the Vainshtein-screened limit of a
ghost-free Hassan--Rosen/dRGT completion, not a new derivative interaction
between metrics.

## Why this is a materially new construction

The failed v6D action generated a retarded tensor response with a response field
and a Lagrange multiplier.  Each localized pair produced one positive and one
negative kinetic eigenvalue.

V7A instead makes the orientation carrier an ordinary propagating massive
spin-2 field.  Around a proportional flat background, write the quadratic
mass-eigenstate action schematically as

$$
S_2={1\over2}H\mathcal EH+{1\over2}M\mathcal EM
-{m_\Sigma^2\over4}(M_{\mu\nu}M^{\mu\nu}-M^2)
+{1\over M_{\rm Pl}}(H_{\mu\nu}+\alpha_\Sigma M_{\mu\nu})T_b^{\mu\nu}.
$$

The Fierz--Pauli tuning supplies the constraint that removes the sixth massive
polarization.  The spectrum has

$$
2\text{ massless spin-2 modes}+5\text{ massive spin-2 modes},
$$

all with positive quadratic kinetic sign.  This is the linear spectrum of
ghost-free bimetric gravity, whose nonlinear completion uses two Einstein--
Hilbert kinetic terms and the special Hassan--Rosen/dRGT potential
([Hassan & Rosen](https://arxiv.org/abs/1109.3515)).

Matter still couples to one physical metric.  The second spin-2 state is a
gravitational carrier, not a supplied matter-density map; its solution must be
fixed by baryons and universal boundary conditions.

## Static response fixed by spin two

Let

$$
a=\alpha_\Sigma^2\ge0,
\qquad x={r\over L_\Sigma},
\qquad K_Y(x)=(1+x)e^{-x}.
$$

Exchange between conserved sources gives the point-source force factors

$$
E_\Psi=1+{4a\over3}K_Y(x),
$$

$$
E_\Phi=1+{2a\over3}K_Y(x),
$$

and consequently

$$
E_{\rm lens}={E_\Psi+E_\Phi\over2}=1+aK_Y(x).
$$

The `4/3`, `2/3`, and `1` coefficients are not independent galaxy, lensing, or
cluster parameters.  They are the helicity structure of a massive spin-2
exchange.  In the short-range unsuppressed limit,

$$
\gamma={\Phi\over\Psi}
={1+2a/3\over1+4a/3}.
$$

This is useful conceptually: the carrier genuinely predicts different massive-
tracer and photon responses, but it does so with a fixed ratio rather than a
lensing multiplier.

## The two pre-data obstructions

### 1. Solar consistency removes useful amplitude

The exact Cassini bound gives

$$
a\le {3\,|\gamma-1|_{\max}\over2-4|\gamma-1|_{\max}}
=3.45016\times10^{-5}.
$$

The project's stricter unscreened high-acceleration force gate gives

$$
{4a\over3}\le10^{-5}
\quad\Longrightarrow\quad
a\le7.5\times10^{-6}.
$$

Even at the maximum of the Yukawa kernel, the largest permitted lensing factor
is therefore

$$
\boxed{E_{\rm lens}<1.0000075.}
$$

A carrier range of kiloparsecs or more has `r/L << 1` in the Solar System, so
changing the range cannot evade this result.  Making the range shorter than the
Solar System instead removes the mode from galaxies and clusters.

### 2. A positive Yukawa mode turns off, not on

The force kernel obeys

$$
{dK_Y\over dx}=-xe^{-x}\le0.
$$

After calibrating Newton's constant in the inner unsuppressed regime,

$$
{g(r)\over g_{\rm local}}
={1+(4a/3)K_Y(r/L_\Sigma)\over1+4a/3}
$$

can only decrease with distance.  Positive spectral weight cannot reveal a
larger gravitational strength outside the calibration region.  A negative
residue could reverse the behavior, but it is precisely a negative-norm pole,
undoing the reason v7A was selected.

## Prior-art boundary

This is not a claim to have invented bimetric or massive gravity.

- Ghost-free nonlinear massless-plus-massive spin-2 interactions are the
  Hassan--Rosen/dRGT class.
- A two-metric MOND interaction based on connection differences is BIMOND
  ([Milgrom](https://arxiv.org/abs/0912.0790)).
- Generic BIMOND derivative mixing does not share the ghost-free bimetric
  constraint.  Its extra Boulware--Deser mode and the exceptional constrained
  nonmetricity limit are discussed explicitly by
  [D'Ambrosio, Garg & Heisenberg](https://arxiv.org/abs/2004.00888).
- More generally, a nonlinear Lorentz-invariant derivative interaction between
  spin-2 fields is not an available repair: the two-derivative no-go result
  leaves the Einstein--Hilbert kinetic term as the healthy option
  ([de Rham, Matas & Tolley](https://arxiv.org/abs/1311.6485)).

The project-specific question is narrower: can a healthy, universally screened
spin-2 carrier add baryon-predicted shear topology to the already-established
low-acceleration trace response without becoming a fitted halo?

## What advances to v7B

The linear carrier fails, but its failure identifies the only admissible escape:

1. keep the positive massless-plus-massive spin-2 spectrum;
2. use the existing ghost-free potential to derive nonlinear Vainshtein
   screening around high-acceleration sources;
3. require a single range and mixing to screen the Solar System while creating
   a useful window in dwarfs, disks, and clusters;
4. test the mass and size scaling analytically before any map or rotation-curve
   fit; and
5. reject the carrier if one universal window cannot separate Solar, galaxy,
   and cluster regimes or if it merely adds a constant force rescaling.

No observational array or untouched holdout was opened for v7A.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v7a_positive_carrier.py
python -m pytest -q tests/test_sigma_v7_positive_carrier.py
```

Machine-readable evidence is stored in
`results/sigma_v7a_positive_carrier_gate/report.json`.

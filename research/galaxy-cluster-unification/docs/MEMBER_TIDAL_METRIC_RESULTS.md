# Member-tidal metric test

## Outcome

The member-only tensor branch fails its frozen advancement test. The qualifying
grid selects `t = 0`, so the directional member term switches itself off. On
the two validation clusters the resulting held-out image-position RMS is
18.432 arcsec, essentially identical to the locked scalar metric-slip result
(18.432 arcsec) and 1.845 times the compact-halo comparator (9.989 arcsec).

This is a useful failure rather than a numerical null result. The constructed
member tensors are nonzero, their unit-coupling deflection corrections have
order-one-arcsecond RMS amplitudes, and the optimizer responds to them. The
problem is transfer and image-root stability: no nonzero primary coupling gives
a better complete exact-root score on the selection systems than `t = 0`.

## Formula tested

The starting point remains the fixed-RAR matter law and its locked scalar
metric slip:

\[
 \Phi=\Phi_N+\phi_\Sigma,\qquad
 \Psi=\Phi_N+(1+s)\phi_\Sigma,\qquad s=5.
\]

Here `Phi_N` is the ordinary baryonic Newtonian potential and `phi_Sigma` is
the extra potential required by fixed RAR. Matter responds to `Phi`, while
light responds to the sum of the time and spatial metric potentials. The
locked value `s=5` makes the extra part of the lensing field 3.5 times its
dynamical value.

The new test asks whether the non-circular environment can change how the
extra potential propagates:

\[
 \partial_i\!\left[(\delta^{ij}+tQ_{\rm env}^{ij})
 \partial_j\phi_\Sigma\right]=\text{source}_\Sigma .
\]

In plain language:

- `delta` is ordinary, direction-independent propagation.
- `Q_env` is a direction map made from the observed positions and relative
  light of cluster-member galaxies.
- `t` is one universal number controlling how strongly that direction map
  changes the extra field.
- The circular average is removed from `Q_env`, so a round environment gives
  zero and does not change the already successful galaxy radial law.

The calculation is performed to first order in `t`:

\[
 \nabla^2\delta\phi=-\partial_i
 (Q_{\rm env}^{ij}\partial_j\phi_\Sigma).
\]

Solving this equation for a scalar potential is important: it guarantees that
the predicted lens deflection remains a gradient. Directly rotating gravity
vectors with a position-dependent matrix could create a curl that no static
scalar lens potential can produce.

## Data and split

The map uses the Caminha et al. CLASH member catalogs, containing 63--120
members per system. Positions and relative photometric light weights are used;
no member mass normalization is fitted. The smoothing is fixed at 10 kpc, with
5 and 20 kpc checks.

- Coupling selection: MACS0329 and MACS0429.
- Cross-cluster validation: MACS1115 and MACS1931.
- Within each source family with at least three images, the last image is held
  out.
- Cluster lens geometry is nuisance-fitted, but there are no per-cluster
  gravity parameters.

These image catalogs were used in earlier project work. The frozen formula and
split are prospective relative to this tensor test, but the systems are not
untouched-project validation.

## Numerical checks

The tensor has eigenvalue magnitude no greater than one by construction. For
the primary maps:

- maximum boundary tensor magnitude: 0.0514, below the frozen 0.10 limit;
- maximum normalized curl RMS: 3.35e-17, below the 0.001 limit;
- 5--20 kpc smoothing changes the selected `t=0` validation result by only
  0.0027%, which is trivially expected once the tensor switches off.

Thus the primary failure is not caused by an FFT boundary artifact or a
nonphysical curl.

## Exact scores

| Test | Held-out validation RMS | Complete roots? | Status |
|---|---:|---:|---|
| Frozen selected tensor, `t=0` | 18.432 arcsec | yes | primary failure |
| Scalar metric slip, independent prior run | 18.432 arcsec | yes | same result |
| Compact cluster halo | 9.989 arcsec | yes | comparator wins |
| Post-result `t=-0.6` | 18.015 arcsec | yes | 2.3% improvement; nonqualifying |
| Post-result `t=+0.9` | not comparable | no | one MACS1931 image root lost |
| Post-result `t=+1.2` | not comparable | no | one MACS1931 image root lost |

The post-result settings were deliberately run only to diagnose the failure.
They cannot replace the frozen result. The moderate negative coupling produces
a real but small improvement and remains 80% worse in RMS than the compact
halo. Stronger positive anisotropy destabilizes the actual multiple-image
solutions.

## What this teaches us

1. Member layout is not the missing universal cluster variable by itself.
   Its preferred directional response differs across systems.
2. A lower local optimizer cost is not enough. Some large couplings improve a
   linearized residual while destroying an exact lens-equation root.
3. The fixed-RAR galaxy result is preserved, but only because the selected
   tensor strength is zero; this branch does not create the needed cluster
   improvement.
4. This does not yet test the most physically plausible full matter tensor.
   The smooth and asymmetric hot-gas distribution is absent, even though gas
   is a major measured baryonic component in clusters.

## Next gate

The immediate next step is not another coupling scan. RX J2129 now has a fully
passing XMM response package, but it still lacks the frozen X5 joint spectral
posterior. X5 must infer temperature and electron density without lensing,
hydrostatic-equilibrium, dark-matter, or new-gravity priors. Only after that
posterior passes can the projected gas surface-density map be combined with the
BCG/member map to build a genuinely baryonic environmental tensor.

That gas-inclusive map should then be frozen and scored on raw image positions
and the same-object stellar dynamics. Until then, this project should retire
the **member-only** tensor, not the broader gas-inclusive tensor concept.

## Reproduce

```powershell
python -m pytest tests/test_tidal_metric.py -q
python scripts/run_member_tidal_metric.py
python scripts/run_member_tidal_nonzero_diagnostic.py
```

Machine-readable outputs are in `results/member_tidal_metric/`.

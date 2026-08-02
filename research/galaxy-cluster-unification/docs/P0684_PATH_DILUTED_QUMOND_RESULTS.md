# P0684 path-diluted potential-channel QUMOND

Frozen before scores: 2026-08-02  
Verdict: numerical integrity **pass**; primary **does not advance**; two diagnostic rows generate a new frozen topology candidate

## Formula

P0684 keeps the single QUMOND-style source equation from P0683 and adds one
dimensionless geometric quantity derived only from the baryonic field:

\[
\nabla^2\Phi=\nabla\!\cdot\left[
\nu_0\!\left({|\nabla\Phi_N|\over a_0}\right)^{p(\chi_b,\eta_b)}
\nabla\Phi_N\right],
\]

\[
\eta_b={|\Phi_b|\over r|\nabla\Phi_N|},\qquad
S_n={\chi_b^n\over\chi_b^n+\chi_t^n},
\]

\[
p=1+N_{\rm extra}S_n\,[\max(\eta_b,1)]^{-q}.
\]

`eta_b` is the number of local radial-acceleration lengths represented by the
baryonic potential. The primary interpretation treats coherent amplitude as
an inverse-square-root survival process:

\[
N_{\rm extra}=3,\quad n=2,\quad q={1\over2}.
\]

Only `chi_t` is a fitted universal setting. No galaxy or cluster has its own
gravity parameter.

## Frozen coverage

- 1,155 formula rows;
- 131 spent galaxies and 968 outer rotation points;
- six spent cluster radial-deflection targets, with five primary and three
  reliability-qualified systems;
- fixed Solar force proxies;
- no raw image-root, topology, 3D field, or sealed-target score.

## Primary outcome

The primary again selects `chi_t=3e-6`.

| Metric | P0683 | P0684 primary | Frozen gate | Result |
|---|---:|---:|---:|---|
| Galaxy RMSE / fixed RAR | 1.035 | **1.034** | `<=1.05` | pass |
| All-five cluster log RMS | 0.285 | **0.240 dex** | `<=0.200` | fail |
| Reliable-three log RMS | 0.309 | **0.187 dex** | `<=0.200` | pass |
| Fixed-RAR cluster gap closed | 51.2% | **58.8%** | `>=75%` | fail |
| Solar force proxies | pass | pass | frozen limits | pass |

The geometric term therefore changes the response in the intended direction
and repairs the reliable-three score. The dimension-fixed primary still
underpredicts MACS0329, MACS0429, and MACS1115 enough to fail the all-five and
gap-closure rules. It cannot advance.

## Predeclared diagnostic result

Two diagnostic rows satisfy every numerical galaxy, cluster, gap, and Solar
threshold:

| Extra channels | onset power | path power | `chi_t` | galaxy/RAR | all-five dex | reliable-three dex | gap closed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 0.50 | `1e-6` | 1.030 | 0.145 | 0.165 | 75.09% |
| 2 | 4 | 0.75 | `1e-6` | 1.029 | 0.145 | 0.156 | 75.21% |

They were explicitly diagnostic and cannot be promoted as P0684 results.
They can generate a new equation for a new observable test.

The inverse-square-root row is chosen as the development formula generator,
despite the slightly lower score of `q=0.75`, because `q=1/2` is the frozen
coherent-amplitude rule and adds no empirically selected fractional power. Its
two extra channels also have a concrete spatial interpretation: one radial
local channel plus two transverse spatial channels. The quartic onset remains
a phenomenological sharp transition and must not be described as derived.

## Next test

The exact generated equation is

\[
p=1+2\,{\chi_b^4\over\chi_b^4+(10^{-6})^4}
{1\over\sqrt{\max(\eta_b,1)}}.
\]

It has no adjustable value left for the spent topology run. The correct next
step is not another radial refit. It is to freeze this equation and implement
it in the registered 3D QUMOND solver, then require on spent RX J2129:

1. solver residual and grid-convergence gates;
2. all source families recover image multiplicity and both parities;
3. critical curves exist in the observed strong-lens region;
4. heldout image RMS is below `3 arcsec`;
5. results retain sign and root count on three grids and fixed stellar/gas
   mass sensitivities; and
6. no per-object gravity or photon amplitude is introduced.

Only a topology and robustness survivor may open P0633/P0640 once.

## Reproduction

```powershell
python scripts/run_p0684_path_diluted_potential_channel_qumond.py
python -m pytest tests/test_potential_channel_qumond.py -q
```

Artifacts are in `results/p0684_path_diluted_potential_channel_qumond/`.


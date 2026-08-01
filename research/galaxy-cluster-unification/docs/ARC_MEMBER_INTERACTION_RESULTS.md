# Arc/member interaction: a directional field test

## Question

Can the observed locations of cluster galaxies redirect a fixed baryon-sourced
gravity field toward strong-lensing constraints without adding radial mass?
This is the most direct test so far of the idea that some of the field assigned
to a smooth dark halo is baryonic gravity returning along curved paths.

The test uses the two strongest potential/path scalar parents from the previous
stage:

- `P0554`, the best galaxy result below 0.2 dex CLASH error; and
- `P0396`, the strongest zero-slip CLASH result.

## Controlled equation

The thin-lens deflection is

\[
\boldsymbol\alpha = \boldsymbol\alpha_{\rm parent}
+ f_{\rm route}\left(
\boldsymbol\alpha_{\rm resolved\ members}
-\left\langle\boldsymbol\alpha_{\rm same\ members}\right\rangle_\theta
\right).
\]

The subtracted term spreads every member around a circular ring while retaining
its mass, cluster-centric radius, and softening. Therefore the bracket changes
direction but has zero azimuthally averaged radial mass. The total effective
member stellar mass stays fixed at `2.0674e11 Msun`.

The frozen 722-law screen varied:

| Variable | Values | Physical question |
|---|---:|---|
| routing fraction | -1 to 2 | Does the field avoid or concentrate near the measured member directions? |
| member mass power | 0.5 to 1.5 | Do many small galaxies or a few massive galaxies dominate routing? |
| softening scale | 0.5, 1, 2 | Is routing compact or diffuse around each galaxy? |
| radial dressing | none, dynamical, photon | Does routing strength follow either scalar-parent enhancement? |

Selection used only 15 RX J2129 training images. Seven images remained withheld
until the formula and nuisance fits were selected. Both parents received the
same optimizer effort.

## Results

Both parents independently selected the same training-only rule:

\[
f_{\rm route}=2,\qquad \eta_{\rm mass}=0.5,\qquad
s_{\rm soft}=0.5,\qquad D(r)=1.
\]

In ordinary language, the training images preferred a strong, compact return
field shared more evenly across low- and high-mass member galaxies. It did not
prefer either parent-based radial dressing.

| Parent | Variant | Training RMS | Held-out RMS | Held-out roots | Outcome |
|---|---|---:|---:|---:|---|
| P0554 | scalar baseline | 0.729 arcsec | 1.256 arcsec | 7/7 | reference |
| P0554 | selected routing | 0.568 arcsec | 1.307 arcsec | 7/7 | 4.0% worse |
| P0396 | scalar baseline | 0.493 arcsec | 1.306 arcsec | 7/7 | reference |
| P0396 | selected routing | 0.290 arcsec | undefined | 6/7 | failed root |

The apparent training gain does not transfer. For P0554, 47.3% of 128
radius-preserving random-angle layouts were at least as good as the measured
layout at fixed geometry. With one-start geometry refits, the empirical value
was 23.5%. Neither passes the predeclared 5% specificity gate.

The failure is not uniform: routing improves four of the seven P0554 held-out
images. One image in source family 5 worsens by 0.538 arcsec and dominates the
net loss. A single local member kernel therefore has the wrong global coupling
even though some individual directions are useful.

The marginal training-screen impact ranking was:

1. routing fraction: 0.231 arcsec span;
2. scalar parent: 0.226 arcsec;
3. softening: 0.032 arcsec;
4. member mass power: 0.016 arcsec; and
5. radial dressing: 0.008 arcsec.

## What this teaches us

The member directions can absorb training residuals, but direct local
`resolved minus circularized` routing is not a predictive field law. The
failure is not caused by the galaxy or Solar-System scalar response: this
operator leaves both exactly unchanged. It is specifically a failure to
predict the withheld two-dimensional image geometry.

The strongest new information is that radial scalar dressing contributes very
little. If field arcs are real, their endpoints cannot be inferred merely by
multiplying each visible member by the local dynamical or photon enhancement.
The next justified model should infer a smooth, divergence- and curl-controlled
route field from baryon origins to independently reconstructed lensing-potential
features, and then lock that operator across several clusters. It should also
separate BCG/ICL, gas, and satellite origins and include projected-depth
uncertainty. More tuning of the present member overlay is not justified.

## Scope

RX J2129 is a spent exploratory holdout after earlier project development. The
member catalog is a stellar-baryon tracer, not a direct dark-matter map, and the
equation above is a thin-lens phenomenology rather than a covariant field
equation. This negative result rejects this local directional implementation;
it does not reject all nonlocal gravity-path models.

Reproduce with:

```powershell
python scripts/run_arc_member_interaction.py
python -m pytest tests/test_member_routing.py tests/test_arc_member_interaction_results.py -q
```

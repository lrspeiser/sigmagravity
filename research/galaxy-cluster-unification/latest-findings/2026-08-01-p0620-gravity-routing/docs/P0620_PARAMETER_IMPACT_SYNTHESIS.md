# P0620: parameter-impact synthesis

## Current tested formula

The most promising current diagnostic is

\[
\boldsymbol\alpha_{\rm test}(\mathbf x)
=\boldsymbol\alpha_{0554}(r)
+{Q^2\over1+\Delta_{80}}
\,\mathcal R_{90}\!\left[\delta\boldsymbol\alpha_{\rm route}(\mathbf x)\right],
\]

where \(\Delta_{80}=\alpha_{0554}(R_{80})/\alpha_b(R_{80})-1\), the conservative
route fraction is \(\Delta_{80}/(1+\Delta_{80})\), its width is
\(0.23R_{80}\sqrt{1+Q^2}\), its return length is \(0.36R_{80}\), and
\(\mathcal R_{90}\) rotates the positive route template by a shared 90 degrees
before annular monopole removal. No gravity parameter is fitted per object.

This is a phenomenological diagnostic, not a field equation or promoted
theory. P0554 still carries the galaxy result; the angular layer is defined to
vanish for the axisymmetric galaxy and Solar point-source limits used here.

## What changes the results most

| Finding | Evidence | Interpretation |
|---|---:|---|
| Spatial width/support is the most recurrent coordinate | rank 1 across 13 prior stages and all 3 domains | It is the best general sensitivity coordinate, not a universal correction. |
| Angular phase is the largest new lens response | shared +90 degrees gives +1.685% mean on five frozen systems and +8.241% on RXJ2129 | The original inward/radial convention was likely the wrong broad phase. |
| Routed fraction is the most explosive coordinate | 1.111-root mean span in P0613 | Larger effects are often destructive; strength alone is not the solution. |
| Width/return coupling is the strongest local support interaction | 0.289 percentage-point response span | It scales the effect but does not fix the sign split. |
| Center-crossing rule is low leverage | 0.095 percentage-point span | Overshooting the center is not the main issue in these tests. |
| Smooth contrast cap is mainly an interaction safeguard | zero marginal root and SPARC span in the bounded factorial | Retain it for boundedness, not explanatory power. |

## Cross-domain status

| Domain | Current result | Honest conclusion |
|---|---:|---|
| SPARC outer rotation | 12.592 km/s RMSE vs fixed RAR 10.348 and simple MOND 10.385 | Close enough to remain useful, but 21.7% worse than RAR and entirely carried by P0554. |
| Solar proxies | all pass; Mercury -1.730 mas/century within 3.1 margin | Compatibility is real for the tested proxies, but mostly supplied by symmetry/screening. |
| Five frozen-geometry clusters | +1.685% mean, 18/18 roots, 3/5 improve | Phase is impactful but not universal. |
| A383 full-refit transfer | +0.174%, 9.081 arcsec RMS | The radial-to-tangential sign improvement transferred, but absolute accuracy is inadequate. |
| MS2137 transfer | incomplete control and candidate roots | Inconclusive. |
| Raw validation vs compact halo | 19.076 vs 9.989 arcsec RMS | Current baryon-only construction remains about 1.91 times worse. |

## Universal lessons from this stage

1. A scalar enhancement can approach galaxy rotation data, but it does not
   predict the cluster's angular lensing structure.
2. Conservative routing changes caustics and exact roots long before it fixes
   the missing absolute convergence.
3. Strength, width, and return length regulate magnitude; phase determines
   whether the correction helps or hurts a particular residual geometry.
4. A shared tangential-like phase is the first recent modification to turn the
   A383 radial loss into a gain under a frozen full refit, but two of five
   development clusters still resist it.
5. The next parameter should not be another fitted scalar. It should be an
   independently observed baryonic direction—gas/stellar offset, tidal axis,
   or resolved multipole orientation—frozen before raw lens scoring.

The stage objective is complete: the highest-impact coordinates have been
separated from low-leverage ones, and the failure boundary is sharper. The
formula is not promoted.

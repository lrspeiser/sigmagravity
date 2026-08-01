# P0571-P0572: tidal cancellation at apparent-gravity locations

## Result in one paragraph

Apparent-dark residual peaks in standard cluster convergence maps preferentially
occur where member-galaxy gravity vectors partially cancel and the baryonic
tidal field has comparable stretching and compression. The null-safe empirical
activation

\[
A(\mathbf x)=\sqrt{1-C(\mathbf x)}\,B_T(\mathbf x)
\]

transferred across all three held-out peak systems, all three separate pilot
systems, and a GLAFIC reconstruction. A forward map selected only on seven
development clusters failed its three held-out clusters. However, a clearly
post-hoc 50-kpc tidal-weighted version improved three pilot normalized lens
maps by 21.8% and all 300 posterior realizations. The supported observation is
therefore a repeatable **location/shape coordinate**, not a law for absolute
gravity strength and not a galaxy-rotation solution.

## The baryon-only invariants

For every catalogued member galaxy at \(\mathbf x_i\), P0571 constructed

\[
\mathbf g_i(\mathbf x)=
w_i^\gamma\frac{\mathbf x_i-\mathbf x}
{(|\mathbf x_i-\mathbf x|^2+s^2)^{(p+1)/2}}.
\]

The vector coherence is

\[
C=\frac{|\sum_i\mathbf g_i|}{\sum_i|\mathbf g_i|}.
\]

It is one when every contribution points the same way and decreases as
different galaxy fields cancel. The local tidal Jacobian \(T\) supplies

\[
B_T=\frac{\mathrm{shear}(T)}
{\mathrm{shear}(T)+|\mathrm{tr}(T)|/2}.
\]

`B_T` is high when stretching in one direction and compression in the other
are comparable. It is a saddle/balance descriptor, not an additional force.

P0571 tested 480 combinations of eight features, five distance exponents,
four softenings, and three light-weight powers. Each real residual peak was
ranked against 71 locations at the same cluster-centric radius. This removes
the simplest radial-concentration explanation.

## Broad invariant screen

The strongest individual feature was high tidal balance with \(p=2.5\),
50-kpc softening, and linear F160W weights.

| Diagnostic | Centered-rank effect |
|---|---:|
| Seven development systems | 0.239 |
| Three held-out systems | 0.174 |
| GLAFIC method control | 0.155 |

All three held-out systems had the development-selected direction. Several
related features told the same qualitative story: peaks were farther from the
nearest member, in lower local density and weaker tidal amplitude, but had
more effective contributors and more balanced tidal structure.

That broad search did not pass its search-aware control. Same-radius
pseudo-peaks produced an equally large best-of-480 effect with empirical
probability 0.249. Tidal balance alone also does not vanish around one source.
No forward formula was authorized from P0571.

## Null-safe interaction

P0571B froze the physically motivated interaction

\[
A=(1-C)^\alpha B_T^\beta
\]

at the ordinary inverse-square field exponent \(p=2\), 50-kpc softening, and
linear light weights. Only nine \((\alpha,\beta)\) choices were searched. The
selected setting was

\[
\boxed{A=\sqrt{1-C}\,B_T}.
\]

| Diagnostic | Result |
|---|---:|
| Development centered-rank effect | 0.275 |
| Search-aware empirical p | 0.00389 |
| Earlier held-out effect | 0.224 |
| Separate three-pilot effect | 0.410 |
| Pilot systems in selected direction | 3/3 |
| GLAFIC effect | 0.171 |

The cancellation factor supplies an exact conceptual null: for one centered
source, or for net axisymmetric baryonic component fields, \(C=1\) and
\(A=0\). The candidate therefore adds no Solar force and no axisymmetric
galaxy force. This makes it safe as an angular activation, but also means it
cannot explain SPARC rotation without a separate radial field law.

## Prospective forward-map failure

P0572 asked the harder question: can the invariant generate a destination map
without seeing the lens target? Three arrival carriers were frozen:

\[
D=A,\qquad D=A\sum_i|\mathbf g_i|,
\qquad D=A\|T\|_F.
\]

After 20, 50, or 100-kpc smoothing, a universal fraction \(f\) replaced the
locked 100-kpc local-light map:

\[
\Sigma_{\rm pred}=(1-f)B_{100}+f\,\widehat D.
\]

All maps were nonnegative and unit normalized, so this test redistributed map
shape rather than adding total normalized strength.

Development selection chose the field-weighted carrier, 100-kpc smoothing,
and the boundary value \(f=1\). It improved development JS from 0.07884 to
0.06877, but failed every prospective transfer gate:

| Held-out test | Result |
|---|---:|
| Local-control JS | 0.03882 |
| Selected JS | 0.04976 |
| Change versus local | 28.2% worse |
| Systems improved | 0/3 |
| Lenstool realizations improved | 0/300 |
| GLAFIC change | 44.6% worse |

The dominant parameter was arrival smoothing: its development main-effect
span was 10.0%, compared with 4.1% for carrier and 2.9% for routed fraction.
This repeats a major earlier lesson: spatial extent is more consequential than
moderate changes in amplitude.

## Post-hoc stability lead and pilot replication

Fourteen P0572 settings happened to improve both the development and held-out
means versus local. They concentrate around a 50-kpc tidal-weighted destination.
The lowest-development-error member of that post-hoc subset was frozen before
P0572B:

\[
D=A\|T\|_F,\qquad w=50\ {\rm kpc},\qquad f=0.8.
\]

Transferred to A2537, MACS J0417, and MACS J0949, it produced:

| Pilot metric | Local | Arrival | Change |
|---|---:|---:|---:|
| Equal-system JS | 0.05503 | 0.04303 | 21.8% better |
| Mean Pearson | 0.6686 | 0.7201 | +0.0515 |
| Systems improved | — | 3/3 | pass |
| Posterior realizations improved | — | 300/300 | pass |

This is a strong replication stress result but not untouched validation.
P0571B had already inspected peak locations from these pilot convergence maps,
and all systems are globally spent. It earns a genuinely fresh map test, not a
raw-lensing or theory claim.

## What this says in ordinary language

The data repeatedly point to the quiet gravitational intersections between
galaxies: places where several galaxy fields oppose each other while spacetime
would be stretched one way and compressed the other. Those locations resemble
where the standard lens models put part of their nonlocal mass.

But knowing likely locations is not enough. The first honest attempt to turn
that clue into a complete arrival map selected the wrong carrier and scale for
new clusters. A narrower tidal-weighted construction looks much better, but it
was found after inspecting the first transfer. The missing piece is a
constitutive rule that determines **how much** response occupies a saddle and
how it connects to the radial field.

## Universal truths from this stage

1. Vector cancellation alone is too weak and does not transfer reliably.
2. Tidal balance alone transfers descriptively but fails a large-search control
   and is not Solar/axisymmetric null.
3. Their product is much more stable than either constituent and has an exact
   isolated-point-source null. P0573 later showed that this does **not** extend
   to a resolved axisymmetric disk.
4. A peak-location correlation does not guarantee a full forward map.
5. The arrival carrier and smoothing scale are more important than moderate
   changes in routed fraction.
6. The 50-kpc tidal-weighted arrival map is the current candidate worth testing
   on entirely uninspected clusters.
7. The activation was initially described too broadly as axisymmetric-null.
   Only its one-source null had actually been tested here. P0573 found a 0.439
   RMS activation for a resolved circular disk, requiring a separate symmetry
   factor before the layer can coexist with an axisymmetric galaxy law.

## Next decisive test

Acquire a fresh cluster set with member-light catalogs and independent
Lenstool/GLAFIC or free-form reconstructions. Lock
`tidal_weighted, 50 kpc, f=0.8` without inspecting peaks or maps. Require all
clusters and posterior realizations to improve a fixed local-light control.
Only after that should the same potential be propagated through raw multiple
images and weak shear. A failure would reduce the P0572B result to selection
history; a pass would justify deriving an absolute-strength and radial closure.

## Reproduce

```powershell
python scripts/run_p0571_apparent_peak_baryon_invariant.py
python scripts/run_p0571b_null_safe_tidal_cancellation.py
python scripts/run_p0572_tidal_cancellation_arrival_forward.py
python scripts/run_p0572b_pilot_arrival_transfer.py
python -m pytest -q tests/test_p0571_p0572_tidal_cancellation_results.py
```

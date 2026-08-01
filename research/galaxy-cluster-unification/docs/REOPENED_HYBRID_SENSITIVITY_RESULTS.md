# Reopened hybrid sensitivity investigation

## Outcome

This program deliberately treated the proposed terms as new-physics
hypotheses. A formula was not rejected because it conflicts with an argument
in the literature. Each idea was varied, refitted with one universal parameter
set, and tested against galaxy rotation, cluster dynamics and lensing, and
Solar-System constraints.

The project-wide disposition vocabulary is now stricter: an untested parent
idea, a proxy-tested version, an empirically unidentified parameter, and a
formula disfavored over a stated tested range are different outcomes. See
[`RECENT_NEW_PHYSICS_IDEA_AUDIT.md`](RECENT_NEW_PHYSICS_IDEA_AUDIT.md). A
failure of one proxy or parameterization is not a rejection of the broader
new-physics idea.

The main result is not a finished theory. It is a map of which mathematical
changes actually matter:

1. The high-acceleration screen exponent is the dominant Solar-System control.
2. Saturating the extra force is the dominant galaxy control, but it also
   removes the cluster enhancement that the model needs.
3. Moving the density transition with acceleration changes galaxies in the
   desired direction while leaving cluster scores nearly unchanged, but the
   effect is still too small.
4. The interaction coefficient between Sigma and refracted-gravity terms is
   almost completely redundant with the fitted Sigma amplitude.
5. Raw lens-root existence is a crucial test. Several apparently good cluster
   scores disappeared when the lens geometry search was made more thorough.
6. Separating the two saturation channels exposes opposite controls:
   Sigma-only saturation modestly improves galaxies without materially moving
   raw lensing, while RG-only saturation improves raw lensing and makes
   galaxies worse.
7. A scalar can classify galaxies and clusters extremely well without being a
   useful gravity gate. Equivalent enclosed mass reached AUC 0.981 but worsened
   every cross-domain formula tested.
8. Local-to-mean density controls which RG branch survives: cluster-side
   gating improves lensing, while galaxy-side gating improves galaxies and
   consistently loses held-out lens roots.
9. Dimensionless tidal-shape ratios have much larger leverage than scalar
   class indicators, but the orientations that improve galaxies and lensing
   still oppose one another.
10. The best tidal compromise depends materially on mixing an axisymmetric
    galaxy closure with a spherical cluster closure. Using one spherical
    closure everywhere worsens its galaxy error from 68.26 to 101.68 km/s.
11. Under spherical closure, every tested scale-free tidal invariant is
    exactly a nonlinear re-expression of local-to-mean density ratio. It does
    not provide independent directional information.
12. Radial profile history is an independent lever. Inner-to-outer memory
    improves galaxy predictions, whereas reversing the direction worsens them;
    placing memory before the Solar screen largely switches the effect off.
13. Memory scale changes raw lens-root existence as well as error. A moderate
    two-log-radius scale is the best stable fractional-memory compromise,
    while longer memory improves galaxies further but loses held-out roots.
14. What the memory transports is more important than its amplitude alone.
    Weighting its source by baryonic acceleration and radius moves galaxy and
    cluster predictions by much larger amounts than small strength changes.
15. A data-derived effective radial power, (q+p,d\ln g_N/d\ln r), orders
    the response in both domains. Because measured SPARC profiles fall faster
    than CLASH profiles, one universal pair (p,q) affects them differently
    without using galaxy or cluster labels.
16. The useful directions still oppose one another: low-acceleration or
    exterior weighting improves galaxies while degrading the cluster bridge
    and usually raw lensing; positive acceleration weighting improves lensing
    while over-accelerating galaxies.
17. Smooth pointwise slope interpolation mostly behaves like another fixed
    blended carrier. Its best stable ratio is 4.19, slightly worse than the
    preceding fixed-carrier ratio of 4.11.
18. Slope-gate sharpness is a very high-leverage anti-galaxy/lensing control.
    A hard switch improves stable raw lensing to 22.89 arcsec but raises galaxy
    error to 319.90 km/s.
19. The hard-switch penalty concentrates in gas-rich, low-stellar-mass
    galaxies. It is not a special failure of bulges or one morphology class.
20. Exact duplicate formulas reveal bridge-fit non-identifiability: nearly
    identical bridge and galaxy scores can conceal large universal-parameter
    changes and different lens branches.
21. A universally smoothed local slope removes the raw-grid extrapolation
    problem of a whole-profile slope, but the best pivot keeps moving until
    the gate is effectively always on.
22. Five independent refits of the exact always-on endpoint recover every
    raw-lens root and reproduce the saturated-pivot galaxy prediction. The
    present gain therefore does not require a slope coordinate.
23. Memory length remains more consequential than smoothing bandwidth. In the
    endpoint sweep it changes raw lensing by 2.86 arcsec and crosses a root
    boundary, whereas the finite-pivot and bandwidth effects are small.
24. Extending the endpoint source to
    \(X=F(g_N/g_{\rm ref})^p(r/{\rm kpc})^q\) identifies a broad, measurable
    exponent ridge rather than one preferred pair of exponents.
25. Across that ridge, memory strength is the largest galaxy lever, radial
    power \(q\) is the largest raw-lensing lever, and acceleration power \(p\)
    provides the cleanest differential control because SPARC and CLASH
    profiles have different measured slopes.
26. A shorter memory length is now bracketed: performance worsens on both
    sides of the useful range. This is stronger evidence than an optimum at a
    tested boundary.
27. The apparent fixed-\(p\) turnover near \(q=5\) is a coordinate artifact.
    Allowing \(p\) to move with \(q\) continues the improvement until the
    correlated ridge is bracketed near \(q=9\)--10.
28. Constant-SPARC-effective-power paths preserve the galaxy score to within
    0.14--0.24 km/s over \(q=8\)--12, while a constant-CLASH-effective-power
    path preserves the bridge to within \(9.9\times10^{-5}\) dex and moves
    galaxies by 6.40 km/s.
29. Effective power is therefore a strong local organizing coordinate, but it
    is not a global invariance: the constant-power stress paths worsen again
    at extreme exponents because real profiles are not single power laws.
30. At high \(q\), memory length becomes the leading refinement. Moving it
    from 0.5 to 0.4 improves the stable cross-domain ratio from 3.540 to 3.487.
31. A balanced 81-cell local factorial separates the coupled controls. Memory
    strength explains 60.5% of sampled galaxy variation, effective power
    22.6%, memory length 9.3%, and \(q\) only 1.2%.
32. The cluster bridge is almost entirely a memory-range response in this
    neighborhood: length explains 85.2% and strength 14.1%; the two exponent
    coordinates together explain less than 0.01%.
33. Raw lensing is different: no main effect explains even 5% of its sampled
    variation. Strength-length and exponent-strength interactions are larger
    than any single exponent effect, and higher-order combinations retain
    53.3% of the diagnostic variation.
34. The local optimum is a plateau, not a measured point. Thirteen cells lie
    within 1% of the best ratio and span all tested \(q\) and memory lengths;
    only high effective power and high memory strength are consistently kept.
35. Eight-start lensing recovered 18 shallow root failures. Eighty-four of 85
    rows retain all roots, so most apparent failures in this local region were
    optimizer failures rather than equation failures.
36. Remembering the bounded tidal-channel gate does not improve the global
    compromise. Gate orientation explains 87.2% of sampled galaxy variation,
    99.8% of bridge variation, and 80.0% of raw-lensing variation; memory
    strength explains only 1.9%, less than 0.1%, and 3.7%, respectively.
37. In the galaxy-favored orientation, inner-to-outer gate memory improves the
    local zero-memory galaxy score while slightly improving raw lensing, but
    the best stable result is still 68.83 km/s and 17.69 arcsec. The best raw
    branch is 8.21 arcsec with gate memory turned off and a catastrophic
    288.88 km/s galaxy error.
38. Replacing the one-sided gate by exact complementary middle-band and
    two-tail gates does not reconcile the endpoints. Zero of 16 nonmonotonic
    cells beats both the galaxy-favored and raw-lensing-favored monotonic
    controls at matched sharpness.
39. Gate topology is nevertheless high leverage within that nonmonotonic
    factorial: it explains 60.0% of sampled galaxy variation and 34.7% of
    robust raw-lensing variation. The effect changes predictions strongly but
    in the wrong correlated direction for unification.
40. The best topology-stage compromise remains monotonic and scores 68.24
    km/s on galaxies and 17.79 arcsec on robust raw lensing. The best raw row
    is the opposite monotonic orientation at 8.40 arcsec and 288.89 km/s.
41. Moving the same tidal gate from the RG ceiling to radial-memory strength
    does not improve the global compromise. Zero of 27 placement rows improves
    both galaxy and robust raw-lensing error over the ungated full-memory
    control, and zero beats the prior 3.475 reference ratio.
42. In the balanced both-placement factorial, cap orientation explains 73.2%
    of sampled galaxy variation, 37.0% of bridge variation, and 63.1% of the
    Solar-proxy variation. Memory orientation is secondary at 8.5%, 0.3%, and
    14.4%, respectively.
43. Raw lensing again behaves differently. Maximum memory strength is the
    largest main effect at 29.3%, but memory-orientation by sharpness explains
    14.9% and memory-orientation by strength explains 10.9%; 24.3% remains in
    higher-order structure.
44. The most attractive new stable tradeoff that gates both mechanisms uses
    the low-tidal orientation for both cap and memory at sharpness 10 and full
    strength. It reaches 65.01 km/s on galaxies, 0.1098 dex on the bridge, and
    18.82 arcsec on robust raw lensing, for a ratio of 6.09. This is useful
    leverage but substantially worse than global full memory.
45. A memory-only low-tidal row initially looked more balanced at 45.61 km/s
    and 19.44 arcsec, but the eight-start replay lost a held-out lens root. A
    memory-only high-tidal row recovered its shallow missing root but retained
    a worse stable ratio of 5.31.
46. Twenty-two of 27 placement fits put at least one universal bridge
    parameter on a declared bound. Predictions identify placement effects more
    strongly than they identify a unique decomposition into
    ((\epsilon_0,\rho_c,Q,B)).
47. A symmetric no-flux diffusion in log radius supplies a genuine spatial
    derivative while conserving the integral of its transported carrier. It
    does not improve the complete-root cross-domain result over its exact
    local or one-sided-memory controls.
48. The physical quantity being redistributed matters. Added acceleration is
    the most lensing-favored carrier and carries the largest galaxy penalty;
    short-range circular-speed-squared redistribution is nearly galaxy-neutral
    but produces only a small raw-lensing gain.
49. In the balanced diffusion factorial, carrier explains 23.3% of galaxy and
    38.4% of bridge variation. Raw lensing is interaction-dominated:
    carrier-by-scale explains 23.5%, carrier-by-strength 14.8%, and
    higher-order structure 27.3% of the diagnostic variation.
50. Symmetric diffusion after the preceding best one-sided memory improves
    galaxy RMSE as far as 30.87 km/s from 37.11 km/s, but all nine such rows
    lose at least one held-out lens root in the eight-start replay.
51. All 38 diffusion-stage bridge refits touch at least one universal-parameter
    boundary. The tiny new global ratio improvement comes from the exact
    zero-diffusion control under a new optimizer seed, not from diffusion.

No tested setting simultaneously reaches the fixed-RAR galaxy reference and
the compact-halo raw-lensing reference. The lowest eight-start raw score with
complete held-out roots is now 8.21 arcsec from the soft
middle-tidal-eigenvalue gate with gate memory turned off. That is 9.3% below
the compact-halo reference on the primary equal-system RMS, and its pooled
reduced chi-square is 140.69 versus 142.62. The same setting has a galaxy error
of 288.88 km/s, or 27.0 times the fixed-RAR error. The best observed
eight-start cross-domain row remains the same physical endpoint:
\(p=1.927395\), \(q=9\), median-SPARC effective power 6, memory length 0.35,
full one-sided memory, and now explicitly zero symmetric diffusion. A new
exact control refit scores 37.11 km/s on galaxies and 27.94 arcsec on raw
lensing, for a worse-reference ratio of 3.474. This is only a 0.039%
improvement over the preceding 3.475 result and is attributable to a new
optimizer seed rather than a new formula term.

Its raw score is 1.82% worse than baryons, 0.89% better than simple MOND, and
3.09 times the compact-halo error. Turning memory off produces the strongest
raw-lensing result in this stage, 18.96 arcsec, but worsens galaxies to 69.66
km/s and therefore has a poor combined ratio of 6.52. Eight exact copies of
the \(p=2.56986,q=10,\ell=0.5\) equation have a median combined ratio of
3.540 and raw scores spanning only 0.061 arcsec, although their fitted
parameters remain nonidentified. The gain is a repeatable exponent-memory
ridge, not a unique determination of microscopic constants or a unified
solution.

## Formula family tested

The ordinary baryonic acceleration is \(g_N\). The refracted-gravity and Sigma
excesses are

\[
R=\epsilon_{\rm RG}^{-1}-1,
\qquad
S=B\,h(g_N).
\]

Two ways of combining them were tested:

\[
F_{\rm int}=R+S+\eta RS,
\]

\[
F_p=(R^p+S^p)^{1/p}.
\]

An optional smooth ceiling limits the combined excess:

\[
F_{\rm sat}=L\tanh(F/L).
\]

The continuation also tested ceilings before combination:

\[
R_{\rm eff}=L_R\tanh(R/L_R),
\qquad
S_{\rm eff}=L_S\tanh(S/L_S).
\]

This allows the RG and Sigma responses to be varied independently instead of
forcing one common cap to suppress both.

The geometry-gated tests mix the uncapped and capped channel continuously:

\[
w(q)=\left[1+\exp\{-s\,o(q-q_0)\}\right]^{-1},
\]

\[
R_{\rm eff}=R+w_R\left[L_R\tanh(R/L_R)-R\right],
\]

with an analogous expression for \(S_{\rm eff}\). Here \(q\) is a
formula-facing geometry measurement, \(q_0\) is a fixed pivot, \(s\) is a
fixed sharpness, and \(o=\pm1\) chooses which side receives more of the cap.

For the directional stage, the absolute tidal eigenvalues are ordered
\(a_1\leq a_2\leq a_3\). The tested dimensionless coordinates include

\[
D=\frac{a_3}{a_1+a_2+a_3},
\qquad
M=\frac{a_2}{a_3},
\qquad
Z=\frac{|\lambda_z|}{a_1+a_2+a_3}.
\]

They describe the *shape* of the local baryonic tidal field rather than its
overall strength. The mixed-closure stage uses an axisymmetric SPARC midplane
and spherical cluster/Solar calculations. A separate control forces every
domain to use the same spherical-density closure.

That control exposes an exact identity. With

\[
\bar\rho=\frac{3g_N}{4\pi G r},
\qquad
\delta=\frac{\rho}{\bar\rho},
\]

the spherical tidal eigenvalues obey

\[
\frac{(\lambda_r,\lambda_t,\lambda_t)}{g_N/r}
=(3\delta-2,1,1).
\]

Consequently, every dimensionless spherical tidal invariant is a deterministic
function of the already-tested density ratio \(\delta\). Genuine new
directional information requires nonspherical maps or independent spatial
derivatives.

The radial-memory stage supplies one such nonlocal profile variable. For
ordered radii, its running excess is

\[
M_i=e^{-\Delta\ln r_i/\ell}M_{i-1}
 +\left(1-e^{-\Delta\ln r_i/\ell}\right)F_i,
\]

\[
F_{{\rm eff},i}=(1-\mu)F_i+\mu M_i.
\]

Here \(\mu\) is a universal memory strength and \(\ell\) is a universal scale
in log radius. The default ordering is from inner to outer radius; the reverse
ordering tests an exterior-pressure interpretation. Memory can be applied
before or after the local high-acceleration screen. A one-point profile and
\(\mu=0\) both reduce exactly to the local formula.

The tidal-gate-memory control remembers the bounded channel classification
rather than the force excess. If \(w_i\) is the local tidal gate,

\[
M^w_i=e^{-\Delta\ln r_i/\ell}M^w_{i-1}
 +\left(1-e^{-\Delta\ln r_i/\ell}\right)w_i,
\]

\[
w_{{\rm eff},i}=(1-\mu_w)w_i+\mu_wM^w_i.
\]

The effective gate places the existing RG and Sigma ceilings before channel
combination and Solar screening. It does not remember acceleration or add a
new force amplitude. This is a proxy for persistent geometry: SPARC uses an
axisymmetric midplane tidal closure, while the bridge, raw clusters, and Solar
System use a spherical closure.

The gate-placement continuation can instead modulate how much of the existing
force memory is retained at each radius. For a local channel weight (w_i),

\[
\mu_{{\rm eff},i}=\mu_{\max}w_i
\quad\hbox{or}\quad
\mu_{{\rm eff},i}=\mu_{\max}(1-w_i),
\]

\[
F_{{\rm eff},i}=F_i+\mu_{{\rm eff},i}(M_i-F_i).
\]

The two orientations apply the same maximum memory strength preferentially to
opposite measured tidal regimes. Independent switches allow the RG and Sigma
ceilings to remain global while the coordinate gates memory, so cap placement
and memory placement can be measured separately.

The symmetric spatial-derivative stage instead transports a positive carrier
\(X\) with the no-flux heat equation in log radius,

\[
\frac{\partial X}{\partial\tau}
=\frac{\partial^2X}{\partial(\ln r)^2},
\qquad
\left.\frac{\partial X}{\partial\ln r}\right|_{r_{\min},r_{\max}}=0,
\]

and blends the diffused profile \(D_\ell[X]\) with its local value,

\[
X_{\rm eff}=(1-\mu_D)X+\mu_DD_\ell[X].
\]

The finite-volume implementation preserves positivity and the quadrature
integral \(\int X\,d\ln r\). The three carriers are fractional excess
\(X=F\), added acceleration \(X=Fg_N\), and added circular-speed-squared
\(X=Fg_Nr\). A one-point profile and \(\mu_D=0\) are exact identities.

The nonmonotonic topology control replaces the one-sided logistic gate with an
exact complementary pair on the same coordinate (x):

\[
w_{\rm band}=\sigma[k(x-x_L)]\,\sigma[k(x_U-x)],
\qquad
w_{\rm tails}=1-w_{\rm band}.
\]

The band gate places the RG ceiling mainly on intermediate tidal shapes; the
two-tail gate places it on both low and high tidal shapes. Because the two
weights sum exactly to one, their difference isolates channel placement rather
than a change in total gate normalization.

The generalized carrier stage replaces the remembered source (F) by

\[
X=F\left(\frac{g_N}{g_{\rm ref}}\right)^p
    \left(\frac{r}{1\ {\rm kpc}}\right)^q.
\]

The same running-memory operation is applied to (X), and the result is
divided by the two weighting factors to return a fractional force excess.
Thus (p=q=0) is the original fractional-memory law, (p=1,q=0) transports
added acceleration, and (p=q=1) transports the added contribution to
circular speed squared. Memory was also placed on the combined screened
response, RG alone, Sigma alone, or RG and Sigma independently.

If a local profile behaves as (g_N\propto r^s), the transported source has
the effective radial factor

\[
X\propto F r^{q+ps}.
\]

This identity makes the exponent pair testable against the measured profile
slopes rather than interpreting (p) and (q) only as abstract knobs.

The local-slope stage then tested whether the carrier powers could interpolate
smoothly between the balanced and galaxy-favoring endpoints. With

\[
s(r)=\frac{d\ln g_N}{d\ln r},
\qquad
w(r)=\mu_s\left[1+e^{-k(s_0-s(r))}\right]^{-1},
\]

the pointwise powers are

\[
p_{\rm eff}=p_0+w(p_1-p_0),
\qquad
q_{\rm eff}=q_0+w(q_1-q_0).
\]

The transported source remains

\[
X=F\left(\frac{g_N}{g_{\rm ref}}\right)^{p_{\rm eff}}
    \left(\frac{r}{1\ {\rm kpc}}\right)^{q_{\rm eff}}.
\]

Here the tested base endpoint is (p_0=-1,q_0=-0.5), the steep-profile
endpoint is (p_1=-0.5,q_1=1.5), and the universal gate controls are strength
(\mu_s), pivot (s_0), and sharpness (k). Setting (\mu_s=0) reproduces
the preceding carrier exactly. One-point profiles also remain exactly local;
the formula never invents a missing slope.

The high-acceleration screen is

\[
W_a=
\left(
\frac{a_s m}{a_s m+g_N}
\right)^n,
\]

where \(a_s\) is a fixed acceleration scale and \(m\) is the tested scale
multiplier. The predicted acceleration is

\[
\frac{g}{g_N}=1+W_aF.
\]

The optional moving density transition is

\[
\rho_{c,\mathrm{eff}}
=
\rho_c
\left(\frac{g_N}{g_{\rm ref}}\right)^\alpha .
\]

For each formula variant, the same four gravity parameters
\((\epsilon_0,\rho_c,Q,B)\) were fitted to the bridge data and then transferred
unchanged to galaxies, raw lensing, and the Solar System. The varied structural
knobs—\(n,L,\eta,p,\alpha,m\)—were fixed universally within a run, never fitted
separately for an individual galaxy or cluster.

## Data and validation

| domain | data used | validation |
|---|---:|---|
| BCG and CLASH bridge | 44 BCG systems plus 20 CLASH clusters; 116 total profile points | five-fold, complete-system holdout |
| Galaxy rotation | 131 SPARC galaxies; 968 outer rotation points | fixed transfer; no gravity refit |
| Raw strong lensing | four clusters; 11 held-out image positions | gravity fixed; only common lens geometry, shear, and source positions optimized |
| Solar System | solar-limb through Saturn force-fraction scan, Earth proxy, and Mercury precession | fixed transfer |

The consolidated program now contains 913 scored rows representing 801 distinct
formula settings and 832 formula/evaluation contexts. Five hundred forty-six
selected settings were rerun with eight raw-lens geometry starts to check
whether the held-out image roots were stable.

## Reference scores

Lower is better in every column.

| reference | galaxy outer RMSE | bridge RMSE | raw lens RMS |
|---|---:|---:|---:|
| Fixed RAR | 10.68 km/s | — | — |
| Baryons only | — | — | 27.44 arcsec |
| Simple MOND | — | — | 28.19 arcsec |
| Compact halo | — | — | 9.05 arcsec |

These references do not have equal status. RAR uses a fixed galaxy relation;
the compact halo is allowed the sort of concentrated cluster mass component
that the universal modified-gravity model is trying to avoid.

## Most informative formula settings

| universal setting | galaxy RMSE | bridge RMSE | raw lens RMS | Solar result | interpretation |
|---|---:|---:|---:|---|---|
| Screen \(n=1.5\), no ceiling | 141.27 km/s | 0.101 dex | 17.85 arcsec, eight-start stable | passes all proxies; Mercury \(-1.19\) mas/century | best repeatable raw-lens improvement, but galaxy boost is far too large |
| Screen \(n=2.0\), no ceiling | 141.05 km/s | 0.115 dex | 18.05 arcsec, eight-start stable | passes | safer Solar suppression does not cure galaxies |
| Screen \(n=1.5\), ceiling \(L=8\) | 90.33 km/s | 0.131 dex | 18.80 arcsec, eight-start stable | passes | partial galaxy improvement with a modest cluster penalty |
| Screen \(n=1.5\), ceiling \(L=3\) | 21.87 km/s | 0.350 dex | 27.38 arcsec in the two-start run; eight-start roots fail | passes | comes within a factor of 2.05 of RAR by removing the cluster response |
| Screen \(n=1.5\), moving threshold \(\alpha=0.75\) | 133.11 km/s | 0.099 dex | 17.81 arcsec in the two-start run | passes | correct separation direction, only a 5.8% galaxy improvement |
| Screen \(n=1.5\), ceiling \(L=8\), \(\alpha=1.5\) | 88.68 km/s | 0.131 dex | 18.78 arcsec in the two-start run | passes | threshold adds little once saturation is present; fitted parameters reach bounds |
| Sigma-only ceiling \(L_S=6.5\) | 132.75 km/s | 0.133 dex | 17.88 arcsec, eight-start stable | passes | improves galaxies 6.0% with essentially no raw-lens change |
| RG-only ceiling \(L_R=2.75\) | 201.29 km/s | 0.164 dex | 15.06 arcsec, eight-start stable | passes | earlier stable raw improvement, but galaxy error worsens 42.5% |
| \(L_R=2,\ L_S=1.5\) | 69.66 km/s | 0.176 dex | 18.95 arcsec, eight-start stable | passes | best robust local compromise, but still 6.52 times the worse reference ratio and fitted \(\epsilon_0,B\) hit bounds |
| Density-ratio gate, cluster-low orientation | 226.06 km/s | 0.163 dex | 14.60 arcsec, eight-start stable | passes | best scalar-gate raw score, but even worse galaxies |
| Same density gate, reversed orientation | 119.05 km/s | 0.103 dex | 20.58 arcsec; roots fail | passes | improves galaxies, but the cluster branch cannot produce every held-out image |
| RG-only cluster gate, sharpness 4 | 210.94 km/s | 0.139 dex | 14.84 arcsec, eight-start stable | passes | confirms RG placement, not Sigma placement, drives the lensing gain |
| RG-only galaxy gate, sharpness 8 | 114.63 km/s | 0.110 dex | 20.62 arcsec; roots fail | passes | strongest galaxy trend in this gate family, incompatible with raw lens roots |
| Middle tidal-ratio gate, pivot 0.85 | 290.16 km/s | 0.123 dex | 8.72 arcsec, eight-start stable | passes | compact-halo-level lensing branch with a catastrophic galaxy prediction |
| Middle tidal-ratio gate, sharpness 5 | 288.89 km/s | 0.107 dex | 8.29 arcsec, eight-start stable | passes | best verified raw score and no fitted parameter at a boundary, but galaxy error is 27.0 times RAR |
| Reversed middle tidal-ratio orientation | 68.26 km/s | 0.115 dex | 17.84 arcsec, eight-start stable | passes | preceding best compromise; fitted Sigma amplitude reaches its lower boundary |
| Same reversed middle ratio, common spherical closure | 101.68 km/s | 0.115 dex | 17.84 arcsec, eight-start stable | passes | changing only the galaxy closure worsens the cross-domain ratio from 6.39 to 9.52 |
| Common-spherical positive fraction, pivot 0.82 | 234.80 km/s | 0.176 dex | 9.13 arcsec; roots fail at eight starts | passes | attractive two-start lens score does not survive the root test |
| Common-spherical traceless fraction, sharpness 40 | 241.23 km/s | 0.177 dex | 9.65 arcsec, eight-start stable | passes | best stable nonlinear spherical gate; near halo lensing but still 22.6 times the RAR galaxy error |
| Dual-cap local law, no radial memory | 69.63 km/s | 0.176 dex | 18.97 arcsec in the two-start run | passes | local control for the memory stage |
| Dual-cap memory, scale 0.25 | 64.13 km/s | 0.177 dex | 19.35 arcsec, eight-start stable | passes | short memory improves galaxies modestly and preserves roots |
| Dual-cap memory, scale 2 | 57.99 km/s | 0.201 dex | 23.73 arcsec, eight-start stable | passes | new best verified compromise; galaxy gain is traded for worse bridge and lensing scores |
| Dual-cap memory, scale 8 | 56.79 km/s | 0.236 dex | 28.41 arcsec; roots fail at eight starts | passes | longer memory keeps improving galaxies but destroys the stable lens branch |
| Unsaturated full memory | 130.57 km/s | 0.102 dex | 18.08 arcsec, eight-start stable | passes | repeatable but much too strong in galaxies |
| Carrier (p=-0.5,q=1.5) | 43.89 km/s | 0.291 dex | 28.53 arcsec, eight-start stable | passes | new best cross-domain ratio, but its gain is only on galaxies and fitted universal parameters touch bounds |
| Carrier (p=0.5,q=0) | 78.16 km/s | 0.174 dex | 18.91 arcsec, eight-start stable | passes | opposite tradeoff: better cluster lensing but much worse galaxies |
| Carrier (p=1,q=0) | 117.11 km/s | 0.147 dex | 16.64 arcsec; roots fail at eight starts | passes | added-acceleration transport pushes lensing further but loses a held-out image and over-accelerates galaxies |
| Slope gate off, balanced carrier | 47.70 km/s | 0.204 dex | 25.05 arcsec, eight-start stable | passes | stable control that improves lensing over baryons while retaining most of the carrier galaxy gain |
| Smooth slope gate, pivot 0 | 44.78 km/s | 0.264 dex | 28.63 arcsec, eight-start stable | passes | best stable adaptive setting, but its 4.19 ratio does not beat the fixed carrier |
| Hard slope gate, sharpness 16 | 319.90 km/s | 0.200 dex | 22.89 arcsec, eight-start stable | passes | repeatable lensing improvement paired with catastrophic low-mass, gas-rich galaxy errors |

The stable \(n=1.5\) case reduces raw lensing RMS by about 35% relative to
baryons and 37% relative to simple MOND. It is still about 1.97 times the
compact-halo error, and its galaxy error is about 13.2 times the fixed-RAR
error. This is why it is evidence for a useful cluster-sensitive response, not
evidence for a unified solution.

## Parameter leverage learned from the data

### 1. Screen exponent

Changing \(n\) from 0.7 to 1.5 reduced the largest Solar fractional force
change from \(2.46\times10^{-4}\) to \(1.55\times10^{-8}\), a factor of roughly
16,000. The Mercury proxy changed from about \(-1.22\) million to
\(-1.19\) milliarcseconds per century. The pass boundary in this implementation
lies near \(n=1.5\).

Reducing only the screen scale did not reproduce this. With \(n=1\), the
Mercury prediction remained thousands of milliarcseconds per century over the
tested scale range. The shape of the turnoff matters more than simply moving
it.

### 2. Saturation ceiling

The ceiling is the strongest galaxy lever. Lowering \(L\) from 12 to 3 reduced
the galaxy RMSE from 129.07 to 21.87 km/s. At the same time, bridge error
worsened from 0.102 to 0.350 dex and the raw-lensing score returned almost to
the baryons-only value.

This shows a concrete structural conflict: the same large excess that helps
the cluster calculation over-accelerates galaxy outskirts. A global scalar
ceiling cannot distinguish the two geometries.

### 3. Moving density transition

With no saturation, increasing \(\alpha\) from 0 to 0.75 reduced galaxy RMSE
from 141.27 to 133.11 km/s while bridge and raw-lens scores were essentially
unchanged. This is the desired direction because it separates galaxy response
from cluster response without an object label.

The gain is modest, and after adding saturation the threshold shift is mostly
absorbed by the other fitted parameters. At ceiling \(L=8\), changing
\(\alpha\) from 0 to 1.5 improves the galaxy score by only 1.64 km/s.

### 4. Sigma–RG interaction

Changing \(\eta\) from 0 to 1 changed the bridge error by only
\(-0.000011\) dex, the galaxy error by \(-0.087\) km/s, and raw lensing by
\(+0.062\) arcsec after refitting. The Sigma amplitude \(B\) compensates for
the interaction. In this formula, \(\eta\) does not supply a new observable
degree of freedom.

### 5. Power-mean combination

Changing the power \(p\) moved the raw-lensing field and galaxy prediction
more than the interaction coefficient did. Some choices improved a two-start
raw score but either increased the galaxy error or lost held-out lens roots.
This knob changes real structure, but the tested scalar power mean does not
resolve the domain conflict.

### 6. Lens geometry robustness

The \(n=1.5\) score was stable: 17.86 arcsec with two starts and 17.85 arcsec
with eight. The \(n=1.6\), \(L=3\), \(L=6\), and \(L=12\) candidates lost one
or more held-out roots in the eight-start replay. A lower optimizer cost is not
automatically a better physical lens if the fitted mapping ceases to produce
an image at the held-out location.

### 7. Channel-specific saturation

Sigma-only saturation has a small but clean separation effect. A ceiling of
\(L_S=6.5\) reduced galaxy RMSE from 141.27 to 132.75 km/s, a 6.0% gain, while
the eight-start raw score changed only from 17.87 to 17.88 arcsec. The fitted
Sigma amplitude reached its upper bound, so the data prefer a harder saturated
Sigma channel than the current parameter range can identify.

RG-only saturation acts in the opposite direction. A ceiling of \(L_R=2.75\)
improved eight-start raw lensing from 17.87 to 15.06 arcsec, a 15.7% gain, but
worsened galaxy RMSE to 201.29 km/s. The fitted low-density permittivity
\(\epsilon_0\) reached its lower bound.

The raw response contains a sharp image-root boundary. Ceilings 2.0, 2.25, and
2.5 still lost held-out roots after eight starts. At 2.75, all roots were
recovered, but the honest RMS rose from the misleading two-start value of
10.84 to 15.06 arcsec. Thus the nearly compact-halo two-start scores below
2.75 are not valid successes.

Crossing both ceilings creates a smooth galaxy-versus-lensing tradeoff but no
joint optimum. At fixed \(L_R=2\), increasing \(L_S\) from 1.0 to 2.0 worsened
galaxy RMSE from 47.71 to 90.28 km/s while improving the complete-root raw
score toward roughly 18–19 arcsec. The first locally complete-root setting,
\(L_S=1.5\), still remains 6.52 times its worse reference error.

### 8. Geometry-indicator audit

Eight label-free indicators were calculated on 131 SPARC galaxies, 44 BCG
systems, and 20 CLASH clusters. Object labels were used only afterward to
measure separation.

| indicator | equal-system SPARC/CLASH AUC | CLASH direction |
|---|---:|---|
| equivalent enclosed baryonic mass | 0.981 | higher |
| radius | 0.972 | higher |
| local-to-mean baryonic density | 0.927 | lower |
| local baryonic density | 0.905 | lower |
| baryonic acceleration | 0.898 | higher |
| source concentration | 0.852 | lower |
| equivalent-mass slope | 0.852 | higher |
| tidal curvature | 0.745 | lower |

The high-AUC mass result did not transfer into a useful formula. Even the
softest mass gate gave 195.31 km/s on SPARC, 0.097 dex on the bridge, and
17.77 arcsec in the eight-start raw test. This demonstrates that separating
catalog classes is not enough: the selector must correlate with the missing
acceleration in the correct radial locations.

### 9. Independent geometry-gate topology

The local-to-mean density ratio was then allowed to place each channel cap
independently.

- Cluster-side RG sharpness increased the raw-lensing response: at sharpness 4,
  eight-start RMS reached 14.84 arcsec, while galaxy RMSE worsened to
  210.94 km/s.
- Galaxy-side RG sharpness improved SPARC monotonically from 137.26 to
  114.63 km/s, but every setting lost held-out image 2c in MACS J0429.
- Moving the Sigma cap from the cluster side to the galaxy side changed SPARC
  RMSE by only 0.0045 km/s and raw RMS by 0.002 arcsec. The bridge fit drove
  the Sigma amplitude toward zero, making its placement observationally
  redundant in this construction.
- The best complete-root geometry-gate raw score, 14.60 arcsec, is about 47%
  better than baryons and 48% better than simple MOND, but remains 1.61 times
  the compact-halo reference and has a 226.06 km/s galaxy error.

The impactful quantity is therefore not simply mass, radius, or “cluster
likeness.” It is the side of the density-ratio field on which the RG
redistribution acts. That switch changes one domain in the desired direction
only by moving the other onto a non-image-producing or high-error branch.

### 10. Directional tidal-shape gates

The directional audit calculated nine dimensionless tidal-eigenvalue
invariants at 1,084 points in 195 systems. The strongest equal-system
SPARC/CLASH separators were:

| tidal-shape indicator | equal-system AUC | CLASH direction |
|---|---:|---|
| largest-eigenvalue dominance \(D\) | 0.995 | lower |
| middle-to-largest ratio \(M\) | 0.992 | higher |
| smallest-to-largest ratio | 0.973 | higher |
| third-axis absolute fraction \(Z\) | 0.957 | lower |

These gates have far more numerical leverage than the scalar mass gate. For
the middle ratio, changing only its pivot moved SPARC RMSE across a
369.83 km/s span and raw lensing across a 16.43 arcsec span. The change is
therefore affecting the field shape, not being absorbed as another amplitude.

The deeper lensing replay found two stable formulas with all 11 held-out roots
below the compact-halo equal-system RMS: pivot 0.85 gave 8.72 arcsec and
sharpness 5 gave 8.29 arcsec. The latter is about 70% below the reported
baryons-only and simple-MOND RMS values in this four-cluster raw test; simple
MOND itself misses a held-out root. The tidal formula recovered 19 of 20
training roots for MACS J0329 and every training root in the other three
clusters, the same training-root pattern as the compact-halo reference. Its
pooled reduced chi-square, 143.00, is essentially tied with but slightly worse
than the compact halo's 142.62. It has no fitted universal parameter at a
bound.

The cross-domain conflict remains. The sharpness-5 formula predicts
288.89 km/s galaxy error. Reversing the middle-ratio gate improves the galaxy
error to 68.26 km/s, but lensing returns to 17.84 arcsec and the fitted Sigma
amplitude goes to its lower boundary. The empirically expected orientation,
in which the high middle ratio receives more of the RG cap, is worse in both
domains at the audited pivot: 405.43 km/s on galaxies and 29.68 arcsec in the
two-start lens test.

This is stronger evidence than class separation alone: a directional
coordinate can select a compact-halo-level lensing branch. It is not yet
evidence for a common physical tensor rule because the galaxy eigenvalues use
an axisymmetric midplane closure while the cluster calculation uses a
spherical closure. The next decisive test must calculate the same tidal tensor
from registered baryonic maps in both domains.

### 11. Common-closure control and spherical identity

The exact 34-formula control changed only the SPARC tidal reconstruction from
axisymmetric to spherical. Bridge parameters, raw-lensing fields, and Solar
predictions were unchanged. Across all 34 formulas the maximum bridge and raw
RMS changes were exactly zero, isolating the effect to galaxy geometry.

The strongest mixed-closure middle-ratio separation fell from AUC 0.992 to
0.903. More importantly, the same best-compromise formula changed as follows:

| closure | galaxy RMSE | eight-start raw RMS | worse reference ratio |
|---|---:|---:|---:|
| axisymmetric SPARC + spherical clusters | 68.26 km/s | 17.84 arcsec | 6.39 |
| spherical in every domain | 101.68 km/s | 17.84 arcsec | 9.52 |

Thus about one third of the apparent galaxy improvement in that setting comes
from the closure choice, not from a geometry-independent law.

The spherical identity was then verified at all 1,084 data points:

\[
\frac{(\lambda_r,\lambda_t,\lambda_t)}{g_N/r}
=
\left(3\frac{\rho}{\bar\rho}-2,1,1\right).
\]

All nine invariant reconstructions matched this one-density-ratio prediction
to a maximum numerical error of \(1.22\times10^{-15}\). Largest-eigenvalue
dominance, middle ratio, determinant shape, positive fraction, trace,
traceless fraction, and the other spherical coordinates are therefore not
independent physical ideas in this approximation.

Forty-eight small nonlinear reparameterizations were nevertheless tested to
measure numerical leverage. Pivot changes moved galaxy RMSE by as much as
206.76 km/s, and positive-fraction pivots moved raw RMS by 8.71 arcsec. The
best apparent two-start raw result, 8.74 arcsec, lost a held-out root with
eight starts. The best stable result was 9.65 arcsec with a 241.23 km/s galaxy
error. The best stable common-spherical compromise remained the original
closure-control formula at 101.68 km/s and 17.84 arcsec.

This establishes a useful limit: nonlinear reshaping of the spherical density
ratio has large numerical impact but does not resolve the galaxy/lensing
conflict. The next independent variable must come from genuine nonspherical
structure, multiple mass centres, vertical thickness, or a spatial derivative
not fixed by spherical Poisson closure.

### 12. Radial profile memory

The radial-memory stage tested 32 controlled variants: memory strength, scale,
direction, and placement for both the unsaturated and dual-cap base laws. The
four gravity amplitudes remained universal and were refitted only to the same
bridge data; no galaxy or lensing gravity parameter was fitted.

The direction result is strong and repeatable. For the dual-cap base,
inner-to-outer memory reduced galaxy RMSE from 69.63 to 59.29 km/s at unit
strength and scale 1. Reversing the same memory increased it to 84.37 km/s.
The useful term therefore carries the inner field outward; it does not behave
like a simple outer boundary pushing inward.

Memory placement identifies where the leverage enters. Post-screen memory
gave 59.29 km/s, whereas pre-screen memory returned 69.64 km/s, essentially
the 69.63 km/s local control. The useful operation therefore remembers the
already screened excess rather than averaging a raw excess that is screened
again locally.

Scale created a continuous galaxy trend and a discrete lensing boundary:

| log-radius scale | galaxy RMSE | eight-start raw RMS | all held-out roots |
|---:|---:|---:|---|
| 0.25 | 64.13 km/s | 19.35 arcsec | yes |
| 1 | 59.19 km/s | 21.81 arcsec | no |
| 2 | 57.99 km/s | 23.73 arcsec | yes |
| 4 | 57.37 km/s | 28.37 arcsec | no |
| 8 | 56.79 km/s | 28.41 arcsec | no |

The nonmonotonic root pattern is why an apparently better galaxy score cannot
be promoted by itself. Scale 2 is the best complete-root cross-domain
compromise: its worse benchmark ratio is 5.43, compared with 6.39 for the
preceding directional compromise. The price is a worse bridge score
(0.201 dex) and raw-lensing error (23.73 arcsec).

The fixed-parameter development audit reached the same directional lesson
before the refit: longer inward-running memory improved galaxy transfer in
both base laws. That makes the effect less likely to be only a parameter-fit
artifact. It still is not a physical derivation. Forty-six of the 64 bridge
systems contain only one radial point and reduce exactly to the local law, so
the bridge weakly constrains this new degree of freedom.

### 13. Quantity transported by radial memory

The carrier investigation first screened 132 fixed-parameter variants, then
ran 58 full four-parameter bridge fits across all domains. Eighteen selected
settings received an eight-start raw-lensing replay. The exponent surface
contained 19 unique (p,q) points after exact duplicate settings were given
equal weight.

Measured baryonic profiles supplied the central regularity. Across 131 SPARC
galaxies, the median (s=d\ln g_N/d\ln r) is (-1.557). Across the 18 CLASH
systems with multiple radial points, it is (-0.448); the probability that a
random CLASH slope is higher than a random SPARC slope is 0.873. Consequently,
negative (p) weights galaxy outskirts much more strongly than cluster
outskirts, whereas positive (q) weights both outward.

On the frozen power surface, the Spearman correlation between SPARC effective
power and galaxy RMSE is (-0.875). The correlation between CLASH effective
power and bridge RMSE is 0.996. These correlations do not prove the carrier
law, but they compress a two-parameter numerical sweep into a simple empirical
statement: **the response follows how strongly the remembered source is
weighted outward in that domain**.

The best stable compromise is (p=-0.5,q=1.5): 43.89 km/s on SPARC, 0.291 dex
on the bridge, and 28.53 arcsec on raw lensing, with all held-out roots found.
Its worse reference ratio is 4.109, 24.3% below the preceding stable ratio of
5.429. The opposite direction (p=0.5,q=0) obtains 18.91 arcsec on lensing but
worsens SPARC to 78.16 km/s. At (p=1,q=0), lensing reaches 16.64 arcsec but
loses a root and SPARC rises to 117.11 km/s. The exponent is therefore a strong
galaxy-versus-lensing lever, not yet a reconciliation.

The approximately CLASH-neutral setting $p=-1,q=-0.5$ is a more balanced
alternative. It scores 47.70 km/s on SPARC, 0.204 dex on the bridge, and 25.50
arcsec in the eight-start lens test with every root recovered. That is a 7.1%
raw-lensing improvement over baryons while retaining most of the galaxy gain.
Its worse reference ratio is 4.47 rather than 4.11, so which candidate is
preferred depends on whether the objective is the minimax benchmark or
improvement in both domains relative to local/baryonic controls.

A separate 35-point fixed-parameter audit varied $p$ from -0.5 to -1.5 and
$q$ in quarter steps from -1 to 0.5. All settings passed the current Solar
proxies. It reproduced the continuous tradeoff: increasing $q$ improves
galaxy error and worsens the bridge, while moving toward CLASH neutrality
preserves more cluster response. Those rows are range-selection evidence only;
they were not promoted without full refitting and lens tests.

Channel placement also mattered. Fractional memory on the combined screened
response produced the galaxy improvement. Moving the same operation to RG or
Sigma separately returned close to the local-law galaxy score. RG-specific
added-acceleration and speed-squared memory produced four Solar-proxy failures,
while their combined-channel counterparts passed. This is direct evidence that
the transported channel is a meaningful structural choice.

Raw-lensing topology remained fragile. Seven candidates reversed root status
between two and eight geometry starts. Exact duplicate settings were
internally repeatable: four (p=0,q=1.5) runs spanned only 0.0047 arcsec, and
the two (p=-1,q=1.5) runs spanned 0.0022 arcsec. The reversals are therefore
associated with lens branches rather than ordinary run-to-run score noise.

### 14. Local-slope adaptive carrier

This stage tested 36 fixed-parameter variations and 29 full bridge-refitted
variations. Fifteen settings received eight-start raw-lensing replays. The
gate strength, slope pivot, sharpness, and both carrier endpoints were varied
in small universal steps. Every setting passed the current Solar proxies.

The central result is that **where the interpolation occurs matters more than
the fact that it is smooth**. At sharpness 0.25, the median gate changes by
only 0.040 across a SPARC galaxy and 0.024 across a resolved CLASH profile.
That setting behaves mainly like one global intermediate carrier. Its best
stable version scores 44.78 km/s on galaxies and 28.63 arcsec on lensing, for
a worse-reference ratio of 4.193. This is 2.0% worse than the fixed-carrier
ratio of 4.109.

Increasing sharpness creates a much larger response, but in the wrong joint
direction. Galaxy RMSE rises from 44.96 km/s at (k=0.25) to 319.90 km/s at
(k=16). Bridge RMSE improves only from 0.259 to 0.200 dex. The (k=16)
formula produces a stable 22.89 arcsec raw-lensing score, 16.6% below baryons,
but its galaxy error is 29.95 times the fixed-RAR reference.

The profile-geometry audit explains the numerical leverage without treating it
as an assumed theoretical objection. For the fixed steep endpoint, the median
SPARC 90th-percentile absolute log-source derivative is 2.45. At (k=4), the
adaptive value rises to 4.60, its across-galaxy 90th percentile reaches 17.38,
and individual galaxies exceed 90. The corresponding CLASH median remains
0.87. Point-dependent powers therefore create source gradients much larger
than either fixed endpoint specifically in structured galaxy profiles.

The per-galaxy audit found that the hard-switch penalty is strongest in
gas-rich and low-stellar-mass systems. At (k=16), error growth has Spearman
correlation (+0.464) with gas fraction and approximately (-0.40) with
stellar mass and luminosity. This is especially damaging because those are the
systems where a universal modified-gravity relation must remain reliable.

Five exact duplicate settings were kept deliberately. Their bridge RMSE spans
only (2.24\times10^{-5}) dex, but their bridge-only fits span 2.73 dex in
critical density and 22.84 in (Q). Their eight-start lens RMS values span
28.94--32.58 arcsec, and every duplicate loses a held-out root. This establishes
that a nearly identical bridge score does not identify one universal parameter
branch or one viable lens mapping.

The tested pointwise exponent interpolation is therefore disfavored as a
unifying mechanism over this range. The parent slope-dependent idea remains
open. The next distinct forms should hold the exponents constant within a
profile or blend bounded memory outputs, avoiding derivatives of the exponents
themselves.

### 15. Profile-level and bounded slope responses

This stage tested the two alternatives identified by the preceding failure.
It added four controlled modes: pointwise exponent interpolation as the exact
control; one exponent pair selected from a single log-linear slope for the
whole profile; a bounded blend of two completed memory responses using that
whole-profile slope; and the same bounded response blend using a pointwise
slope. The completed-response construction first evaluates both carriers and
then mixes their predicted fractional forces, so every output is between the
two endpoint outputs at that radius.

The result changes the interpretation of the prior failure. With identical
gravity parameters and the same carrier endpoints, the pointwise exponent
mode scores 171.47 km/s on SPARC. The profile-exponent, profile-response, and
point-response modes score 45.08, 45.10, and 45.06 km/s respectively. Removing
point-dependent exponent derivatives therefore cuts the fixed-parameter
galaxy error by 73.7%. The slope premise survives this test; the specific
operation of differentiating a changing exponent does not.

The full experiment contains 43 fixed-parameter variations, 27 universal
bridge refits, and 15 eight-start raw-lens replays. All settings pass the
current Solar proxies. The best derivative-safe cross-domain compromise is the
profile-response blend with memory log-scale 0.5: 44.17 km/s on SPARC, 0.181
dex on the bridge, and 24.62 arcsec on robust raw lensing with all roots. Its
worse-reference ratio is 4.136, only 0.64% above the prior fixed-carrier value
of 4.109, while its raw-lens error is 13.7% lower.

The strongest raw-lensing setting uses a half-strength profile-response gate.
It scores 22.63 arcsec with every root, 17.5% below baryons and 19.7% below
fixed simple MOND on this raw four-cluster test. It remains 2.50 times the
compact-halo error, and its 45.83 km/s galaxy error is 4.29 times the fixed-RAR
reference. Thus the bounded blend is a real lensing lever without the former
galaxy catastrophe, but it does not yet close the galaxy gap.

The whole-profile slope also exposes a new measurable limitation. Median
measured slopes are -1.56 for 131 SPARC galaxies and -0.45 for 18 CLASH
clusters, a useful separation. For the four raw-lens clusters, however, the
median slope changes from -0.54 on measured baryonic anchors to -1.24 on the
0.1--1,000,000 kpc ray-integration grid. At gate sharpness 4, the median
galaxy-like weight consequently changes from 0.14 to 0.71. A global slope is
therefore sensitive to extrapolation range; the pointwise completed-response
mode avoids that particular dependency.

Six candidates reverse held-out root status between two and eight geometry
starts, and every full refit reaches at least one universal-parameter bound.
The appropriate next experiment is a fine interaction grid around memory
scale 0.5, gate strength 0.5, and nearby pivots, followed by a smoothed local
slope whose averaging length is itself universal. These are empirical
follow-ups, not literature-based rejections or confirmations.

### 16. Fine slope-response neighborhood and exact-formula repeatability

The fine investigation added 42 universal bridge refits and 37 eight-start
raw-lens replays across the local scale, strength, and pivot neighborhood.
Every formula continued to use one setting for every galaxy and cluster, and
all 42 passed the current Solar proxies.

Two small structural changes had the largest stable effect: moving the
whole-profile slope pivot from -1 toward 0, and shortening the memory length
from two to approximately 0.8 natural-log radius units. The best single
development branch, at pivot 0 and memory scale 0.8, scores 42.32 km/s on
SPARC, 0.211 dex on the bridge, and 28.85 arcsec on the eight-start raw lens
test with every root. Its worse-reference ratio is 3.962, 3.58% below the
preceding fixed-carrier value of 4.109.

That best branch was then treated as a numerical hypothesis rather than a
result to trust. The exact structural formula was bridge-refitted five times
with independent optimizer seeds and each fit received an eight-start lens
replay. Four of five fits recovered every lens root. Among those four, the
median SPARC error is 42.53 km/s, the median raw-lens error is 28.97 arcsec,
and the median worse-reference ratio is 3.982. The repeatability-adjusted gain
over the prior fixed carrier is therefore 3.09%, slightly smaller than the
optimistic single-branch gain.

The predictions are more repeatable than the fitted parameters. Across the
five identical formulas, the fitted critical density spans 2.45 dex and
\(Q\) spans 7.60--16.99, while the Sigma amplitude reaches its upper bound of
30 every time. This means the response direction is partly identified but its
decomposition into the four bridge parameters is not. Fourteen settings also
reverse root status between two and eight lens starts, confirming that a
shallow geometry search is not a reliable pass/fail test.

The data therefore support shorter memory and a pivot near zero as useful
local directions in this formula, but do not establish the lucky 3.962 branch
as uniquely reproducible. The conservative value carried forward is 3.982,
and the next experiment replaces the global slope with a universally smoothed
local slope to remove radial-extrapolation dependence without returning to the
unstable pointwise exponent law.

### 17. Universally smoothed local slope

The smoothed-local experiment evaluated 32 fixed variants, 39 universal
bridge refits, and 25 eight-start raw-lens replays. The local slope at each
radius was estimated by a Gaussian-weighted linear fit in log radius, with one
universal bandwidth. Narrow bandwidths up to 0.5 were nearly invariant to the
artificial outer cutoff used by raw ray integration. Broad bandwidths of two
to four log-radius units separated the SPARC and CLASH samples more strongly,
but reintroduced cutoff dependence.

Within the refitted grid, response strength and memory length moved the data
far more than smoothing bandwidth. The best stable observed point moved the
slope pivot to 1.25 and scored 41.86 km/s on SPARC and 28.42 arcsec on the
eight-start raw-lens test, for a ratio of 3.919. A half-strength gate gave the
best raw score, 26.96 arcsec, but worsened galaxies to 44.78 km/s. All 39
refits reached at least one bridge-parameter bound, and exact duplicates again
occupied different parameter branches.

The important observation was that the best pivot lay at the upper boundary.
A fixed-parameter extension showed the galaxy score continuing to improve as
the pivot rose from 1.25 to approximately four, then plateauing. At that point
the slope gate selects the steep endpoint almost everywhere. This turned the
next experiment into a mechanism control: compare the finite slope gate to an
exact endpoint formula with the slope term removed.

### 18. Finite slope pivot versus the exact endpoint

The endpoint control added 23 universal refits and replayed all 23 with eight
lens-geometry starts. It included finite pivots from 1.25 to 6, four smoothing
bandwidths, five memory lengths, five exact endpoint memory lengths, and five
independent refits of the exact scale-0.8 endpoint.

All five exact endpoint refits retained every raw-lens root. Their eight-start
lensing scores span only 0.00019 arcsec around 28.205 arcsec, while their
galaxy scores span 0.348 km/s. The fitted critical density spans 2.51 dex and
\(Q\) spans 11.98--21.97; the Sigma amplitude reaches its upper bound of
30 in every copy. Predictions are therefore much more repeatable than the
parameter decomposition.

The saturated finite-pivot prediction at pivot 6 differs from the median exact
endpoint galaxy score by only \(6.7\times10^{-7}\) km/s. Its raw score is 0.21
arcsec worse, so the exact endpoint is at least as good on these data. The
empirical conclusion is narrow but clear: the improvement in this formula is
an endpoint-response effect, not identified evidence for local-slope physics.
Other slope-dependent laws remain untested rather than rejected.

The single best endpoint optimizer branch gives ratio 3.901, but the other
four copies give the representative ratio 3.934. Memory scale remains the
largest structural lever and crosses a lens-root boundary at scale 0.6. Two
settings even reverse from complete roots at two starts to incomplete roots
at eight starts, so root status remains an optimizer-stability diagnostic and
not a theory-wide rejection criterion.

### 19. Endpoint source power and memory

The next stage first screened 134 fixed-parameter rows, then ran 66 universal
refits and eight-start raw-lensing replays for all 66. Fifty-three retained
complete roots. The transported source was

\[
X=F\left(\frac{g_N}{g_{\rm ref}}\right)^p
  \left(\frac{r}{\mathrm{kpc}}\right)^q,
\]

with an inner-to-outer running average in \(\ln r\). For a local profile slope
\(s=d\ln g_N/d\ln r\), its effective radial power is \(e=q+ps\). The measured
median slopes differ substantially: \(s=-1.557\) in SPARC and \(-0.448\) in
CLASH. One universal \((p,q)\) pair can therefore produce different radial
responses without an explicit galaxy/cluster label.

Controlled paths holding the CLASH effective power approximately fixed moved
the cluster bridge very little while changing galaxies; paths holding the
SPARC effective power fixed did the converse more strongly. On normalized
median spans, memory strength was the largest galaxy lever, \(q\) the largest
raw-lensing lever, memory length the largest bridge lever, and \(p\) a smaller
but cleaner differential lever. This is an empirical coordinate system for
future formulas, not proof that the microscopic law must be a power law.

### 20. Bracketed short memory and the higher-power ridge

A 68-row fixed audit bracketed the useful memory scale, and 47 universal
refits plus 47 eight-start replays refined the neighborhood. Forty-five rows
retained every raw-lens root. The best stable observed row reached 38.21 km/s
on galaxies, 0.204 dex on the cluster bridge, and 28.50 arcsec on raw lensing.
The best stable raw row reached 27.74 arcsec while retaining 39.10 km/s on
galaxies.

Within the fully refitted \(q=2.5\)--3.5 surface, larger \(q\) improved the
galaxy score monotonically, so that surface alone did not bracket the radial
power. Two additional fixed-parameter audits extended to \(q=8\). They found
the galaxy response turning over near \(q=4.5\)--5.5, while the best \(p\)
shifted along a shallow ridge from negative toward positive values. Those
fixed audits disfavored indefinitely increasing \(q\) only at fixed \(p\).
The next stage followed the moving \((p,q)\) ridge and showed that this first
turnover was not a bracket on the joint response.

Seventeen rows reversed lens-root status between two and eight starts. Most
reversals recovered missing roots, while two lost roots after the deeper
search selected a different parameter branch. Consequently, a missing root
in one shallow optimization is evidence against that fitted branch, not a
reason to discard the underlying new-physics family.

### 21. Moving high-power ridge and matched effective-power paths

The high-power follow-up used five fixed audits totaling 181 rows to avoid
mistaking another grid boundary for an optimum. A fixed-\(p\) scan first
appeared to turn over near \(q=5\), but the best \(p\) moved systematically
with \(q\). Following that moving ridge continued the galaxy improvement
through \(q=8\). A second extension and constant-SPARC-effective-power stress
test finally bracketed it near \(q=9\)--10; scores then worsened through
\(q=20\).

Forty formulas were frozen before new raw-lensing inspection. They included
15 local ridge perturbations, nine matched effective-power controls, memory-
scale and memory-strength sweeps, a prior endpoint control, and exact repeats.
All 40 passed the Solar proxies. After eight-start lens replays, 37 retained
every root; one shallow root failure was recovered, while memory scales 0.6
and 0.7 and memory strength 0.6 remained incomplete.

The matched paths expose a useful universal relationship. At fixed median
SPARC effective power, changing \((p,q)\) across \(q=8,10,12\) moves galaxies
by only 0.14--0.24 km/s and robust raw lensing by 0.07 arcsec. At fixed median
CLASH effective power, the bridge moves by less than \(10^{-4}\) dex while
galaxies move by 6.40 km/s and raw lensing by 0.46 arcsec. Thus
\(e=q+p\,d\ln g_N/d\ln r\) captures much of the local response, and the
different measured profile slopes supply real cross-domain leverage without
an object-class label.

The most impactful control remains memory strength: its sweep spans 32.04
km/s on galaxies and 10.90 arcsec on raw lensing. Memory length spans 2.50
km/s, 0.041 dex on the bridge, and 1.98 arcsec on raw lensing. The exponents
select which radial information is transported; strength and length determine
how much of that information survives and over what range. These are coupled
empirical coordinates, not independently measured physical constants.

### 22. Balanced exponent-memory interaction factorial

The interaction stage froze a complete \(3^4=81\)-cell design before new
lensing inspection. It crossed \(q=9,9.5,10\), median-SPARC effective powers
5.3, 5.65, and 6, memory lengths 0.35, 0.4, and 0.45, and strengths 0.85,
0.925, and 1. Four extra copies of the prior best setting brought the stage to
85 universal refits and 85 eight-start raw-lensing replays. Every row passed
the Solar proxies.

Because the design is balanced, classical orthogonal sums of squares measure
local main effects and pair interactions without conflating one coordinate
with another. Galaxy error is primarily a memory-strength response: strength
accounts for 60.5% of sampled variation, effective power 22.6%, memory length
9.3%, and \(q\) 1.2%. The largest interaction is strength by length at 4.1%.
For the cluster bridge, memory length accounts for 85.2% and strength 14.1%;
the exponents have essentially no local bridge leverage after refitting.
Solar-proxy variation is likewise dominated by strength, 83.3%, and length,
13.5%.

Raw lensing does not reduce to one knob. The largest main effect, strength,
accounts for only 4.9% of diagnostic RMS variation. Strength by length is
9.9%, effective power by strength 8.0%, and \(q\) by strength 6.1%; pairwise
terms plus higher-order structure dominate. This explains why one-dimensional
searches repeatedly fixed galaxies while breaking lensing, or vice versa.
Raw RMS values from the one incomplete-root cell remain diagnostic only and
are excluded from successful-fit ranking.

The best stable cell scores 37.12 km/s on galaxies, 0.191 dex on the bridge,
and 27.94 arcsec on raw lensing, ratio 3.475. The improvement over the prior
best is just 0.34%. Thirteen factorial cells are within 1% of this ratio and
span every tested \(q\) and memory length. The data therefore identify a broad
plateau with effective power at least 5.65 and strength at least 0.925, not a
unique \(q\), \(p\), or memory length.

Only one of 81 factorial cells remains root-incomplete after eight starts,
whereas 19 appeared incomplete after two starts. All 18 reversals recovered
roots. Five exact repeats of the prior best formula have a median ratio 3.487,
raw span 0.064 arcsec, and galaxy span below \(10^{-10}\) km/s, even though
\(Q\) spans 19.49 and the fitted critical density spans 1.78 dex. Predictions
are repeatable; the universal parameter decomposition is not.

### 23. Tidal-channel gate memory

The next frozen 24-row factorial moved memory from the force response to the
bounded middle-tidal-eigenvalue channel gate. It crossed two gate orientations,
inner-to-outer memory lengths of 0.1, 0.35, and 1 log-radius units, strengths of
0, 0.5, and 1, and an outer-to-inner direction control at length 0.35. Every
row was replayed with eight raw-lens geometry starts; 18 of 24 retained every
held-out image root. All rows passed the current Solar proxies.

The best stable cross-domain row uses the galaxy-favored `cluster_low`
orientation, inner-to-outer memory, length 0.35, and full strength. It scores
68.83 km/s on SPARC, 0.1148 dex on the bridge, and 17.69 arcsec on raw lensing,
for a worse-reference ratio of 6.44. This is substantially worse than the
existing global 3.475 compromise and is not a new program winner. Relative to
the matching zero-memory orientation, the memory term improves galaxy transfer
and raw lensing together, so it is a real local effect, just not a large enough
one.

The balanced decomposition shows why. Gate orientation accounts for 87.2% of
galaxy variation, 99.8% of bridge variation, 80.0% of raw-lensing variation,
and 92.5% of Solar-proxy variation on this grid. Memory strength accounts for
only 1.9%, less than 0.1%, 3.7%, and 0.9%, respectively. The best raw-lensing
row is the opposite `cluster_high` orientation with memory strength zero:
8.21 arcsec, slightly better than the 9.05-arcsec compact-halo equal-system
reference, but with a 288.88 km/s galaxy error and a pooled reduced chi-square
of 140.69. The data identify the already-known orientation conflict more
strongly than a persistent-gate mechanism.

This result has a strict claim boundary. Under the spherical cluster closure,
the middle-to-maximum tidal ratio is only a nonlinear expression of
local-to-mean density. Independent directional information exists only on the
axisymmetric SPARC side. The test therefore motivates a registered-map
calculation; it does not establish that real gravity remembers a tidal state.

### 24. Nonmonotonic tidal-gate topology

The following frozen 22-row stage tested whether the opposing orientations
were an artifact of requiring a one-sided monotonic gate. Sixteen cells crossed
exact complementary middle-band and two-tail topologies, lower pivots 0.45 and
0.60, upper pivots 0.80 and 0.90, and sharpness 5 and 10. Four matched
monotonic controls and two constant-weight controls completed the design. Every
row was replayed with eight raw-lens geometry starts; 15 retained every held-out
root, including 11 nonmonotonic rows. All rows passed the Solar proxies.

No nonmonotonic cell beat both matched monotonic endpoints. The best stable
cross-domain row is still the monotonic galaxy-favored orientation at sharpness
10: 68.24 km/s on SPARC, 0.1191 dex on the bridge, and 17.79 arcsec on robust
raw lensing, for a worse-reference ratio of 6.389. The best stable raw row is
the opposite monotonic orientation at sharpness 5: 8.40 arcsec, but with a
288.89 km/s galaxy error. This stage therefore does not improve the global
3.475 compromise.

On the balanced nonmonotonic grid, topology explains 60.0% of sampled galaxy
variation and 34.7% of robust raw-lensing variation. Sharpness explains 42.0%
of bridge variation, while the upper pivot explains 19.7% of raw-lensing
variation. These are large response changes, but they move the domains in a
correlated tradeoff rather than creating a common optimum. Two rows also
reversed root status between the two- and eight-start searches, reinforcing
that topology scores require robust geometry optimization.

The exact complement construction makes the negative result narrower and
cleaner: the conflict is not merely caused by monotonic ordering or by changing
the total gate normalization. It remains possible that the measured coordinate
is useful when applied to memory, redistribution direction, or a common tensor
map rather than only to an RG amplitude ceiling. The cluster-side coordinate
is still a spherical density-ratio proxy, so this does not reject a genuinely
nonmonotonic directional field.

### 25. Tidal placement on channel ceilings versus radial memory

The next frozen 27-row stage separated a distinction that earlier tests had
confounded: a measured tidal coordinate can decide where an RG ceiling acts,
where radial memory acts, both, or neither. The 16-cell balanced factorial
crossed high- versus low-tidal cap orientation, high- versus low-tidal memory
orientation, sharpness 5 versus 10, and maximum memory strength 0.5 versus 1.
Four memory-only, four cap-only, and three global-memory controls completed the
design. All 27 universal bridge fits were transferred unchanged to SPARC, raw
lensing, and Solar proxies, then replayed with eight lens-geometry starts.
Sixteen rows retained every held-out image root, and every row passed the Solar
proxies.

The pointwise memory rule was

\[
F_{{\rm eff},i}=F_i+\mu_{\max}w_i(M_i-F_i)
\]

for the high-tidal orientation and the same expression with (1-w_i) for the
low-tidal orientation. Independent switches kept the RG and Sigma ceilings
global in the memory-only controls. The generalized carrier was frozen at the
preceding best endpoint, (p=1.927395), (q=9), and memory length 0.35, so
the stage measures placement rather than reopening the exponent ridge.

The ungated full-memory control remains the best stable row: 37.14 km/s on
SPARC, 0.1913 dex on the bridge, and 27.96 arcsec on robust raw lensing, for a
worse-reference ratio of 3.477. It closely repeats the preceding global winner
at 37.12 km/s and 27.94 arcsec but does not improve it. No placement row
improves both domains over that control, beats the prior global ratio, or meets
both the fixed-RAR and compact-halo references.

Cap orientation is the dominant refitted galaxy control, accounting for 73.2%
of sampled variation and spanning 120.72 km/s. It also accounts for 37.0% of
bridge variation, while sharpness accounts for 32.4% and their interaction for
27.7%. At the locked baseline parameters the dominance is even clearer: cap
orientation explains 85.9% of galaxy and 97.9% of bridge variation. The bridge
refit therefore compensates for much, but not all, cap-placement leverage.

Raw lensing is controlled differently. Maximum memory strength is its largest
main effect at 29.3%, but memory-orientation by sharpness and
memory-orientation by strength account for 14.9% and 10.9%. The best stable
raw row in this stage gates both the cap and memory toward the high-tidal side,
reaching 17.78 arcsec while over-accelerating galaxies to 243.81 km/s. The best
stable both-placement compromise instead gates both mechanisms toward the
low-tidal side at sharpness 10 and full strength: 65.01 km/s, 0.1098 dex, and
18.82 arcsec. It is a meaningful local tradeoff but has a ratio of 6.09.

The deeper root search changes the scientific conclusion for the most tempting
memory-only row. Low-tidal memory at sharpness 10 initially retained all roots
at 45.61 km/s and 19.44 arcsec, but the eight-start optimization selected a
different branch that loses a held-out root. Conversely, high-tidal memory at
sharpness 5 recovers its shallow missing root but remains worse than global
memory at 56.69 km/s and 27.30 arcsec. Placement affects lens topology, not
only smooth error.

The result narrows the next target. The useful cross-domain distinction is not
obtained by making the same scalar coordinate decide how much memory survives.
Further local pivot or sharpness tuning would polish a tradeoff whose direction
is already measured. A stronger next test must let registered nonspherical
baryonic structure change redistribution direction or spatial derivatives,
rather than only a scalar cap or scalar memory amplitude.

### 26. Conservative symmetric profile diffusion

The frozen 38-row stage tested a spatial second derivative without adding a
new net carrier normalization. Twenty-seven cells crossed three carriers
(fractional excess, added acceleration, and circular-speed-squared), three
log-radius diffusion lengths (0.15, 0.35, and 0.7), and three strengths (0.25,
0.5, and 1). Nine further cells applied fractional diffusion after the
preceding best one-sided-memory response. Exact local and one-sided-memory
controls completed the design. Every row received an eight-start lens replay;
28 retained every held-out image root, and every row passed the Solar proxies.

The local control scores 69.66 km/s on SPARC, 0.1763 dex on the bridge, and
18.95 arcsec on robust raw lensing. The strongest complete-root raw diffusion
transports added acceleration over 0.7 log radius at full strength. It reaches
18.65 arcsec, an improvement of 0.30 arcsec, but worsens galaxies by 8.74 km/s
to 78.40 km/s. The cleanest low-cost row transports circular-speed-squared over
0.15 log radius at half strength: it improves raw RMS by 0.089 arcsec while
changing galaxy RMSE by only +0.022 km/s. That small effect is worth retaining
as a control, but it is far too small and development-sample-dependent to be a
cross-domain advance.

The pure-diffusion factorial makes the leverage explicit. Carrier, scale, and
strength explain 23.3%, 20.6%, and 19.1% of sampled galaxy variation. Carrier
explains 38.4% of bridge variation and carrier-by-scale another 32.0%. Raw
lensing is not controlled by one main effect: carrier-by-scale explains 23.5%,
carrier-by-strength 14.8%, and higher-order structure 27.3%. The raw variance
decomposition is diagnostic because the longest, strongest speed-squared row
loses a held-out root.

Adding diffusion after the best one-sided memory has a larger galaxy effect.
Full diffusion at scale 0.35 reduces galaxy RMSE from 37.11 to 30.87 km/s.
Memory-diffusion strength explains 78.3% of the sampled galaxy variation, while
scale explains 60.7% of its small bridge variation. However, every one of the
nine memory-plus-diffusion rows loses a held-out image root. The exact
zero-diffusion memory control recovers every root at 27.94 arcsec and remains
the best complete-root cross-domain row.

All 38 bridge fits place at least one of \((\epsilon_0,\rho_c,Q,B)\) on a
declared boundary. Symmetric profile curvature is therefore a measurable
response lever, but it reinforces rather than resolves the established
anti-galaxy/lensing direction. The next directional test still requires
registered nonspherical gas, BCG, ICL, member, and galaxy maps.

## What these tests do not establish

- The CLASH bridge uses NFW-deprojected published profiles. It is useful for
  ranking formulas but is not a theory-neutral measurement of the field.
- The SPARC stellar nuisance parameters were frozen from the RAR analysis.
  That makes transfer strict, but this was not a joint nuisance refit.
- Raw lensing used a zero-slip, pseudo-elliptical construction based on
  spherical interpolated profiles. It is not yet a covariant ray-tracing
  implementation of the proposed new physics.
- The Cassini comparison is a force-fraction proxy with \(\gamma=1\), and the
  Mercury calculation is a first-order weak-field diagnostic.
- The stages were selected sequentially after viewing earlier stages. They are
  sensitivity experiments, not a preregistered final holdout claim.
- The mixed directional stage compares an axisymmetric SPARC tidal closure
  with a spherical bridge/lensing closure. The common-spherical control
  removes that inconsistency only by imposing a poor approximation on flat
  galaxies; neither control replaces registered baryonic maps.
- Radial memory is a phenomenological ordered-profile response, not a derived
  claim that gravity literally propagates radially. Most BCG bridge systems
  have only one point and cannot identify its scale or direction.
- Tidal-gate memory is also a phenomenological proxy. It remembers a bounded
  classifier, not force, and mixes an axisymmetric SPARC closure with spherical
  cluster and Solar closures. It is not a common tensor-map solution.
- Tidal-gated radial memory uses that same mixed closure and a frozen endpoint
  carrier. Its failure disfavors scalar placement on that carrier, not a
  spatially nonlocal field equation or memory constructed from registered maps.
- Conservative profile diffusion preserves the transported radial-carrier
  integral, but it remains a one-dimensional no-flux proxy. Its failure does
  not reject anisotropic redistribution or a common map-based tensor equation.
- The carrier correlations use measured log-linear profile slopes and a frozen
  exploratory exponent surface. They identify leverage, not a discovery
  significance or a unique microscopic interpretation.
- A failed carrier exponent or channel only disfavors that equation over the
  tested range; it does not reject history-dependent or nonlocal gravity.
- The smoothed-local slope is robust to the raw radial cutoff only for narrow
  bandwidths. The high-pivot optimum saturates to the exact endpoint, so these
  data do not identify a nontrivial slope transition.
- Local baryonic derivatives mix real structure, measurement noise, and
  interpolation choices. The current data do not isolate those contributions.
- Nothing here establishes that this formula is novel in the literature or
  derives it from an action principle.

## Next data-driven formula experiments

The next changes should attack the measured conflict rather than add another
globally interchangeable amplitude:

1. Retire purely scalar class-separation gates as the main selector. Mass and
   density ratio distinguish the samples but choose mutually incompatible RG
   branches.
2. Replace both analytic closures with registered two-dimensional or
   three-dimensional baryonic maps in galaxies and clusters. The common
   spherical control proves that another scale-free spherical invariant cannot
   add independent information.
3. The common radial second-derivative control is now complete. Apply the next
   directional coordinate to a registered nonspherical redistribution tensor;
   the lens-root failures show that radial field shape alone is insufficient.
4. Use registered two-dimensional cluster maps containing gas, BCG,
   intracluster light, and member galaxies, then ray-trace the predicted
   potential directly.
5. Use galaxy stellar and gas maps with vertical scale information so the same
   tensor or nonlocal response can be tested across flat, bulge-dominated, and
   low-surface-brightness systems.
6. Replace the phenomenological radial running average with a field equation
   or action whose static solution produces a measurable history or spatial-
   derivative term. Test it first on resolved BCG and cluster baryon profiles,
   because single-point bridge systems cannot constrain memory.
7. Freeze the formula after these development data, then evaluate untouched
   galaxies and clusters. A universal setting only becomes persuasive when the
   formula itself is also held fixed.
8. Stop locally polishing \(p\), \(q\), scalar gate pivots, scalar memory
   placement, and symmetric radial diffusion. The exponent ridge, placement,
   and diffusion stages improved no complete-root cross-domain formula. Carry
   only their measured strength/length controls into a map-based redistribution
   direction, where raw lensing shows unresolved interaction structure.

## Reproducibility artifacts

- `results/reopened_hybrid_program/combined_scores.csv` contains all 913 scored
  rows.
- `results/reopened_hybrid_program/program_summary.json` records the exact
  sensitivity differences, references, input hashes, and claim boundaries.
- `results/reopened_hybrid_channel_saturation/` and
  `results/reopened_hybrid_channel_saturation_fine/` contain the broad and
  fine channel sweeps.
- `results/reopened_hybrid_tidal_gate_memory/`,
  `results/reopened_hybrid_tidal_gate_memory_raw_robustness/`, and
  `results/reopened_hybrid_tidal_gate_memory_analysis/` contain the 24-row
  gate-memory factorial, all eight-start lens replays, and its balanced-effects
  analysis.
- `results/reopened_hybrid_tidal_gate_topology/`,
  `results/reopened_hybrid_tidal_gate_topology_raw_robustness/`, and
  `results/reopened_hybrid_tidal_gate_topology_analysis/` contain the 22-row
  exact band/tails topology test, all eight-start replays, and its balanced
  topology and endpoint-reconciliation analysis.
- `results/reopened_hybrid_tidal_memory_placement/`,
  `results/reopened_hybrid_tidal_memory_placement_raw_robustness/`, and
  `results/reopened_hybrid_tidal_memory_placement_analysis/` contain the
  27-row cap-versus-memory placement design, every eight-start replay, and the
  balanced main, interaction, root, parameter, and matched-control analysis.
- `results/reopened_hybrid_profile_diffusion/`,
  `results/reopened_hybrid_profile_diffusion_raw_robustness/`, and
  `results/reopened_hybrid_profile_diffusion_analysis/` contain the 38-row
  conservative diffusion design, every eight-start replay, joined scores, and
  the carrier/scale/strength variance decomposition.
- `results/reopened_hybrid_channel_raw_robustness/`,
  `results/reopened_hybrid_channel_fine_raw_robustness/`, and
  `results/reopened_hybrid_channel_low_rg_raw_robustness/` contain the
  eight-start channel replays.
- `results/reopened_geometry_indicator_audit/` contains the 195-system
  label-free indicator audit.
- `results/reopened_tidal_shape_indicator_audit/` contains the 1,084-point
  directional invariant audit.
- `results/reopened_hybrid_geometry_gate/` and
  `results/reopened_hybrid_geometry_gate_topology/` contain the scalar gate and
  independent-placement sweeps.
- `results/reopened_hybrid_tidal_shape_gate/` and
  `results/reopened_hybrid_tidal_shape_gate_raw_robustness/` contain the
  34-variant directional sweep and eight-start replay.
- `results/reopened_tidal_shape_common_spherical_audit/` and
  `results/reopened_spherical_tidal_identity/` contain the common-closure audit
  and the 1,084-point algebraic identity verification.
- `results/reopened_hybrid_tidal_shape_common_spherical/` and
  `results/reopened_hybrid_tidal_shape_common_spherical_adaptive/` contain the
  exact 34-formula closure control and 48 nonlinear reparameterizations.
- `results/reopened_radial_memory_transfer_audit/` contains the fixed-parameter
  development audit.
- `results/reopened_hybrid_radial_memory/` and
  `results/reopened_hybrid_radial_memory_raw_robustness/` contain the
  32-variant full sweep and eight-start replay.
- `results/reopened_hybrid_memory_carrier_audit/` contains the 132-variant
  fixed-parameter carrier screen.
- `results/reopened_profile_slope_audit/` contains the measured SPARC and
  CLASH profile-slope comparison.
- `results/reopened_hybrid_memory_carrier/` and
  `results/reopened_hybrid_memory_carrier_raw_robustness/` contain the
  58-variant full sweep and the first 14 eight-start lens replays.
- `results/reopened_hybrid_memory_carrier_slope_neutral_audit/` contains the
  corrected 35-point quarter-step fixed-parameter neighborhood.
- `results/reopened_hybrid_memory_carrier_slope_neutral_raw_robustness/`
  contains four additional eight-start lens replays around the measured
  CLASH-neutral direction.
- `results/reopened_hybrid_memory_carrier_analysis/` contains augmented scores,
  the deduplicated 19-point exponent surface, and the consolidated findings.
- `results/reopened_hybrid_slope_adaptive_carrier_audit/` contains the
  36-variant fixed-parameter slope-gate screen.
- `results/reopened_slope_gate_geometry_audit/` and
  `results/reopened_slope_gate_failure_modes/` contain the direct profile-
  gradient and per-galaxy morphology diagnostics.
- `results/reopened_hybrid_slope_adaptive_carrier/` and
  `results/reopened_hybrid_slope_adaptive_carrier_raw_robustness/` contain the
  29 full cross-domain fits and 15 eight-start lens replays.
- `results/reopened_hybrid_slope_adaptive_carrier_analysis/` contains the
  consolidated score, duplicate-branch, root, and claim-boundary analysis.
- `results/reopened_hybrid_slope_response_modes_audit/` contains the
  43-variant fixed-parameter comparison of all four slope-response modes.
- `results/reopened_slope_response_range_audit/` contains measured profile
  slopes and the raw-lens radial-cutoff sensitivity calculation.
- `results/reopened_hybrid_slope_response_modes/` and
  `results/reopened_hybrid_slope_response_modes_raw_robustness/` contain the
  27 universal refits and 15 eight-start raw-lensing replays.
- `results/reopened_hybrid_slope_response_modes_analysis/` contains the
  consolidated derivative-safe comparison and claim boundaries.
- `results/reopened_hybrid_slope_response_fine/`,
  `results/reopened_hybrid_slope_response_pivot_extension/`, and their raw
  robustness directories contain the 37 fine and pivot refits and 32
  eight-start lens replays.
- `results/reopened_hybrid_slope_response_best_repeatability/` and
  `results/reopened_hybrid_slope_response_best_repeatability_raw/` contain five
  independent refits and five eight-start replays of one exact structural
  formula.
- `results/reopened_hybrid_slope_response_fine_analysis/` consolidates all 42
  fine, pivot, and repeatability refits, including parameter spread and root
  reversals.
- `results/reopened_smoothed_slope_geometry_audit/`,
  `results/reopened_hybrid_smoothed_local_slope/`, and
  `results/reopened_hybrid_smoothed_local_slope_raw_robustness/` contain the
  local-bandwidth audit, 39 universal refits, and 25 eight-start replays.
- `results/reopened_hybrid_smoothed_local_slope_analysis/` consolidates the
  smoothing, pivot, strength, memory, and repeatability effects.
- `results/reopened_hybrid_smoothed_local_pivot_extension/` and
  `results/reopened_hybrid_smoothed_local_pivot_extension_raw_robustness/`
  contain the 23 finite-pivot and exact-endpoint comparisons and their full
  eight-start replay.
- `results/reopened_hybrid_smoothed_local_pivot_extension_analysis/` records
  the five endpoint repeats, parameter ranges, root reversals, and the direct
  slope-versus-endpoint comparison.
- `results/reopened_hybrid_endpoint_power_memory/` and its raw-robustness
  directory contain 66 universal source-power/memory refits and 66 eight-start
  replays; the companion analysis combines them with 134 fixed audit rows.
- `results/reopened_hybrid_endpoint_boundary_refinement/` and its
  raw-robustness directory contain the 47 short-memory/high-power refits and
  replays; the companion analysis records the five exact repeats and 17 root
  reversals.
- `results/reopened_hybrid_endpoint_high_q_audit/` and
  `results/reopened_hybrid_endpoint_high_q_p_bracket_audit/` contain the
  first fixed-parameter extensions; the moving-ridge and constant-power audit
  directories contain the corrections that bracket the correlated ridge.
- `results/reopened_hybrid_endpoint_high_q_ridge/` and its raw-robustness
  directory contain all 40 frozen full refits and eight-start replays.
- `results/reopened_hybrid_endpoint_high_q_ridge_analysis/` contains the
  matched-path comparisons, memory leverage, exact-repeat ranges, and claim
  boundaries.
- `results/reopened_hybrid_endpoint_interaction_factorial/` and its
  raw-robustness directory contain all 85 universal refits and eight-start
  replays.
- `results/reopened_hybrid_endpoint_interaction_factorial_analysis/` contains
  the balanced main/pair effect decomposition, root-completion response,
  near-optimal plateau, and exact-repeat analysis.
- The reusable formula implementation is
  `src/voidscreen/reopened_hybrids.py`.
- The main sweep and raw robustness runners are
  `scripts/run_reopened_hybrid_sensitivity.py` and
  `scripts/run_reopened_hybrid_raw_robustness.py`.

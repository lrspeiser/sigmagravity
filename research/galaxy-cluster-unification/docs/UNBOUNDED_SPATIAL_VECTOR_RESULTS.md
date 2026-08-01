# Mass-conserving spatial-vector lens test

## Question

The spherical multi-cluster test could not represent the fact that several
galaxies bend a light ray from different directions.  This test asks whether
the two locked curvature-running laws failed simply because those measured 2D
directions were omitted.

This is a structural diagnostic on already used systems, not an external
validation.  The Caminha et al. member catalogs provide positions and lens-model
magnitudes, but not complete stellar masses; numerical gas and intracluster-light
maps are also absent.

## Mass-conserving vector construction

For member positions $\boldsymbol\theta_i$ and light weights

$$
w_i={10^{-0.4m_i}\over\sum_j10^{-0.4m_j}},
$$

the softened thin-lens deflection template is

$$
\boldsymbol\alpha_{\rm mem}(\boldsymbol\theta)=
{D_{ls}\over D_s}\sum_i {4G w_iM_{b,\rm out}\over c^2D_l}
{\boldsymbol\theta-\boldsymbol\theta_i
\over |\boldsymbol\theta-\boldsymbol\theta_i|^2+s^2}.
$$

$M_{b,\rm out}=g_{\rm bar}(r_{\rm out})r_{\rm out}^2/G$ is the baryonic mass
at the outermost Tian anchor.  It is a normalization for the angular template,
not added mass.  At every radius the numerical circular mean
$\langle\boldsymbol\alpha_{\rm mem}\cdot\hat{\boldsymbol r}\rangle_\phi$ is
subtracted:

$$
\Delta\boldsymbol\alpha_{\rm mem}=
\boldsymbol\alpha_{\rm mem}-
\langle\boldsymbol\alpha_{\rm mem}\cdot\hat{\boldsymbol r}\rangle_\phi
\hat{\boldsymbol r}.
$$

The tested lens field is

$$
\boldsymbol\alpha=
\boldsymbol\alpha_{\rm locked,spherical}
+fD(r)\Delta\boldsymbol\alpha_{\rm mem}
+\boldsymbol\alpha_{\rm shear}.
$$

Two dressings were frozen:

- `GR_linear`: $D(r)=1$;
- `locked_running`: $D(r)=g_{\rm locked}(r)/g_{\rm bar}(r)$.

The full vector lens equation and its numerical Jacobian are used, so the
directional field and its local gradient tensor affect the image roots.  This
is nevertheless a thin-lens closure, not a covariant field equation.

The universal grid was $f=0,0.025,0.05,0.1,0.2,0.4,0.8$ and
$s=0.2,0.5,1,2,5,10$ arcsec.  Every point was screened using only training
images at the previous training geometry.  Four points per model and dressing
were refitted on training images, and the winning universal setting was then
scored on 11 untouched within-family images across four clusters.

## Predictive result

| variant | selected $f$ | $s$ (arcsec) | equal-system held-out RMS (arcsec) | change from spherical parent | compact-halo ratio |
|---|---:|---:|---:|---:|---:|
| additive $\alpha=10$, running-dressed | 0.025 | 0.2 | **18.216** | -0.3% | 2.01 |
| additive $\alpha=10$, GR-linear | 0.200 | 0.2 | 18.342 | -1.0% | 2.03 |
| curvature $p=2$, GR-linear | 0.100 | 0.2 | 18.688 | -0.3% | 2.07 |
| curvature $p=2$, running-dressed | 0.025 | 0.2 | 18.724 | -0.5% | 2.07 |
| compact one-halo comparator | -- | -- | **9.048** | -- | 1.00 |

A negative change means worse performance.  Every exact image root converged,
so this failure is not a root-finding artifact.  No variant passed any joint
advance path; geometry parameters also remained at boundaries in several
systems.

The selected correction was small compared with the failed field.  Its median
magnitude at an observed image was about 0.03--0.04 arcsec for MACS0329,
MACS0429, and MACS1115, and 0.16--0.32 arcsec for MACS1931.  The corresponding
locked spherical deflections were about 12--25 arcsec, while held-out errors
were 6--26 arcsec.

Per-system behavior is decisive:

| system | additive running-dressed (arcsec) | spherical parent | compact halo |
|---|---:|---:|---:|
| MACS0329 | 25.295 | 25.279 | 11.167 |
| MACS0429 | 6.206 | 6.122 | 1.797 |
| MACS1115 | 24.621 | 24.622 | 14.057 |
| MACS1931 | 6.534 | 6.093 | 1.401 |

The one-halo comparator is itself inadequate for MACS0329 and MACS1115, which
confirms that those systems need multi-component structure.  More importantly,
the spatial candidates remain far behind the halo in MACS0429 and MACS1931,
where the compact halo is already adequate.  Complex systems alone therefore
do not explain the failure.

## Post-failure number oracle

After the predictive result was locked, all 148 universal grid combinations
were ranked directly on held-out images at the fixed parent geometry.  This is
explicitly forbidden as prediction and can only diagnose missed number
selection.  Requiring every root to converge gives:

| variant | held-out oracle $f,s$ | oracle RMS | parent RMS |
|---|---:|---:|---:|
| additive, GR-linear | 0, 0.2 | 18.165 | 18.165 |
| additive, running-dressed | 0, 0.2 | 18.165 | 18.165 |
| $p=2$, running-dressed | 0.4, 5 | 18.337 | 18.630 |
| $p=2$, GR-linear | 0.8, 5 | 18.601 | 18.630 |

Thus the most favorable converged oracle improves $p=2$ by only 1.6% and still
has twice the compact-halo error.  Choosing a different number from the frozen
grid cannot rescue this construction.

## Common-aperture correction

The primary audit exposed one non-residual comparability issue: MACS1931 has a
600 kpc Tian anchor while the other three profiles stop at 200 kpc.  Its primary
member template was consequently normalized with $4.83\times10^{13}M_\odot$,
whereas its 200 kpc baryonic mass is $1.18\times10^{13}M_\odot$.  The
mass-conserving subtraction remained correct, but the universal fraction did
not describe a common aperture.

A second protocol therefore fixed every template to the exact shared 200 kpc
anchor without changing the grid, split, gravity laws, or optimization.  The
held-out images were already known, so this is a robustness control rather than
a new prediction.

| common-200-kpc variant | selected $f$ | $s$ (arcsec) | held-out RMS (arcsec) | roots |
|---|---:|---:|---:|---|
| additive, GR-linear | 0.40 | 0.5 | **18.210** | all |
| additive, running-dressed | 0.05 | 0.2 | 18.233 | all |
| $p=2$, GR-linear | 0.20 | 0.2 | 18.652 | all |
| $p=2$, running-dressed | 0.05 | 0.2 | failed aggregate | 10/11 |

The best common-aperture control remains slightly worse than the 18.165 arcsec
additive spherical parent and twice the 9.048 arcsec halo comparator.  The
unequal primary aperture therefore does not explain the null result.

## What was learned

1. Observed member directions are real lens structure, but adding their
   mass-conserving vector contrast is not the missing unifying variable for
   either locked radial law.
2. Training favored the sharpest allowed 0.2-arcsec member cores, but the small
   training gain did not transfer.  That is the signature of local adjustment,
   not a stable physical law.
3. The dominant error is the large-scale radial and multi-component field, not
   a small satellite-galaxy angular correction.
4. The available catalog cannot test the stronger physical claim that gas,
   BCG, ICL, and satellites source a nonlinear tensor field differently.  That
   requires component surface-density maps with a common astrometric frame.
5. The next defensible cluster calculation is a complete baryonic map or a
   simulation-based field solve.  Increasing $f$, adding a lensing-only
   multiplier, or choosing it separately per cluster would only hide the
   missing mass in a fitted amplitude.

## Reproducible artifacts

- `configs/unbounded_running_spatial_vector_protocol.json`
- `configs/unbounded_running_spatial_vector_common200_control.json`
- `scripts/run_unbounded_running_spatial_vector.py`
- `scripts/run_unbounded_running_spatial_vector_oracle.py`
- `src/voidscreen/spatial_lensing.py`
- `results/unbounded_running_spatial_vector/report.json`
- `results/unbounded_running_spatial_vector/member_audit.csv`
- `results/unbounded_running_spatial_vector/grid_screen.csv`
- `results/unbounded_running_spatial_vector/selection_refits.csv`
- `results/unbounded_running_spatial_vector/predictions.csv`
- `results/unbounded_running_spatial_vector/spatial_vector.png`
- `results/unbounded_running_spatial_vector_oracle/report.json`
- `results/unbounded_running_spatial_vector_oracle/oracle_grid.csv`
- `results/unbounded_running_spatial_vector_common200/report.json`
- `results/unbounded_running_spatial_vector_common200/member_audit.csv`

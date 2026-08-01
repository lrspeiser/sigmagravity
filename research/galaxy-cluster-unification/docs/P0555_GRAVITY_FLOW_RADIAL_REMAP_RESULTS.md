# P0555: gravity-flow radial remap and coherence results

## The question tested

The motivating picture was that baryonic matter produces the gravity, but the
field can be redirected through space before it is observed.  A dark-matter
halo center would then be an apparent destination or crossing point of gravity
originating in luminous matter, rather than a new gravitating substance.

This stage tested two separable consequences of that picture:

1. Does gravity fit the raw lens data better when individual baryon-to-baryon
   routes remain separate, or when neighboring baryonic vectors sum coherently?
2. Can a universal radial relocation of the extra response improve galaxy
   rotation, cluster lensing, and Solar-System limits at the same time?

No formula is promoted.  The useful outcome is a tighter description of what a
future field equation must and must not do.

## Formula 1: conservative radial flux remap

For an already-computed extra acceleration, define an effective enclosed flux

\[
F_X(<r) = a_X(r)r^2.
\]

The factor of \(G\) is omitted because it cancels.  Move a universal fraction
\(f\) to a new radial scale \(\lambda\):

\[
F_{X,\mathrm{new}}(<r)
=(1-f)F_X(<r)+fF_X(<r/\lambda),
\]

\[
g_{\mathrm{new}}(r)=g_b(r)+\frac{F_{X,\mathrm{new}}(<r)}{r^2}.
\]

- \(\lambda<1\): the extra response is compressed inward.
- \(\lambda>1\): the extra response is expanded outward.
- \(f=0\) or \(\lambda=1\): exactly the P0554 parent.
- No new acceleration amplitude is introduced.

The screen contained 49 fixed settings: the parent plus 48 combinations of
six route fractions and eight radial scales.  One setting was selected on 91
SPARC galaxies and 13 CLASH systems, then transferred to 40 galaxies, seven
CLASH systems, and raw RXJ1347 image coordinates.

## Constant-remap result

The discovery rule selected \(f=0.01,\lambda=0.8\).  It passed all analytic
Solar proxies, including a Mercury proxy of -1.744 mas/century.  Its changes
relative to the exact P0554 parent were:

| Test | Gain versus parent |
|---|---:|
| 91-galaxy discovery | +0.0029% |
| 13-cluster derived-lensing discovery | +0.0964% |
| 40-galaxy formula holdout | +0.1554% |
| 7-cluster derived-lensing formula holdout | -0.1677% |
| raw RXJ1347 pair holdout | -7.9750% |

Only one of the 48 non-parent settings improved both discovery metrics.  None
improved discovery galaxies, derived clusters, and raw RXJ1347 together.

The best setting for each individual domain pointed in conflicting directions:

| Domain | Best setting | Gain |
|---|---|---:|
| discovery galaxies | \(f=0.03,\lambda=1.2\), outward | +0.0616% |
| discovery derived clusters | \(f=0.1,\lambda=0.8\), inward | +0.5462% |
| raw RXJ1347 | \(f=0.03,\lambda=1.2\), outward | +10.6611% |

The apparently two-dimensional \((f,\lambda)\) grid was effectively only one
coordinate.  A quadratic in

\[
\epsilon=f\ln\lambda
\]

explained 99.82% of the discovery-galaxy response, 99.81% of the derived
cluster response, and numerically 100% of the RXJ1347 response.  Future tests
should therefore use one signed displacement parameter, not pretend that
\(f\) and \(\lambda\) are independently identified by these data.

## Five additional raw clusters

All 49 settings were transferred to RXJ2129, MACS0329, MACS0429, MACS1115,
and MACS1931 at archived ordinary geometry.  Sources were re-profiled on
training images, but no gravity or geometry parameter was refitted.

The frozen scalar-selected setting improved the four-parent-complete matched
RMS by only 0.0567%.  The best complete setting improved it by 0.2385%, while
worsening discovery galaxies by 0.2669% and raw RXJ1347 by 70.19%.

The preferred radial direction was not universal:

- MACS0329, MACS0429, and MACS1115 had better minima inward.
- RXJ2129, MACS1931, and RXJ1347 had better minima outward.
- MACS1931's parent failed one held-out root, so its RMS direction is
  descriptive and topology-sensitive.

## Formula 2: smooth potential-dependent direction

The object-level audit found that higher baryonic potential tended to prefer
the outward direction.  In 20 derived clusters, median potential depth had
Spearman \(\rho=0.589\) with an FDR-adjusted \(q=0.0227\).  Galaxies showed the
same sign much more weakly (\(\rho=0.196, q=0.120\)); the preregistered
cross-domain promotion gate therefore failed.

The lead was nevertheless tested with one new universal parameter:

\[
\ln\lambda(r)=A\tanh\left[\ln\left(\frac{\Phi_b(r)}{2\times10^{-6}}\right)\right],
\]

\[
F_{X,\mathrm{new}}(<r)=F_{X,\mathrm{parent}}(<r/\lambda(r)).
\]

The pivot is the existing P0554 dimensionless potential scale.  No object type
appears in the equation.  Nineteen amplitudes from -0.2 to +0.2 were tested.

No nonzero amplitude improved both discovery domains.  The least-bad
non-parent value, \(A=-0.005\), produced:

| Test | Gain versus parent |
|---|---:|
| discovery galaxies | +0.0171% |
| discovery derived clusters | -0.0257% |
| formula-holdout galaxies | -0.2874% |
| formula-holdout derived clusters | -0.3041% |
| six-system matched raw lenses | +0.0089% |

It remained Solar-safe, but it worsened RXJ1347 from 0.873 to 1.037 arcsec and
RXJ2129 from 1.096 to 1.198 arcsec.  The smooth potential transition is not a
universal radial solution.

## Coherent versus separately preserved routes

The separate baryonic-network experiment and its continuous interpolation gave
a stronger structural result than the radial tests:

- A coherent local-vector field beat the explicit branch network on the five
  discovery clusters.
- Replacing any fraction of the coherent field with separately preserved
  baryon-to-baryon branches worsened the five-system RMS monotonically.
- The same monotonic worsening transferred to the untouched RXJ1347 formula
  comparison.

This does not prove microscopic quantum coherence.  It says that the data favor
summing the neighboring baryonic contributions into a local field before the
response is propagated, rather than assigning persistent source-to-destination
gravity tubes.

## What has been learned

1. **The inferred halo locations are still useful backtracking targets.** Six
   of seven published halo components lie within 24 kpc of cataloged luminous
   baryons.  The difficult MACS1931 component remains the main long route.
2. **A global radial relocation is too simple.** Its best possible gains are
   sub-percent and its direction changes between systems and observables.
3. **Two radial parameters collapse to one.** The identified coordinate is a
   signed logarithmic displacement, \(\epsilon=f\ln\lambda\).
4. **Baryonic potential depth is a lead, not a universal law.** It predicts
   some sign variation in derived clusters but fails the galaxy cross-domain
   criterion and the direct transfer test.
5. **Raw image topology carries information absent from a radial acceleration
   curve.** A formula can improve a derived CLASH profile and worsen exact
   multiple-image roots, or the reverse.
6. **Coherent local vector summation is the most stable new structural clue.**
   Explicit branch identity and radial-only remapping are both disfavored.

## Tidal-field equation audit

A conservative two-dimensional constitutive response whose anisotropy is
generated by the baryonic tidal field is still the natural mathematical form:

\[
\nabla_i\!\left[\left(\delta^{ij}+s(I_1,I_2)Q^{ij}\right)
\nabla_j\Phi\right]=4\pi G\rho_b,
\]

where \(Q^{ij}\) is the trace-free, normalized Hessian of the baryonic
potential and \(s\) is a universal screened scalar of local invariants.

The repository audit found that this is **not an untested next step**.  The
member-only version was already implemented in both contrast-only and full
radial-plus-angular forms.  Both frozen tests selected zero coupling; nonzero
couplings either gave small system-dependent changes or destroyed required
image roots.  Therefore the member-only tensor is retired rather than being
rerun under new notation.

The form remains interesting because:

- it reduces to an ordinary scalar response in spherical or isolated fields;
- multiple baryonic vectors are summed locally before the anisotropic response;
- it can move angular convergence without forcing the same radial shift in
  every system;
- its divergence form provides a conservation audit;
- Solar suppression can be tested with the same invariant, not imposed by an
  object label.

The unresolved version requires a registered baryonic map containing hot gas,
BCG, diffuse intracluster light, and member galaxies.  Current member catalogs
do not supply that map.  Until such data are assembled, more member-only tensor
couplings would repeat an exhausted family.

## Reproducible artifacts

- `src/voidscreen/radial_route.py`
- `scripts/run_p0554_radial_flux_remap.py`
- `scripts/run_p0554_radial_flux_remap_multicluster_raw.py`
- `scripts/run_p0554_radial_flux_remap_forensics.py`
- `scripts/run_p0554_potential_transition_remap.py`
- `results/p0554_radial_flux_remap/`
- `results/p0554_radial_flux_remap_multicluster_raw/`
- `results/p0554_radial_flux_remap_forensics/`
- `results/p0554_potential_transition_remap/`

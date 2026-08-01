# Photon-sector investigation

## Outcome

The tested idea that galaxy rotation is largely an illusion created by photon
frequency or angular-propagation effects is not supported.

This does not exclude new physics. It changes the target: a viable extension
should alter the gravitational field that moves matter and bends light, while
retaining a single set of universal constants.

## Tests completed

| test | sample | primary result | conclusion |
|---|---:|---:|---|
| Milky Way raw-channel forward fit | 95 masers, 190 velocity channels | constant photon term 0.22 km/s; bootstrap 95% -5.88 to 6.03 | the required 30-61 km/s frequency illusion is absent |
| SPARC propagation-distance test | 139 galaxies | -0.23 km/s change per distance decade; bootstrap 95% -7.78 to 7.30 | an unsaturated path-accumulation law is absent |
| M33 angular-versus-H I rotation | two maser regions | H I minus astrometric motion -9.93 km/s; 95% -59.62 to 39.59 | direct angular rotation agrees with spectroscopic rotation |

The M33 comparison uses the published relative proper motion of
30.8 +/- 4 microarcsec/year, the H I-model transverse component of
106 +/- 20 km/s, and the independent TRGB distance of 794 +/- 23 kpc.
The underlying primary source is
[Brunthaler et al. (2005)](https://arxiv.org/abs/astro-ph/0503058).

## Why the first Milky Way diagnostic changed

The first source-by-source comparison divided measured velocities by circular
projection factors. That produced very large outliers and an apparent radial
trend. The replacement fit forward-modeled the measured proper-motion and
radial-velocity channels, fitted a shared rotation curve and Solar motion, and
never divided by a projection. The apparent constant photon amplitude then
collapsed from 7.4 km/s to 0.22 km/s and failed held-out validation.

## What remains logically open

1. A static scalar angular remapping multiplies both proper motion and annual
   parallax locally, so it cancels from transverse velocity. It cannot produce
   the needed effect.
2. An anisotropic or time-dependent optical metric can transform annual
   parallax and secular motion differently. Testing it requires raw epoch
   astrometry, parallax-ellipse residuals, and multi-frequency observations.
3. A path law that saturates before roughly 1 Mpc is not identifiable with the
   SPARC distance range. Such a law must still explain why Milky Way and M33
   angular motion agree with Doppler motion.
4. A coordinated theory that changes Doppler shifts, astrometry, and
   electromagnetic distance indicators remains logically possible, but it is
   no longer a minimal explanation.

## New-physics direction retained

The next candidate modifies the real gravitational field rather than only the
photon readout. In the weak-field limit, define

\[
\nabla_i\left(K^{ij}\nabla_j\Phi\right)=4\pi G\rho_b,
\]

\[
K^{ij}=\left[\exp\left(-\kappa S_a Q\right)\right]^{ij},
\quad
S_a=\frac{a_0^2}{a_0^2+|\nabla\Phi_N|^2},
\quad
Q^i{}_j=
\frac{\mathcal E^i{}_k\mathcal E^k{}_j}
{\mathcal E_{mn}\mathcal E^{mn}},
\]

\[
\mathcal E_{ij}=
\partial_i\partial_j\Phi_N-
\frac{1}{3}\delta_{ij}\nabla^2\Phi_N.
\]

Here:

- \(\Phi_N\) is the ordinary baryonic Newtonian potential.
- \(a_0\) is the already fixed low-acceleration scale.
- \(\mathcal E_{ij}\) is the traceless baryonic tidal tensor.
- The normalized squared tidal tensor supplies direction without an
  object-type label.
- The acceleration factor screens the modification in the Solar System.
- \(\kappa\geq0\) is one universal response strength.
- The matrix exponential keeps \(K^{ij}\) positive definite for every finite
  \(\kappa\).

Matter follows \(\Phi\), and the first relativistic closure gives photons the
same potential. This automatically respects the M33 result. Disk, bulge, and
multi-centre cluster geometries can respond differently because \(K^{ij}\) is
tensorial, not because a galaxy or cluster label is supplied.

This is a research candidate, not a novelty claim. Density-dependent scalar
gravitational permittivity already exists in refracted gravity, including a
[covariant scalar-tensor formulation](https://arxiv.org/abs/2109.11217).
The proposed normalized tidal-tensor response is a distinct testable
specialization, but a full literature and prior-art review is still required.

### Spherical proxy result

The original linear mapping was retired because its spherical low-acceleration
boost cannot exceed 3, below the CLASH median requirement of about 7.9.
Reciprocal and exponential positive-definite mappings remove that ceiling.
Six one-parameter combinations of mapping and acceleration gate were
cross-validated on 131 SPARC galaxies and 20 CLASH clusters.

No spherical formula passed both domains. The exponential \(n=2\) gate reached
0.135 dex on CLASH, but its shared cluster-strength setting produced a
94.8 km/s SPARC error. The reciprocal \(n=2\) compromise produced 90.5 km/s
on SPARC and 0.144 dex on CLASH. The separately preferred CLASH coupling was
1.62 times the SPARC coupling for the best exponential family.

Therefore a scalar or spherical reweighting is not enough. The only reason to
advance the tensor branch is to test its defining prediction: full 3-D
directional field redistribution must suppress the response in disk galaxies
while retaining it in distributed cluster geometries.

## Required next data

- Axisymmetric stellar and gas density maps with vertical scale information
  for a training and held-out SPARC subset.
- Two-dimensional cluster baryon maps: X-ray gas, BCG, intracluster light, and
  member galaxies on one registered grid.
- Raw multiple-image positions and redshifts for untouched cluster holdouts.
- Raw VLBI epoch positions for the parallax-ellipse optical-metric null test.

## Advancement gates

- SPARC outer held-out RMSE no worse than 10.4 km/s without per-galaxy gravity
  parameters.
- CLASH derived-profile error no worse than 0.139 dex.
- Raw cluster image-plane error below 10 arcsec initially, with 2 arcsec as the
  decisive target.
- Solar-System fractional deviation below the Cassini-scale limit.
- One universal \(\kappa\), positive-definite response, and stable solutions.

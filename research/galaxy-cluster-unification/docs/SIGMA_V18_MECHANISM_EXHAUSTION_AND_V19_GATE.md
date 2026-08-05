# Sigma v18 mechanism-exhaustion audit and v19 gate

## Decision

The v18D failure does **not** authorize another generic memory, diffusion, or
tensor-carrier formula. Those broad ideas have already been explored in
materially distinct ways. The next admissible question is narrower:

> Does a directly measured baryonic assembly-history variable contain halo
> size, offset, or orientation information that is absent from the current
> density, temperature, and line-of-sight velocity-stress snapshots?

V19 is therefore a measurement-first observability gate. It will not choose a
field equation or fit a lensing target until the proposed history variable is
constructed and frozen from baryonic data alone.

## Mechanisms already covered

| Family | Representative tests | Outcome relevant to v19 |
|---|---|---|
| One isotropic spatial memory | v3D-v4C | Lost distributed shear phase or required different cluster scales; three-formulation reset triggered |
| Conservative path routing and diffusion | P0646-P0657 | Some spent-image improvement, but no universal all-root cross-validation solution |
| Tensor AQUAL and squared coherence | P0659-P0677 | Numerically coherent and strongly separates galaxy/cluster structure, but raw cluster multiplicity remained missing and the confinement branch was retired |
| Pure-metric retarded memory | v6A-v6D | Causal response could be prescribed, but its minimal variational localization contained six negative kinetic directions |
| Aether/tidal tensor carrier | v10A-v10D | Static, nonlinear-kinetic, and physical tensor-cone failures triggered a reset |
| Anisotropic scalar and material memory | v11A-v11C | Three finite-background kinetic-rank failures triggered a reset |
| Preferred clock/ADM trace | v12-v13 | Three bounded-Hamiltonian failures triggered a reset |
| Local covariant gauge-tidal carrier | v14A-v14C | Curvature obstruction, Weyl obstruction, and opposite-residue spin-two pole triggered a reset |
| Instantaneous thermal and member stress | v17E and v18D | Approximate halo extent survived, but universal amplitude, phase, and shear did not |

Changing an exponent, smoothing kernel, range, or carrier name inside these
families is not a new mechanism.

## The remaining empirical clue

The conventional residual-field energy is diagnosed by

\[
\mathcal E_{\rm req}(\mathbf x)
=\Delta\kappa^2+\Delta\gamma_1^2+\Delta\gamma_2^2.
\]

Its cumulative radii \(R_{50}\) and \(R_{80}\) cannot be changed by an overall
source coefficient. Thermal stress predicted all four cross-transfer radii to
within about 5.4%-9.5%. Collisionless stress predicted three of four within
10%, with one 29.6% miss. Yet neither source predicted the full field.

This is consistent with—but does not prove—a lagged response: the apparent
halo extent may remember where baryonic energy and momentum have propagated,
while the current snapshot no longer fixes its amplitude and direction.

## What v19 must measure before proposing an equation

A causal assembly source requires more information than a density map. The
minimum target-blind data package is:

1. member positions, secure redshifts, and uncertainties;
2. X-ray surface density and resolved temperature with uncertainties;
3. a shock, cold-front, or radio-relic geometry with a measured propagation
   speed or a quantitative upper bound;
4. BCG, intracluster-light, stellar, and gas mass maps on a common WCS;
5. an explicit bound on unobserved transverse velocity and line-of-sight depth;
6. the same construction for at least two spent clusters in different merger
   stages, followed by at least four systems before holdout.

For the present spent pair, published work supplies a useful contrast. MACS
J0416 has a detailed member-velocity/lensing analysis and reported component
offset structure, while PLCK G287 has double radio relics and a Chandra shock
front. These facts establish data plausibility, not a fitted Sigma result.

## Frozen v19 source-level null tests

Before opening any lensing target, a proposed history coordinate must pass all
of these:

| Gate | Required outcome |
|---|---|
| Definition | One formula from baryonic observables; no cluster label, lensing map, halo catalog parameter, or fitted center |
| Units | Dimensionless state or a scale derived from observables and universal constants; no per-cluster response length |
| Time information | Changes when two systems have identical instantaneous density/stress but different declared assembly histories |
| Projection robustness | The sign and principal direction survive the declared line-of-sight/depth uncertainty ensemble |
| Resolution | All scored source observables change by at most 2% when numerical resolution doubles after common-PSF matching |
| Identifiability | At least one history statistic is measured at five-sigma in both development clusters |
| Independence | The source maps and every allowed sensitivity are hashed before any lensing target is loaded |

Failure of the time-information gate means the proposal is only another
instantaneous stress law. Failure of projection or identifiability means the
available data cannot test it; that is a data limitation, not evidence for the
physics.

## The equation class only a source pass would authorize

A source pass would authorize deriving—not fitting—a covariant state equation
of the schematic form

\[
\boxed{
\mathcal D_{\rm causal}[g,T_b]\,\mathcal S_{\mu\nu}
=\mathcal P_{\mu\nu}[T_b,\nabla T_b]
}
\]

together with one physical-metric equation

\[
\boxed{
G_{\mu\nu}+\mathcal K_{\mu\nu}[g,\mathcal S]
={8\pi G\over c^4}T^{(b)}_{\mu\nu}.
}
\]

These boxes define required jobs, not an accepted v19 theory. The first says
that baryonic flow forces a causal state; the second says that the same state
changes the metric seen by both matter and light. The action must fix the
homogeneous state by one universal cosmological/no-incoming prescription and
must contain no freely shaped halo-equivalent initial condition.

The apparent halo remains an output,

\[
\rho_{\Sigma,\mathrm{eff}}
=-{1\over4\pi G}\nabla\cdot(\mathbf g+\mathbf D),
\]

and its size is measured from the predicted one-metric field—not inserted as
\(R_\Sigma(M_b)\), selected from a halo catalog, or adjusted per cluster.

## Advancement and reset rules

1. If no target-blind assembly variable passes the observability gates, stop
   the causal-history branch and report the missing measurements.
2. If a source passes but the same one-coefficient spent-pair transfer gates as
   v18D fail, retire baryonic causal history as the direct missing-field source.
3. If it passes source and transfer gates, derive the complete covariant action,
   Hamiltonian, characteristics, weak-field metric potentials, and Solar limit
   before opening a holdout.
4. No new range, exponent, orientation, shear, or cluster amplitude may be
   added after seeing a failed target.

The first v19 task is therefore a public-data and projection-identifiability
audit, not another formula sweep.

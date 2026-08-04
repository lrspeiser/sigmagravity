# Sigma v11C Biot-stretch kinetic falsification and mechanism reset

## Decision

Exact Sigma v11C is retired before observational data. Replacing v11B's Green
strain by the finite Biot strain does repair the one-dimensional negative
quartic that killed v11B. It does not give a globally positive relativistic
kinetic matrix. A finite orientation-preserving anisotropic stretch makes one
rank-one material-shear stiffness exceed the time-kinetic budget on a tilted
slice. The physical material-coordinate Hessian crosses zero and becomes
negative while the material remains exactly comoving with the timelike aether.

V11A, v11B, and v11C are three materially distinct post-v10-reset closures
failing the same nonlinear kinetic-rank gate. The preregistered stopping rule
therefore triggers: do not add a v11D strain cutoff or fitted saturation. Reset
the material-memory mechanism.

## Candidate action

For the three material scalars `X^I`, define

$$
Q^I=A^\mu\nabla_\mu X^I,
\qquad
C^{IJ}=q^{\mu\nu}\nabla_\mu X^I\nabla_\nu X^J,
$$

and use the positive right stretch and Biot strain

$$
U=\sqrt C,\qquad S=U-I.
$$

The proposed Sigma term is

$$
\mathcal L_\Sigma
={1\over2}Q_IQ^I
-s\left[S_{\rm TF}:S_{\rm TF}+b(\operatorname{tr}S)^2\right],
$$

with the already derived coefficients

$$
s={3\over11},\qquad b={17\over24}.
$$

Around `D=I`, it has two shear phonons with squared speed `3/11` and one
longitudinal phonon with squared speed `3/4`. No new constant is introduced.

## Why the obvious repair looked successful

On the exact v11B one-stretch path,

$$
D=\operatorname{diag}(1+\gamma v w,1,1),
$$

the Biot strain is linear rather than quadratic in `w`. Its Hessian is

$$
H_{\rm one}=\gamma^2
\left[1-v^2\,2s\left({2\over3}+b\right)\right]
=\gamma^2\left(1-{3v^2\over4}\right)>0
$$

for every subluminal tilt. At `v=1/2`, it is `13/12`. This repairs the exact
channel that falsified v11B, but a one-dimensional check is not a global
kinetic proof.

## Exact mixed-shear configuration

Use Minkowski space and

$$
A^\mu=\gamma(1,v,0,0),\qquad v={1\over2}.
$$

Choose the finite material map

$$
X^1=e\gamma(x-vt),\qquad X^2=e y,\qquad X^3=M z,
$$

with `e=1/10`, and perturb

$$
X^2\rightarrow X^2+w t.
$$

In an orthonormal aether frame,

$$
Q(w)=\gamma w\,e_2,
\qquad
D(w)=
\begin{pmatrix}
e&\gamma v w&0\\
0&e&0\\
0&0&M
\end{pmatrix}.
$$

At `w=0`, `Q=0`: the material is exactly comoving with the aether, hence its
flow is timelike. Also

$$
\det D=e^2M>0,
$$

so the map lies strictly inside `GL+(3)` rather than on a singular boundary.

The two-by-two block gives the exact identities

$$
\operatorname{tr}U=M+\sqrt{4e^2+(\gamma v w)^2},
$$

$$
\|U-I\|^2=\|D\|_F^2-2\operatorname{tr}U+3.
$$

For the rank-one direction `e_1 tensor e_2`, the Biot energy curvature is

$$
K=s\left[
2+{-2+2(b-1/3)(M+2e-3)\over2e}
\right].
$$

The coordinate-velocity Hessian is therefore

$$
\boxed{H=\gamma^2(1-v^2K).}
$$

The finite rank surface is

$$
M_*=3-2e+
{1+e[1/(sv^2)-2]\over b-1/3}
={398\over45}=8.844444\ldots.
$$

At `0.99`, `1`, and `1.01` times this axial stretch:

| `M/M_*` | Hessian | `det D` | material speed relative to aether |
|---:|---:|---:|---:|
| 0.99 | `+0.0301515` | positive | `0` |
| 1.00 | `0` | positive | `0` |
| 1.01 | `-0.0301515` | positive | `0` |

At the simple rational counterexample `M=10`,

$$
K={57\over11}>4={1\over v^2},
\qquad
\boxed{H=-{13\over33}}.
$$

The normalized Lagrangian is finite. A central finite difference agrees with
the analytic Hessian within the frozen `10^-5` tolerance.

## Why this is decisive

This is one physical material-coordinate Rayleigh direction with metric and
aether velocities fixed. If a quadratic form is negative on one direction,
off-diagonal mixing cannot make it positive definite. The continuous finite
path through `M_*` also contains a singular Legendre surface before any
material flow approaches the light cone.

A restriction such as `e>e_min`, `M<M_max`, or a hand-selected strain
saturation would be a new state cutoff. It would not follow from the action,
would add exactly the kind of hidden regime rule prohibited by the project,
and would merely move the problem.

## Relation to existing elasticity

The strain `U-I` and energy quadratic in it are established quadratic-Biot
elasticity, not a new Sigma invention; see Vitral and Hanna's
[Quadratic-stretch elasticity](https://arxiv.org/abs/2104.11714). The broader
mathematical warning is also known: quadratic strain energies in this family
are not globally rank-one convex, and the planar Biot energy requires a
quasiconvex relaxation; see Martin et al.,
[Quasiconvex relaxation of planar Biot-type energies](https://arxiv.org/abs/2501.10853),
and the general ellipticity/rank-one equivalence for isotropic energies in
[Martin et al.](https://arxiv.org/abs/2008.11631).

Our result is narrower and relativistic. It does not infer failure merely from
the published nonconvexity result. The explicit Sigma counterexample has
*positive but excessive* spatial rank-one curvature. On a foliation tilted
relative to the aether, that curvature subtracts from the `Q^2` time kinetic
term and makes the physical velocity Hessian negative. That exact Lorentzian
failure and its coefficients are specific to this candidate.

## Consequence

The post-v10 material-memory sequence is closed:

1. v11A: bounded alignment times an unbounded finite memory gradient;
2. v11B: Green-strain square creates a negative tilted-velocity quartic;
3. v11C: Biot stretch repairs that quartic but has excessive anisotropic
   rank-one stiffness on a finite comoving background.

The next candidate must obtain environmental/tidal memory from a constraint or
degenerate structure whose all-background kinetic rank is an identity, rather
than from another positive spatial strain energy attached to independent
material clocks. No observational product or holdout was opened.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v11c_biot_stretch_rank.py
python -m pytest -q tests/test_sigma_v11c_biot_stretch_rank.py
```

Machine-readable evidence is in
`results/sigma_v11c_biot_stretch_rank/report.json`.

# Sigma v6A: causal metric-memory action envelope

## Decision

Sigma v6A advances **only** to a full closed-time-path variation and spectrum
audit. It is not authorized to see galaxy or cluster observations. The selected
mechanism is a baryon-forced, retarded, pure-metric memory whose scalar amplitude
is supplemented by the trace-free spatial Hessian of that memory.

This is materially different from the retired local v5 rows:

- v5A used an invalid signed symmetric-teleparallel scalar base;
- v5B added a local causal polarization but lost the nonlinear lapse constraint;
- v5C used a healthy luminal DHOST envelope but its fixed row returned to GR or
  an attractive Yukawa law outside the baryons.

The point of v6A is to retain a source-determined response in vacuum without a
free halo-like field configuration and to make the orientation of separated
baryonic components enter the equation before nonlinear summation.

## Prior-art boundary

The scalar part is not new. Deffayet, Esposito-Farese, and Woodard constructed
pure-metric nonlocal models that recover MOND and sufficient spherical lensing,
including a high-field-suppressed activation of the same general form used below
([arXiv:1106.4984](https://arxiv.org/abs/1106.4984)). Aether/vector theories are
also not a clean novelty route: generalized Einstein-Aether theory already
reproduces MOND, and published Bullet Cluster work obtains displaced lensing by
allowing coherent vector-field concentrations only weakly tied to baryons—close
to reintroducing a halo-like independent state
([arXiv:0806.4319](https://arxiv.org/abs/0806.4319)). AeST is an even stronger
modern published control with MOND galaxies and a cluster-scale departure, so it
must be compared rather than relabeled
([arXiv:2007.00082](https://arxiv.org/abs/2007.00082),
[arXiv:2312.00889](https://arxiv.org/abs/2312.00889)).

The only potentially project-specific element is a **healthy, baryon-forced
trace-free-Hessian orientation closure**. No originality claim is made unless a
full action passes the health gates and a dedicated literature audit finds no
equivalent term.

## Why the action must be an in-in effective action

A traditional variation of a nonlocal single-history action symmetrizes its
kernel. The resulting equation contains advanced as well as retarded support.
This is not a minor numerical choice: it means the field can respond before its
source. The issue has been shown explicitly for broad functions of nonlocal terms
([arXiv:1601.03808](https://arxiv.org/abs/1601.03808)).

The provisional action envelope is therefore a closed-time-path (Schwinger–
Keldysh, or “in-in”) effective action,

$$
\Gamma[g_+,g_-]
=S_{\rm EH}[g_+]-S_{\rm EH}[g_-]
+S_b[g_+]-S_b[g_-]+\Gamma_\Sigma[g_+,g_-].
$$

The $+$ and $-$ metrics are two histories used to compute a causal expectation
value, not two physical metrics. The physical equation is obtained by varying
with respect to the difference history and then setting $g_+=g_-=g$. Ordinary
matter and light therefore still use one physical metric.

At quadratic response order the influence part must have the schematic causal
form

$$
\Gamma_\Sigma
\supset {M_{\rm Pl}^2a_\Sigma^2\over2}
\int d^4x\,d^4x'\sqrt{-g(x)}\sqrt{-g(x')}
\,\Delta I_A(x)K_R^{AB}(x,x')\bar I_B(x'),
$$

where $K_R(x,x')=0$ unless $x'$ lies in the causal past of $x$. All homogeneous
memory data are fixed to zero on the declared cosmological initial surface. Thus
zero baryonic/metric source gives zero Sigma response; a freely chosen vector or
scalar “cloud” is disallowed.

## Frozen invariant envelope

Let $u^\mu[g]$ be a future-timelike direction built from the metric and its past
light cone, and $h_{\mu\nu}=g_{\mu\nu}+u_\mu u_\nu$. A representative retarded
memory scalar is

$$
U_R=\Box_R^{-1}
\left(R_{\mu\nu}u^\mu u^\nu-\frac12R\right).
$$

Define a scalar amplitude and trace-free orientation tensor,

$$
X={c^4\over a_\Sigma^2}
h^{\mu\nu}\nabla_\mu U_R\nabla_\nu U_R,
$$

$$
T_{\mu\nu}=h_\mu{}^\alpha h_\nu{}^\beta
\left(\nabla_\alpha\nabla_\beta
-\frac13h_{\alpha\beta}h^{\rho\sigma}
\nabla_\rho\nabla_\sigma\right)U_R,
\qquad
Z={c^8\over a_\Sigma^4}T_{\mu\nu}T^{\mu\nu}.
$$

The frozen two-constant envelope is

$$
\chi=X+\lambda_\Sigma\sqrt Z,
\qquad
f(\chi)=\chi e^{-\sqrt\chi}.
$$

The two constants are $a_\Sigma$ and $\lambda_\Sigma$. At small $\chi$,

$$
f(\chi)=\chi-\chi^{3/2}+O(\chi^2),
$$

so the quadratic term can cancel the corresponding weak-field Einstein term and
leave the cubic scaling needed for a MOND/BTFR-like exterior. At high field the
correction is exponentially suppressed. These statements concern only the action
envelope; the coefficient and sign must emerge correctly from the complete
variation.

The Hessian is not a label for “cluster.” It is zero or simple in a smooth
spherical field and becomes directionally structured when multiple baryonic
concentrations overlap. Because the response is nonlinear,

$$
\mathcal N(T_1+T_2)\ne\mathcal N(T_1)+\mathcal N(T_2),
$$

which is the mathematical opening needed to change critical-curve topology rather
than merely multiply a broad radial lens.

## Completed construction checks

The executable audit found:

| Check | Result | Gate |
|---|---:|---:|
| Retarded response before an impulse | $0$ | $\le10^{-14}$ |
| Time-symmetric response before the same impulse | nonzero | must expose the ordinary-action problem |
| Zero-source response with fixed zero state | $0$ | exactly zero |
| Maximum rotation-covariance residual, 1,000 trials | recorded in report | $\le10^{-12}$ |
| Median nonlinear superposition residual, 1,000 trials | recorded in report | $\ge0.01$ |
| High-field correction at $g/a_\Sigma=10^5$ | recorded in report | $\le10^{-5}$ |
| Universal physical constants | 2 | $\le5$ |

These are necessary construction checks, not a field-theory validation.

## Required next result

Before any observational fit, v6A must pass all of the following in one report:

1. vary the complete closed-time-path action and prove the physical equation is
   covariantly conserved;
2. localize the response for calculation without promoting fixed auxiliary
   histories into free halo-like initial data;
3. derive the full scalar, vector, and tensor quadratic operators on Minkowski and
   FLRW backgrounds;
4. show positive spectral density, no exponentially growing poles, a well-posed
   retarded initial-value problem, and $|c_T/c-1|\le10^{-15}$;
5. derive the static spherical equation and its coefficient, rather than inserting
   RAR by hand;
6. show that the tensor/Hessian term avoids the published tensor instabilities of
   simple nonlocal Weyl-square models
   ([arXiv:1512.06373](https://arxiv.org/abs/1512.06373)).

Failure of any one of these retires the exact v6A envelope before galaxy or cluster
data are opened.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/select_sigma_v6a_metric_memory.py
python -m pytest -q tests/test_sigma_v6_metric_memory.py
```

Machine-readable evidence is in
`results/sigma_v6a_metric_memory_selection/report.json`.

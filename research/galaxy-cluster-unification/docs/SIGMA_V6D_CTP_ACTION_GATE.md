# Sigma v6D closed-time-path action gate

## Decision

Sigma v6D is retired as a fundamental field theory before observational data. It
can still be retained as a causal effective-response control.

The doubled-history construction can be made formally covariant and causal, and
its diagonal diffeomorphism identity implies conservation. However, the minimal
variational localization of its inverse operators contains six negative kinetic
directions and six additional multiplier configurations. Calling those variables
“fixed auxiliaries” removes their free initial data only by returning to a
non-fundamental prescribed nonlocal response. That does not satisfy the project's
positive-energy action gate.

## Complete minimal localization

On each closed-time-path branch, introduce:

- the one physical metric $g_{\mu\nu}$;
- scalar response $U$ and multiplier $A$;
- spatial symmetric trace-free tensor response $Q_{\mu\nu}$ and multiplier
  $B_{\mu\nu}$, each with five components;
- algebraic multipliers enforcing spatiality and tracelessness.

Suppressing projection multipliers, the branch action required to reproduce the
v6D definitions is

$$
\begin{aligned}
S_{\rm loc}={}&S_{\rm EH}[g]+S_b[g,\psi_b]\\
&+{M_{\rm Pl}^2\over2}\int d^4x\sqrt{-g}\Big\{
a_\Sigma^2f_D(X,C_Q)
+A(\Box U-\mathcal S_g)\\
&\hspace{38mm}
+B^{\mu\nu}\left[(\Box-m_\Sigma^2)Q_{\mu\nu}
-T_{\mu\nu}[U,g]\right]
\Big\},
\end{aligned}
$$

where

$$
\mathcal S_g=R_{\mu\nu}u^\mu u^\nu-\frac12R,
$$

and $f_D$ is the cubic-only orientation correction selected in v6D.

The closed-time-path action is

$$
\Gamma_{\rm CTP}
=S_{\rm loc}[g_+,U_+,A_+,Q_+,B_+]
-S_{\rm loc}[g_-,U_-,A_-,Q_-,B_-],
$$

supplemented by an initial density functional and final $+$/$-$ matching. Both
must be invariant under the diagonal diffeomorphism. The physical equation is
obtained after variation and the limit $g_+=g_-=g$.

This is the smallest complete variational action for the declared inverse-
operator definitions. Merely writing $U=\Box_R^{-1}\mathcal S_g$ inside
$f_D$ is not an action variation: traditional variation symmetrizes the kernel.

## Ward identity

Let $E_g^{\mu\nu},E_U,E_A,E_Q^{\alpha\beta},E_B^{\alpha\beta}$ denote branch
Euler expressions. Branch diffeomorphism invariance gives

$$
\begin{aligned}
0={}&-2\nabla_\mu E_g^\mu{}_{\nu}
+E_U\nabla_\nu U+E_A\nabla_\nu A\\
&+E_Q^{\alpha\beta}\nabla_\nu Q_{\alpha\beta}
-2\nabla_\alpha(E_Q^{\alpha\beta}Q_{\nu\beta})\\
&+E_B^{\alpha\beta}\nabla_\nu B_{\alpha\beta}
-2\nabla_\alpha(E_B^{\alpha\beta}B_{\nu\beta}),
\end{aligned}
$$

plus terms proportional to the algebraic projection equations. Hence

$$
\nabla_\mu E_g^\mu{}_{\nu}=0
$$

when every auxiliary equation holds. Applying the identity to both branches and
taking the physical limit produces the diagonal CTP conservation identity.
Minimal baryonic coupling independently gives

$$
\nabla_\mu T_b^{\mu}{}_{\nu}=0
$$

on the matter equations. Thus conservation is not the failure.

## Kinetic obstruction

Integrating the constraint terms by parts exposes

$$
\mathcal L_{\rm kin}
\supset-\nabla_\mu A\nabla^\mu U
-\nabla_\rho B_{\mu\nu}\nabla^\rho Q^{\mu\nu}.
$$

For each component, the velocity Hessian is proportional to

$$
H_{\rm pair}=\begin{pmatrix}0&1\\1&0\end{pmatrix},
$$

with eigenvalues $+1$ and $-1$. Changing the sign of the constraint only exchanges
the eigenvectors.

There is one scalar pair and five spatial trace-free tensor pairs. The complete
minimal response/multiplier kinetic Hessian therefore has

$$
\boxed{6\text{ positive directions},\qquad6\text{ negative directions}.}
$$

It is full rank, so these are not gauge-null directions. The localization has 12
configuration components and 24 second-order Cauchy data, compared with the six
desired retarded responses.

This agrees with the known warning that localizing inverse-d'Alembertian gravity
actions generically exposes ghostlike auxiliary combinations
([De Felice & Sasaki](https://arxiv.org/abs/1412.1575)). Some special nonlocal
form factors admit a finite, controlled initial-value formulation, but that must
be proven for the specific kernel rather than assumed
([Calcagni, Modesto & Nardelli](https://arxiv.org/abs/1803.00561)).

## Causality does not remove the kinetic sign

Fixed zero initial data can select the retarded response

$$
G_R(t,t')=\Theta(t-t')
{\sin[\omega(t-t')]\over\omega}.
$$

The adjoint solution associated with an ordinary endpoint variation has advanced
support. A closed-time-path contour can select retarded physical equations, as is
standard for nonlocal gravitational effective actions
([arXiv:1709.10435](https://arxiv.org/abs/1709.10435)).

But this creates a strict choice:

1. Treat $A,B$ as local fields with an initial state: evolution is causal, but the
   kinetic matrix has physical negative directions.
2. Declare $A,B$ fixed response variables with no state: the negative modes are
   not independently excited, but the construction is an effective nonlocal rule,
   not the required positive-energy fundamental action.

The exact v6D action cannot satisfy both sides of the goal simultaneously.

## Scope and next requirement

This result does not reject all causal nonlocal gravity. It rejects this exact
constraint-localized v6D construction as the project's fundamental theory.

Any successor must obtain the scalar and orientation response by integrating out
a manifestly positive local carrier whose state is fixed universally—not by
imposing retarded inverse operators with multiplier pairs. That carrier must also
avoid becoming a freely shaped, halo-like invisible component.

No galaxy, cluster, Solar-System, or holdout data were accessed in reaching this
decision.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v6d_ctp_action.py
python -m pytest -q tests/test_sigma_v6_ctp_action.py
```

Machine-readable evidence is in
`results/sigma_v6d_ctp_action_gate/report.json`.

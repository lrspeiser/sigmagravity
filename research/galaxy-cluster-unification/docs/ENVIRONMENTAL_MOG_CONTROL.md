# Environmental MOG control: derivation result

Status: **retired at the pre-fit checkpoint**. No MOG parameter was fit.

The frozen five-parameter EMOG-Q0 action, complete variation, conservation
identity, spherical solution, and joint structural test are in
[the derivation report](ENVIRONMENTAL_MOG0_DERIVATION.md). Its canonical
chameleon avoids EA-Q0's omitted/oversized reciprocal source, but produces a
different fatal inconsistency: its metric enhancement must be monotone with
baryonic density while the CLASH target contains a pair that forces at least a
59.0% error for any such response. The universal Yukawa correction has an
approximately $1/r$ slope over only 0.055 dex, and the changing scalar metric
strength cannot stay locked to the fixed conserved vector charge. The declared
5% gate therefore fails before Stage 3 or Stage 4.

## Prior-art boundary

Scalar--tensor--vector gravity and its attractive metric plus repulsive Proca
weak-field law are prior art; see
[Moffat (2005)](https://arxiv.org/abs/gr-qc/0506021). This project does not claim
that field content or the Yukawa force law as new. The narrower test is whether
one conserved action can make its scalar/vector state respond to the already
measured full baryonic environment and then predict galaxy dynamics, BCG
dynamics, and lensing without an object-class rule.

## Starting envelope

The next cycle starts from

$$
S=S_g[g,G]+S_\phi[g,\phi_a,\mu,G]
+S_s[g,G,\mu]+S_m[g,\phi_a,\psi_m],
$$

with one metric $g_{ab}$, a positive-energy massive vector $\phi_a$, and the
minimum scalar content needed to make $G$ and the vector range $\mu^{-1}$
dynamical. A universal matter current may couple to $\phi_a$ only if its charge-
to-mass ratio is derived and composition independent. Light must receive its
lensing prediction from the same metric solution; there is no lensing-only
normalization.

The constant-field spherical control is

$$
g(r)=-\frac{G_NM}{r^2}
\left[1+\alpha-\alpha(1+\mu r)e^{-\mu r}\right].
$$

The short-distance cancellation and the sign of the vector energy must follow
from the action. The formula is a diagnostic limit, not permission to assign
$\alpha$ or $\mu$ separately to each system.

## First required outcomes

1. Freeze an action with at most five global physical parameters, no per-object
   values, no class label, and no lensing-only term.
2. Vary the metric, vector, scalar, and matter equations and prove the on-shell
   stress-energy/current conservation identity.
3. Derive the weak-field signs, $r\ll\mu^{-1}$ cancellation, large-radius
   attraction, and the metric lensing potentials.
4. Determine whether the scalar equations can use baryonic boundary data
   without repeating EA-Q0's reciprocal-source failure or inserting the
   measured $\chi_b$ by hand.
5. Reject the control before fitting if it cannot approach the observed
   galaxy $1/r$ acceleration over the measured radial support, if it requires a
   per-system range, or if its kinetic/gradient spectrum is unhealthy.

Only a passing derivation could freeze a Stage 3 fit and the unchanged 34-host
Stage 4 replay. EMOG-Q0 did not pass. The project now moves to a premise-level
rethink rather than another interpolation function.

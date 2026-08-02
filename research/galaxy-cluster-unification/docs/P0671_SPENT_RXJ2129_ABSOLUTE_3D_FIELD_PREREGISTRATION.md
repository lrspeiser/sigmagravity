# P0671 spent RX J2129 absolute 3D field preregistration

P0671 solves two equations on exactly the same P0670 physical baryonic cube and
simple-MOND boundary. The control is scalar AQUAL (`sigma=0`); the candidate is
the P0669 tensor equation with its already-built `sigma` and direction. Both
use the same numerical solver, `a0`, source, grid, nonlinear tolerance, and
zero-slip photon rule.

The line-of-sight integral produces a physical angular deflection before the
source-redshift factor `Dds/Ds`. No amplitude is fitted. The frozen structural
questions are:

- do both nonlinear equations converge below `1e-5` residual;
- is the boundary retained exactly and the constitutive tensor positive;
- is the strong-lens deflection finite and physically nonzero;
- is the tensor change distinguishable from numerical zero but smaller than a
  10 percent instability; and
- do both deflection maps remain curl-free to numerical tolerance?

P0671 computes no raw lens residual, root, parity, multiplicity, or topology.
A pass merely authorizes a separately frozen spent-data topology audit.

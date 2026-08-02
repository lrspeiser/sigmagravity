# P0681 compact-halo Hessian step-convergence preregistration

P0680 establishes exact Hessian symmetry and identical negative-Jacobian point
counts, but its `0.01 arcsec` direct comparison misses `1e-5` agreement by a
factor of roughly two. P0681 is the final derivative audit.

At the same 92 points, direct steps `0.01, 0.005, 0.002, 0.001 arcsec` are
compared to the unchanged exact Hessian. Convergence and Jacobian-determinant
errors must decrease at every step and finish below `1e-6`; normalized curl at
the smallest step must be below `2e-9`; all four steps must retain exactly six
negative-Jacobian points; and all values must remain finite.

A pass qualifies P0678's numerical target specification without changing any
historical failure. Failure ends derivative refinement and leaves P0678's
strength decomposition provisional. No scientific formula or lens root is
scored.

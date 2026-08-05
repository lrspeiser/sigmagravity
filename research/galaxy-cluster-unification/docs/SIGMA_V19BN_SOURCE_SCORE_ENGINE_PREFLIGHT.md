# Sigma V19BN source-score engine preflight

V19BN implements the decision layer between the future regional I4/I5 maps and
an action-derivation decision. It applies the V19BL thresholds without reading
lensing or halo data.

The engine requires every candidate gradient, computes I4 tensor amplitude and
axis or I5 scalar activation for every posterior draw, applies the fixed
quadratic density-control PRESS test draw by draw, checks every registered
resolution and aperture simultaneously, and performs posterior-median
leave-one-region-out stability.

Manufactured controls behave correctly: a response constructed from the
density predictors is rejected in every draw; independent spatial structure
passes the novelty threshold; a 2% amplitude perturbation transfers; a
45-degree tensor-axis rotation fails; and a uniform tensor survives every
single-region omission. No observed source score has been computed yet.

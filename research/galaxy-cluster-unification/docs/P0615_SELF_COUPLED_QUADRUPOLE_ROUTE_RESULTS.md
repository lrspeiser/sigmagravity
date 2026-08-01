# P0615: self-coupled quadrupole routing

P0615 removes the independent extent-gate strength that failed the P0614
composite audit. At the baryonic `R80`, define

\[
\Delta_{80}={\alpha_{P0554}\over\alpha_b}-1,
\qquad f_{\rm self}={\Delta_{80}\over1+\Delta_{80}}.
\]

The same scalar excess that supplies extra radial gravity now determines the
fraction eligible for endpoint routing. The angular amplitude is constructed
from the centroided baryonic spin-2 asymmetry `Q`. The main candidate is

\[
\epsilon={Q^2\over1+\Delta_{80}}.
\]

This has a simple symmetry motivation: `Q^2` is the lowest even rotational
scalar made from a quadrupole, and the denominator is the baryonic fraction of
the total P0554 field. There is no fitted route coefficient.

Five related exponent choices and a scalar control are scored on the four
P0581 raw clusters and RX J2129. Galaxy rotation remains the P0554 scalar score
because the route layer is an exact axisymmetric null; the Solar route layer is
also exactly zero. All tests use frozen ordinary geometry and source positions
to isolate field response.

This is an opened-data, post-failure diagnostic. Even if a law improves all
five systems, it requires a new raw-cluster transfer before it can be treated
as evidence.

## Measured outcome

The linear law `Q/(1+Delta80)` improves the four-cluster fixed-geometry RMS by
0.341% but loses one RX J2129 root. The quadratic law
`Q^2/(1+Delta80)` preserves all **18/18** held-out roots, improves the
four-cluster aggregate by **0.105%**, and improves RX J2129 by **0.826%**. Its
derived amplitudes range from 0.0062 to 0.0211 in the four-cluster group and
equal 0.00769 in RX J2129. No strength is fitted.

The closely related routed quadratic `Q^2 Delta80/(1+Delta80)^2` gives the
largest RX J2129 gain, 1.002%, but a smaller four-cluster gain, 0.085%. The
predeclared safety-first diagnostic ranking therefore selects the simpler
quadratic-over-total law.

This is a consistency clue, not a competitive lens fit. The four-cluster
scalar control at the frozen geometry has 22.514 arcsec RMS and the quadratic
law has 22.491 arcsec; both remain far from compact-halo accuracy. The useful
result is that one baryon-derived invariant gives the same small, root-safe
direction of change in both opened cohorts. A different-cluster refit is the
required next test.

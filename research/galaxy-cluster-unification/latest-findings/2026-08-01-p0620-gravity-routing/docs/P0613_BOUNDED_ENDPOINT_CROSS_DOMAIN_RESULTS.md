# P0613: bounded endpoint cross-domain response

P0613 is the first experiment chosen from the machine-readable P0612 impact
atlas. It fixes endpoint residence, the return distance, and the baryonic
extent gate. It crosses only three universal settings:

\[
\eta\in\{0.18,0.23,0.28\},\qquad
q\in\{0.5,0.75,1.0\},\qquad
A\in\{5,10,20\}.
\]

Here `eta R80` is the deposition width, `q` multiplies the fixed
concentration-derived routed fraction, and `A` is the ceiling of the smooth
angular response `A tanh(x/A)`. No gravity setting is fit per galaxy or cluster.

The cluster calculation uses four real raw strong-lensing systems and exact
nonlinear image roots at the already-fitted P0581 geometry. The galaxy column
uses the matching conservative endpoint profiles for 131 SPARC galaxies and
968 outer points. The Solar routing layer is exactly zero for a point source,
so its fractional force change and extra Mercury precession are both zero.

The resulting table deliberately distinguishes:

- a large RMS change;
- losing or recovering an image root;
- a real galaxy-velocity change;
- an exact galaxy or Solar null.

See `results/p0613_bounded_endpoint_cross_domain/report.json` for the measured
winner, main effects, comparators, and gates. This is a spent-data response
experiment. Its winner is not a validated theory candidate.

## Measured outcome

Only **1 of 27** universal settings preserves all 11 held-out roots in all four
clusters: `eta=0.23`, `q=1`, and `A=20`. At the fixed P0581 geometry its
equal-cluster RMS is **19.159 arcsec**. On the three systems where the scalar
baseline also has every root, this is a **0.92%** improvement.

The same conservative endpoint translation gives **70.926 km/s** SPARC outer
RMSE, compared with **72.399 km/s** for Newtonian gravity and **10.348 km/s**
for fixed RAR. It therefore changes galaxy predictions in the right direction
but remains **6.85 times** the fixed-RAR error. The Solar routing layer remains
exactly null.

Universal strength has the largest mean root-count span (1.11 roots) and the
largest galaxy span (0.305 km/s); width follows at 1.00 root and 0.157 km/s.
Saturation has zero marginal effect on both total and per-system mean root
counts and zero galaxy effect, but its conditional interactions with width and
strength have root-count RMS values of 0.222 and 0.257. Cap 20 is necessary in
the lone fully safe three-way combination. This is an interaction, not evidence
that the cap is independently powerful.

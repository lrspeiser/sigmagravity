# P0616: frozen self-coupled route transfer

P0616 freezes the P0615 law

\[
f_{\rm self}={\Delta_{80}\over1+\Delta_{80}},\qquad
\epsilon={Q^2\over1+\Delta_{80}}
\]

before computing either quantity or any route response for A383 and MS2137.
The systems are not pristine project holdouts, but they are chronological
formula holdouts: other field families used them, while this scalar-excess plus
quadrupole law did not.

Both the P0554 scalar control and self-coupled endpoint candidate receive 16
ordinary-geometry starts per cluster with paired random starts. No gravity
coefficient is fitted to either cluster. The route layer remains an exact
galaxy-axisymmetry and Solar point-source null, so inherited SPARC and Solar
scores are reported separately from the raw test.

The result file records training and held-out root completeness before RMS.
An apparent improvement from a model that loses a root is not counted.

## Result

The formula did **not** transfer.

- A383 completed every training and held-out root under both variants. Its
  held-out RMS changed from 9.097 to 9.137 arcsec, a 0.442% deterioration.
- MS2137 is inconclusive for model comparison because both the P0554 scalar
  control and the routed candidate converged only 7/8 training and 2/3 held-out
  roots. The incompleteness is therefore not evidence that the route itself
  destroyed a previously valid solution.
- The derived route amplitudes were 0.0178 for A383 and 0.0408 for MS2137;
  neither was fitted to its lensing residuals.
- The one complete matched system had aggregate held-out RMS 9.097 arcsec for
  the control and 9.137 arcsec for the candidate. The 2 arcsec absolute gate,
  root-completeness gate, and non-worsening gate all fail.

The useful lesson is narrower than a theory claim: a quadratic member-light
quadrupole is gentle enough to preserve fixed-geometry roots, but its tiny local
gain is not stable once conventional lens geometry can move. It is not promoted
as a gravity law. The next useful variations should change the *spatial support
or phase of the routed field*, not merely rescale this amplitude.

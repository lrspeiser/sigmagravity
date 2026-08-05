# Sigma V19AB Bessel-to-NSC transformation commissioning plan

## Purpose

V19AA proved that the Bullet paper's rounded coordinates cannot produce a
secure position-only counterpart.  V19AB asks a narrower question before any
ambiguous candidate is scored: can the paper's Bessel `B/R/I` photometry
predict NSC `g/r/i/z` well enough to distinguish otherwise similar objects?

The answer is needed for source reconstruction, not gravity fitting.  V19AB
cannot read lensing or halo data, infer stellar mass, construct a current map,
or change a gravity formula.

## Provisional anchors and honest limitation

The frozen source inventory contains 15 Bullet cones with complete published
`B/R/I`, exactly one NSC candidate, and complete NSC `g/r/i/z`.  These rows are
useful calibration anchors but are not identity ground truth.  A chance object
can still be the sole catalog source in a cone.  The robust loss limits the
influence of a small number of such contaminants, and the final claim remains
an internal commissioning result.

Some singleton photometry was inspected while designing this protocol.  That
fact is recorded in the configuration.  No transform has been fitted and no
validation retrieval score has been computed before the freeze.

## Frozen transformation

The published predictors are the two independent colors `B-R` and `R-I`.  For
each NSC band the model predicts an offset from the nearest published band:

\[
 \begin{pmatrix}g-B&r-R&i-I&z-I\end{pmatrix}
 =
 \begin{pmatrix}
 1 & (B-R-2.4) & (R-I-1.1)/0.5
 \end{pmatrix}\,C .
\]

The four columns of `C` are fitted independently with a fixed `soft_l1` robust
scale of 0.25 mag and ridge penalty 0.25 on the two color slopes.  Predictive
scales are the robust development residual scales with a fixed 0.15-mag floor.
There is no candidate-specific transformation.

The split is deterministic: sort by `sha256("SIGMA-V19AB:" + object_id)`;
the first ten rows are development and the remaining five validation.  The
validation IDs and every input hash are frozen in the configuration.

## Concrete validation gates

The transformation advances only if all four validation offset median
absolute errors are at most 0.45 mag.  Each validation Bessel row is also
matched blindly against all five validation NSC detections:

- the full magnitude-offset score must retrieve at least 4/5 provisional pairs
  at rank one and have mean reciprocal rank at least 0.80;
- the color-only score must retrieve at least 3/5 at rank one and have mean
  reciprocal rank at least 0.65; and
- every optimizer and exact split/inventory check must pass.

A pass authorizes a new, separately frozen likelihood on ambiguous NSC
candidates.  It does not itself select a counterpart.  A failure leaves all
V19AA ambiguous states intact and sends the project to synthetic SED/filter
curves or new source imaging rather than a looser retrospective gate.

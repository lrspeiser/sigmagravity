# Sigma V19AZ probabilistic member-position/current ensemble plan

## Why this protocol exists

The Bullet Cluster member catalog supplies published Bessel B/R/I photometry
and line-of-sight velocities for 72 of its 78 spectroscopic members.  Those
quantities already belong to the spectroscopic members; we do not need to
rediscover them from noisy DECam counterparts.  What remains uncertain for 57
of those members is the precise sky coordinate inside the broad rounding cell
of the published position.

V19AZ therefore stops treating counterpart identity as a winner-take-all
problem.  It constructs a probability distribution over all candidate and
private-null positions, while enforcing that a real catalog object cannot be
assigned to two spectroscopic members at once.  The resulting ensemble is the
appropriate input for a directional, current-sensitive, or long-wavelength
gravity test because it propagates the actual positional ambiguity instead of
turning it into false precision.

## Exact joint posterior

For member `i`, candidate `j` retains its frozen V19AU positional probability
`p_ij` at the primary counterpart prior `Q=0.90`.  The private-null probability
is

\[
 p_{i0}=1-\sum_j p_{ij}.
\]

For an assignment vector `a`, the joint posterior is

\[
 P(a)={1\over Z}\prod_i p_{i,a_i}
 \prod_{i<k}{\bf 1}(a_i=0\;\hbox{or}\;a_k=0\;\hbox{or}\;a_i\ne a_k).
\]

The indicator is the one-to-one rule: private nulls never conflict, but one
real `candidate_id` can belong to at most one member.  Candidate sharing links
only a small subset of the 57 members.  V19AZ enumerates every valid state in
each connected conflict component and uses log-sum-exp normalization.  It is
therefore an exact finite posterior, not an MCMC estimate and not a global MAP
counterpart list.

The 15 validated anchor coordinates are fixed.  If a null state is selected,
the member position is drawn uniformly over its original coordinate-rounding
rectangle: half of one RA time-second east-west and half of one arcsecond
north-south.  The member still exists in a null state; only its sub-arcsecond
catalog identity is unknown.

## What is carried by each ensemble point

The relative Bessel-I luminosity is

\[
 \ell_I=10^{-0.4(I-20)}.
\]

The reference magnitude is only a fixed normalization.  No absolute
luminosity, K correction, extinction correction, stellar-population model, or
mass-to-light ratio is inferred.

For every draw, each measured redshift is perturbed by its catalog uncertainty.
The cluster rest-frame line-of-sight velocity is

\[
 v_{\parallel,i}={cz_i-\mathrm{median}(cz)\over
 1+\mathrm{median}(cz)/c},
\]

and the stored current proxy is `ell_I * v_parallel`.  This is only a
luminosity-weighted line-of-sight current.  The transverse velocities remain
explicitly unmeasured rather than being set to zero.

## Frozen decision gates

The protocol must recover exactly 78 spectroscopic members, 72 finite-BRI
members, 15 anchors, 57 probabilistic members, 640 candidate hypotheses, 568
unique candidates, and six explicit missing-photometry members.  All exact
component probabilities and member marginals must normalize within `1e-12`,
candidate occupancy cannot exceed one, and every one of 8,192 deterministic
ensemble draws must be one-to-one and contain all 72 BRI members.

As a check on the serialized ensemble, its state frequencies must agree with
the exact marginals to a maximum absolute difference of `0.03`.  Failure closes
the protocol without changing the prior, null kernel, seed, sample count, or
threshold after seeing the result.

V19AZ reads no lensing observations, inferred dark-matter maps, halo fits,
gravity residuals, or Sigma parameters.  It is a baryonic-input uncertainty
product, not evidence that any gravity equation works.

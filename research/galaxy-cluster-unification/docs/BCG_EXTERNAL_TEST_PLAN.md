# Post-discovery external BCG check

Status: score frozen before execution, 2026-07-26. This is not a blind
preregistration: the Tian et al. paper and its reported cluster-scale RAR were
known before the test. The U0 formula and bounds were frozen and fitted only on
SPARC+CLASH before this table was acquired.

## Purpose

U0 passed the first joint cross-validation, but its potential threshold lies
between most SPARC and CLASH points and some fold parameters contact their
bounds. The 50 MaNGA brightest-cluster galaxies in Tian et al. 2024 occupy the
intermediate regime and use stellar dynamics rather than lensing. They test
whether U0 learned a physical-looking transition or merely separated the two
development samples.

## Frozen procedure

1. Fit U0 once to all 131 SPARC galaxies and all 20 CLASH clusters with the
   already frozen summed likelihood and 16 deterministic starts.
2. Do not use any MaNGA BCG acceleration in fitting or model selection.
3. The BCG table supplies one outer kinematic point per object. Apply the
   already declared point-mass tail at that point:

   $$
   |\Phi_{\rm bar}|=g_{\rm bar}r_{\rm last},\qquad
   \chi=|\Phi_{\rm bar}|/c^2.
   $$

   This uses the BCG's observed baryons only. No unmeasured host-cluster
   potential is inserted after the fact.
4. Predict $g_{\rm obs}$ with the frozen U0 closure. Compare with two zero-fit
   references: the galaxy RAR at $a_0=1.2\times10^{-10}$ m/s² and the published
   cluster-scale RAR at $a_0=2.0\times10^{-9}$ m/s².
5. Propagate the reported $g_{\rm bar}$ uncertainty through the local model
   slope and add it in quadrature with the reported $g_{\rm obs}$ uncertainty.
   Report measurement-only $\chi^2$/point, RMS dex, median absolute dex, and a
   paired BCG bootstrap for U0 minus fixed galaxy RAR.

## Interpretation

U0 is supported as an empirical bridge only if it improves on fixed galaxy RAR
for these 50 untouched BCGs. Matching the separately labeled cluster-scale RAR
would be stronger, but it is not required because U0 is constrained by both
development domains rather than tuned to BCGs. Failure cannot be repaired by
adding the host potential without registering a new model and obtaining another
external sample.

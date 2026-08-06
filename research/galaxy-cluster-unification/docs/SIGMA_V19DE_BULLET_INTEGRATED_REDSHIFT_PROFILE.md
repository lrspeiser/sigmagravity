# Sigma V19DE Bullet integrated redshift profile

## Frozen preflight

The payload-blind preflight passes. It verifies the four integrated product
sizes and schedules 101 coarse plus 101 fine redshift points with two starts
per point for both APEC and MEKAL. It opened no source PHA or response
scientific array and fit no source quantity. The config SHA-256 is
`e5f49f693f466d6ada9ec08a38956bd4043839d5404f32599e747771afae74ec`,
the runner SHA-256 is
`1ded9c3a1d152fc8276eec4970702f358c99630aeabd6685d9ad36ee31213924`,
and the preflight-report SHA-256 is
`cdaba4c88d39449de6851fdcdad226a623f565f5bf96616e312ffbe2f579a096`.

V19DE is the first source-line fit authorized by the completed Bullet response
and gain chain. It commissions the fitting engine on the integrated,
known-outcome Bullet spectrum before any regional velocity pattern is opened.

Both frozen branches use the ungrouped source and associated blank-sky
background with WStat in 2--10 keV. The primary model is
`tbabs*(apec_1+apec_2)` and the published-model robustness branch is
`tbabs*(mekal_1+mekal_2)`. The two thermal components share one redshift;
temperature, abundance, normalization and Galactic absorption nuisance values
are refit at every fixed redshift. Exchange-symmetric components are reported
canonically with `T1 <= T2`.

Each branch evaluates 101 coarse points across optical redshift plus or minus
0.05 and 101 fine points across plus or minus 0.005 from the coarse global
minimum. Every point uses a warm start plus one rotating frozen anchor. The
profile must have a finite interior Delta-WStat=1 interval, no distinct
secondary minimum within Delta-WStat 6.63, remain within 0.01 of the optical
redshift, and agree across APEC and MEKAL within 0.003.

The integrated gain covariance is transported independently from the nine
ObsID calibration measurements using the already frozen Fe-K response
contributions. It is reported separately from the deterministic spread of the
nine fitted mean gain corrections.

This stage does not perform the 500-draw posterior-predictive goodness test or
the 4,096-point thermal-mixture Sobol propagation. A pass authorizes that next
integrated stage, not regional velocities. No regional source line, ObsID 554,
Abell 2146, lensing, halo, gravity or action payload is authorized.

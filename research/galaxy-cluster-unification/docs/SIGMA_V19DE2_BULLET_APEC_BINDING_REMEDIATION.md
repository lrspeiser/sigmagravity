# Sigma V19DE2 Bullet APEC-binding remediation

## Terminal scientific result

The remediated source execution completed, but the unchanged V19DE scientific
gate **fails**. This is no longer the invalid missing-AtomDB result: all 404
APEC/MEKAL profile points have finite two-start fits, both actual source
sessions retain the frozen AtomDB binding, and every configured component has
finite positive probe flux.

Six of the seven terminal gates pass. APEC and MEKAL give primary minima at
`z=0.3008` and `z=0.2999`, respectively, a difference of `0.0009` below the
frozen `0.003` limit. Their Delta-WStat=1 intervals are
`[0.2983406, 0.3012871]` and `[0.2973302, 0.3029507]`; both lie inside the
profile domain and both minima are within `0.01` of the optical redshift.
The corresponding nominal velocities relative to `z=0.296` are 1,110 and
902 km/s. These values are diagnostic only and are not admitted measurements.

The failed gate is profile uniqueness. The APEC coarse profile has a distinct
secondary minimum at `z=0.3050`, separated from the primary fine minimum by
`0.0042` in redshift and only `Delta WStat=1.7802` above the coarse global
minimum. This is inside the preregistered rejection threshold of `6.63`. Both
the warm and independent anchor starts converge successfully in that basin.
The APEC fine profile is still descending at its `z=0.305` upper edge, while
the coarse profile rises again at `z=0.306`, so the primary fine window does
not establish a unique isolated solution. Both primary plasma-model fits also
place the cool and hot temperatures effectively on the frozen 3.5 and 27 keV
bounds; this is an additional diagnostic, not a post-hoc gate.

The integrated gain covariance remains finite and positive semidefinite, with
220.4 km/s one-sigma uncertainty and 561.1 km/s weighted mean-correction
dispersion. It does not resolve the spectral ambiguity.

The report SHA-256 is
`6e7b4859ffc9ea835b76a4141ca76c2667edb18b4d61fd349cecb7eabc2cfa9d`;
the terminal checkpoint SHA-256 is
`e597abfd5e7d14bc7187749ad98595cbe8a87e2710a24f7aff50915f5385f324`.
The outer host wrapper timed out after one hour, but the same checkpointed WSL
process completed normally; no duplicate execution was launched.

No posterior-predictive simulation, thermal-mixture Sobol propagation,
regional source line or velocity, ObsID 554, Abell 2146, lensing, gravity, or
action payload was opened. The integrated systematic/goodness stage and every
regional stage remain unauthorized. The exact integrated two-temperature
Chandra closure is retained as a scientific failure and may not be retuned to
rescue this source route.

## Frozen preflight

The payload-blind remediation preflight passes. The exact 204,059,520-byte
continuum and 209,629,440-byte line tables match their frozen hashes. With
`APECROOT` bound to AtomDB 3.0.9, `xsapec` and `xsmekal` return finite positive
probe fluxes of `2.493875e-4` and `2.490728e-4`, respectively. No source PHA or
response scientific array was opened.

The config SHA-256 is
`eb9ba22888f0dff3b696834613caa890b2958820fd664876b13f8df13fbd1dcc`,
the runner SHA-256 is
`57b76be67f1d7325ec2eda78a3f86dee461a49917d5747a5d1ed214944216c1d`,
and the preflight-report SHA-256 is
`272ed6ed5fc77eddc357dea433692aef343159c6dcaec74ab1c5925b18eb1e05`.

V19DE2 repairs only the XSPEC model-data failure discovered by V19DE. The
scientific method remains the exact hash-frozen V19DE config: the source,
background, response, APEC/MEKAL models, fit band, WStat likelihood, nuisance
bounds, two-start rule, coarse/fine grids and all terminal gates are unchanged.

The remediation binds `APECROOT` to AtomDB 3.0.9 as declared by the active
XSPEC installation. It freezes the byte sizes and SHA-256 hashes of
`Xspec.init`, `apec_v3.0.9_coco.fits`, and `apec_v3.0.9_line.fits`. Before any
source rerun, both `xsapec` and `xsmekal` must return finite positive flux in
two fixed 2.0--2.5 keV probe bins. The same check is repeated on the actual two
components in each configured source session.

The V19DE invalid report is a mandatory parent. V19DE2 cannot change a
scientific setting or open a regional spectrum, ObsID 554, Abell 2146,
lensing, gravity or action payload. A successful environment preflight must be
committed before the integrated source profile is rerun.

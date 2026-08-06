# Sigma V19DE2 Bullet APEC-binding remediation

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

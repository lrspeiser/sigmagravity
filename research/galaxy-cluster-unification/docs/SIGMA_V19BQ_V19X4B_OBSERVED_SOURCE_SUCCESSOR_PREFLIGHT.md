# Sigma V19BQ V19X4B observed-source successor preflight

V19BQ completes the preregistered handoff from the V19W5-authorized spectral
chain to the source-physics decision without altering any frozen historical
file. It will consume terminal V19X4B gas posteriors and V19BMB stellar ranks,
then apply the exact V19BP I4/I5 candidates, thresholds and decision rules.

The eventual test contains two merging clusters, three temperature--density
correlation branches and six smoothing/aperture variants. Each of I4 and I5 is
therefore evaluated under 36 conditions, for 72 candidate evaluations. No
cluster, uncertainty branch or spatial scale may be averaged away.

I4 must supply a detected, narrow and stable projected direction everywhere.
Only after that requirement passes may either I4 amplitude or scalar I5 satisfy
the strength requirement. This prevents a scalar correlation from being
mistaken for the direction-carrying source needed to reproduce cluster shear
and critical-curve topology.

The preflight passes with terminal gas, stellar and source results sealed.
Lensing, halo maps, galaxy rotations, action placement, gravity parameters and
holdouts remain closed. A future source pass will authorize deriving the
least-field-content covariant action; it will not itself prove a gravity law.
If the source gate fails, the project records that falsification before looking
at lensing targets.

## Reproduction

```powershell
python scripts/check_sigma_v19bq_v19x4b_observed_source_successor_preflight.py
python -m pytest tests/test_sigma_v19bq_v19x4b_observed_source_invariant_scoring.py -q
```

The frozen report is
`results/sigma_v19bq_v19x4b_observed_source_successor_preflight/report.json`.

# P0554 structural exact-refit results

## Outcome

Eight predeclared structural formulas were tested after refitting six ordinary
lens-geometry nuisances in each of five raw clusters. Every fit used eight
starts; source positions were reprofiled, and no gravity parameter was fit.

The fixed-geometry image-root recoveries do not survive. The P0554 parent and
all seven structural variants solve exactly 17 of the same 18 held-out images;
MACS1931 remains incomplete in every case. No variant has a valid all-five
accuracy score and none is promoted.

Two continuous effects remain informative on the four systems complete for
both parent and candidate:

- lowering the dynamics-addition softness to 0.98 gives the largest matched
  aggregate improvement, 2.50%, but worsens galaxy rotation accuracy; and
- lowering the lensing-addition softness to 0.98 improves CLASH, RX J2129, and
  the three other complete raw clusters without changing galaxies or Mercury,
  but fails to recover MACS1931 and slightly worsens the only directly matched
  historical-validation system.

## Frozen shortlist

The shortlist was selected from the 73-formula structural screen before any
new exact score:

| Structural test | Reason included |
|---|---|
| screen softness 0.98 | smallest partially shared fixed-geometry direction |
| screen softness 0.90 | first fixed-geometry all-root screen result |
| dynamics addition 1.02 | galaxy/Solar-favored direction |
| dynamics addition 0.98 | CLASH-favored direction, still Solar safe |
| lensing addition 0.98 | CLASH/RX fixed-geometry direction |
| lensing addition 1.02 | other-four fixed-geometry direction |
| potential-to-radius coupling +0.01 | tiny SPARC/CLASH/RX direction and root recovery |

The P0554 parent was included as the eighth formula. All SPARC, CLASH, and
Solar scores were recomputed using the exact nonlinear dynamical response.

## Raw exact-refit results

| Formula | RX J2129 RMS | Other-three matched RMS | All-four matched change | Roots |
|---|---:|---:|---:|---:|
| P0554 parent | 1.256 arcsec | 21.391 arcsec | reference | 17/18 |
| screen softness 0.98 | 1.370 | 21.405 | -0.08% | 17/18 |
| screen softness 0.90 | 1.132 | 21.390 | +0.02% | 17/18 |
| dynamics addition 1.02 | 1.703 | 21.348 | +0.15% | 17/18 |
| dynamics addition 0.98 | 1.122 | 20.859 | **+2.50%** | 17/18 |
| lensing addition 0.98 | **1.114** | 20.941 | **+2.11%** | 17/18 |
| lensing addition 1.02 | 2.570 | 21.345 | +0.03% | 17/18 |
| potential-to-radius +0.01 | 1.303 | 21.400 | -0.05% | 17/18 |

The table's aggregate change uses RX J2129, MACS0329, MACS0429, and MACS1115,
which are complete for both parent and candidate. It does not treat missing
MACS1931 as zero error.

## Which fixed directions survived

### Screen shape

The smallest screen change looked favorable in fixed RX geometry but reverses
after refitting: 1.256 becomes 1.370 arcsec. The larger 0.90 change retains an
RX improvement, to 1.132 arcsec, but its other-cluster change is only +0.0065%
and it no longer recovers the MACS1931 root.

Thus screen softness remains a scalar/transition control but its apparent
topology advantage was a frozen-geometry artifact.

### Dynamics addition

Lowering the dynamics softness to 0.98 survives in both raw scopes:

- RX improves by 10.72%;
- the other three complete clusters improve by 2.49%;
- CLASH improves from 0.19908 to 0.19650 dex; and
- Mercury remains inside the analytic margin at -2.872 mas/century.

The cost is SPARC: error rises from 12.571 to 12.708 km/s. Increasing the
softness to 1.02 improves SPARC to 12.462 and Solar safety, but worsens CLASH
and RX J2129. The galaxy/cluster tradeoff therefore survives ordinary geometry
freedom.

### Lensing addition

Lowering lensing softness to 0.98 is the most coherent surviving lensing-only
change:

- SPARC and Mercury are exactly unchanged;
- CLASH improves from 0.19908 to 0.19641 dex;
- RX J2129 improves from 1.256 to 1.114 arcsec; and
- the other three complete clusters improve from 21.391 to 20.941 arcsec.

The fixed other-cluster direction actually reverses from a 0.67% worsening to
a 2.10% improvement after refitting. This shows why geometry freedom was
necessary.

It is not a universal cluster result. MACS1931 still lacks one root, and on the
historical MACS1115+1931 validation pair only MACS1115 is comparable; it
worsens by 0.0066%. A light-only nonlinear response also remains a
phenomenological slip rather than a derived field equation.

### Potential-to-radius coupling

The +0.01 coupling's fixed RX improvement reverses to a 3.74% worsening after
refitting. Its all-system matched change is -0.05%, and its root recovery
disappears. This operator is largely absorbed by ordinary geometry at the
tested scale.

## Topology and geometry lessons

The most important negative result is exact:

> All eight formulas return to 17/18 roots after ordinary geometry refitting.

The earlier root creation was not a robust structural prediction. It reflected
where a fixed geometry placed one image relative to a caustic. This sharpens
the requirement for a future two-dimensional operator: it must predict image
multiplicity after nuisance geometry is allowed to respond.

Thirty of the 40 fitted geometries touch at least one declared nuisance bound,
including four of five parent fits. That signals limited geometry closure in
these simplified radial models and restricts the strength of continuous RMS
claims. It does not explain away the root result, because the same incomplete
MACS1931 topology persists across every formula.

No exact formula has all five systems complete, so there is no valid
all-five comparison with the 9.989-arcsecond limited compact-halo validation
aggregate.

## Universal conclusions retained

1. Fixed-geometry root recovery is not robust evidence for a structural term.
2. The dynamics addition law's galaxy/cluster tradeoff survives geometry
   refitting.
3. A softer lensing addition law produces the broadest continuous cluster-
   lensing improvement among complete systems, but not universal topology or
   validation success.
4. Small screen and potential-to-radius effects are substantially absorbed by
   ordinary geometry.
5. The missing physics is still two-dimensional field topology rather than
   another radial coefficient.

## Limits

The shortlist and clusters are spent exploratory data, CLASH accelerations are
derived through conventional GR/NFW models, and the Solar checks are analytic
proxies. The six-parameter lens model is deliberately simple, and most fits
touch a nuisance boundary. The compact-halo comparator is limited rather than
a full modern dark-matter analysis.

## Reproduction

```powershell
python scripts/run_p0554_structural_exact_refit.py
python -m pytest tests/test_p0554_structural_exact_refit.py -q
```

Machine-readable outputs are in `results/p0554_structural_exact_refit/`.

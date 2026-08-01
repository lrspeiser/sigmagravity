# P0600: intermediate-potential BCG bridge

P0599 places a bounded potential threshold between the measured SPARC and
CLASH ranges. P0600 applies that formula unchanged to 34 SPIDERS-MaNGA BCGs:
11 use direct Tian et al. accelerations and 23 use a disjointly calibrated
DynPop/NSA proxy.

The BCG catalog does not contain a resolved full-host baryon profile. The test
therefore brackets three missing quantities without fitting the formula:

- BCG-only, BCG plus eRASS median/p90 gas, or cosmic-baryon host potential;
- a Hernquist BCG radial shape, the measured CLASH median shape, or neutral
  shape response;
- screening by local central acceleration or the weak host limit.

The primary bracket uses BCG plus median eRASS gas potential, CLASH median
radial shape, and weak-host screening. Differences among brackets measure which
new observation is needed most; a favorable post-hoc bracket does not replace
the frozen primary.

Run:

```powershell
python scripts/run_p0600_bcg_potential_gap.py
python -m pytest tests/test_p0600_bcg_potential_gap_results.py -q
```

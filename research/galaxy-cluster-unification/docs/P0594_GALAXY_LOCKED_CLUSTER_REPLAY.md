# P0594: galaxy-locked cluster replay

P0594 applies the spatial settings selected only from P0593B's galaxy discovery
sample to the ten-cluster morphology data. It compares three maps: unsmeared
member light, the galaxy-locked spatial law, and the independently selected
cluster law.

The galaxy acceleration gate cannot be evaluated because the ten-cluster input
is relative member light without a physical total baryonic mass normalization.
The replay therefore uses the favorable low-acceleration limit, `S=1`, so the
spatial mixing fraction is its maximum 0.25. This is an upper-bound transfer
test, not a measurement of cluster activation. P0592 had also already disclosed
the main ensemble score of this grid point; the new checks are its full
realization behavior and independent GLAFIC comparison.

Run:

```powershell
python scripts/run_p0594_galaxy_locked_cluster_replay.py
python -m pytest tests/test_p0594_galaxy_locked_cluster_replay_results.py -q
```

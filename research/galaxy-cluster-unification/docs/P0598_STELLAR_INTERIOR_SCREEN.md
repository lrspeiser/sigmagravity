# P0598: stellar-interior acceleration screening

P0597's ungated law is almost invisible to exterior planets because a spherical
redistribution remains enclosed. That does not make it safe inside the source.
P0598 therefore applies the same response to a uniform-density Solar-mass,
Solar-radius sphere and compares acceleration-gate powers `n=0,1,2,4`.

The source gate is

`S_n = 1 / (1 + (g_b(R80)/a0)^n)`.

The experiment also computes the gate at the physically normalized MACS J0416
baryon field. This tests the desired separation: negligible response in a
high-acceleration star, retained response in a low-acceleration cluster, and
minimal loss of galaxy accuracy.

The Solar profile is only a screening diagnostic. A real theory would still
need a standard-solar-model or helioseismic calculation and a covariant metric.

Run:

```powershell
python scripts/run_p0598_stellar_interior_screen.py
python -m pytest tests/test_p0598_stellar_interior_screen_results.py -q
```

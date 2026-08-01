# P0608B optimizer-basin audit

**Superseded:** a one-start fit with an explicit initial geometry is deterministic,
so these repeats did not sample independent basins. The corrected random-start
audit is in `results/p0608c_tomography_random_start_robustness/`.

Across 48 one-start refits, the best gamma=0 versus gamma=1 training difference is 0.000063 arcsec, while the pooled 16-84% basin span is 0.000063 arcsec. The corresponding held-out difference/span is 0.000050/0.000050 arcsec.

The current angular route is too small and too degenerate with structural lens geometry to identify a redshift exponent.

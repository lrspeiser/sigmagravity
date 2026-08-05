# V19CK single-RMF diagnostic

V19CK is an operational diagnostic for the one Chandra response cell that failed twice while 5,081 other cells passed. The two ordinary `specextract` failures produced byte-identical logs, so another blind retry is forbidden.

The protocol copies only the already materialized source and blank-sky event subsets into a new, non-admitted workspace. It repeats the same response settings once with maximum CIAO verbosity and, if needed, calls `mkacisrmf` directly on the generated PHA WMAP to expose the underlying error that `specextract` normally wraps.

The diagnostic may not modify a recovery checkpoint, drop the cell, move the response position, change weighting or energy settings, resume the source-only chain, or access any lensing, halo, action, gravity, or holdout payload. Even if it creates a valid response, that response is diagnostic and cannot enter the final archive. Any remedy requires a separately frozen, physically equivalent protocol after the exact cause is known.

## Result

The direct diagnostic captured the swallowed error. The `det=8` WMAP has two nonzero detector-bin centers, `(4044.5, 4196.5)` and `(4044.5, 4212.5)`. Both fall outside valid chip coordinates, so `mkacisrmf` reports: `No non-zero pixels map to valid chip coordinates in the supplied wmap`.

This is the documented ACIS chip-edge WMAP failure mode, not an absence of events or an astrophysical failure. CIAO documents that weighted RMFs use the spectrum WMAP and that chip-edge coordinate mapping can fail. V19CK still authorizes no remedy. V19CM separately freezes a finer `det=1` WMAP test that keeps all science selections and weighting unchanged.

Official references: [CIAO WMAP definition](https://cxc.cfa.harvard.edu/ciao/dictionary/wmap.html), [mkacisrmf chip-edge caveat](https://cxc.cfa.harvard.edu/ciao/ahelp/mkacisrmf.html), and [specextract WMAP parameters](https://cxc.cfa.harvard.edu/ciao/ahelp/specextract.html).

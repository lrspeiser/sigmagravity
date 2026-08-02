# P0687 system path-coordinate snapshot

Frozen before scores: 2026-08-02

## Bottom line

A single baryon-derived system concentration coordinate repairs the
outward-rising exponent that failed raw topology, but loses too much of the
spent cross-cluster amplitude fit. It does not advance.

The fixed formula remains `1.029x` fixed RAR on 131 galaxies and every cluster
exponent decreases outward. Its cluster score is `0.234 dex` on five systems
and `0.201 dex` on the reliable three, closing 59.9% of the fixed-RAR gap.
The frozen requirements were `<=0.20 dex` for both subsets and at least 75%
gap closure.

The capped-local control scores `0.154/0.177 dex` but still increases outward,
showing that the local radial information is useful and the central hollow is
the specific defect.

The next generator is the parameter-free inward monotone majorant

\[
p_{\rm env}(r)=\max_{s\ge r}p_{\rm local}(s),
\]

which fills the hollow without lowering the successful outer response.

## Hosted researcher model

The deployment design remains Vercel for the researcher interface and typed
gateway, with isolated Cloud Run Jobs or Modal workers for formula evaluation,
3D field solves, and lens-root searches. Named real systems and seeded
synthetic galaxies/clusters will use immutable data, solver, formula, and
comparator hashes.

See [`the public simulator/API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md).

## Canonical evidence

- [`P0687 results`](../../docs/P0687_SYSTEM_PATH_COORDINATE_QUMOND_RESULTS.md)
- [`P0685-P0686 3D/topology results`](../../docs/P0685_P0686_LOCKED_PATH_QUMOND_RESULTS.md)
- [`P0684 radial generator`](../../docs/P0684_PATH_DILUTED_QUMOND_RESULTS.md)

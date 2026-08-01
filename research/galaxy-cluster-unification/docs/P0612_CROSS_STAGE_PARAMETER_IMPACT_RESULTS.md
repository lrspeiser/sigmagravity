# P0612: cross-stage parameter-impact synthesis

## Question

Which small changes to the gravity-routing formulas repeatedly move real galaxy
or cluster observables, and which merely produce a large but destructive
response?

## Method

P0612 reads the completed one-coordinate and factorial response tables from
P0554 through P0611. It does not refit any object. Because the source metrics
include arcseconds, km/s, Jensen-Shannon divergence, and image-root counts, each
span is divided by the largest span in the same stage and metric. A value of 1
therefore means “most sensitive coordinate in this particular comparison,” not
“100% improvement.”

The atlas retains the later transfer outcome for every stage. This prevents a
coordinate such as routed fraction from being ranked as good physics simply
because it moves the answer dramatically.

## Results

Run `python scripts/run_p0612_cross_stage_parameter_impact.py` for the current
machine-readable counts and ranks. The stable qualitative result is:

1. Routed strength or fraction is highly sensitive in exact raw lensing, but
   the response is usually destructive. P0606 selects zero routing on its
   training score, and five of eight fraction variants lose an image root.
2. Width or spatial support is the most recurrent geometric coordinate. It
   matters in reconstructed cluster maps, exact raw positions, and SPARC, but
   no tested width law has yet transferred as a universal correction.
3. Endpoint residence is repeatedly the least-bad return rule for raw heldout
   positions and galaxy rotation. Conservation still leaves its SPARC error
   about 6.85 times fixed RAR, so endpoint preference is a geometric clue, not
   a solution.
4. Path length is a real cross-domain lever. Its preferred value changes with
   observable and does not rescue the amplitude problem.
5. Anisotropy strength can move individual clusters strongly, while its
   preferred sign changes by cluster. Tensor orientation itself is weak.
6. Gates and saturation can determine whether exact roots exist even when
   their smooth RMS or map effect looks small. The frozen dual-component
   misalignment gate nevertheless failed its P0611 transfer.
7. Solar safety has mostly been supplied by a high-acceleration screen or an
   exact symmetry null. That is necessary bookkeeping, not evidence that the
   cluster mechanism is correct.

## Next bounded formula test

The next experiment should not reopen member-to-member networks, tensor
orientation, or the dual-misalignment gate. Those branches have already failed
transfer. The narrow remaining combination is a bounded endpoint-deposition
law with:

- width expressed as a universal fraction of baryonic `R80`;
- smooth saturation to prevent root loss;
- one universal routed strength;
- no cluster- or galaxy-specific gravity setting.

It must report four outputs separately: SPARC outer velocity error, raw-cluster
heldout RMS, exact image-root count, and Solar/Mercury proxies. A strong change
with failed roots is a rejection, not an improvement.

## Claim limits

This is a synthesis of project-spent experiments. It measures recurrence and
sensitivity, not discovery significance. Reconstructed convergence maps and
raw multiple-image positions are intentionally kept as separate observables.

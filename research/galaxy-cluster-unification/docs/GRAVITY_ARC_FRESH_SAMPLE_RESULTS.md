# Gravity-arc tomography: frozen ten-cluster test

## Bottom line

The exact C0351 center-return formula discovered in the first three clusters
does not transfer strongly enough to ten untouched RELICS clusters. It fails
both the preregistered Lenstool confirmation gate and the independent-method
GLAFIC gate. The failure is informative: the data are much more sensitive to
the **direction and spatial width** of the return than to the redistributed
fraction, distance exponent, or a modest change in return distance.

One small change—widening the return endpoint from 50 to 60 kpc—improves C0351
in 8/10 systems under both reconstruction methods. It is a promising new
candidate, not a confirmed result, because these ten systems revealed it.

## What was frozen

The discovery clusters A2537, MACS J0417, and MACS J0949 were excluded. Before
downloading or viewing the new convergence maps, ten replacement systems and
the following three primary candidates were locked:

| Candidate | Meaning |
|---|---|
| LOCAL75 | Put the galaxy light where it is and smooth it by 75 kpc |
| CENTRAL100 | Ignore individual galaxies and place a 100 kpc Gaussian at the baryonic luminosity center |
| C0351 | Return 50% of each galaxy's influence toward the baryonic center, using $L_0=250$ kpc, $\nu=-0.5$, and a 50 kpc endpoint width |

Twelve additional candidates changed exactly one ingredient at a time. They
were frozen as parameter-impact diagnostics and were not allowed to replace
C0351 in the confirmation test.

The sample contains 46,917 catalog entries, 4,281 usable F160W galaxies inside
300 kpc, 832 strict photo-z members, 1,000 Lenstool uncertainty maps, and ten
GLAFIC best maps. All map geometry covered at least 99.96% of the common
aperture before any convergence values were inspected.

## Locked C0351 result

Lower Jensen--Shannon (JS) divergence is better.

| Test | Lenstool ensemble | GLAFIC best |
|---|---:|---:|
| Median improvement over local light | 4.4% | 7.8% |
| Systems better than local light | 6/10 | 5/10 |
| Median improvement over central halo | -2.7% | 25.3% |
| Systems better than central halo | 4/10 | 8/10 |
| Median Pearson change versus best null | -0.046 | -0.015 |
| Frozen gate | fail | fail |

The two reconstruction methods agree on the sign of C0351's advantage over the
best null in 7/10 systems, exactly at that sub-gate's threshold. They disagree
about which simple null is strongest: Lenstool often favors the central halo,
whereas GLAFIC more often lets C0351 beat that halo but favors local light in
half the systems. The median Lenstool--GLAFIC target disagreement is JS 0.0165
with Pearson 0.962, but RXC J0600.1-2007 reaches JS 0.0614. Reconstruction
choice is therefore not a small technical detail for every cluster.

## Which formula changes matter

The table reports the span in median JS produced by the frozen variants. It is
an effect-size ranking, not a new fit.

| Changed ingredient | Lenstool JS span | GLAFIC JS span | Interpretation |
|---|---:|---:|---|
| Direction | 0.0332 | 0.0335 | Dominant; center return is much better than rings, neighbors, or summed external-field direction |
| Endpoint width | 0.0164 | 0.0140 | Second strongest; 40 kpc hurts, 60 kpc usually helps |
| Endpoint versus tube | 0.0072 | 0.0086 | Depositing along the whole path is generally worse than depositing near the return point |
| Return scale | 0.0066 | 0.0073 | Moderate sensitivity between 200 and 300 kpc |
| Distance exponent | 0.0034 | 0.0021 | Weak sensitivity near $\nu=-0.5$ |
| Returned fraction | 0.0006 | 0.0011 | Almost unidentified by normalized morphology between 40% and 60% |

Changing the direction to isotropic rings worsens median JS by 0.0332 in
Lenstool and 0.0335 in GLAFIC. Strongest-neighbor and summed-external-field
directions are also worse. This is the clearest transferable finding: a
coherent cluster-scale center matters much more than the exact amount of
influence placed in the return channel.

The favorable W060 perturbation keeps every C0351 parameter fixed except
$w=60$ kpc. It improves median JS by 0.00558 in Lenstool and 0.00372 in GLAFIC,
winning in 8/10 systems in each. If it had been the preregistered primary, its
Lenstool medians would have been 12.6% better than local light and 13.9% better
than the central halo, with seven wins against each. It would still miss the
required eight local wins and the Pearson-loss gate. It is close, not a hidden
pass.

## A more useful scale-free expression

The ten systems have median baryonic $R_{80}=259$ kpc. Thus the favorable
cluster numbers are approximately

$$
L_0\simeq R_{80},\qquad w\simeq0.23R_{80}.
$$

That suggests replacing fixed cluster units with a dimensionless kernel:

$$
\ell_i=\lambda R_b
\left(\frac{r_i}{R_b}\right)^\nu,
\qquad
w=\eta R_b,
$$

with exploratory values $\lambda\sim1$, $\eta\sim0.23$, and $\nu\sim-0.5$.
This is a cleaner cross-domain hypothesis because $R_b$ can be measured for a
cluster, galaxy, or compact system. The current data do not establish that
these ratios are universal.

## What differs between successful and unsuccessful clusters

A post-confirmation morphology analysis tested 117 correlations. The strongest
clue is that C0351's advantage over local light grows when the baryonic member
distribution is more radially extended and less sharply concentrated:

$$
\rho_{\rm Spearman}=0.855
$$

for $R_{50}/R_{80}$ versus the Lenstool improvement. The sign remains positive
in every leave-one-system-out repetition. Rounder member distributions also
tend to help. Wider 60 kpc endpoints appear most useful for lopsided systems.

These are hypotheses, not discoveries. With only ten systems and 117 searched
correlations, no relationship survives a 5% false-discovery-rate correction;
the leading adjusted value is $q=0.130$.

## Physical interpretation

The tested kernel can be written schematically as

$$
K(\mathbf y|\mathbf x)=
(1-f)G_{w_l}(\mathbf y-\mathbf x)+
fG_w\!\left[\mathbf y-\mathbf x-
\ell(r)\hat{\mathbf c}(\mathbf x)\right],
$$

where $\hat{\mathbf c}$ points toward the baryonic luminosity center. The
results favor a broad, coherent central return, but that can imitate a smooth
central halo. They do not yet show literal curved gravity trajectories, and the
endpoint preference over a projected tube weakens that literal interpretation.

There is also a cross-domain limitation. A normalized redistribution conserves
the total effective source and cannot by itself maintain excess gravity far
outside a galaxy's baryonic extent. Galaxy rotation curves would require either
path multiplicity/residence time or a field-strength closure. Such an amplitude
term must be screened in the Solar System. Those additions are not licensed by
this shape-only cluster result and must be tested independently.

## Next falsification

The next clean candidate is the scale-free W060 form:

$$
\lambda=1,quad \eta=0.23,quad \nu=-0.5,quad f=0.5,
$$

with center-directed endpoints. It should be locked and tested on the remaining
unused RELICS systems, without changing its parameters. In parallel, the
project should derive a path-multiplicity amplitude and test it against SPARC
rotation curves and Solar-System bounds. If one environmental screen is needed,
its parameters must be common across all three domains.

## Reproduction

```powershell
python scripts/download_gravity_arc_fresh_sample.py
python scripts/audit_gravity_arc_fresh_sample.py
python scripts/run_gravity_arc_fresh_sample.py
python scripts/analyze_gravity_arc_fresh_drivers.py
pytest tests/test_gravity_arc_fresh_sample_results.py
```

The complete numerical outputs are in `results/gravity_arc_fresh_sample`.

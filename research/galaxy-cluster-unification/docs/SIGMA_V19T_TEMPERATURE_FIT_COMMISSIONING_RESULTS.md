# Sigma V19T temperature-fit commissioning result

V19T **passed every frozen fit-commissioning gate**.  The response-and-fit
production pipeline is now technically authorized.

The model and gates were pushed before Sherpa opened the grouped spectrum:

- `xstbabs * xsapec` over 0.5–7.0 keV;
- Asplund et al. 2009 abundances;
- fixed HI4PI \(N_H=4.38\times10^{20}\ {\rm cm^{-2}}\);
- fixed Bullet redshift \(z=0.296\);
- fixed abundance \(Z=0.3\,Z_\odot\) for commissioning;
- temperature bounded to 1–30 keV and normalization to \(10^{-10}\)–1;
- 25-count grouping, background subtraction, `chi2xspecvar`, and the inherited
  LevMar → Nelder–Mead → LevMar fallback sequence;
- independent starting temperatures of 3, 8 and 15 keV.

## Result

| Check | Result |
|---|---:|
| Best-fit temperature | 15.2383 keV |
| 68% profile interval | 11.0934–22.7730 keV |
| Fractional 68% half-width | 0.3832 |
| Fit statistic / degrees of freedom | 16.9733 / 23 |
| Reduced statistic | 0.7380 |
| Temperature from 3 keV start | 15.2383 keV |
| Temperature from 8 keV start | 15.2383 keV |
| Temperature from 15 keV start | 15.2383 keV |
| Fractional multistart spread | \(7.59\times10^{-12}\) |
| Parameter on a bound | no |
| All frozen gates | pass |

The high temperature is plausible for hot Bullet intracluster gas, but this
number is **not** being promoted as a regional or global scientific
measurement.  It comes from only one observation/CCD contribution to one
adaptive region, selected for response commissioning by maximum source count.
A physical temperature map requires combining every supporting observation and
CCD for every admitted region under the same frozen rules.

The main conclusion is narrower and important: the exact V19Q event support,
V19R calibrated response, V19S Galactic absorption, grouping, background
scaling, XSPEC plasma model, optimizer, and confidence calculation form a
working end-to-end chain.  No thermal-stress map, lensing prediction, or gravity
parameter was constructed or changed.

# P0637 theory-blind photometric metadata

## Purpose

P0637 fixes the physical scale and viewing geometry needed to turn the 13 sealed
LITTLE THINGS moment-0 and optical images into baryonic mass maps. It does this
without using a target velocity field, rotation curve, or kinematic orientation.

## Frozen rules

The source is the Hunter and Elmegreen (2006) VizieR catalog
`J/ApJS/162/49`. For each frozen P0633 galaxy:

1. distance and foreground reddening come from the unique table 1 row;
2. PA, axis ratio, optical center, and pixel scale come from the unique V-band
   table 2 geometry;
3. total V luminosity and B-V color use the largest table 3 aperture;
4. inclination is calculated from the photometric axis ratio with the source
   catalog's single irregular-galaxy thickness, `q0=0.3`;
5. table 5 inclination is only a rounding audit of that calculation;
6. stellar mass initially uses one universal V-band mass-to-light value of 0.5,
   with mandatory universal 0.25 and 1.0 sensitivity runs. It is never fitted
   separately by galaxy.

The inclination rule is

```text
cos(i)^2 = clip(((b/a)^2 - q0^2) / (1 - q0^2), 0, 1).
```

## Outcome

All 13 galaxies have a complete, unique photometric input row. The computed
inclinations reproduce the catalog's rounded values within one degree. No
sealed kinematic product has been downloaded, parsed, or scored.

P0637 therefore closes the geometry-leakage risk. The next operation is a
deterministic registration of the H I and V-band maps onto physical disk grids,
followed by Newtonian, QUMOND, and AQUAL predictions while the target rotation
curves remain sealed.

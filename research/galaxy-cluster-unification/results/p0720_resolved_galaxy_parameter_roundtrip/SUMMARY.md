# P0720 resolved-galaxy parameter round trip

- Status: **PASS**
- Real resolved galaxies: **13**
- Gravity parameters used during extraction: **0**
- Observed velocity targets used during extraction: **no**
- Total-map median normalized error: **0.168**
- Total-map worst normalized error: **0.257**
- Total-map median pixel correlation: **0.986**
- Maximum 2D mass-closure error: **3.978e-16**
- Maximum 3D-to-2D projection error: **4.155e-16**
- Median input-cell / numeric-representation ratio: **8.4×**

This is a representation and generation result, not a gravity result.  Gas and
stellar maps were reduced to radial/Fourier structure plus signed local
features, then regenerated without reading a rotation curve or fitting a
gravity parameter.  The 3D products are ensembles of declared vertical priors:
different thickness and flaring choices project to the same 2D mass map, which
is the physically honest treatment of the missing depth information.

The stellar maps inherit P0639's fixed V-band mass-to-light assumption of 0.5.
The commissioning gates were informed by exploratory work on these same maps;
they must not be described as blind validation.

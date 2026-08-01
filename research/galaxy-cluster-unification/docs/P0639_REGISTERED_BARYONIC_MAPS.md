# P0639 registered physical baryonic maps

## Purpose

P0639 converts the permitted H I moment-0 and V-band images for all 13 sealed
LITTLE THINGS galaxies into registered face-on physical mass maps. These are
the inputs to the blind Newtonian, QUMOND, AQUAL, and geometric-transport
predictions; no target velocity product is opened here.

## Universal construction rules

- The independently measured distance, photometric center, PA, axis ratio, and
  inclination come from P0637.
- P0638 Gaia WCS places each optical image on the radio celestial frame.
- The grid contains an odd 65 to 513 cells per axis. Its half-width is 1.25
  times the larger of the 99.5%-flux H I radius and published V-band aperture,
  with a universal 0.75 kpc floor. Cell count rises automatically until the
  radio beam is sampled by approximately one cell; it is not selected from a
  target outcome.
- The 21-cm map supplies its own integrated H I mass. A single factor of 1.33
  adds helium.
- V-band light uses one universal mass-to-light ratio of 0.5; 0.25 and 1.0 are
  mandatory universal sensitivity cases. No value is fitted by galaxy.
- Foreground peaks are bounded by the smaller of a 99.5th-percentile cap and
  50 times the measured sky noise, applied identically to every target.

The registered map is permitted a small interpolation correction to reproduce
the directly integrated moment-0 mass. A large correction fails the stage.
Mass on outer grid cells is also bounded, preventing the field solver from
mistaking a truncated source for a physical edge.

## Outputs

Each object receives a compressed `axis_kpc`, `gas`, `stars`, and `total` map.
The audit records mass conservation, beam resolution, gas fraction, H I
concentration, lopsidedness, clumpiness, inclination, and the measured gas-star
centroid offset. These outcome-blind coordinates will later determine whether
a candidate fails preferentially for asymmetry, clumpiness, or geometry.

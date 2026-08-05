# Sigma V19AX DELVE DR3 coadd acquisition plan

## Purpose

V19AW showed that another catalog crossmatch cannot reproduce most of the
frozen HSC candidate identities. V19AX therefore asks a narrower question: are
the homogeneous DELVE DR3 coadd pixels, masks and weights available over every
candidate position so that forced photometry can later be tested without a
second catalog identity?

This stage acquires pixels but does not measure any source.

## Public products and preflight

[NOIRLab describes DELVE DR3](https://datalab.noirlab.edu/data/delve) as a
release of coadded `griz` images and catalogs from the DESDM processing. Its
[Simple Image Access documentation](https://datalab.noirlab.edu/docs/manual/UsingAstroDataLab/DataAccessInterfaces/SimpleImageAccessSIA/SimpleImageAccessSIA.html)
lists the public `https://datalab.noirlab.edu/sia/delve_dr3` endpoint.

A metadata-only center query returned 27 rows. Small center cutouts established
that each standard `griz` coadd has three usable planes:

- extension 1: calibrated image, `MAGZERO=30`, in counts per second;
- extension 2: integral mask; and
- extension 3: inverse-variance weight.

The preflight did not measure flux at an anchor or candidate position. The
standard coadds are selected uniquely; parallel `_nobkg` products are excluded
before the full-field pixels are opened.

## Frozen field and gates

One 0.17-degree square cutout centered at
`(104.6247543743987, -55.94659781854907)` spans the extrema of all 568 frozen
candidate coordinates with at least ten pixels of intended edge margin.

The acquisition passes only if:

- the SIA response still has exactly 27 rows;
- exactly 12 products are selected—image, mask and weight for each `griz` band;
- every product is a 2D celestial-WCS FITS image at `0.26295` arcsec/pixel;
- every product has a dimension between 2,200 and 2,400 pixels and all 12 share
  one shape;
- at least 99% of every plane is finite and at least 99% of every weight plane
  is positive;
- all 568 candidate coordinates are inside every plane with at least a
  ten-pixel margin; and
- no anchor or candidate photometry, association or mass inference is made.

## What a pass permits

A pass permits a new development-only experiment comparing fixed aperture and
simultaneous crowded-field models on the ten development anchors. The five
unchanged validation anchors then choose whether that measurement family is
adequate. Candidate pixels remain uninterpreted until that protocol is frozen.

The coadds are an input to the baryonic source map needed by the
long-wavelength hypothesis. They do not test the gravitational wavelength,
field amplitude or source equation themselves.

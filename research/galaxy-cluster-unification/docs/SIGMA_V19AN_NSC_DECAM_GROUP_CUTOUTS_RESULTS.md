# Sigma V19AN NSC DECam grouped-cutout results

## Decision

**Failed closed at the first transport gate.**  The first frozen request,
`c4d_121208_034504_ooi_r_d2_ext35`, returned HTTP 500 on all three attempts.
No V19AN image file or downstream manifest was written, and V19AN does not
authorize photometry.

The failure report records the exact config, runner and group-plan hashes.

## What failed

The request size was not the cause.  A post-failure transport diagnostic found:

- the same NSC SIA descriptor fails at its original 0.01-degree size;
- removing the cutout parameters returns an empty body;
- a largest-footprint `tu...` control returns a valid 66,286,080-byte FITS
  payload; and
- the pattern is confined to the older `*_d2` descriptor family in this
  sample.

Those stale descriptors account for 22 exposures, 37 exposure/extension
groups and 275 of the 1,032 measurement memberships.  The frozen plan contains
102 non-`d2` groups, but the all-groups rule correctly prevented V19AN from
silently advancing on only those accessible images.

## Authoritative fallback discovered

The NOIRLab Astro Data Archive currently holds the same failed observation as
the calibrated `c4d_121208_034504_ooi_r_bullet.fits.fz` product, MD5
`0fd3776025534b82ef390843b3bb6cef`.  Its public selected-HDU endpoint returns
the primary header plus the matching CCD in 5,296,320 bytes.

Crucially, the archive spatial query maps the failed anchor to HDU 34, whereas
the stale NSC descriptor said extension 35.  Therefore a successor cannot
repair the URLs by renaming the file or subtracting one from every extension.
It must query and freeze the authoritative archive MD5 and HDU identity at the
frozen sky position for every affected group.

## Next protocol

V19AO should preserve the 139-group membership plan but use two source routes:

1. unchanged NSC SIA grouped cutouts for the 102 non-`d2` groups; and
2. authoritative archive file/HDU subsets for all 37 stale groups.

Every fallback mapping must be established from metadata and committed before
pixel retrieval.  All anchors must then pass FITS, celestial-WCS and containment
checks.  No exposure may be dropped, and no photometric, PSF or validation
quantity may select the source route.

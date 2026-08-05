# Sigma V19AM NSC SIA anchor-coverage plan

## Purpose

V19AL closed the FORS1 astrometric strategy because the shared circular PSF
model failed in two of three filters.  V19AM tests a lower-assumption source
route: whether independently calibrated DECam images are addressable for all
per-exposure measurements of the fifteen singleton anchors that were already
opened by V19AB.

NSF NOIRLab documents NSC DR2 image discovery through the Simple Image Access
service at `https://datalab.noirlab.edu/sia/nsc_dr2`.  The service returns image
metadata and a detector-extension cutout URL.  V19AM queries only the metadata;
it does not request any image pixels.

## Frozen sample and matching rule

- Use exactly the ten development and five validation anchors in the hashed
  V19AB commissioning sample.
- Retain all 1,032 V19AC measurement rows for those objects.
- Match a measurement only when its `exposure` equals the basename of the SIA
  `siaRef` value exactly.
- Require one and only one calibrated DECam `InstCal` image descriptor with the
  same band for every measurement.
- Fail rather than choose among zero or multiple matches.

There is no seeing, flag, depth, date, filter, PSF or validation-performance
selection.  All returned exact matches are written to a hashed manifest.

## Gate and boundary

The stage passes only if all 1,032 measurements have one exact descriptor and
all fifteen raw metadata responses are preserved and hashed.  Passing means
the independent-image route is addressable.  It does not mean that image
retrieval, sky subtraction, PSF estimation, deblending or profile photometry
will pass.

The next stage, if authorized by this audit, must freeze a cutout-footprint and
retrieval rule before downloading pixels.  It should exploit repeated
exposure/extension groups without discarding any anchor measurement and should
validate retrieval completeness and FITS/WCS integrity before inspecting
photometric residuals.

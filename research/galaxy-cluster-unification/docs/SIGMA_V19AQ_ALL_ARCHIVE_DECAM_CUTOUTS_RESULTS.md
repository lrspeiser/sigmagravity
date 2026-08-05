# Sigma V19AQ all-Archive DECam cutouts: failed-closed result

V19AQ removed the unreliable SIA cutout service, but it did not complete the
all-group acquisition.

It retrieved and structurally validated 49 selected detectors. At group 50,
the exact frozen `v1` Archive product failed decompression on all three frozen
attempts: CFITSIO reached the end of the compressed byte stream. The invalid
payload was never persisted.

## What the failure means

The sky position, detector WCS, and target membership are not the problem. The
failure is tied to a legacy processing product last updated in 2020. The same
original DECam observation has a more recently updated calibrated `ooi`
product in the Archive.

V19AQ's preference for an exact old basename was useful for lineage but too
strict for reliable retrieval. It can preserve a corrupted processing version
even when the Archive contains a current version of the same observation.

## Required successor rule

Use one version policy for every exposure: identify the original observation
from the already-frozen `c4d` prefix or SIA association ID, restrict candidates
to matching DECam InstCal `ooi` science images, and select the unique most
recently updated valid metadata row. Then independently resolve each detector
by the unique full-header WCS containing all frozen anchors.

This is a processing-version choice made without flux, PSF, morphology,
lensing, halo, or gravity values. It must be frozen before any successor image
payload is opened.

# Sigma V19AP header-WCS-resolved DECam cutouts: failed-closed result

V19AP fixed the stale-detector indexing problem but did not complete the image
acquisition.

It successfully retrieved and structurally validated the first 49 of 139
groups, including all 37 header-WCS-resolved Archive subsets. It then stopped
at group 50. The NSC SIA service returned HTTP 200 but only 28,800 bytes of FITS
headers and no declared image body. Four independent requests reproduced the
same header-only response. The invalid payload was never persisted.

## Additional integrity finding

The 12 readable SIA products persisted before the failure all contain FITS
`CHECKSUM` and `DATASUM` keywords that fail verification. The 37 Archive
selected-HDU reconstructions are structurally readable but omit those checksum
keywords. Therefore, later reports must distinguish:

- FITS structural readability and WCS/pixel containment;
- validity of any embedded FITS checksum keywords; and
- the SHA-256 we compute for the exact downloaded payload.

V19AP cannot claim that all groups or all checksum gates passed.

## Successor direction

The Archive selected-HDU route was successful for every group on which it was
used and returns one detector rather than a 33.5 MB full SIA product. The next
protocol will resolve all 82 exposures through the Archive, using exact frozen
filenames where available and the frozen SIA observation association for
renamed or stale products. Every detector will still be selected only by the
unique full-header WCS containing all frozen anchors. This removes the failed
SIA transport dependency without selecting on flux, PSF, morphology, or any
gravity outcome.

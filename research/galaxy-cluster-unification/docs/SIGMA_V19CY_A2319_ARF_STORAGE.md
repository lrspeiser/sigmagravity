# A2319 ARF storage manifest

The final P2 raytrace is stored losslessly as:

`data/processed/sigma_v19cy_a2319_response_aware_spectral/response_arfs/000102_open_0_cross_obsid/image_raytrace.fits.gz`

The uncompressed FITS is 2,156,051,520 bytes, which exceeds GitHub LFS's
2 GiB per-object limit by 8,567,872 bytes. The gzip archive is 1,433,363,371
bytes and has SHA-256:

`d0be1411271fb99ebec05d0867afcd72820efaab73fef71d699015283f9cc70a`

Restore the exact reported artifact with:

```bash
gzip -dc image_raytrace.fits.gz > image_raytrace.fits
```

The restored FITS must have SHA-256:

`2796d446cabbc95454497f2dac0a61ee1b906a0d792413e170db3861ca1d3171`

That is the hash recorded by the frozen ARF report. The archive was verified by
streaming decompression into SHA-256 before publication; no scientific bytes
were changed.

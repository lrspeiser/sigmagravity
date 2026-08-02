# P0653 compact nonlocal transport

## Conservation correction

P0653 reuses the P0652 path average, bounded activation, unit amplitude, and
50--58-arcsecond support. The only change is that the same compact taper is
applied again after path transport, before taking the divergence. This forces
the transported flux to vanish at the declared physical support boundary.

The correction succeeds numerically:

- integrated source fraction falls from `0.0495` to `1.09e-17`;
- maximum edge flux falls to exactly zero;
- normalized curl is `2.44e-17`; and
- all training, CV, and spent-heldout roots converge.

## Accuracy retained, but not enough to pass

The compact field retains most of the P0652 development signal:

- lambda-zero CV RMS: `2.760255 arcsec`;
- matched `m=3` CV RMS: `2.599360 arcsec`;
- P0652 open-path CV RMS: `2.075148 arcsec`;
- P0653 compact-path CV RMS: `2.154175 arcsec`;
- improvement versus zero field: `21.96%`; and
- improvement versus the matched multipole: `17.13%`.

Thirteen of fourteen gates pass. The sole failure is the predeclared
spent-heldout safety gate: `2.087516 arcsec`, `15.31%` worse than P0599 versus
the allowed 10%.

The compact taper reduces unit deflection RMS from `0.6384` in P0652 to
`0.4421 arcsec`. It therefore fixes conservation by changing more than an
irrelevant boundary zero mode; it suppresses part of the transported field.
The compact operator is rejected and does not advance to robustness.

## Remaining numerical question

The underlying P0652 map spans only -60 to +60 arcseconds, while the existing
streamline trace can extend 48 cells. A more faithful numerical boundary test
is to zero-pad the computational domain by those same 48 cells, carry the
unmodified P0652 path flux on that larger domain, and solve before sampling the
original region. This introduces no physical support, length, or fit. It tests
whether P0652's source leakage and some of its heldout error were finite-box
artifacts rather than reasons to impose a second physical taper.

No physical support radius may be retuned, and no blind outcome may be opened
for that numerical test.

## Reproduction

```powershell
python scripts/run_p0653_compact_nonlocal_transport.py
python -m pytest tests/test_p0653_compact_nonlocal_transport.py -q
```

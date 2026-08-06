# Sigma V19DD Bullet gain-weight transport

## Frozen preflight

The payload-blind preflight passes with 43 regions and 3,483 response cells.
It opened no source PHA or ARF payload and fit no source quantity. The frozen
config SHA-256 is
`c7143bff88665f5aa8602ba7febb9daea3083c71cdb4e813ed4b4d0268b81f1c`,
the runner SHA-256 is
`2eb59a3ad14073b81293824e73f041f4da190506b2cc54583dc3e695c0a78223`,
and the preflight-report SHA-256 is
`3237cdf936d43ae1a87eabb7fbd9bc7362ddbe7b079483375b198e9a1ce618a0`.

V19DD freezes the last mechanical input needed before source redshift fitting.
It builds the nine-ObsID integrated Bullet spectrum from all 3,483 primary
cells and derives each region's relative ObsID contribution at observed Fe-K.

For a cell, the contribution is its source exposure multiplied by its ARF at
`6.7 / (1 + 0.296)` keV. Contributions are summed within region and ObsID and
normalized across the nine observations. This is the same `exposure * area`
quantity that controls a narrow-line count contribution and the V19CW response
hierarchy. A direct-cell sum must reproduce the observation-hierarchical
integrated value to relative tolerance `1e-6`.

For independent gain measurements with parameters `p_i=(b_i,s_i)`, covariance
transport is

\[
p_{\rm eff}=\sum_i w_i p_i,
\qquad
C_{\rm eff}=\sum_i w_i^2 C_i.
\]

The squared weights are essential: directly assigning the median per-ObsID
gain error to a nine-observation combined spectrum would overstate the error,
while ignoring gain would understate it.

The transport also reports the response-weighted RMS dispersion of the nine
fitted mean gain corrections. This is not a calibration uncertainty: it
measures the known line broadening that can arise when differently shifted
observation responses are mixed. It remains separate from the covariance term
so the later fitter cannot count it twice.

No source line, temperature, abundance, redshift or velocity is fitted. ObsID
554, Abell 2146, lensing, gravity and action payloads remain sealed.

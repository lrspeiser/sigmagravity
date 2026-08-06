# Sigma V19DD Bullet gain-weight transport

## Outcome

The frozen transport passes every gate. The integrated spectrum uses all
3,483 primary cells exactly once and conserves all 674,283 full-PHA source
counts. Its source, background, ARF and RMF links are exact. All 387 regional
ObsID weights exist, every regional weight vector sums to one, and the maximum
direct-cell versus observation-hierarchical Fe-K response difference is
`3.25001e-8`, below the frozen `1e-6` gate.

The response-weighted regional gain covariance corresponds to 195--241 km/s
one-sigma uncertainty at observed Fe-K, with median 221 km/s. Separately, the
weighted RMS spread of the nine fitted mean gain corrections is 472--587 km/s,
with median 563 km/s. The latter is a possible response-mixture broadening,
not uncertainty in the weighted mean, and requires the preregistered
gain-corrected sign-topology robustness branch.

The result report SHA-256 is
`7d7156ebe888a1d249dc3c6e07a1a8311d7ba349e4d018f57bc9eee1330cc095`.
V19DD authorizes the frozen Bullet source-redshift fitter. It is not a source
velocity or gravity result.

## Frozen preflight

The payload-blind preflight passes with 43 regions and 3,483 response cells.
It opened no source PHA or ARF payload and fit no source quantity. The frozen
config SHA-256 is
`c7143bff88665f5aa8602ba7febb9daea3083c71cdb4e813ed4b4d0268b81f1c`,
the runner SHA-256 is
`2eb59a3ad14073b81293824e73f041f4da190506b2cc54583dc3e695c0a78223`,
and the preflight-report SHA-256 is
`3237cdf936d43ae1a87eabb7fbd9bc7362ddbe7b079483375b198e9a1ce618a0`.

V19DD froze the last mechanical input needed before source redshift fitting.
It built the nine-ObsID integrated Bullet spectrum from all 3,483 primary
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

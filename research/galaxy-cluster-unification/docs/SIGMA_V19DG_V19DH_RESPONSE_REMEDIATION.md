# Sigma V19DG--V19DH response-combination remediation

## Terminal result

The original V19X2 commissioning did not reach a spectral fit. CIAO
`addresp` failed while directly merging the 3,812-cell Bullet integrated
response. Its header merger reported inconsistent metadata and a missing
time keyword at one position even though the independently audited response
archive remained complete and hash exact.

Two materially separated implementation tests followed.

1. **V19DG/V19DG2 hierarchical CIAO combination failed.** It preserved source
   counts exactly, background counts to about $10^{-13}$, exposure exactly and
   ARFs to about $1.2\times10^{-7}$ relative. However, the second CIAO merge
   changed sparse RMF groups and channel spans. The frozen exact-structure gate
   therefore failed in all four controls. This method is rejected and is not
   eligible for a full successor.
2. **V19DH direct response-array parity passed.** It independently applies the
   published `addresp` weights to every input response without recursively
   recompressing an intermediate sparse RMF:

   \[
   A(E)=\frac{\sum_i t_i A_i(E)}{\sum_i t_i},
   \qquad
   R(E,c)=
   \frac{\sum_i t_i A_i(E)R_i(E,c)}{\sum_i t_i A_i(E)}.
   \]

   The first reconstruction used the two registered commissioning regions and
   two 64-cell manifest prefixes and is explicitly exploratory. V19DH then
   froze two nonoverlapping controls: the last 128 manifest cells of each
   cluster. Both passed every prospective gate.

| Prospective diagnostic | Bullet suffix 128 | Abell 2146 suffix 128 | Gate |
|---|---:|---:|---:|
| Maximum ARF relative difference | $5.87\times10^{-8}$ | $5.69\times10^{-8}$ | $\le10^{-6}$ |
| Maximum dense RMF absolute difference | $9.999\times10^{-7}$ | $9.995\times10^{-7}$ | no element $\ge10^{-6}$ |
| RMF row-sum maximum difference | $1.155\times10^{-5}$ | $8.671\times10^{-6}$ | $\le10^{-4}$ |
| Worst of four folded-spectrum relative L1 differences | $8.794\times10^{-6}$ | $6.235\times10^{-6}$ | $\le10^{-5}$ |

The residual RMF difference is bounded by CIAO's declared `addresp` sparse
cutoff of $10^{-6}$. V19DH evaluates flat, two power-law and 5-keV thermal
incident spectra so array agreement is also checked in detector-count space.

## What this authorizes

V19DH authorizes freezing a separately named writer and full-combination
successor using the direct formulas. The successor must still:

- construct valid OGIP ARF/RMF products and prove Sherpa can load them;
- sum source and ASCA-scaled background spectra without invoking `addresp`;
- conserve every source PHA count exactly;
- fit the two integrated and two preregistered regional commissioning spectra
  under the unchanged V19X2 model and gates;
- stop before all 494 regional fits unless every commissioning gate passes.

This is engineering evidence only. No temperature, thermodynamic-gradient
stress, baroclinicity, lensing, halo, action, gravity parameter or holdout
result was opened. The V19DG2 hierarchical failure remains terminal even
though the independent V19DH method passed.

The terminal reports are:

- `results/sigma_v19dg2_hierarchical_response_equivalence/report.json`
- `results/sigma_v19dh_direct_response_parity/report.json`

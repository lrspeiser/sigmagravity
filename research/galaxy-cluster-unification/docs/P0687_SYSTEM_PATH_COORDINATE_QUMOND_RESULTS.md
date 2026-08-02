# P0687 baryonic system path-coordinate QUMOND

Frozen before scores: 2026-08-02  
Verdict: integrity **passes**; fixed global-system primary **does not advance**

## Why this test exists

P0685-P0686 showed that the local coordinate

\[
\eta_b(r)={|\Phi_b(r)|\over r g_b(r)}
\]

produces enough total bending but creates a hollow cluster response. Its
channel exponent increases outward across the strong-lens region, and the raw
RX J2129 audit loses image roots, multiplicity, and parity.

P0687 replaces the local value with one scale-free coordinate calculated from
the whole baryonic system:

\[
\eta_{\rm sys}={\max_r |\Phi_b(r)|\over \max_r[r g_b(r)]}.
\]

The locked primary is

\[
p(r)=1+2{\chi_b(r)^4\over\chi_b(r)^4+(10^{-6})^4}
[\max(\eta_{\rm sys},1)]^{-1/2}.
\]

No galaxy or cluster has a fitted gravity value. The previously failed local
operator and a capped-local form were frozen as non-advancing controls.

## Baryonic coordinate

The coordinate separates the spent profiles without using their target
outcomes. The 131 galaxies have median `1.170` and range `0.880-1.854`.

| Cluster | `eta_sys` |
|---|---:|
| MACS0329 | 3.604 |
| MACS0429 | 4.256 |
| MACS1115 | 2.354 |
| MACS1931 | 3.175 |
| RXJ1347 | 3.014 |
| RXJ2129 | 2.993 |

## Frozen result

| Variant | Galaxy/RAR | All-five cluster RMS | Reliable-three RMS | Gap closed | Exponent nonincreasing |
|---|---:|---:|---:|---:|---|
| global system primary | **1.029** | **0.234 dex** | **0.201 dex** | **59.9%** | yes |
| local P0685 control | 1.030 | 0.145 dex | 0.165 dex | 75.1% | no |
| capped-local diagnostic | 1.030 | 0.154 dex | 0.177 dex | 73.7% | no |

The global coordinate fixes the topology-derived shape constraint and keeps
galaxies essentially at fixed RAR. It fails the frozen `0.20 dex` all-five and
reliable-three gates and the 75% gap-closure gate. The miss on the reliable
three is small but still a preregistered failure. It does not advance.

The capped-local diagnostic shows that useful information remains in the
radial coordinate: it retains most of the radial accuracy but also retains an
outward-rising exponent. It cannot be promoted.

## What this rules out

One object-wide concentration scalar is not enough to satisfy both the spent
cross-cluster amplitude targets and the topology-derived nonhollow constraint
with the locked P0685 constants. A per-cluster amplitude or an RX J2129 core
radius is still prohibited.

## Next operator

The next parameter-free generator retains local radial information but applies
the minimum shape repair required by P0686. If `p_local(r)` is the P0685
exponent, define its inward monotone majorant

\[
p_{\rm env}(r)=\max_{s\ge r}p_{\rm local}(s).
\]

This fills an inner hollow without lowering the previously successful outer
response and guarantees that `p_env` cannot increase outward. A spherical
spent-data screen must be frozen before it is evaluated. A radial pass would
only permit a separately frozen 3D implementation on baryonic potential-depth
shells.

## Reproduction

```powershell
python scripts/run_p0687_system_path_coordinate_qumond.py
python -m pytest tests/test_potential_channel_qumond.py tests/test_spatial_qumond_3d.py -q
```

Artifacts are in `results/p0687_system_path_coordinate_qumond/`.

## Claim boundary

P0687 uses spent galaxy and cluster profiles and was generated from a spent
topology failure. It is mechanism development, not validation. The galaxy
coordinate is estimated on the scored baryonic profile rather than a new
full-disk field. P0633 and P0640 remain sealed.

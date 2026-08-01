# R1 next stage: RX J2129 strict observable package

This is the active residual-blind route after the completed raw-dynamics acquisition cycle. It does not authorize a gravity law, a Weyl-response reconstruction, or a negative-gravity fit.

## Why the route changed

The same-system ledger still contains 15 evaluated systems, 13 with local numerical dynamics, 12 with local observable lens positions, two structural 3+3 passes (MACS J1206 and RX J2129), and zero strict-ready systems. No system has both complete baryonic forward inputs and a theory-neutral joint covariance.

The completed raw-dynamics cycle added zero structural promotions:

- Abell 1689 failed the frozen pPXF systematic grid at a 36.6% maximum dispersion shift versus the 10% ceiling.
- Abell 2261 failed the frozen continuum-center range gate.
- A383 failed its 0.200-Angstrom arc-RMS ceiling.
- MS2137 failed its continuum-registration and opposite-half geometry gates before pPXF.
- A2537, predeclared only as a disturbed engineering control, passed acquisition and environment gates but failed C2a: its two CuAr RMS values were 0.2248 and 0.2205 Angstrom versus the frozen 0.2000-Angstrom ceiling. No science frame was processed.

Repeating the same spectroscopy route would not address the two promoted systems' actual missing inputs. The next experiment therefore tries to finish RX J2129, whose four-bin MUSE dynamics and covariance already pass.

## R1B3-P1: feasibility and acquisition

The frozen feasibility gate passes:

| Input | Frozen result |
|---|---:|
| XMM observation | ObsID 0093030201, archived/public |
| Gross XMM duration | 58,916 s |
| HEASARC located-data estimate | 705,896,733 bytes |
| Local public S3 manifest | 2,824 objects, 645,765,075 bytes |
| Raw ODF | 284 files, 139,942,725 bytes |
| PPS | 2,255 files, 476,825,638 bytes |
| HST images | local F814W and F125W, each 5000x5000 |
| Lens coordinates | 21 spectroscopic images, seven families |
| Inner common-support images | 5.2, 6.3, and 8.2 from three families |

Every S3 object has an archive ETag/MD5 check and a local SHA-256 in `data/raw/r1_rxj2129_xmm/provenance.json`. No XMM pixel was inspected during acquisition.

## R1B3-P2: two independent measurement gates

### XMM gas reduction

Freeze and install the official Ubuntu 24.04 SAS 22.1.0 build plus its HEASoft prerequisite and a dated Current Calibration File snapshot. Then process the ODF only under recorded commands and logs.

Advance only if at least two of MOS1, MOS2, and pn each retain:

- at least 15,000 s cleaned exposure;
- at least 25% of gross exposure;
- background scales in the open interval (0.5, 2.0);
- accepted calibration, field-center, flare-filter, and point-source-mask gates.

### HST measurement covariance

The old 0.5-arcsec diagonal lens covariance remains a likelihood convention, not a measurement error. Freeze the execution protocol before remeasuring any arc pixel. Fit local PSF-convolved flexible light models independently in F814W and F125W with 500 noise/background bootstraps plus shared WCS-registration draws.

Advance only if:

- all three inner images pass;
- at least 18 of 21 field images pass;
- cross-band centroid differences are at most 0.2 arcsec;
- per-coordinate standard errors lie in [0.02, 0.30] arcsec;
- the resulting full 42x42 covariance is positive semidefinite and contains no mass-model residual term.

## R1B3-P3: concrete success

The XMM likelihood must supply at least five accepted annuli from 10-500 kpc, at least 2,000 total net counts, S/N at least 5 per annulus, density uncertainty at most 20%, temperature uncertainty at most 30%, at least 2,000 joint posterior draws, and a positive-semidefinite covariance. The same gas draws must generate spherical and projected gas terms.

Success was originally defined as one RX J2129 package containing accepted dynamics covariance, marginalized stellar/ICL and satellite inputs, XMM gas density/temperature covariance, and HST measurement-level image/redshift covariance. Its local stage rule would have authorized only a separate dynamical-versus-Weyl identifiability audit. The later, independently frozen ten-system public-data ceiling now supersedes that local authorization: H2 and X4 may finish and be preserved, but H3, X5, and response reconstruction are not authorized in any outcome branch.

The pre-ceiling failure rule would have moved the same availability screen to MACS J1206. That rule is now superseded. If either XMM or HST fails, RX J2129 remains structural-only and no replacement system is selected. Thresholds are not changed, Chandra is not revived, and no gravity formula is fit.

## Current execution gate (2026-07-27)

The environment, calibration, flare filtering, immutable source mask, and two-part quantitative background gate now pass. MOS2 and pn are retained; MOS1 is excluded because its CCD5 FWC/corner scale is 0.35334, below the unchanged open lower bound of 0.5. The local 650-900 kpc transfer scales are 1.31342 for MOS2 and 0.74195 for pn, so the required two instruments pass full X2.

X3 passes without changing an edge, energy band, instrument, or threshold: all six immutable annuli pass, the conservative total net count is 76,279.260, and the minimum annular S/N is 84.091. This closes the count-information risk but does not authorize a spectral fit.

The active XMM action is full X4 response production. The quarantined direct, a02-to-a01 cross-region, and central-source interface passes for MOS2 and pn. The nominal 650 full-field map request is ineligible because SAS realized it as 642x642 with 81-detector-unit pixels, above the frozen 80-unit ceiling, and its response changes also exceeded the 2% limits. The representative a04-to-a04 920-versus-1302 promoted comparison now passes: MOS2 changes are 1.803% integrated, 1.841% median-shape, and 2.034% p95-shape; pn changes are 0.464%, 0.463%, and 0.523%. Rebuild every direct MOS2+pn response, construct the complete 6x6 input-annulus to output-annulus PSF cross-region ARF matrix for each detector, and construct the central unresolved-source response vector. The wide outer X3 region exceeded the default detector-map coverage, so no X3 RMF/ARF may enter a temperature likelihood. Advance to the frozen joint gas-likelihood implementation only after all X4 products pass coverage, task-error, energy-grid, finiteness, and matrix-completeness checks.

In parallel, HST H1 now passes its measurement-only calibration gate: 178 mutual registration matches, 0.024001-arcsec leave-one-out RMS against the 0.12-arcsec ceiling, 607/612 successful F814W PSF fits, 203/203 successful F125W fits, a frozen union segmentation, and every 500-draw bootstrap complete. The failed v0.3 engineering run remains preserved. The exact H2 implementation of the declared two-band centroid model now passes its synthetic recovery, checksum, immutable-ledger, and static no-pixel audits and is executing on all 21 rows. Its first attempt stopped before writing any centroid when Astropy rejected an equivalent but non-identical coordinate frame; only the explicit sky-frame conversion was corrected, logged, re-hashed, and re-audited. H2 must accept all three inner images and at least 18 total images to pass its local gate. The global ceiling nevertheless withholds H3 in either H2 branch. Neither branch may use a lens residual or gravity prediction. A gas temperature/density fit, dynamical/Weyl response, and gravity-law fit remain unauthorized.

## External-search checkpoint: J1402 closes the one-off route

SDSS J1402+6321 exactly reproduces the released Project Dinos stored-chain likelihood on 25,807 retained pixels: the total log-likelihood difference is exactly zero, and the three full-mask reduced chi-squares are 1.102, 0.868, and 0.917. Its six frozen sector holdouts also pass the ordinary predictive threshold, with pooled reduced chi-square from 0.897 to 1.361 and an aggregate value of 1.089. All three deliberately corrupted coordinate mappings are strongly rejected.

The model nevertheless fails the separately frozen coherent-residual gate. The maximum PSF-matched heldout residual is 28.391 sigma in F435W sector 0, above the unchanged 5-sigma ceiling. No optimizer, sector, mask, coordinate, or threshold was rerun after that result. J1402 therefore stops before a lens-response Jacobian or KCWI reduction and does not count toward the ten-system target.

This is the third completed external candidate without a promotion after J0946 and ESO 325-G004, so the rethink checkpoint frozen in the J1402 protocol is now active. No fourth one-off external target is authorized. The active experiment remains RX J2129; after its XMM X4 and HST H2 gates finish, the project must reassess the ten-system public-data premise and state a formal sample-size/identifiability ceiling before choosing any further acquisition or gravity-law work.

The residual-blind public-data reassessment is now frozen and complete before either active RX J2129 outcome. It covers 45 source-screened BCG hosts, the 15-system same-object ledger, all 14 SLACS-KCWI systems, and the three completed external one-off candidates. The audited universe has a structural ceiling of three systems and zero strict-ready systems; at least seven genuinely new rank-three systems would still be required even if every ceiling system were repaired. No public numerical kinematic map was identified for any of the 14 SLACS-KCWI objects, and none of J0946, ESO 325-G004, or J1402 promoted. This establishes the frozen hard public-data shortfall for a ten-system population test.

An RX J2129 pass can therefore produce at most a single-system pipeline and identifiability demonstration; it cannot authorize population cross-validation or a universal one- versus two-potential claim. An RX J2129 failure produces no response reconstruction. In either branch, the ten-system freeze and R2 population test remain empirically unidentifiable with the audited public data, and the unification claim is withheld. The machine-readable decision is `results/r1_ten_system_public_data_ceiling/report.json`, and the updated full-goal ledger is `results/r0_r2_goal_progress/report.json`.

## Frozen terminal closure

The terminal RX J2129 protocol was frozen before either final H2 or X4 gate was visible. Once both reports exist, the closure verifier independently re-hashes the four H2 outputs, all 11 immutable H2 input artifacts, all 108 X4 response products, and all eight X4 detector maps. It also binds X4 to the exact protocol, map-convergence report, production runner, and audit implementation; checks the immutable 21-image/42-band H2 attempt count and the exact 12-RMF, 12-direct-ARF, 12-central-ARF, and 72-cross-ARF X4 counts; and records the corresponding pass/fail branch. A hidden local watcher may run only the X4 product audit, terminal disposition, master goal-ledger refresh, and closure tests. It contains no command that can rerun H2/X4, start H3/X5, select a system, or fit a theory.

The final master ledger distinguishes two claims. `premise_passed` remains false unless every requested population-stage operation passes. `full_goal_complete` becomes true only when the explicit stop rule is terminally satisfied: population R2 is empirically unidentifiable with the audited public data, the unification claim is withheld, and both new-system selection and force/action fitting remain prohibited. This treats a negative audit as a completed scientific audit without representing it as a successful unification premise. The supporting 11-requirement ledger is `results/r0_r2_completion_evidence/report.json`; it independently re-hashes all 133 scored source files, all 32 cycle-three BCG profile artifacts, and every embedded input of the public-data ceiling before permitting terminal closure.

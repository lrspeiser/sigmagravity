# Sigma V19DI--V19DN: direct responses and the Bullet soft-spectrum failure

Status: terminal evidence through V19DN, 2026-08-06. No lensing, halo, action,
gravity parameter, I4, I5, validation, or holdout payload was opened.

## Result in one paragraph

The direct OGIP response writer and deterministic FITS canonicalizer now pass
their prospective controls. The full 5,082-cell combination conserves counts,
uses every registered cell exactly once, preserves links and checksums, and
produces successful registered-region fits. The full commissioning nevertheless
fails because a one-temperature model is inadequate for the integrated Bullet
Cluster spectrum (`reduced chi2 = 2.7937`). A frozen two-temperature diagnostic,
rerun after correcting a statistic-initialization defect, also fails
(`2.8023`) and is disfavored by BIC. A further frozen band diagnostic localizes
the problem: Bullet fails badly at 0.5--2 keV (`4.1623`) but passes at 2--7 keV
(`1.2200`). No 494-region production or thermodynamic source construction is
authorized. The next observation-model test must address the soft band and
avoid treating the entire merging cluster as one plasma behind one averaged
response.

## Why this work was needed

The leading physical candidates for a directional baryonic Sigma source use
thermodynamic-gradient stress and baroclinicity. Those quantities require a
spatially resolved X-ray temperature field with response propagation that is
reproducible, auditable, and independent of lensing outcomes. Earlier CIAO
`addresp` and hierarchical `combine_spectra` routes did not preserve the frozen
sparse response structure for the Bullet workload. V19DI--V19DN therefore test
the observation machinery before constructing either candidate invariant.

## Frozen sequence

| Stage | Terminal result | Meaning |
|---|---|---|
| V19DI | direct OGIP writer preflight passed | direct ARF/RMF construction and OGIP writing passed the registered small controls |
| V19DJ | execution failed | the response-free full PHA merge was terminated during the Bullet integrated workload; no scientific result |
| V19DK | execution failed | the legacy comparator expected an `_src.pi` alias for the grouped source; no scientific result |
| V19DK2 | grouped canonicalization preflight passed | two independent runs of both registered regions are byte-identical after canonicalization; arrays, links, checksums, Sherpa load, and forward folding pass |
| V19DL | commissioning gate failed | all engineering gates and both registered regional fits pass; the Bullet integrated 1T goodness gate fails |
| V19DM | result discarded | the 2T branch inherited Sherpa's `chi2gehrels` default instead of explicitly restoring frozen `chi2xspecvar` |
| V19DM2 | valid minimal-mixture failure | statistic parity is restored; the identical 2T model, starts, bounds, and gates still fail for Bullet |
| V19DN | residual localization completed | the Bullet failure is soft-band dominated; hard-band 1T is adequate, but this diagnostic cannot authorize production |

## Engineering evidence

V19DK2 applied deterministic FITS canonicalization only after science arrays
were constructed. For both Bullet region 169 and Abell 2146 region 62, two
independent executions produced byte-identical grouped source PHA, background
PHA, ARF, and RMF products. The science arrays remain exact, FITS `CHECKSUM` and
`DATASUM` cards validate, and Sherpa loads and forward-folds both response sets.

V19DL then combined all 5,082 registered response cells:

- Abell 2146: 1,270 integrated cells and 10 cells in registered region 62;
- Bullet: 3,812 integrated cells and 9 cells in registered region 169;
- every cell is used exactly once;
- source/background event counts and full-PHA counts are conserved exactly;
- the grouped source links point to the exact registered background, ARF, and
  RMF; and
- every frozen snapshot has valid deterministic checksums.

The direct response arithmetic is therefore not the reason V19DL fails.
Runtime is still a production concern: the response construction is fast, but
the CIAO Bullet PHA merge takes hours. A direct PHA writer requires its own
parity preflight before any large successor.

## Spectral evidence

### V19DL one-temperature commissioning

| Spectrum | Temperature (keV) | Abundance (solar) | Reduced chi2 | Gate |
|---|---:|---:|---:|---|
| Abell 2146 integrated | 8.1255 | 0.4976 | 1.2232 | pass |
| Bullet integrated | 16.0636 | 0.3836 | 2.7937 | **fail** |
| Abell 2146 region 62 | 10.2086 | fixed from integrated | 0.7508 | pass |
| Bullet region 169 | 15.2068 | fixed from integrated | 1.0103 | pass |

The failure is specifically the assertion that one integrated plasma model is
adequate for both entire clusters. It is not a failure of the two registered
local spectra.

### V19DM and V19DM2 minimal thermal mixture

V19DM preregistered one cluster-label-free rule: retain 1T where it passes;
otherwise admit two tied-abundance APEC components only if reduced chi2 is at
most 1.5, `Delta BIC <= -10`, both normalization fractions are at least 5%,
the temperatures differ by at least a factor 1.2, and every free parameter is
strictly inside its inherited bound.

The first execution is scientifically discarded because `ui.clean()` reset
the two-temperature statistic to `chi2gehrels`. V19DM2 changed only that
initialization and explicitly restored `chi2xspecvar`; all scientific choices
were unchanged.

| Cluster | 1T reduced chi2 | Best 2T reduced chi2 | Delta BIC (2T-1T) | Selection |
|---|---:|---:|---:|---|
| Abell 2146 | 1.2232 | 1.2287 | +12.2006 | retain 1T |
| Bullet | 2.7937 | 2.8023 | +10.4164 | none; 2T rejected |

The Bullet 2T solution places about 93.6% of its normalization near 15.63 keV
and 6.4% at 27.25 keV. It neither repairs goodness nor earns its two extra
parameters. This rejects the minimal two-temperature repair. It does not show
that the gas is literally isothermal or that every differential-emission-
measure model must fail.

### V19DN frozen energy-band localization

Every band uses the same absorbed 1T APEC model, bounds, `chi2xspecvar`, data,
and response products. Band selection is diagnostic and cannot define the
production source.

| Cluster | 0.5--7 | 0.7--7 | 1--7 | 2--7 | 0.5--2 |
|---|---:|---:|---:|---:|---:|
| Abell 2146 | 1.2232 | 1.1974 | 1.0318 | 0.9653 | 1.3860 |
| Bullet | **2.7937** | **2.6578** | **1.5666** | 1.2200 | **4.1623** |

In the Bullet soft-only fit, temperature rises to 28.88 keV and abundance hits
its maximum of 2 solar. The hard 2--7 keV fit is adequate at 13.91 keV and
0.3865 solar. The mismatch is therefore concentrated below about 2 keV, with
some residual influence extending through 1--2 keV. It is not a broadband
failure of the hard thermal continuum.

## Scientific interpretation

Three possibilities remain open and must be separated rather than hidden with
additional plasma components:

1. soft-background or calibration mismatch across the heterogeneous Bullet
   observations;
2. invalidity of representing thousands of spatial/observation responses by
   one merged spectrum and one effective response when the plasma varies; or
3. genuinely distributed absorption, temperature, abundance, or foreground
   emission that requires an observation-resolved hierarchical likelihood.

The most defensible next experiment is a frozen observation-resolved joint
fit. It should preserve each observation's response and background, share only
the physically common quantities, and allow local temperature/normalization
variation. A preceding source/background-count audit should show whether the
soft failure tracks background fraction, detector, epoch, or observation.

## Consequence for Sigma Gravity

This sequence changes no gravity result. It improves the evidential foundation
for testing thermodynamic-gradient stress (`I4`) and baroclinicity (`I5`), but
neither invariant is admitted yet. The leading formula lesson remains that
cluster lensing likely needs directional, component-sensitive baryonic state,
not another scalar amplitude. A valid temperature map is one possible input to
that state; it cannot be manufactured by accepting a globally inadequate
spectrum.

## Reproduction

The frozen configurations are:

- `configs/sigma_v19di_direct_ogip_writer_preflight.json`
- `configs/sigma_v19dj_direct_response_commissioning.json`
- `configs/sigma_v19dk_fits_canonicalization_preflight.json`
- `configs/sigma_v19dk2_grouped_canonicalization_preflight.json`
- `configs/sigma_v19dl_canonicalized_direct_response_commissioning.json`
- `configs/sigma_v19dm_minimal_thermal_mixture_diagnostic.json`
- `configs/sigma_v19dm2_statistic_parity_remediation.json`
- `configs/sigma_v19dn_integrated_residual_localization.json`

The terminal reports are the same-named directories under `results/`. Tests
verify runner hashes, terminal statuses, sealed-payload boundaries,
canonicalizer determinism, and the reported scientific gates.

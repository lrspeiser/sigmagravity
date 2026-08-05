# Sigma V19W2 exact-binmap response commissioning plan

## Purpose

Commission the smallest implementation correction that addresses all three
failure classes exposed by the live V19W audit while preserving the frozen
event partition and every scientific setting.

The correction replaces polygon event filtering with an exact binary mask
defined by the already-frozen V19M binmap. The final response reference is the
actual selected event nearest the DETX/DETY centroid, mapped through the
observation's unreprojected detector frame and required to remain on the
declared CCD. Runtime-only directories use the short production-index token so
CIAO's internal Unix socket remains below the operating-system path limit.

The first invocation stopped during Python import because Astropy is absent
from the CIAO environment. It created no scratch directory or scientific
product. Protocol 1.0.1 therefore makes a pre-execution dependency correction:
CIAO `dmimgcalc` writes the already-declared binary mask while propagating the
frozen binmap WCS, and CIAO `pycrates` plus NumPy independently checks every
output pixel. No selection, count, response or gravity setting changes.

Protocol 1.0.1 then stopped in the manifest-validation gate, also before any
scratch directory or product existed: the background count for
`BULLET_bin154_obs4985_ccd3` had been transcribed as 7 rather than the frozen
V19U manifest value 14. Protocol 1.0.2 corrects only that metadata transcription;
the executable mask and event-count gates remain unchanged.

Protocol 1.0.2 then reached the CIAO mask writer after creating only the four
deterministic background-geometry prerequisites. CIAO rejected a direct
comparison because Boolean images are not a supported `dmimgcalc` output type;
no exact mask, selected spectrum, response or report existed. Protocol 1.0.3
uses the documented two-step form: `dmimgcalc` makes a WCS-preserving image of
ones, and `dmimgthresh` retains one only for the closed integer interval
`bin_id:bin_id`, writing zero elsewhere. The independent pixel equality check
remains unchanged.

Protocol 1.0.3 proved that all five binary masks reproduce their frozen event
counts. It then exposed a CIAO interaction: `dmextract` propagated a `MASK`
image rather than producing the `WMAP` block required by the weighted-response
stage. A spent diagnostic confirmed the exact workaround. `dmcopy` first
materializes the already-verified event subset; `specextract` then sees a
physical event file and writes a valid WMAP while retaining the identical 0.5--7
keV count. The nonzero-background boundary case completed both PHAs and both
weighted responses.

The path-length cell has zero blank-sky rows at every energy. For that declared
edge case, source-only `specextract` creates the weighted ARF/RMF, `dmextract`
creates a valid zero-count background PHA, and `dmhedit` adds the ordinary
`BACKFILE` link before the same blank-sky scaling audit. Each command and hash
is retained. Protocol 1.1.0 uses a new scratch root so no failed response is
reused.

Protocol 1.1.0 passed the three boundary cells and the zero-background/path
cell. The remaining two-event case demonstrated that a reprojected SKY
centroid is not a valid detector calibration reference: its midpoint mapped
through the observation aspect to CCD 2 even though both measured events have
`CCD_ID=3`. Protocol 1.2.0 therefore uses detector coordinates, which
`reproject_events` does not alter. It selects the actual 0.5--7 keV event
nearest the exact subset's DETX/DETY centroid, converts that detector medoid
through the observation's unreprojected V19H source-excluded frame, and
requires the result to map back to the declared CCD. The coordinate is derived
calibration metadata, never a fitted position. The spent off-CCD diagnostic
completed the full weighted response at that derived reference.

## Why the binmap remains authoritative

The V19P/V19Q manifest assigns every event to exactly one integer binmap pixel.
The alternative region polygons include a handful of shared boundary events.
Changing the manifest to match those polygons would introduce inconsistent or
duplicated membership when spectra are later combined. V19W2 therefore changes
the executable filter, not the science partition.

## Frozen commissioning cells

| Cell | Failure represented |
|---|---|
| `BULLET_bin45_obs554_ccd3` | source-only +1 polygon boundary event |
| `BULLET_bin135_obs3184_ccd3` | background-only +1 boundary event |
| `BULLET_bin232_obs4984_ccd3` | source and background each +1 |
| `BULLET_bin154_obs4985_ccd3` | two-event centroid maps to adjacent CCD |
| `BULLET_bin36_obs4984_ccd3` | AF_UNIX path too long in both base attempts |

The products remain under `/home/henry/sv19w2_v120` and are not promoted into the
live V19W archive.

## Advancement rule

Only an all-cell pass authorizes a separately frozen missing-cell recovery
after the base V19W process has exited. That successor must identify missing
tasks from the final base report rather than from this commissioning list and
must finish with an independent 5,082-cell archive audit before V19X can run.

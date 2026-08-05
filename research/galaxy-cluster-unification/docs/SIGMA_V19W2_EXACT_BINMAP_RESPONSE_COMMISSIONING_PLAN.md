# Sigma V19W2 exact-binmap response commissioning plan

## Purpose

Commission the smallest implementation correction that addresses all three
failure classes exposed by the live V19W audit while preserving the frozen
event partition and every scientific setting.

The correction replaces polygon event filtering with an exact binary mask
defined by the already-frozen V19M binmap. It leaves `refcoord` unset and uses
CIAO's supported `resp_pos=CENTROID` pixel-mask mode. Runtime-only directories
use the short production-index token so CIAO's internal Unix socket remains
below the operating-system path limit.

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

The products remain under `/home/henry/sv19w2` and are not promoted into the
live V19W archive.

## Advancement rule

Only an all-cell pass authorizes a separately frozen missing-cell recovery
after the base V19W process has exited. That successor must identify missing
tasks from the final base report rather than from this commissioning list and
must finish with an independent 5,082-cell archive audit before V19X can run.

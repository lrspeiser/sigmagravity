# P0619: frozen tangential transfer

P0619 freezes the P0618 safety-first +90-degree route before evaluating it on
A383 or MS2137. The amplitude remains \(Q^2/(1+\Delta_{80})\), the width is
\(0.23R_{80}\sqrt{1+Q^2}\), and no gravity coefficient is fitted to either
cluster. Six ordinary lens-geometry variables and training source positions are
refit with the same 16-start seeds used by the P0616 scalar control.

The transfer is chronological relative to this particular formula, not a
pristine project holdout. A383 supplies a complete matched baseline. MS2137 is
reported but cannot make an RMS comparison unless both the scalar and routed
variants recover every required root.

## Result

A383 retained all 10 training and 4 held-out roots. Its held-out RMS changed
from 9.0966 arcsec for the P0554 scalar control to 9.0808 arcsec for the frozen
tangential route, a 0.174% improvement. The prior radial P0615 route had
worsened A383 by 0.442%, so the universal phase change contributes a 0.616
percentage-point improvement relative to that radial result.

MS2137 remained incomplete under both variants: 7/8 training roots and 2/3
held-out roots. Its optimizer cost fell from 324.50 for the scalar replay to
322.99 for the tangential route, but an incomplete-root cost is not accepted as
a lensing improvement.

The phase clue therefore transfers on the one clean chronological comparison,
but the formula does not pass. A383's 9.081 arcsec RMS is far above the frozen
2 arcsec absolute gate, and MS2137 supplies no valid RMS comparison. The result
supports further testing of a tangential-like operator; it does not establish
an adequate cluster lens model.

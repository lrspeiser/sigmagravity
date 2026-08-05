# V19CM fine-WMAP edge diagnostic

V19CM tests the least invasive response remedy implied by V19CK. The source has valid ACIS events, but the ordinary `det=8` WMAP aliases its two occupied bins beyond the chip boundary. V19CM changes only the WMAP sampling to `det=1`. It keeps every event and all region, background, energy, calibration, weighted-ARF, weighted-RMF, response-position, and response-reference settings unchanged.

The products are diagnostic. Passing requires all four nonempty spectral products, finite positive ARF response, finite nonzero RMF matrix, correct PHA links, and byte-identical recovery inputs after execution. Passing can authorize only a new preregistered recovery-and-equivalence protocol; it cannot itself fill the missing V19W5 checkpoint or resume the scientific chain.

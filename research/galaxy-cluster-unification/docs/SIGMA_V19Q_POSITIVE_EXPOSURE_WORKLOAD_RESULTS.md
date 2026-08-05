# Sigma V19Q positive-exposure workload result

V19Q **passed every frozen gate** and authorizes regional response
commissioning.

The universal event-support rule is

\[
I_{\rm event}=I(500\le E\le7000\ {\rm eV})
I({\rm exact\ flux\_obs\ FOV})
I({\rm event\ pixel\ is\ on\ grid})
I({\rm exact\ broad\ exposure}>0).
\]

It was applied without change to all 20 observations, 494 admitted regions,
and every supported CCD.  There is no observation, cluster, region, CCD,
position, or event-identity exception.

| Check | Bullet Cluster | Abell 2146 |
|---|---:|---:|
| Admitted regions | 366 | 128 |
| Response task keys | 3,812 | 1,270 |
| Task keys changed from V19N | 0 | 0 |
| Zero-exposure events rejected inside regions | 1 | 0 |
| Science count delta | 0 | 0 |
| Per-observation image sum changed pixels | 0 | 0 |
| Scaled blank-sky count delta | 0 | \(1.46\times10^{-11}\) |
| All frozen gates | pass | pass |

The numerical background remainder is floating-point summation noise and is
well inside the already-frozen \(10^{-5}\)-count arithmetic tolerance; science
event conservation retained exact zero tolerance.

The result confirms a workload of **5,082 response cells**, each expected to
produce a source PHA, background PHA, ARF, and RMF: 20,328 products.  The
conservative planning envelope remains 124.1 GiB, so the next step is one
frozen commissioning cell rather than an immediate full batch.

This is an instrumental data-support result, not evidence for or against a
gravity theory.  No spectrum, response, temperature, density, shock speed,
lensing prediction, or gravity parameter was constructed in V19Q.

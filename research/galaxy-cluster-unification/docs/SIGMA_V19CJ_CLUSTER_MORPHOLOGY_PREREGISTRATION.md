# Sigma V19CJ cluster morphology preregistration

V19CJ freezes how the 162 clean V19CH cluster candidates will be divided by
baryonic source structure before any candidate morphology value is downloaded
or read.

The primary source is the Yuan and Han Chandra catalog of 964 clusters.  Its
signed morphology index combines X-ray profile shape and asymmetry.  With the
published zero boundary and quoted uncertainty, V19CJ defines:

- secure relaxed: `delta + e_delta < 0`;
- secure disturbed: `delta - e_delta > 0`;
- boundary/intermediate: the interval crosses zero.

A secure extreme must also agree with at least two of three independent
directions within the clean crossmatch: relaxed systems have higher
concentration, lower centroid shift and lower third-order power ratio;
disturbed systems have the opposite directions.  Medians are computed only
inside the finite clean crossmatch.  A secondary Planck-name catalog checks
concentration and centroid shift but cannot replace the primary classification
or move a threshold.

The metadata shortlist must contain at least eight systems: at least three
secure relaxed, three secure disturbed and two boundary or discordant cases.
The eventual six-cluster test must preserve at least two relaxed, two disturbed
and one intermediate/discordant system while also spanning mass, redshift and
baryonic layout.

Selection by image multiplicity, lensing RMS, halo properties, Sigma residuals
or parameter preference is forbidden.  V19CJ admits no cluster; it only permits
the two registered morphology tables to be acquired under the frozen rule.

# Data layout

`raw/sparc/` is a byte-for-byte snapshot imported from a local SigmaGravity
checkout. `provenance.json` records its origin and SHA-256 hashes. The import
script never writes to the source checkout.

`raw/cosmicflows4/` contains official Cosmicflows-4 density reconstructions and
the companion group catalog. Its download manifest records source URLs, byte
sizes, and SHA-256 hashes.

`raw/clash_tian2020/` contains the official CDS table of 84 baryonic and
lensing-derived acceleration measurements for 20 CLASH galaxy clusters. It is
downloaded independently even though a matching copy existed in SigmaGravity.

`raw/manga_bcg_tian2024/` contains the hashed arXiv source for the 50 MaNGA BCG
dynamical-acceleration measurements. `derived/manga_bcg_tian2024.csv` is a
reproducible extraction of the paper table for an intermediate-scale test.

`derived/void_scores_cf4.csv` is the frozen, SPARC-aligned external environment
table. `derived/cf4_environment_report.json` records its coordinate convention,
grid hashes, validation, and summary statistics. Rebuild both with
`python scripts/build_cf4_environment.py`. Generated fit outputs belong in
`results/`, not here.

The SPARC source describes 175 galaxies and 3,391 radial mass-model rows. Cite:
Lelli, McGaugh & Schombert, *Astronomical Journal* 152, 157 (2016).

For Cosmicflows-4 cite Courtois et al., *Astronomy & Astrophysics* 670, L15
(2023), DOI 10.1051/0004-6361/202245331, and Tully et al., *Astrophysical
Journal* 944, 94 (2023).

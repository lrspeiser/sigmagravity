# CLASH galaxy-cluster radial acceleration data

`fig2.dat` is the 84-row, 20-cluster machine-readable catalog accompanying
Tian et al., *Astrophysical Journal* 896, 70 (2020), DOI
10.3847/1538-4357/ab8db9. It pairs baryonic acceleration from the observed gas,
BCG, and cluster-galaxy components with total acceleration inferred from joint
strong-lensing, weak-lensing shear, and magnification mass models.

Columns are cluster name, spherical radius (kpc), `log10(gbar/[m s^-2])`,
`log10(gtot/[m s^-2])`, and their standard errors in dex. The catalog was
downloaded directly from CDS; `provenance.json` records its URL and SHA-256.

Run `powershell -File scripts/download_clash_rar.ps1` to reproduce the snapshot.
The SigmaGravity checkout contained the same file and supplied the expected
hash, but this repository downloads its own byte-for-byte copy from CDS.

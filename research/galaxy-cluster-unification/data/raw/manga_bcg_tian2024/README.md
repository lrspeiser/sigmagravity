# MaNGA BCG dynamics data

The arXiv source for Tian et al., *Astronomy & Astrophysics* 684, A180
(2024), DOI 10.1051/0004-6361/202347868, contains the 50-row table of MaNGA
brightest-cluster-galaxy dynamics used for the independent intermediate-scale
check. The table reports the final kinematic radius, baryonic acceleration,
dynamical acceleration from Abel-inverted stellar kinematics, and errors.

Run `powershell -File scripts/download_manga_bcg.ps1`, then
`python scripts/build_manga_bcg_table.py`. The download manifest hashes the
source archive and extracted TeX; the derived CSV retains the published values
and computes only `radius_kpc = effective_radius_kpc * last_radius_re`.

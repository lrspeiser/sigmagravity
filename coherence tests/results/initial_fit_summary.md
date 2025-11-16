# Initial Coherence Fits

- Data: Real SPARC rotation curves (`gravitywavebaseline/sparc_results/*_capacity_test.json`).
- Ïƒ_v: from `data/sparc/sparc_combined.csv` (no edits).
- Î»_gw proxy: `2Ï€R` (geometric circumference; needed because Î»_gw not stored in exports).
- Random search: 600 draws per mechanism/galaxy.

| Galaxy | Model | Ïƒ_v (km/s) | Points | RMSE log10(f) | MAE(f) | Max|Î”f| | Best-fit params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| DDO154 | path_interference | 6.76 | 12 | 0.1734 | 1.5850 | 3.4385 | A=2.86, beta_sigma=0.0733, ell0=2.35, n_coh=4.82, p=1.99 |
| DDO154 | metric_resonance | 6.76 | 12 | 0.0273 | 0.1870 | 0.4764 | A=1.57, beta_sigma=1.07, ell0=5.87, lambda_m0=74.2, log_width=1.29, n_coh=4.74, p=0.451 |
| DDO154 | entanglement | 6.76 | 12 | 0.1786 | 1.6452 | 3.5330 | A=2.82, ell0=2.95, n_coh=4.61, p=2.14, sigma0=46.9 |
| DDO154 | vacuum_condensation | 6.76 | 12 | 0.1864 | 1.6862 | 3.6095 | A=3, alpha=0.985, beta=1.42, ell0=2.08, n_coh=4.5, p=2.35, sigma_c=95.4 |
| DDO154 | graviton_pairing | 6.76 | 12 | 0.1884 | 1.7034 | 3.6292 | A=2.96, gamma_xi=0.0959, n_coh=3.24, p=2.5, sigma0=64.6, xi0=1.37 |
| NGC2403 | path_interference | 20.18 | 73 | 0.0562 | 0.2207 | 1.2953 | A=2.6, beta_sigma=0.39, ell0=10.8, n_coh=2.53, p=2.98 |
| NGC2403 | metric_resonance | 20.18 | 73 | 0.0526 | 0.2098 | 1.1387 | A=2.61, beta_sigma=0.481, ell0=14.6, lambda_m0=59.4, log_width=1.62, n_coh=2.67, p=2.73 |
| NGC2403 | entanglement | 20.18 | 73 | 0.0629 | 0.2667 | 1.2442 | A=2.88, ell0=17.3, n_coh=3.09, p=2.93, sigma0=101 |
| NGC2403 | vacuum_condensation | 20.18 | 73 | 0.0686 | 0.2649 | 1.0962 | A=2.84, alpha=3.51, beta=3.18, ell0=16.6, n_coh=4.26, p=2.39, sigma_c=105 |
| NGC2403 | graviton_pairing | 20.18 | 73 | 0.0892 | 0.3492 | 1.7468 | A=2.94, gamma_xi=0.208, n_coh=4.2, p=2.15, sigma0=63.6, xi0=12.4 |
| NGC5055 | path_interference | 26.93 | 28 | 0.2176 | 0.3808 | 0.6152 | A=0.694, beta_sigma=0.982, ell0=16.8, n_coh=0.567, p=2.82 |
| NGC5055 | metric_resonance | 26.93 | 28 | 0.2318 | 0.4355 | 0.6879 | A=1.6, beta_sigma=1.02, ell0=19, lambda_m0=25.9, log_width=1.76, n_coh=0.554, p=2.48 |
| NGC5055 | entanglement | 26.93 | 28 | 0.2203 | 0.3776 | 0.6153 | A=1.1, ell0=15.7, n_coh=0.362, p=2.44, sigma0=90.1 |
| NGC5055 | vacuum_condensation | 26.93 | 28 | 0.2203 | 0.3566 | 0.6153 | A=1.55, alpha=2.22, beta=0.632, ell0=18.6, n_coh=0.396, p=2.6, sigma_c=102 |
| NGC5055 | graviton_pairing | 26.93 | 28 | 0.2068 | 0.2890 | 0.6152 | A=2.75, gamma_xi=1.11, n_coh=4.45, p=2.83, sigma0=99.4, xi0=19.8 |
| NGC7331 | path_interference | 35.77 | 36 | 0.2102 | 0.3403 | 0.5934 | A=0.172, beta_sigma=0.529, ell0=13, n_coh=0.339, p=3 |
| NGC7331 | metric_resonance | 35.77 | 36 | 0.2131 | 0.3534 | 0.6720 | A=2.79, beta_sigma=0.881, ell0=0.806, lambda_m0=3.28, log_width=0.122, n_coh=4.23, p=2.9 |
| NGC7331 | entanglement | 35.77 | 36 | 0.2088 | 0.3293 | 0.5936 | A=0.517, ell0=16.7, n_coh=0.509, p=2.96, sigma0=62.8 |
| NGC7331 | vacuum_condensation | 35.77 | 36 | 0.2099 | 0.3395 | 0.5936 | A=2.01, alpha=1.16, beta=2.42, ell0=19.2, n_coh=0.826, p=2.44, sigma_c=50.3 |
| NGC7331 | graviton_pairing | 35.77 | 36 | 0.2034 | 0.2987 | 0.5935 | A=1.86, gamma_xi=0.854, n_coh=0.944, p=2.56, sigma0=102, xi0=19.9 |
| UGC02953 | path_interference | 39.45 | 115 | 0.1579 | 0.3696 | 7.0514 | A=2.14, beta_sigma=0.159, ell0=18.8, n_coh=0.48, p=2.44 |
| UGC02953 | metric_resonance | 39.45 | 115 | 0.1802 | 0.4857 | 7.0514 | A=1.74, beta_sigma=0.224, ell0=16.6, lambda_m0=67.6, log_width=1.25, n_coh=0.838, p=2.94 |
| UGC02953 | entanglement | 39.45 | 115 | 0.1565 | 0.3779 | 7.0514 | A=1.8, ell0=18.9, n_coh=0.467, p=2.99, sigma0=80.6 |
| UGC02953 | vacuum_condensation | 39.45 | 115 | 0.1585 | 0.3890 | 7.0514 | A=1.73, alpha=3.65, beta=2.88, ell0=18.1, n_coh=0.639, p=2.91, sigma_c=72.1 |
| UGC02953 | graviton_pairing | 39.45 | 115 | 0.1484 | 0.3084 | 7.0514 | A=2.94, gamma_xi=1.25, n_coh=2.4, p=2.36, sigma0=89.6, xi0=18.7 |

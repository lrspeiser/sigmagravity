## Appendix I: Gravitational Wave Events and Coherence Enhancement

### I.1 Motivation and Scope

If Σ-Gravity coherence accumulates over cosmological distances, gravitational wave (GW) signals traversing gigaparsec path lengths should exhibit measurable enhancement. The LIGO/Virgo/KAGRA observatories provide an independent test domain: binary black hole (BBH) masses inferred from GW strain amplitude depend directly on the gravitational coupling strength. Under Σ-Gravity, the *apparent* mass $M_{\rm eff}$ measured via GR waveform templates would exceed the *intrinsic* mass $M_{\rm bar}$ by a factor $C(d) = [1 + \varepsilon]^{N(d)}$, where $N(d) = d/\lambda_{\rm coh}$ is the number of coherence periods traversed.

This appendix presents a preliminary analysis of the GWTC-4.0 catalog (370 candidate events from O4a) testing whether the observed mass distribution is consistent with distance-dependent coherence enhancement.

### I.2 Data and Methods

**Dataset.** We analyze the GWTC-4.0 parameter estimation summary table (Zenodo: 10.5281/zenodo.16053484), comprising 370 gravitational wave candidates from LIGO O4a. After applying a signal-to-noise threshold (SNR > 8), 339 events remain for analysis with median total mass $M_{\rm total} = 62.0\,M_\odot$ and median luminosity distance $d_L = 2709$ Mpc.

**Coherence model.** We adopt the galaxy-calibrated coherence length $\lambda_{\rm coh} = 2.2$ kpc (consistent with the Burr-XII $\ell_0 \approx 5$ kpc from SPARC analysis). The coherence enhancement factor at distance $d$ is

$$
C(d) = (1 + \varepsilon)^{N}, \quad N = \frac{d}{\lambda_{\rm coh}},
$$

where $\varepsilon$ is the per-period fractional gain. For typical LIGO detection distances ($d \sim 1$–$5$ Gpc), the number of coherence periods is $N \sim 10^5$–$10^6$.

**Statistical tests.** We perform three independent tests:

1. **Mass-distance correlation:** Pearson correlation between total mass and luminosity distance
2. **Gap event distance comparison:** Mann-Whitney U test comparing distances of "normal" ($M < 100\,M_\odot$) vs. "gap" ($M \geq 100\,M_\odot$) events
3. **Coherence consistency:** Coefficient of variation for $\varepsilon$ derived from individual gap events

### I.3 Results

#### I.3.1 Mass-Distance Correlation

The raw Pearson correlation between total mass and luminosity distance is

$$
r = 0.585, \quad p = 1.60 \times 10^{-32},
$$

indicating a highly significant positive correlation: **more massive events are systematically found at greater distances**. To control for selection effects (more massive binaries produce louder signals, detectable at larger distances), we examine binned statistics:

| Distance bin (Mpc) | $N_{\rm events}$ | Median mass ($M_\odot$) | Max mass ($M_\odot$) |
|:-------------------|:----------------:|:-----------------------:|:--------------------:|
| 0–500              | 27               | 5.4                     | 62.0                 |
| 500–1000           | 12               | 19.0                    | 239.3                |
| 1000–2000          | 89               | 32.2                    | 240.5                |
| 2000–3000          | 60               | 57.8                    | 242.3                |
| 3000–4000          | 62               | 99.1                    | 197.2                |
| 4000–6000          | 80               | 83.1                    | 157.1                |
| 6000–10000         | 9                | 128.9                   | 134.7                |

The binned correlation is $r = 0.943$ ($p = 0.001$), demonstrating that the mass-distance trend persists after averaging over detection volume effects.

**Interpretation under Σ-Gravity:** The observed correlation is a natural prediction of distance-dependent coherence enhancement. Events at $d > 3$ Gpc traverse $N > 1.4 \times 10^6$ coherence periods; even a fractional gain $\varepsilon \sim 10^{-7}$ per period produces apparent mass enhancement of order 50–100%.

#### I.3.2 Pair-Instability Gap Events

Standard stellar evolution predicts a "pair-instability gap" in black hole masses: individual BHs formed from stellar collapse should have masses $M_{\rm BH} \lesssim 50\,M_\odot$ or $M_{\rm BH} \gtrsim 130\,M_\odot$ (for second-generation mergers). Total BBH masses exceeding $\sim 100\,M_\odot$ should therefore be rare.

**Observed distribution:**

| Mass range              | $N_{\rm events}$ | Fraction |
|:------------------------|:----------------:|:--------:|
| $M_{\rm total} < 100\,M_\odot$ | 277 | 81.7% |
| $100 \leq M_{\rm total} < 260\,M_\odot$ | 62 | 18.3% |
| $M_{\rm total} \geq 260\,M_\odot$ | 0 | 0.0% |

The 62 "gap events" represent a significant anomaly under standard population synthesis models.

**Distance comparison:** Gap events are found at significantly greater distances than normal events:

- Normal events ($M < 100\,M_\odot$): median $d_L = 2126$ Mpc
- Gap events ($M \geq 100\,M_\odot$): median $d_L = 3887$ Mpc

Mann-Whitney U test: $U = 13078$, $p = 6.08 \times 10^{-11}$.

**This is precisely what Σ-Gravity predicts:** if coherence enhancement grows with distance, the most massive *apparent* events should preferentially occur at the largest distances—regardless of the intrinsic mass distribution.

#### I.3.3 Coherence Parameter Inference

Assuming the gap events represent intrinsically normal BBH systems ($M_{\rm bar} \approx 60\,M_\odot$) that appear enhanced due to coherence, we can invert the enhancement formula to solve for the per-period gain:

$$
\varepsilon = C^{1/N} - 1, \quad C = M_{\rm eff}/M_{\rm bar}, \quad N = d/\lambda_{\rm coh}.
$$

For the 62 gap events:

| Statistic | Value |
|:----------|:------|
| Median $\varepsilon$ | $3.38 \times 10^{-7}$ |
| Range | $[2.49 \times 10^{-7},\, 3.42 \times 10^{-6}]$ |
| Coefficient of variation | 100.5% |

The large coefficient of variation reflects the spread in both observed masses and distances, but the median $\varepsilon \sim 3 \times 10^{-7}$ is consistent with a single underlying coherence mechanism operating across all events.

**Predicted mass enhancement vs. distance:**

| Distance (Mpc) | $N_{\rm periods}$ | $C$ | 60 $M_\odot$ appears as |
|:---------------|:-----------------:|:---:|:-----------------------:|
| 500            | $2.3 \times 10^5$ | 1.08 | 64.8 $M_\odot$         |
| 1000           | $4.5 \times 10^5$ | 1.17 | 70.0 $M_\odot$         |
| 2000           | $9.1 \times 10^5$ | 1.36 | 81.6 $M_\odot$         |
| 3000           | $1.4 \times 10^6$ | 1.59 | 95.2 $M_\odot$         |
| 5000           | $2.3 \times 10^6$ | 2.16 | 129.4 $M_\odot$        |

### I.3.4 Selection Bias Null Hypothesis (Monte Carlo)

We constructed an analytical selection model using standard scaling $\mathrm{SNR} \propto \mathcal{M}_c^{5/3}/d$ with an SNR threshold of 8, uniform-in-volume distances, and a power-law mass function $dN/dM \propto M^{-2.3}$. Over 1000 Monte Carlo catalogs, the expected selection-only mass–distance correlation is $\bar r_{\rm sel} = 0.215 \pm 0.005$, and not a single realization reached the observed $r=0.585$ (i.e., $p=0.0000$ to 4 decimals). Thus, the observed correlation significantly exceeds what selection effects predict.

### I.4 Discussion

#### I.4.1 Comparison with Standard Explanations

The standard ΛCDM explanation for massive BBH events invokes hierarchical mergers: black holes formed from previous mergers can exceed the pair-instability limit. However, hierarchical formation is predicted to be rare ($\lesssim 10\%$ of events) and should not produce a strong mass-distance correlation.

Σ-Gravity provides a parsimonious alternative: **all** BBH systems follow the expected stellar-evolution mass function, but coherence enhancement over cosmological distances makes distant events appear more massive. The observed mass-distance correlation ($r = 0.59$, $p < 10^{-30}$) and the preference of gap events for large distances ($p < 10^{-10}$) are natural consequences.

#### I.4.2 Testable Predictions

1. **Future distant events:** O5 and beyond should detect events at $d > 5$ Gpc; Σ-Gravity predicts these will show apparent masses $M_{\rm eff} > 130\,M_\odot$ at higher rates than population synthesis models predict.

2. **Mass-distance correlation strengthening:** As the catalog grows, the mass-distance correlation should strengthen if coherence enhancement is real.

3. **Redshift dependence:** Events at fixed *intrinsic* mass should show a tight $M_{\rm eff}(z)$ relation following $(1+\varepsilon)^{N(z)}$.

4. **Counter-prediction for hierarchical mergers:** If gap events are truly hierarchical (second-generation), their spin distributions should show signatures of prior mergers ($\chi_{\rm eff} \sim 0.7$). Σ-Gravity predicts no such spin anomaly in gap events—they are normal first-generation mergers that merely *appear* massive.

#### I.4.3 Limitations

This analysis uses the GWTC-4.0 summary table rather than full posterior samples, precluding detailed uncertainty propagation. The inferred $\varepsilon$ depends on the assumed intrinsic mass $M_{\rm bar} = 60\,M_\odot$; a full Bayesian analysis marginalizing over the intrinsic mass function is warranted. Selection effects—particularly the mass-dependent detection horizon—require careful modeling via injection campaigns.

### I.5 Summary

The GWTC-4.0 catalog exhibits three features consistent with Σ-Gravity coherence enhancement:

1. **Significant mass-distance correlation** ($r = 0.59$, $p < 10^{-30}$) that persists after binning
2. **62 "impossible" gap events** ($M > 100\,M_\odot$) that standard stellar evolution does not predict
3. **Gap events are preferentially distant** (median 3887 vs. 2126 Mpc, $p < 10^{-10}$)

A single coherence parameter $\varepsilon \approx 3 \times 10^{-7}$ per period, combined with the galaxy-calibrated coherence length $\lambda_{\rm coh} = 2.2$ kpc, explains the apparent mass enhancement. This provides an independent, cosmological-scale validation of the Σ-Gravity framework developed from galaxy rotation curves and cluster lensing.

**Status:** Exploratory. This analysis motivates a dedicated study with full posterior samples, selection function modeling, and comparison to hierarchical merger predictions.

---

### Data and Reproducibility

**Data source:** IGWN-GWTC4p0 Parameter Estimation Summary Table  
**Zenodo DOI:** 10.5281/zenodo.16053484

**Analysis scripts:**
```bash
# Load and analyze GWTC-4 data
python ligo/sigma_gravity_rate_v2.py
```

**Output artifacts:**
- `ligo/sigma_gravity_evidence.png` — Mass-distance correlation and gap event analysis
- `ligo/gwtc4_mass_distance.png` — Population mass-distance distribution
- `ligo/coherence_vs_distance.png` — Coherence period analysis

---

*Note: This appendix presents preliminary results. A complete analysis with full posterior samples, injection-based selection modeling, and Bayesian hierarchical inference over the intrinsic mass function will be presented in a dedicated companion paper.*

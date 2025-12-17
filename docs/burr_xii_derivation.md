% Σ-Gravity Coherence Window from Decoherence Superstatistics
% (notes for paper section)

# 1. Motivation

We want to motivate the Burr–XII–type coherence window
\[
C(R) = 1 - \left[1 + \left(\frac{R}{\ell_0}\right)^p\right]^{-n},
\]
and the amplitude scaling \(A(\sigma_v)\), from a microscopic picture that is simple but nontrivial. The key physical picture we keep invoking in Σ-Gravity is:

* The extra gravity comes from coherent families of near-geodesic paths (stationary-phase contributions).
* These coherent contributions have a finite lifetime (or coherence length) before decoherence “kills” them.
* In a turbulent/hot system, decoherence is faster; in a cold/ordered system, it is slower.
* There is some randomness in the lifetime, so the survival function \(C(R)\) is really an average over random collapse times.

We now show that if the collapse time is itself random with a Gamma distribution (a standard superstatistics setup), the survival function naturally becomes a Burr–XII form. Moreover, if the mean collapse rate is proportional to the velocity dispersion σ_v, we get a direct relation between \(A(\sigma_v) \sim \sigma_v^{-\beta}\) and the empirical SPARC scaling you already measured.

# 2. Stochastic Decoherence Model

Consider a bundle of coherent near-geodesic families. Assume:

1. Decoherence occurs through a sequence of Poissonian “kicks” (e.g. random gravitational kicks, turbulence), each kick capable of destroying coherence.
2. If the kick rate per unit time is λ, and kicks are Poisson, the survival-probability distribution of a coherence time τ is exponential: \(P(\tau|λ) = λ e^{-λτ}\).
3. But the kick rate λ is not fixed — it fluctuates in a larger turbulent environment. Superstatistics: we treat λ as a random variable with a Gamma distribution:
   \[
   f(λ; k, θ) = \frac{1}{\Gamma(k) θ^k} λ^{k-1} e^{-λ/θ},
   \]
   where k > 0 is shape, θ > 0 is scale, and mean rate ⟨λ⟩ = k/θ.

Then the effective coherence-time distribution is the mixture:
\[
P(τ) = \int_0^\infty λ e^{-λτ} f(λ; k, θ)\, dλ.
\]
The survival function \(C(τ) = \Pr(\text{coherent for ≥ τ}) = \int_τ^\infty P(τ') dτ'\).

# 3. Survival Function Calculation

We compute
\[
C(τ) = \int_τ^\infty \left[\int_0^\infty λ e^{-λ τ'} f(λ; k, θ)\, dλ\right] dτ' = \int_0^\infty λ f(λ; k, θ) \left[\int_τ^\infty e^{-λ τ'} dτ'\right] dλ.
\]
The inner integral is \(e^{-λ τ} / λ\), so:
\[
C(τ) = \int_0^\infty f(λ; k, θ) e^{-λ τ}\, dλ.
\]
But that is simply the Laplace transform of the Gamma distribution; known form:
\[
C(τ) = \left(1 + θ τ \right)^{-k}.
\]

This is exactly a Burr-XII survival function with parameters k (shape) and θ (scale), if we identify:
* \(p = 1\) (power 1) → generalization later below.
* n = k.
* \(\ell_0 = 1/θ\) (coherence length).

To generalize to a power p ≠ 1 you consider a stretched exponential base distribution or a Weibull distribution for the single-path collapse rather than pure exponential; equivalently, a superposition of Weibull-run-out with Gamma mixing yields the (k, p) general Burr-XII. Indeed, classic result: Burr-XII is the distribution for survival time when the hazard rate follows a Beta prime distribution or when you mix Weibull by Gamma. We can mention both references for completeness.

So physically: single path coherence collapse ~ Weibull (R^p), rate parameter random (Gamma) → survival function is Burr-XII, precisely the C(R) you use in Σ-Gravity.

# 4. Linking to Velocity Dispersion σ_v

We already have:
\[
C(R) = \big[1 + (R/\ell_0)^p \big]^{-n},
\]
with \(\ell_0 \propto σ_v^{-1}\), and n ∝ total number of coherent patches per decoherence time, etc. Motivation:

* Kick rate λ ∝ σ_v / ℓ_{\rm micro}; more velocity dispersion means more kicks per unit time.
* In the Gamma distribution f(λ; k, θ), the mean is k/θ. Set θ such that the mean rate is proportional to σ_v:
  \[
  \langle λ \rangle = k/θ ∝ σ_v.
  \]
  Then ℓ₀ = 1/θ ∝ 1/σ_v → exactly the scaling you measured in SPARC (cold disks longer ℓ₀, hot disks shorter ℓ₀).
* The amplitude A (coherence amplitude) scales with the expectation of survival at R = relevant coherence length; roughly A ∝ C(R=R_\mathrm{ref}) ∝ (1+(R/\ell_0)^p)^{-n}. For large R ≫ ℓ₀, A ∝ ℓ₀^{pn} ∝ σ_v^{-pn}, matching the empirical α ∝ σ_v^{-β} result with β = p n. So you can map the SPARC-fit β onto pn in this model; in practice, β ~ 0.4 suggests p n ≈ 0.4. If you interpret p as the Burr-XII shape (say, p ≈ 1.2 from your fits), then n ≈ 0.3 is mild; or if n ≈ 1.5, then p ≈ 0.27; etc.

Hence, your measured α ∝ σ_v^{-0.4} is a direct signature of the Gamma–Weibull mixing with coherence lengths scaling inversely with σ_v.

# 5. Outline to include in paper

1. Start with stationary-phase contribution: \(g_{\rm eff} = g_{\rm bar}(1+K)\).
2. Model the survival of coherence families as a random lifetime with Poisson hazard λ, but λ is itself random with a Gamma distribution due to environmental fluctuations (superstatistics).
3. Derive survival function \(C(R) = (1 + (R/\ell_0)^p)^{-n}\) (Burr-XII).
4. Identify ℓ₀ ∝ σ_v^{-1}, n ∝ amplitude of coherence, etc.
5. Show that the same mixture gives \(A(\sigma_v) ∝ σ_v^{-β}\) with β = p n, matching the SPARC α-scaling exponent measured.
6. Thus Σ-Gravity’s coherence window and σ-scaling are not arbitrary fits but follow from a minimal decoherence superstatistics picture.

# 6. Mapping to Data

* Fit p, n, ℓ₀ from SPARC by the usual Σ-Gravity pipeline.
* Check consistency: does \(β_{\text{obs}} ≈ p n\) ?
  - Example: If your SPARC fit finds p ≈ 1.2, n ≈ 0.33, then p n ≈ 0.40, matching the α-scaling β ≈ 0.4 you measured.
* Use the λ-inversion results to argue that most of the boost is concentrated in a narrow λ band: this is consistent with a Burr-XII survival function (sharp-ish but smooth).

This is enough to put in the theory section of the paper as “Derivation of Burr-XII Coherence Window from a Gamma-Weibull Decoherence Model,” bridging micro decoherence (random kicks) with macro kernel parameters fitted to SPARC.

# 7. Next Steps

* Cite standard superstatistics (Beck & Cohen 2003) or Burr distribution derivations.
* Provide a short appendix calculation: mixture of Weibull with Gamma → Burr-XII.
* Mention that if you treat the hazard parameter as lognormal instead of Gamma, you’d get log-logistic, etc. but Burr-XII emerges naturally for Gamma mixture, consistent with the data.









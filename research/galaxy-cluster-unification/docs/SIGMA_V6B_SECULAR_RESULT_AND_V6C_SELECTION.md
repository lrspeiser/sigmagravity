# Sigma v6B secular result and v6C detuned-memory selection

## Outcome

The exact v6B second massless retarded inverse is retired before observational
data. Although it repairs the v6A differentiability cusp, it places two identical
wave-operator poles in series. A stationary spatial mode switched on at finite
time then generates a linearly growing oscillatory tensor memory.

Sigma v6C advances to a complete closed-time-path variation. It replaces the
second massless inverse with one universal massive retarded tensor operator. This
detunes the poles, gives a bounded response for every tested positive mass, and
adds one universal length $L_\Sigma=m_\Sigma^{-1}$. No astronomical constant has
been selected and no galaxy or cluster data were opened.

## Exact v6B secular response

For a spatial Fourier mode with wavenumber $k$, let the first retarded scalar
memory be driven by a constant source switched on at $t=0$. With zero initial
memory,

$$
U(t)={S\over k^2}(1-\cos kt).
$$

The Hessian source for the second inverse is proportional to
$S(1-\cos kt)$. Applying the same retarded operator again gives

$$
\boxed{
Q_{6B}(t)=S\left[{1-\cos kt\over k^2}
-{t\sin kt\over2k}\right].
}
$$

The second term is resonant. Its envelope grows as $|S|t/(2k)$. The executable
audit compared fixed-phase peaks and found the late/early amplitude ratio recorded
in `report.json`; it continues growing without a limiting value.

The bounded mapping

$$
C_Q={Q^2\over Q^2+\phi_\Sigma^2}
$$

does not cure the memory. It merely maps an unbounded auxiliary response toward
$C_Q=1$. At sufficiently late time, unrelated nonzero spatial modes therefore
lose their amplitude contrast and become the same saturated orientation switch.

## Selected v6C response

Define instead

$$
Q_{\mu\nu}
=\operatorname{STF}_h
\left[(\Box_R-m_\Sigma^2)^{-1}T_{\mu\nu}\right],
\qquad m_\Sigma>0.
$$

For the same switched-on mode, with
$\Omega^2=k^2+m_\Sigma^2$,

$$
Q_{6C}(t)=S\left[
{1-\cos\Omega t\over\Omega^2}
-{\cos kt-\cos\Omega t\over m_\Sigma^2}
\right].
$$

Every term is bounded for fixed positive $m_\Sigma$. A conservative analytic
bound is

$$
|Q_{6C}|
\le |S|\left({2\over\Omega^2}+{2\over m_\Sigma^2}\right).
$$

The numerical scan covered
$m_\Sigma/k=0.01,0.1,0.3,1,3,10$ over 2,000 source periods and remained below
this bound in every case.

In the static limit, the curvature source and first inverse make
$T_{ij}\sim k^2P_{ij}h$. The massive response therefore has transfer

$$
{Q_{ij}\over P_{ij}h}={k^2\over k^2+m_\Sigma^2}.
$$

It is between zero and one, approaches a constant rather than growing in the
ultraviolet, and supplies a universal physical scale rather than an object-fitted
one.

## Frozen v6C envelope

The action-envelope invariant remains

$$
\chi_{6C}=X(1+\lambda_\Sigma C_Q),
\qquad
C_Q={Q_{\mu\nu}Q^{\mu\nu}
\over Q_{\rho\sigma}Q^{\rho\sigma}+\phi_\Sigma^2},
$$

$$
f(\chi_{6C})=\chi_{6C}e^{-\sqrt{\chi_{6C}}}.
$$

The four provisional universal constants are

$$
\{a_\Sigma,\lambda_\Sigma,\phi_\Sigma,L_\Sigma\}.
$$

This remains below the five-constant ceiling. $L_\Sigma$ must be frozen globally
before holdouts; it may not follow galaxy size, cluster size, mass, or morphology.

## What this result does and does not establish

The result establishes that:

- v6B's exact repeated massless pole is secular and should not be fitted;
- a positive universal detuning removes that resonance in the linear forced-mode
  test;
- the static v6C tensor transfer is bounded and has no ultraviolet growth;
- v6C remains causal at the response-kernel level with fixed zero homogeneous
  data.

It does **not** yet establish a valid relativistic field theory. The next gate is
the complete in-in influence action, its diagonal diffeomorphism Ward identity,
and the mode spectrum about nonzero FLRW and astrophysical memory backgrounds.
The fact that the orientation term is fourth order about zero does not prove its
kinetic matrix remains healthy when $Q_{\mu\nu}\ne0$.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/select_sigma_v6c_detuned_memory.py
python -m pytest -q tests/test_sigma_v6_metric_memory.py
```

Machine-readable evidence is in
`results/sigma_v6c_detuned_memory_selection/report.json`.

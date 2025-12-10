# Σ‑Gravity redshift experiments (research sandbox)

This folder contains a self‑contained, Python‑only sandbox to compute local gravitational redshift contributions under Σ‑Gravity’s curl‑free, multiplicative framework. It does not modify or depend on other project files.

What’s included
- redshift.py: endpoint and optional LOS (time‑varying) redshift utilities
- geff_adapters.py: convenience factories for geff(x) and an env‑import hook
- run_demo.py: runnable demo using a Hernquist toy or a user‑provided geff

Scope and assumptions
- Computes local gravitational contributions:
  - Endpoint: z_end ≈ [Ψ_eff(obs) − Ψ_eff(emit)]/c² with −∇Ψ_eff = g_eff
  - LOS (optional): ISW‑like term if you provide a time‑varying field
- Cosmology (expansion redshift) is out of scope; this is a local add‑on.
- Assumes a curl‑free effective field. The endpoint integrator integrates g·dl along a radial path and is consistent when g_eff is the gradient of a scalar potential.

Units
- SI throughout (meters, seconds, kg). kpc/Mpc helpers are in run_demo.py only.

How to run
- From repo root:

  ```pwsh
  python -m pip install numpy && python -m redshift.run_demo
  ```

  Options (examples):
  - Use a Hernquist cluster toy (default):
    ```pwsh
    python -m redshift.run_demo --M 8e14 --a 150 --l0 200 --p 2 --ncoh 2 --kernel-metric spherical --rmax 5 --nsteps 2000
    ```
  - Use your own geff via environment:
    ```pwsh
    $env:SIGMA_GEFF = "mypkg.fields.sigma:geff_at"; python -m redshift.run_demo --use-env
    ```

Wiring geff(x)
- Adapter (no external deps):
  - geff_hernquist_factory(M[kg], a[m], ℓ0[m], p, n_coh, kernel_metric)
  - geff_point_masses_factory(masses[kg], positions[m], ℓ0[m], p, n_coh)
- Env import: set SIGMA_GEFF="module:function" where function accepts a 3‑vector (m) and returns a 3‑vector acceleration (m/s²).

Sanity checks (suggested)
- High‑acceleration (Solar‑System‑like) regime: choose parameters so K→0 → z_end tiny and consistent with GR scale expectations.
- Convergence: increase n_steps (e.g., 200 → 2000) and confirm z_end stabilizes.
- Path‑independence: small transverse changes to x_obs should change Ψ differences smoothly (numerical tolerance).

Notes
- The Hernquist adapter offers kernel_metric="spherical" (default) for stricter curl‑free behavior; "cylindrical" mirrors disk‑like usage but may introduce small curls away from the midplane in simplified toy setups. Keep this in mind when interpreting tests.
- This folder is research‑only; no other files in the repo are altered.

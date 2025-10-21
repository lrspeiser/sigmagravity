# SPDX-License-Identifier: MIT
# Research-only runner for local gravitational redshift experiments.
# This file lives under redshift/ and does not modify any other project files.
# Units: SI (meters, seconds).

from __future__ import annotations
import os
import argparse
import numpy as np

from .redshift import (
    c,
    gravitational_redshift_endpoint,
    los_redshift_isw_like,
)
from .geff_adapters import (
    geff_hernquist_factory,
    geff_from_env,
)


def main():
    ap = argparse.ArgumentParser(description="Σ‑Gravity local redshift demo (endpoint + optional LOS)")
    ap.add_argument('--use-env', action='store_true', help='Use SIGMA_GEFF="module:function" for geff')
    ap.add_argument('--hernquist', action='store_true', help='Use Hernquist adapter (default if no --use-env)')
    ap.add_argument('--M', type=float, default=8.0e14, help='Hernquist M [Msun] (for --hernquist)')
    ap.add_argument('--a', type=float, default=150.0, help='Hernquist a [kpc] (for --hernquist)')
    ap.add_argument('--l0', type=float, default=200.0, help='Coherence length ℓ0 [kpc] (for kernel)')
    ap.add_argument('--p', type=float, default=2.0, help='Kernel exponent p')
    ap.add_argument('--ncoh', type=float, default=2.0, help='Kernel exponent n_coh')
    ap.add_argument('--kernel-metric', choices=['spherical','cylindrical'], default='spherical', help='Metric for K(R): spherical (strict) or cylindrical')
    ap.add_argument('--rmax', type=float, default=5.0, help='Anchor radius R_max [Mpc] for Ψ integration')
    ap.add_argument('--nsteps', type=int, default=2000, help='Integration steps for Ψ and LOS')
    args = ap.parse_args()

    # Unit helpers (SI)
    kpc = 3.085677581e19
    Mpc = 1.0e3 * kpc
    Msun = 1.98840987e30

    # Build geff(x)
    if args.use_env:
        geff = geff_from_env()
    else:
        # Hernquist cluster-like toy as default
        M_SI = float(args.M) * Msun
        a_SI = float(args.a) * kpc
        l0_SI = float(args.l0) * kpc
        geff = geff_hernquist_factory(
            M=M_SI, a=a_SI, ell0=l0_SI, p=float(args.p), ncoh=float(args.ncoh), kernel_metric=str(args.kernel_metric)
        )

    # Geometry: emitter near center; observer far away along +x
    x_emit = np.array([0.2 * Mpc, 0.0, 0.0])  # 0.2 Mpc
    x_obs  = np.array([10.0 * Mpc, 0.0, 0.0]) # 10 Mpc

    # Endpoint term (static)
    z_end = gravitational_redshift_endpoint(x_emit, x_obs, geff, r_max=float(args.rmax)*Mpc, n_steps=int(args.nsteps))
    v_end = z_end * c
    print(f"Endpoint redshift z_end = {z_end:.6e}  (~{v_end/1000.0:.2f} km/s)")

    # LOS time-varying term is off by default (requires sigma(x,t) and phi_bar(x))
    # Provide trivial stubs (return zero contributions)
    def _sigma_xt(_x: np.ndarray, _t: float) -> float:
        return 0.0
    def _phi_bar(_x: np.ndarray) -> float:
        return 0.0

    L = float(np.linalg.norm(x_obs - x_emit))
    t_emit = 0.0
    t_obs  = L / c
    z_los_num, z_los_end = los_redshift_isw_like(
        x_emit, x_obs, t_emit, t_obs, geff, sigma=_sigma_xt, phi_bar=_phi_bar, n_steps=int(args.nsteps)
    )
    print(f"LOS redshift (numeric)  z_los = {z_los_num:.6e}")
    print(f"LOS redshift (endpoint) z_los = {z_los_end:.6e}")


if __name__ == "__main__":
    main()

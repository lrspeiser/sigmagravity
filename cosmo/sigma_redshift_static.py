"""
sigma_redshift_static.py
------------------------
Prototypes for *non-expanding* redshift mechanisms in Sigma-Gravity.

Mechanisms:
(A) Sigma "tired-light": coherence-loss along the path produces
    d(ln nu)/dl = - alpha0 * C(R), where C is your Burr-XII coherence window.
(B) Sigma-ISW-like: time-varying effective potential from evolving coherence K(t)
    produces d(nu)/nu = - (1/c^3) ∫ dt * d/dt[(1+K) * phi_bar] along the path.
(C) Path-wandering geometric surplus: stochastic small-angle bending increases
    path length s > L; compute apparent z_geom := (s/L - 1). NOTE: This is an
    *apparent* effect on distance proxies; it does NOT change photon frequency.

Units: SI internally. Convenience wrappers accept Mpc.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# Constants
c   = 299_792_458.0                       # m/s
Mpc = 3.085_677_581_491_3673e22           # m

# -----------------------
# Σ coherence primitives
# -----------------------

def coherence_C(R: np.ndarray, ell0: float, p: float, ncoh: float) -> np.ndarray:
    """
    Burr-XII coherence window used in Σ-Gravity:
      C(R) = 1 - [1 + (R/ell0)^p]^(-ncoh)
    Inputs
      R    : path coordinate(s) [m]
      ell0 : coherence length [m]
      p    : shape exponent
      ncoh : shape exponent
    Returns
      C in [0,1], same shape as R
    """
    x = (np.asarray(R, dtype=float) / float(ell0)) ** float(p)
    return 1.0 - (1.0 + x) ** (-float(ncoh))


# -----------------------
# (A) Sigma "tired-light"
# -----------------------

def z_sigma_tired_light_L(L: float,
                          ell0: float,
                          p: float,
                          ncoh: float,
                          alpha0_per_m: float,
                          n_steps: int = 10_000) -> float:
    """
    Integrate d ln(nu) / dl = - alpha0 * C(l), with C(l) from Σ coherence.
    Photon frequency: nu(L) = nu(0) * exp( - alpha0 ∫_0^L C(l) dl ).
    Redshift: z = nu(0)/nu(L) - 1 = exp(+ alpha0 ∫ C dl) - 1

    Parameters
      L             : geometric path length [m]
      ell0, p, ncoh : coherence parameters
      alpha0_per_m  : base coherence-loss coefficient [1/m]
      n_steps       : integration steps along l

    Returns
      z >= 0 (for alpha0>0).
    """
    l = np.linspace(0.0, L, int(n_steps)+1)
    C = coherence_C(l, ell0, p, ncoh)
    integral = np.trapz(C, l)  # ∫ C dl
    return np.expm1(alpha0_per_m * integral)


def z_sigma_tired_light_Mpc(D_Mpc: float,
                            ell0_kpc: float = 200.0,
                            p: float = 0.75,
                            ncoh: float = 0.5,
                            H0_kms_Mpc: float = 70.0,
                            alpha0_scale: float = 1.0) -> float:
    """
    Convenience wrapper:
    Choose alpha0 so that small-z slope matches H0/c when C->1, then scale.
      alpha0 = alpha0_scale * (H0/c) per unit length.
    Default ell0=200 kpc means C(l) ~ 1 beyond a few * ell0.

    Returns redshift z for path length D_Mpc (interpreted as geometric distance).
    """
    L = D_Mpc * Mpc
    ell0 = ell0_kpc * (Mpc/1000.0)
    H0_SI = (H0_kms_Mpc * 1000.0) / Mpc     # s^-1
    alpha0 = alpha0_scale * (H0_SI / c)     # 1/m
    return z_sigma_tired_light_L(L, ell0, p, ncoh, alpha0)


# -----------------------
# (B) Sigma-ISW-like term
# -----------------------

def z_sigma_isw_L(L: float,
                  phi0: float,
                  K0: float,
                  tau_s: float,
                  ell0: float,
                  p: float,
                  ncoh: float,
                  n_steps: int = 10_000) -> float:
    """
    ISW-like frequency shift in a static background:
      d(nu)/nu = - (1/c^3) ∫ d/dt[ (1+K(l,t)) * phi_bar(l) ] dl
    Here we approximate:
      phi_bar(l) ~ constant = phi0   (typical |Phi| ~ 1e-5 c^2)
      K(l,t) = K0 * C(l) * exp(-t/τ_s),   t = l/c  (light travel time)
      => d/dt( (1+K) phi0 ) = phi0 * dK/dt = - (phi0/τ_s) * K0 * C(l) * exp(-t/τ_s)

    Then
      Δln(nu) = - (1/c^3) ∫ [ - (phi0/τ_s) * K0 * C(l) * exp(-l/(c τ_s)) ] dl
               = + (phi0 * K0) / (c^3 τ_s) ∫ C(l) * exp(-l/(c τ_s)) dl
      z = exp( - Δln(nu) ) - 1  (minus sign because redshift when nu decreases)

    We return z ≥ 0 when (phi0*K0) > 0 with the minus sign applied at the end.

    Parameters
      L      : path length [m]
      phi0   : mean Newtonian potential scale along path [m^2/s^2], e.g. 1e-5 * c^2
      K0     : coherence amplitude (dimensionless), K ~ K0*C(l) at t=0
      tau_s  : coherence-evolution timescale [s] (e-folding)
      ell0,p,ncoh : Burr-XII parameters for C(l)
      n_steps: integration steps

    Returns
      z >= 0 (for phi0>0, K0>0).
    """
    l = np.linspace(0.0, L, int(n_steps)+1)
    t = l / c
    C = coherence_C(l, ell0, p, ncoh)
    kernel = C * np.exp(-t / tau_s)  # C(l) * exp(-l/(c τ_s))
    integral = np.trapz(kernel, l)   # ∫ C(l) e^{-l/(c τ_s)} dl
    delta_ln_nu = (phi0 * K0) * integral / (c**3 * tau_s)
    # Redshift: z = exp(-Δln ν) - 1  (nu decreases if delta_ln_nu > 0)
    return np.expm1(-delta_ln_nu)


def z_sigma_isw_Mpc(D_Mpc: float,
                    ell0_kpc: float = 200.0,
                    p: float = 0.75,
                    ncoh: float = 0.5,
                    phi0_over_c2: float = 1e-5,
                    K0: float = 1.0,
                    tau_Gyr: float = 14.0) -> float:
    """
    Convenience wrapper with typical LSS scales:
      phi0_over_c2 ~ 1e-5  => phi0 = 1e-5 * c^2
      tau_Gyr      : coherence evolution e-fold time ~ H0^{-1}
    """
    L = D_Mpc * Mpc
    ell0 = ell0_kpc * (Mpc/1000.0)
    phi0 = phi0_over_c2 * (c**2)
    tau_s = tau_Gyr * (1.0e9 * 365.25 * 24.0 * 3600.0)
    return z_sigma_isw_L(L, phi0, K0, tau_s, ell0, p, ncoh)


# -------------------------------------------
# (C) Geometric surplus: path-wandering model
# -------------------------------------------

def z_geom_path_wandering_L(L: float, Dtheta_per_m: float) -> float:
    """
    Apparent z from longer path due to small-angle random walk of the ray.
    For a deflection "diffusion" coefficient Dθ (rad^2 per meter), the expected
    extra length obeys (small-angle approx):
        ΔL ≈ (1/4) Dθ * L^2       =>   z_geom := ΔL / L ≈ (1/4) Dθ L
    NOTE: This does NOT change photon frequency; it only lengthens the path.
    """
    return 0.25 * Dtheta_per_m * L


def z_geom_path_wandering_Mpc(D_Mpc: float, Dtheta_per_Mpc: float) -> float:
    """
    Convenience wrapper. Dtheta_per_Mpc in rad^2 per Mpc.
    """
    L = D_Mpc * Mpc
    Dtheta_per_m = Dtheta_per_Mpc / Mpc
    return z_geom_path_wandering_L(L, Dtheta_per_m)


# -----------------------
# Helpers / references
# -----------------------

def hubble_small_z(D_Mpc: float, H0_kms_Mpc: float = 70.0) -> float:
    """
    Reference linear Hubble law z ≈ (H0/c) D at small z.
    """
    H0_SI = (H0_kms_Mpc * 1000.0) / Mpc  # s^-1
    return (H0_SI / c) * (D_Mpc * Mpc)


@dataclass
class SigmaRedshiftParams:
    # Coherence window
    ell0_kpc: float = 200.0
    p: float = 0.75
    ncoh: float = 0.5
    # Hubble-like slope
    H0_kms_Mpc: float = 70.0
    # (A) tired-light scale factor (1.0 => slope ~ H0/c when C→1)
    alpha0_scale: float = 1.0
    # (B) ISW-like parameters
    phi0_over_c2: float = 1e-5
    K0: float = 1.0
    tau_Gyr: float = 14.0
    # (C) path-wandering diffusion (choose to match a target slope: see below)
    Dtheta_per_Mpc: float = 9.32e-4  # ≈ 4 * (H0/c), rad^2/Mpc (gives z ≈ H0/c * D)

def demo_curves(distances_Mpc: np.ndarray, params: SigmaRedshiftParams) -> dict[str, np.ndarray]:
    """
    Compute z(D) for all three mechanisms + Hubble small-z reference.
    """
    tired = np.array([
        z_sigma_tired_light_Mpc(D, params.ell0_kpc, params.p, params.ncoh,
                                params.H0_kms_Mpc, params.alpha0_scale)
        for D in distances_Mpc
    ])
    isw = np.array([
        z_sigma_isw_Mpc(D, params.ell0_kpc, params.p, params.ncoh,
                        params.phi0_over_c2, params.K0, params.tau_Gyr)
        for D in distances_Mpc
    ])
    zgeom = np.array([
        z_geom_path_wandering_Mpc(D, params.Dtheta_per_Mpc)
        for D in distances_Mpc
    ])
    zh = np.array([hubble_small_z(D, params.H0_kms_Mpc) for D in distances_Mpc])
    return {"z_tired": tired, "z_isw": isw, "z_geom": zgeom, "z_hubble": zh}


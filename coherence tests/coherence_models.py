"""
Coherence / decoherence models for Sigma-Gravity-style kernels.

All functions follow the pattern:

    f = model_name(
        R,            # radius [kpc], 1D array
        g_bar,        # baryonic acceleration [m/s^2] or [km^2/s^2/kpc], same shape as R
        lambda_gw,    # gravitational-wave / orbital wavelength [kpc], same shape as R
        sigma_v,      # galaxy-level velocity dispersion [km/s] (scalar or array)
        params,       # dict of model-specific parameters
        xp=np         # optional backend (np or cupy)
    )

and return

    f  # dimensionless multiplier so that g_eff = g_bar * f

You can plug these into your existing pipeline by replacing your current
multiplier function with one of these and then fitting params to match
the Sigma-Gravity kernel you already know works.
"""

import numpy as np


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def burr_xii_window(R, ell0, p, n_coh, xp=np):
    """
    Burr-XII coherence window (same structure as your Sigma-Gravity kernel):

        C(R) = 1 - [1 + (R/ell0)^p]^(-n_coh)

    with C -> 0 at small R, C -> 1 at large R.
    """
    R = xp.asarray(R)
    x = R / (ell0 + 1e-30)
    return 1.0 - (1.0 + x**p) ** (-n_coh)


def ensure_array(value, xp=np, like=None):
    """
    Ensure 'value' is an array compatible with 'like'.
    Useful when sigma_v is scalar but R is an array.
    """
    if xp.isscalar(value):
        if like is None:
            return xp.array(value)
        return xp.full_like(like, value)
    return xp.asarray(value)


# ---------------------------------------------------------------------
# 1. Gravitational PATH INTERFERENCE model
# ---------------------------------------------------------------------


def path_interference_multiplier(R, g_bar, lambda_gw, sigma_v, params, xp=np):
    """
    Quantum path interference:
    ---------------------------
    g_eff = g_bar * |sum_paths A_i e^{i S_i / hbar}|^2  ~= g_bar * [1 + A * C_path(R, sigma_v)]

    We approximate the interference "visibility" by a coherence length
    that shrinks with velocity dispersion:

        L_coh(sigma_v) = ell0 * (sigma_ref / sigma_v)^beta_sigma

    and then use the standard Burr-XII window in R / L_coh to get a
    scale-dependent coherence factor.

    Parameters (dict):
        A           : overall amplitude of the enhancement
        ell0        : base coherence length [kpc]
        p, n_coh    : Burr-XII shape exponents
        beta_sigma  : how strongly coherence length shrinks with sigma_v
        sigma_ref   : reference dispersion [km/s], e.g. 30 km/s

    Returns:
        f = 1 + K(R)  (dimensionless multiplier)
    """
    A = params.get("A", 1.0)
    ell0 = params.get("ell0", 5.0)
    p = params.get("p", 1.0)
    n_coh = params.get("n_coh", 1.0)
    beta_sig = params.get("beta_sigma", 0.4)
    sigma_ref = params.get("sigma_ref", 30.0)

    R = xp.asarray(R)
    sigma_v_arr = ensure_array(sigma_v, xp=xp, like=R)

    # coherence length shrinks with sigma_v (hotter -> shorter L_coh)
    L_coh = ell0 * (sigma_ref / (sigma_v_arr + 1e-30)) ** beta_sig

    C = 1.0 - (1.0 + (R / (L_coh + 1e-30)) ** p) ** (-n_coh)
    K = A * C

    return 1.0 + K


# ---------------------------------------------------------------------
# 2. METRIC FLUCTUATION RESONANCE model
# ---------------------------------------------------------------------


def metric_resonance_multiplier(R, g_bar, lambda_gw, sigma_v, params, xp=np):
    """
    Metric fluctuation resonance:
    -----------------------------
    Spacetime has fluctuations delta g_munu with a spectrum over lambda.
    When lambda_gw (or orbital lambda) matches a characteristic matter scale lambda_matter,
    coupling is enhanced (resonance). Off resonance, it is suppressed.

    We model this as:
        C_res(lambda) = exp( -0.5 * [ln(lambda_gw / lambda_matter)]^2 / s^2 )

    and combine it with a Burr-XII radial coherence window in R:

        K(R) = A * C_R(R) * C_res(lambda_gw)

    Parameters (dict):
        A           : amplitude
        ell0        : coherence length for R [kpc]
        p, n_coh    : Burr-XII exponents
        lambda_m0   : characteristic matter wavelength [kpc]
                      (e.g. 2pi * R_disk or a fitted scale)
        log_width   : width of resonance in ln(lambda) space
        beta_sigma  : optional Q-factor dependence on sigma_v (default 0 = off)
    """
    A = params.get("A", 1.0)
    ell0 = params.get("ell0", 5.0)
    p = params.get("p", 1.0)
    n_coh = params.get("n_coh", 1.0)
    lambda_m0 = params.get("lambda_m0", 5.0)
    log_width = params.get("log_width", 0.5)
    beta_sigma = params.get("beta_sigma", 0.0)
    sigma_ref = params.get("sigma_ref", 30.0)

    R = xp.asarray(R)
    lambda_gw = xp.asarray(lambda_gw)
    sigma_v_arr = ensure_array(sigma_v, xp=xp, like=R)

    # radial coherence window
    C_R = burr_xii_window(R, ell0, p, n_coh, xp=xp)

    # resonance in wavelength space (log-normal around lambda_m0)
    log_ratio = xp.log((lambda_gw + 1e-30) / (lambda_m0 + 1e-30))
    C_res = xp.exp(-0.5 * (log_ratio / log_width) ** 2)

    # optional Q-factor: colder systems (small sigma_v) get sharper / stronger resonance
    if beta_sigma != 0.0:
        Q_factor = (sigma_ref / (sigma_v_arr + 1e-30)) ** beta_sigma
        C_res = C_res * Q_factor

    K = A * C_R * C_res
    return 1.0 + K


# ---------------------------------------------------------------------
# 3. ENTANGLEMENT-MEDIATED GRAVITY model
# ---------------------------------------------------------------------


def entanglement_multiplier(R, g_bar, lambda_gw, sigma_v, params, xp=np):
    """
    Entanglement-mediated gravity:
    ------------------------------
    Gravity has an extra contribution from matter-geometry entanglement.
    The strength scales with an "entanglement factor" that drops as
    velocity dispersion (effective temperature) rises.

    A toy closure:
        F_ent(sigma_v) = exp( - (sigma_v / sigma_0)^2 )

    and then
        K(R) = A * F_ent * C_R(R; ell0, p, n_coh)

    This encodes the idea that cold, low-sigma_v systems maintain quantum
    correlations with geometry; hot systems do not.

    Parameters (dict):
        A           : amplitude
        ell0        : coherence length [kpc]
        p, n_coh    : Burr-XII exponents
        sigma0      : characteristic dispersion scale [km/s]
    """
    A = params.get("A", 1.0)
    ell0 = params.get("ell0", 5.0)
    p = params.get("p", 1.0)
    n_coh = params.get("n_coh", 1.0)
    sigma0 = params.get("sigma0", 30.0)

    R = xp.asarray(R)
    sigma_v_arr = ensure_array(sigma_v, xp=xp, like=R)

    C_R = burr_xii_window(R, ell0, p, n_coh, xp=xp)
    F_ent = xp.exp(-(sigma_v_arr / (sigma0 + 1e-30)) ** 2)

    K = A * F_ent * C_R
    return 1.0 + K


# ---------------------------------------------------------------------
# 4. STOCHASTIC VACUUM CONDENSATION model
# ---------------------------------------------------------------------


def vacuum_condensation_multiplier(R, g_bar, lambda_gw, sigma_v, params, xp=np):
    """
    Stochastic vacuum condensation:
    -------------------------------
    Vacuum fluctuations condense into a coherent state below a critical
    "temperature" set by sigma_v. This is a phase-transition-like order
    parameter:

        f_cond(sigma_v) = max[ 0, 1 - (sigma_v / sigma_c)^alpha ]^beta

    and the condensate sources extra gravity with a radial coherence
    window C_R.

    Parameters (dict):
        A           : amplitude
        ell0        : coherence length [kpc]
        p, n_coh    : Burr-XII exponents
        sigma_c     : critical dispersion [km/s]
        alpha       : how sharply the condensate turns on
        beta        : extra steepness for the order parameter
    """
    A = params.get("A", 1.0)
    ell0 = params.get("ell0", 5.0)
    p = params.get("p", 1.0)
    n_coh = params.get("n_coh", 1.0)
    sigma_c = params.get("sigma_c", 40.0)
    alpha = params.get("alpha", 2.0)
    beta = params.get("beta", 1.0)

    R = xp.asarray(R)
    sigma_v_arr = ensure_array(sigma_v, xp=xp, like=R)

    C_R = burr_xii_window(R, ell0, p, n_coh, xp=xp)

    x = sigma_v_arr / (sigma_c + 1e-30)
    raw = 1.0 - x**alpha
    f_cond = xp.clip(raw, 0.0, 1.0) ** beta

    K = A * f_cond * C_R
    return 1.0 + K


# ---------------------------------------------------------------------
# 5. NON-LOCAL GRAVITON CORRELATIONS (PAIRING) model
# ---------------------------------------------------------------------


def graviton_pairing_multiplier(R, g_bar, lambda_gw, sigma_v, params, xp=np):
    """
    Non-local graviton correlations / pairing:
    -----------------------------------------
    In cold, coherent systems, gravitons form bound pairs or collective
    modes, enhancing the effective coupling over a coherence length xi.

    A simple closure:
        n_bound(sigma_v) = exp( - sigma_v / sigma_0 )
        C_pair(R)        = Burr-XII with xi as coherence length

    giving
        K(R) = A * n_bound * C_pair(R)

    Parameters (dict):
        A           : amplitude
        xi0         : base coherence length [kpc]
        p, n_coh    : Burr-XII exponents
        sigma0      : dispersion scale for pairing (km/s)
        gamma_xi    : optional scaling of xi with sigma_v
    """
    A = params.get("A", 1.0)
    xi0 = params.get("xi0", 5.0)
    p = params.get("p", 1.0)
    n_coh = params.get("n_coh", 1.0)
    sigma0 = params.get("sigma0", 30.0)
    gamma_xi = params.get("gamma_xi", 0.0)  # >0 -> hotter -> shorter xi

    R = xp.asarray(R)
    sigma_v_arr = ensure_array(sigma_v, xp=xp, like=R)

    # density of bound graviton pairs (falls with sigma_v)
    n_bound = xp.exp(-(sigma_v_arr / (sigma0 + 1e-30)))

    # coherence length for the pair correlations
    xi = xi0 * (sigma0 / (sigma_v_arr + 1e-30)) ** gamma_xi

    C_pair = 1.0 - (1.0 + (R / (xi + 1e-30)) ** p) ** (-n_coh)

    K = A * n_bound * C_pair
    return 1.0 + K


# ---------------------------------------------------------------------
# Convenience: registry and example usage
# ---------------------------------------------------------------------


MODEL_REGISTRY = {
    "path_interference": path_interference_multiplier,
    "metric_resonance": metric_resonance_multiplier,
    "entanglement": entanglement_multiplier,
    "vacuum_condensation": vacuum_condensation_multiplier,
    "graviton_pairing": graviton_pairing_multiplier,
}


def apply_coherence_model(model_name, R, g_bar, lambda_gw, sigma_v, params, xp=np):
    """
    Generic wrapper: pick a model by name and apply it.

    Example:
        f = apply_coherence_model(
                "graviton_pairing",
                R, g_bar, lambda_gw, sigma_v,
                params={"A": 0.6, "xi0": 5.0, "p": 0.75, "n_coh": 0.5}
            )
        g_eff = g_bar * f
    """
    model = MODEL_REGISTRY[model_name]
    return model(R, g_bar, lambda_gw, sigma_v, params, xp=xp)

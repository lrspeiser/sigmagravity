
import numpy as np
from .background import c

def endpoint_grz(psi_emit, psi_obs):
    """Endpoint gravitational redshift (weak field): z_end ≈ [Ψ_obs - Ψ_emit]/c^2"""
    return (psi_obs - psi_emit) / (c**2)

def los_isw_redshift(a_path, A_of_a, phi_bar_of_a, two_potentials_equal=True):
    """
    Integrated (ISW-like) redshift for a parametrized path in scale factor a.
    Δν/ν ≈ -(2/c^2) [ (1+K) Φ ]_emit^obs   if Φ≈Ψ.
    """
    a = np.asarray(a_path, dtype=float)
    K = A_of_a(a)
    phi = np.asarray([phi_bar_of_a(ai) for ai in a])
    combo = (1.0 + K) * phi
    factor = -2.0/(c**2) if two_potentials_equal else -1.0/(c**2)
    return factor * (combo[-1] - combo[0])

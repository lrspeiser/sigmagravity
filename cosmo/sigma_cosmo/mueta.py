
import numpy as np
from .kernel import Rofk, K_of_R_a

def mu_of_k_a(k, a, ell0, p, ncoh, A0=1.0, a_t=0.5, s=2.0, A_form="logistic", two_pi=True):
    """
    Modified Poisson factor μ(k,a) so that k^2 Ψ = -4πG a^2 ρ_b δ_b * μ.
    Here Σ acts as an effective boost: μ = 1 + K(R,a).  (Slip η≈1 by default.)
    """
    R = Rofk(k, two_pi=two_pi)
    K = K_of_R_a(R, a, ell0, p, ncoh, A0=A0, a_t=a_t, s=s, A_form=A_form)
    return 1.0 + K

def eta_of_k_a(k, a):
    """Gravitational slip η = Φ/Ψ.  Σ is curl-free and conservative: set η≈1."""
    return np.ones_like(np.asarray(a, dtype=float))

def Sigma_lensing_of_k_a(k, a, **kwargs):
    """Lensing response Σ_lens ∝ (Φ+Ψ). If η=1 then Σ_lens = μ."""
    return mu_of_k_a(k, a, **kwargs)

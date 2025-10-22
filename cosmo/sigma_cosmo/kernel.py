
import numpy as np

Mpc = 3.0856775814913673e22  # m

def coherence_kernel(R, ell0, p, ncoh):
    """
    Σ coherence window: C(R) = 1 - [1 + (R/ell0)^p]^(-ncoh).  0<=C<1, monotone.
    Parameters in SI (R, ell0 in meters).
    """
    x = (R/ell0)**p
    return 1.0 - (1.0 + x)**(-ncoh)

def AofA(a, A0=1.0, a_t=0.5, s=2.0, form="logistic"):
    """
    Time dependence for the amplitude A(a). Several options:
    - 'constant': A(a)=A0
    - 'logistic': A(a) = A0 / (1 + (a_t/a)^s)  (turns on around a_t)
    - 'growth':   A(a) = A0 * (a / a_t)^s / (1 + (a / a_t)^s)
    """
    a = np.asarray(a, dtype=float)
    if form=="constant":
        return A0*np.ones_like(a)
    if form=="logistic":
        return A0 / (1.0 + (a_t/np.maximum(a,1e-12))**s)
    if form=="growth":
        x = (a/np.maximum(a_t,1e-12))**s
        return A0 * x/(1.0+x)
    return A0*np.ones_like(a)

def Rofk(k, two_pi=True):
    """
    Map wave number k [1/m] to a real-space scale R [m].
    If two_pi=True, use R = 2π/k; else R=1/k.
    """
    k = np.asarray(k, dtype=float)
    return (2.0*np.pi/k) if two_pi else (1.0/np.maximum(k,1e-30))

def K_of_R_a(R, a, ell0, p, ncoh, A0=1.0, a_t=0.5, s=2.0, A_form="logistic"):
    """
    K(R,a) = A(a) * C(R).  (Optionally add geometry gates externally.)
    """
    C = coherence_kernel(R, ell0, p, ncoh)
    # reuse AofA by importing at call site or recomputing here inline
    from .kernel import AofA as _AofA
    A = _AofA(a, A0=A0, a_t=a_t, s=s, form=A_form)
    return A * C

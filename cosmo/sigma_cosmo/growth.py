
import numpy as np
from .background import Cosmology
from .mueta import mu_of_k_a

def _rk4(f, x0, y0, x1, n):
    """Simple RK4 integrator for vector y' = f(x,y)."""
    x = np.linspace(x0, x1, n+1)
    h = (x1 - x0)/n
    y = np.zeros((n+1, len(np.atleast_1d(y0))), dtype=float)
    y[0] = y0
    for i in range(n):
        xi = x[i]; yi = y[i]
        k1 = f(xi, yi)
        k2 = f(xi + 0.5*h, yi + 0.5*h*k1)
        k3 = f(xi + 0.5*h, yi + 0.5*h*k2)
        k4 = f(xi + h,     yi + h*k3)
        y[i+1] = yi + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x, y

def growth_D_of_a(cosmo: Cosmology, k=1e-3, a_ini=1e-3, a_fin=1.0, n_steps=4000,
                  ell0=200.0e3*3.085677581e16, p=0.75, ncoh=0.5,
                  A0=4.6, a_t=1e-2, s=2.0, A_form="logistic", two_pi=True):
    """
    Solve linear growth with modified Poisson (μ). Expressed in ln a.
    D'' + [2 + d ln H / d ln a] D' - (3/2) Ω_matter(a) μ(k,a) D = 0
    Uses Omega_matter(a) if Omega_eff0>0, else Omega_b(a).

    Returns
    -------
    a_grid : array
    D : array normalized to D(a_ini)=a_ini
    f : d ln D / d ln a
    """
    lnai = np.log(a_ini); lnaf = np.log(a_fin)
    def coeffs(a):
        # Use Omega_matter if Omega_eff0 present, else Omega_b
        Om = cosmo.Omega_matter(a) if hasattr(cosmo, 'Omega_eff0') and cosmo.Omega_eff0 > 0 else cosmo.Omega_b(a)
        dlnH = cosmo.dlnH_dlnA(a)
        mu = mu_of_k_a(k, a, ell0, p, ncoh, A0=A0, a_t=a_t, s=s, A_form=A_form, two_pi=two_pi)
        return dlnH, Om, mu

    def rhs(ln_a, y):
        # y = [D, dD/d ln a]
        a = np.exp(ln_a)
        dlnH, Om_b, mu = coeffs(a)
        D, dD = y
        d2D = -(2.0 + dlnH)*dD + 1.5*Om_b*mu*D
        return np.array([dD, d2D])

    # ICs: in matter era, D ~ a -> at a_ini set D=a_ini, dD/d ln a = D
    y0 = np.array([a_ini, a_ini])
    lna, y = _rk4(rhs, lnai, y0, lnaf, n_steps)
    a_grid = np.exp(lna)
    D = y[:,0]
    f = y[:,1]/np.maximum(D,1e-30)
    return a_grid, D, f

def growth_table(cosmo: Cosmology, k_list, **kwargs):
    """Compute D(a) and f(a) on a grid of k. Returns dict {k: (a, D, f)}."""
    out = {}
    for k in np.atleast_1d(k_list):
        a, D, f = growth_D_of_a(cosmo, k=k, **kwargs)
        out[float(k)] = (a, D, f)
    return out

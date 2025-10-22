
import numpy as np

# Physical constants
c = 299792458.0  # m/s
Mpc = 3.0856775814913673e22  # m
HUBBLE_KMS_PER_MPC_TO_SI = 1000.0 / Mpc  # 1 (km/s/Mpc) in s^-1

class Cosmology:
    """
    Minimal FRW background to explore Σ-cosmology.
    By default: flat baryons+Λ(+radiation optional) WITHOUT dark matter.

    Parameters
    ----------
    H0_kms_Mpc : float
        Hubble constant today in km/s/Mpc (e.g. 70.0).
    Omega_b0 : float
        Present-day baryon density fraction.
    Omega_r0 : float
        Present-day radiation fraction (photons+neutrinos). Small but nonzero.
    Omega_L0 : float
        Present-day Λ fraction. Flatness implies Omega_k0 = 1 - Omega_b0 - Omega_r0 - (Omega_DM0 if included).
    """
    def __init__(self, H0_kms_Mpc=70.0, Omega_b0=0.048, Omega_r0=8.6e-5, Omega_L0=0.70, Omega_eff0=0.0):
        self.H0_kms_Mpc = H0_kms_Mpc
        self.H0 = H0_kms_Mpc * HUBBLE_KMS_PER_MPC_TO_SI  # s^-1
        self.Omega_b0 = Omega_b0
        self.Omega_r0 = Omega_r0
        self.Omega_L0 = Omega_L0
        self.Omega_eff0 = Omega_eff0  # Σ-driven geometric background (no particles)
        # Flat: Omega_k0 follows from closure
        self.Omega_k0 = max(0.0, 1.0 - (Omega_b0 + Omega_r0 + Omega_eff0 + Omega_L0))

    def E(self, a):
        """
        Dimensionless Hubble rate E(a) = H(a)/H0.
        E(a)^2 = Omega_r0 a^-4 + (Omega_b0 + Omega_eff0) a^-3 + Omega_k0 a^-2 + Omega_L0
        where Omega_eff0 is the Σ-driven effective matter background.
        """
        a = np.asarray(a, dtype=float)
        return np.sqrt(
            self.Omega_r0 * a**(-4) +
            (self.Omega_b0 + self.Omega_eff0) * a**(-3) +
            self.Omega_k0 * a**(-2) +
            self.Omega_L0
        )

    def H(self, a):
        """H(a) in s^-1."""
        return self.H0 * self.E(a)

    def dlnH_dlnA(self, a, eps=1e-6):
        """
        Numerical derivative d ln H / d ln a = (a/H) * dH/da
        """
        a1 = a * (1 - eps); a2 = a * (1 + eps)
        H1 = self.H(a1); H2 = self.H(a2)
        return (np.log(H2) - np.log(H1)) / (np.log(a2) - np.log(a1))

    def Omega_b(self, a):
        """Omega_b(a) = rho_b(a)/rho_crit(a)."""
        return (self.Omega_b0 * a**(-3)) / (self.E(a)**2)
    
    def Omega_matter(self, a):
        """Omega_matter(a) = (rho_b + rho_eff)(a) / rho_crit(a). Total matter including Ω_eff."""
        return ((self.Omega_b0 + self.Omega_eff0) * a**(-3)) / (self.E(a)**2)

    # Distances
    def comoving_distance(self, z, n_steps=4096):
        """
        Chi(z) = c ∫_0^z dz' / H(z').  [meters]
        """
        z = float(z)
        zz = np.linspace(0.0, z, n_steps+1)
        aa = 1.0/(1.0+zz)
        Hz = self.H(aa)
        integrand = 1.0 / Hz
        chi = c * np.trapz(integrand, zz)  # meters
        return chi

    def lookback_time(self, z, n_steps=4096):
        """
        t_L(z) = ∫_0^z dz' / [(1+z') H(z')].  [seconds]
        """
        z = float(z)
        zz = np.linspace(0.0, z, n_steps+1)
        aa = 1.0/(1.0+zz)
        Hz = self.H(aa)
        integrand = 1.0/((1.0+zz)*Hz)
        tL = np.trapz(integrand, zz)
        return tL

    def z_of_chi(self, chi, zmax=10.0, n_steps=10000):
        """
        Invert chi(z) by tabulation (simple but robust).
        """
        zz = np.linspace(0.0, zmax, n_steps+1)
        chis = np.array([self.comoving_distance(z) for z in zz])
        return np.interp(chi, chis, zz)

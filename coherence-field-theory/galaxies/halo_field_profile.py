"""
Solve scalar field equation for galaxy halos.

Given:
- Scalar potential V(φ) (same as cosmology)
- Matter coupling β
- Baryon density profile ρ_b(r)

Compute:
- Scalar field profile φ(r)
- Effective halo density ρ_φ(r)
- Effective halo parameters (ρ_c0, R_c)
"""

import numpy as np
from scipy.integrate import odeint, solve_bvp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz
import warnings

# Physical constants
G = 4.30091e-6  # (km/s)^2 kpc / M_sun
M_PL = 2.435e18  # Reduced Planck mass (kg) ~= 1.22e19 GeV
M_PL_KPC = M_PL * 1.989e30 / (3.086e19)**3  # Convert to M_sun/kpc^3 (approximate)


class HaloFieldSolver:
    """
    Solve static Klein-Gordon equation for scalar field in galaxy background.
    
    Equation:
        (1/r²) d/dr (r² dφ/dr) = dV_eff/dφ
    
    where:
        V_eff(φ) = V(φ) + (β/M_Pl) ρ_b(r)
    """
    
    def __init__(self, V0, lambda_param, beta, M4=None, phi_inf=None):
        """
        Initialize field solver.
        
        Parameters:
        -----------
        V0 : float
            Exponential potential scale (energy density)
        lambda_param : float
            Exponential potential slope
        beta : float
            Matter coupling (dimensionless)
        M4 : float, optional
            Chameleon mass scale (if None, use pure exponential)
        phi_inf : float, optional
            Cosmological field value at infinity (if None, use 0)
        """
        self.V0 = V0
        self.lambda_param = lambda_param
        self.beta = beta
        self.M4 = M4
        self.phi_inf = phi_inf if phi_inf is not None else 0.0
        
        # Potential function
        if M4 is not None:
            self.V_func = self._V_chameleon
            self.dV_dphi = self._dV_chameleon_dphi
            self.d2V_dphi2 = self._d2V_chameleon_dphi2
        else:
            self.V_func = self._V_exponential
            self.dV_dphi = self._dV_exponential_dphi
            self.d2V_dphi2 = self._d2V_exponential_dphi2
    
    def _V_exponential(self, phi):
        """Exponential potential: V(φ) = V₀ e^(-λφ)."""
        return self.V0 * np.exp(-self.lambda_param * phi)
    
    def _dV_exponential_dphi(self, phi):
        """dV/dφ for exponential potential."""
        return -self.lambda_param * self.V0 * np.exp(-self.lambda_param * phi)
    
    def _d2V_exponential_dphi2(self, phi):
        """d²V/dφ² for exponential potential."""
        return self.lambda_param**2 * self.V0 * np.exp(-self.lambda_param * phi)
    
    def _V_chameleon(self, phi):
        """Chameleon potential: V(φ) = V₀ e^(-λφ) + M⁵/φ."""
        if np.any(phi <= 0):
            # Regularize near φ=0
            phi_safe = np.maximum(phi, 1e-6)
            return self.V0 * np.exp(-self.lambda_param * phi) + self.M4**5 / phi_safe
        return self.V0 * np.exp(-self.lambda_param * phi) + self.M4**5 / phi
    
    def _dV_chameleon_dphi(self, phi):
        """dV/dφ for chameleon potential."""
        if np.any(phi <= 0):
            phi_safe = np.maximum(phi, 1e-6)
            return -self.lambda_param * self.V0 * np.exp(-self.lambda_param * phi) - self.M4**5 / phi_safe**2
        return -self.lambda_param * self.V0 * np.exp(-self.lambda_param * phi) - self.M4**5 / phi**2
    
    def _d2V_chameleon_dphi2(self, phi):
        """d²V/dφ² for chameleon potential."""
        if np.any(phi <= 0):
            phi_safe = np.maximum(phi, 1e-6)
            return (self.lambda_param**2 * self.V0 * np.exp(-self.lambda_param * phi) + 
                    2 * self.M4**5 / phi_safe**3)
        return (self.lambda_param**2 * self.V0 * np.exp(-self.lambda_param * phi) + 
                2 * self.M4**5 / phi**3)
    
    def Veff(self, phi, rho_b):
        """
        Effective potential: V_eff(φ) = V(φ) + (β/M_Pl) ρ_b.
        
        Parameters:
        -----------
        phi : float or array
            Scalar field value
        rho_b : float or array
            Baryon density (M_sun/kpc^3)
            
        Returns:
        --------
        Veff : float or array
            Effective potential (in cosmology units: H0²)
        """
        # Coupling term: (β/M_Pl) * ρ_b
        # In cosmology units, we need to convert ρ_b to cosmology units first
        # ρ_b is in M_sun/kpc^3, need to convert to cosmology units (relative to ρ_crit)
        # Critical density: ρ_crit = 3H0²/(8πG) ≈ 1.45e3 M_sun/kpc^3
        H0_kms_kpc = 70.0 / 306.6  # km/s/kpc
        H0_squared = H0_kms_kpc**2
        rho_crit = 3 * H0_squared / (8 * np.pi * G)  # M_sun/kpc^3
        
        # Convert ρ_b to cosmology units (fraction of critical density)
        rho_b_cosm = rho_b / rho_crit
        
        # Coupling: β * (ρ_b / ρ_crit) in cosmology units
        # This gives coupling term in same units as V(φ) (H0²)
        coupling_term = self.beta * rho_b_cosm
        
        return self.V_func(phi) + coupling_term
    
    def dVeff_dphi(self, phi, rho_b):
        """
        dV_eff/dφ = dV/dφ + (β/M_Pl) dρ_b/dφ.
        
        Since ρ_b doesn't depend on φ, this is just dV/dφ.
        """
        return self.dV_dphi(phi)
    
    def _KG_rhs(self, y, r, rho_b_interp):
        """
        Right-hand side of Klein-Gordon equation.
        
        System:
            dφ/dr = y[1]
            d²φ/dr² = dV_eff/dφ - (2/r) dφ/dr
        
        Parameters:
        -----------
        y : array [φ, dφ/dr]
            Field and its gradient
        r : float
            Radius
        rho_b_interp : callable
            Interpolated baryon density ρ_b(r)
        """
        phi, dphi_dr = y[0], y[1]
        
        # Regularize at r=0
        if r < 1e-6:
            r = 1e-6
        
        # Get baryon density
        if callable(rho_b_interp):
            rho_b = rho_b_interp(r)
        else:
            rho_b = rho_b_interp
        
        # Ensure scalar values
        if np.isscalar(rho_b):
            rho_b = float(rho_b)
        else:
            rho_b = float(rho_b[0] if len(rho_b) > 0 else 0.0)
        
        d2phi_dr2 = self.dVeff_dphi(phi, rho_b) - (2.0 / r) * dphi_dr
        
        return np.array([dphi_dr, d2phi_dr2])
    
    def solve(self, rho_baryon, r_grid, method='bvp', phi_init=None):
        """
        Solve φ(r) for given baryon profile.
        
        Parameters:
        -----------
        rho_baryon : callable or array
            Baryon density profile ρ_b(r) (M_sun/kpc^3)
            Can be function or array (interpolated)
        r_grid : array
            Radial grid (kpc)
        method : str
            'bvp' (boundary value problem) or 'shooting'
        phi_init : array, optional
            Initial guess for φ(r) (for BVP method)
            
        Returns:
        --------
        phi : array
            Scalar field profile φ(r)
        dphi_dr : array
            Gradient dφ/dr
        """
        if not callable(rho_baryon):
            # Interpolate if array provided
            rho_interp = interp1d(r_grid, rho_baryon, kind='linear', 
                                 bounds_error=False, fill_value=(rho_baryon[0], 0.0))
        else:
            rho_interp = rho_baryon
        
        if method == 'bvp':
            return self._solve_bvp(rho_interp, r_grid, phi_init)
        else:
            return self._solve_shooting(rho_interp, r_grid)
    
    def _solve_bvp(self, rho_interp, r_grid, phi_init=None):
        """Solve using boundary value problem solver."""
        def bc(ya, yb):
            """Boundary conditions: dφ/dr = 0 at r=0, φ = φ_inf at r_max."""
            return [ya[1],  # dφ/dr = 0 at r=0
                   yb[0] - self.phi_inf]  # φ = φ_inf at r_max
        
        def ode_system(r, y):
            # BVP solver passes arrays, need to handle scalar r
            if isinstance(r, np.ndarray):
                r = float(r[0] if len(r) > 0 else r_grid[0])
            return self._KG_rhs(y, r, rho_interp)
        
        # Initial guess
        if phi_init is None:
            # Linear interpolation between boundaries
            y_init = np.zeros((2, len(r_grid)))
            y_init[0, :] = np.linspace(0.0, self.phi_inf, len(r_grid))  # φ
            y_init[1, :] = 0.0  # dφ/dr
        else:
            y_init = phi_init
        
        # Solve BVP
        try:
            sol = solve_bvp(ode_system, bc, r_grid, y_init, tol=1e-6, max_nodes=1000)
            if sol.success:
                phi = sol.y[0]
                dphi_dr = sol.y[1]
            else:
                warnings.warn(f"BVP solver did not converge: {sol.message}")
                # Fall back to shooting
                return self._solve_shooting(rho_interp, r_grid)
        except Exception as e:
            warnings.warn(f"BVP solver failed: {e}, falling back to shooting")
            return self._solve_shooting(rho_interp, r_grid)
        
        return phi, dphi_dr
    
    def _solve_shooting(self, rho_interp, r_grid):
        """Solve using shooting method (integrate from outside inward)."""
        r_min, r_max = r_grid[0], r_grid[-1]
        
        # Start from outside: φ(r_max) = φ_inf, dφ/dr(r_max) = 0
        # But integrate inward is tricky, so integrate outward and adjust
        
        def objective(phi0):
            """Objective: minimize |dφ/dr| at r=0."""
            # Ensure scalar
            phi0 = float(phi0[0] if isinstance(phi0, (list, np.ndarray)) else phi0)
            # Integrate outward from r=0
            y0 = np.array([phi0, 0.0], dtype=float)  # φ(0) = phi0, dφ/dr(0) = 0
            
            # Wrapper for ODE solver to handle scalar r
            def rhs_wrapper(y, r):
                return self._KG_rhs(y, float(r), rho_interp)
            
            sol = odeint(rhs_wrapper, y0, r_grid, atol=1e-8, rtol=1e-8)
            phi_final = sol[-1, 0]
            dphi_dr_final = sol[-1, 1]
            
            # Want φ(r_max) ≈ φ_inf, dφ/dr(r_max) ≈ 0
            return (phi_final - self.phi_inf)**2 + dphi_dr_final**2
        
        # Find φ(0) that satisfies boundary conditions
        result = minimize(objective, self.phi_inf, method='L-BFGS-B', 
                         bounds=[(self.phi_inf - 10, self.phi_inf + 10)])
        phi0 = result.x[0]
        
        # Integrate final solution
        y0 = np.array([phi0, 0.0], dtype=float)
        
        # Wrapper for ODE solver
        def rhs_wrapper(y, r):
            return self._KG_rhs(y, float(r), rho_interp)
        
        sol = odeint(rhs_wrapper, y0, r_grid, atol=1e-8, rtol=1e-8)
        
        phi = sol[:, 0]
        dphi_dr = sol[:, 1]
        
        return phi, dphi_dr
    
    def effective_density(self, phi, dphi_dr, convert_to_mass_density=True):
        """
        Compute effective density: ρ_φ(r) = ½(∇φ)² + V(φ).
        
        Parameters:
        -----------
        phi : array
            Scalar field profile
        dphi_dr : array
            Gradient dφ/dr
        convert_to_mass_density : bool
            If True, convert V(φ) from energy density to mass density
            
        Returns:
        --------
        rho_phi : array
            Effective density (M_sun/kpc^3)
        """
        # Kinetic term: ½(∇φ)² = ½(dφ/dr)²
        # Units: (dφ/dr)² has units of energy density if φ is dimensionless
        kinetic = 0.5 * dphi_dr**2
        
        # Potential term: V(φ)
        # Units: V(φ) typically in energy density units
        potential = self.V_func(phi)
        
        # Convert energy density to mass density if needed
        # In cosmology, V(φ) is in units where H0² = 8πG/3 * ρ_crit
        # So V has dimensions of energy density = ρ_crit (in cosmology units)
        # Need to convert to physical mass density (M_sun/kpc³)
        if convert_to_mass_density:
            # Critical density: ρ_crit = 3H0²/(8πG)
            # H0 ≈ 70 km/s/Mpc ≈ 0.228 km/s/kpc
            H0_kms_kpc = 70.0 / 306.6  # km/s/kpc
            H0_squared = H0_kms_kpc**2  # (km/s)²/kpc²
            # G = 4.30091e-6 (km/s)² kpc / M_sun
            rho_crit = 3 * H0_squared / (8 * np.pi * G)  # M_sun/kpc³
            # V(φ) in cosmology units is relative to ρ_crit
            # So multiply by ρ_crit to get physical density
            potential = potential * rho_crit  # Convert to mass density (M_sun/kpc³)
            kinetic = kinetic * rho_crit  # Convert kinetic term too
        else:
            # Assume V(φ) already in M_sun/kpc^3
            pass
        
        # Total density
        rho_phi = kinetic + potential
        
        return rho_phi
    
    def fit_halo_parameters(self, rho_phi, r_grid, r_fit_min=None, r_fit_max=None):
        """
        Fit pseudo-isothermal profile to ρ_φ(r).
        
        Parameters:
        -----------
        rho_phi : array
            Effective density from field solution
        r_grid : array
            Radial grid
        r_fit_min : float, optional
            Minimum radius for fitting (default: 0.5 * median(r))
        r_fit_max : float, optional
            Maximum radius for fitting (default: 5 * median(r))
            
        Returns:
        --------
        rho_c0 : float
            Central density parameter
        R_c : float
            Core radius parameter
        chi2 : float
            Chi-squared of fit
        """
        if r_fit_min is None:
            r_fit_min = 0.5 * np.median(r_grid)
        if r_fit_max is None:
            r_fit_max = 5.0 * np.median(r_grid)
        
        # Select fitting region
        mask = (r_grid >= r_fit_min) & (r_grid <= r_fit_max)
        r_fit = r_grid[mask]
        rho_fit = rho_phi[mask]
        
        if len(r_fit) < 3:
            # Not enough points
            return np.nan, np.nan, np.inf
        
        def chi_squared(params):
            """Chi-squared for pseudo-isothermal fit."""
            rho_c0, R_c = params
            if rho_c0 <= 0 or R_c <= 0:
                return 1e10
            
            # Pseudo-isothermal: ρ(r) = ρ_c0 / [1 + (r/R_c)²]
            rho_model = rho_c0 / (1 + (r_fit / R_c)**2)
            
            # Use relative error
            chi2 = np.sum(((rho_fit - rho_model) / (rho_fit + 1e-6))**2)
            return chi2
        
        # Initial guess
        rho_c0_guess = rho_fit[0]  # Central density
        R_c_guess = np.median(r_fit)  # Typical scale
        
        bounds = [(1e-3, 1e12),  # rho_c0 (M_sun/kpc^3) - wider range (allow very low)
                 (0.01, 50.0)]   # R_c (kpc) - allow very small
        
        result = minimize(chi_squared, [rho_c0_guess, R_c_guess], 
                         method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            rho_c0, R_c = result.x
            chi2 = result.fun
        else:
            rho_c0, R_c = np.nan, np.nan
            chi2 = np.inf
        
        return rho_c0, R_c, chi2


def test_halo_solver():
    """Test the halo solver with a simple exponential disk."""
    print("=" * 80)
    print("TESTING HALO FIELD SOLVER")
    print("=" * 80)
    
    # Test parameters
    V0 = 1e-6  # Energy density scale
    lambda_param = 1.0
    beta = 0.1  # Weak coupling
    M_disk = 1e10  # M_sun
    R_disk = 3.0  # kpc
    
    # Create solver
    solver = HaloFieldSolver(V0, lambda_param, beta, M4=None)
    
    # Exponential disk density: ρ(r) = (M_disk / (2π R_disk²)) exp(-r/R_disk)
    def rho_disk(r):
        """Exponential disk density."""
        if np.isscalar(r):
            r = np.array([r])
        rho = (M_disk / (2 * np.pi * R_disk**2)) * np.exp(-r / R_disk)
        return rho if len(rho) > 1 else rho[0]
    
    # Radial grid
    r_grid = np.logspace(-1, 2, 200)  # 0.1 to 100 kpc
    
    print(f"\nSolving for exponential disk:")
    print(f"  M_disk = {M_disk:.2e} M_sun")
    print(f"  R_disk = {R_disk:.2f} kpc")
    print(f"  V0 = {V0:.2e}")
    print(f"  lambda = {lambda_param:.2f}")
    print(f"  beta = {beta:.2f}")
    
    # Solve
    try:
        phi, dphi_dr = solver.solve(rho_disk, r_grid, method='shooting')
        
        # Compute effective density
        rho_phi = solver.effective_density(phi, dphi_dr)
        
        # Fit halo parameters
        rho_c0, R_c, chi2 = solver.fit_halo_parameters(rho_phi, r_grid)
        
        print(f"\nSolution found!")
        print(f"  phi(0) = {phi[0]:.6f}")
        print(f"  phi(inf) = {phi[-1]:.6f}")
        print(f"  max(|dphi/dr|) = {np.max(np.abs(dphi_dr)):.6e}")
        
        print(f"\nFitted halo parameters:")
        print(f"  rho_c0 = {rho_c0:.2e} M_sun/kpc^3")
        print(f"  R_c  = {R_c:.2f} kpc")
        print(f"  chi^2   = {chi2:.2f}")
        
        # Check physicality
        if rho_c0 > 0 and R_c > 0:
            print(f"\n[OK] Solution is physical!")
        else:
            print(f"\n[WARNING] Solution may be unphysical (fit failed)")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_halo_solver()


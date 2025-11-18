"""
r_of_z_weyl.py
--------------
Utility for distance inversion in Weyl-integrable redshift model.

Provides:
- Newton solver for r(z) (distance from redshift)
- Distance modulus computation
- Alcock-Paczyński ratio computation
- Interface for Pantheon+ fitting
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sigma_redshift_derivations import SigmaKernel, WeylModel

class WeylDistanceInverter:
    """
    Utility class for computing distances from redshifts in Weyl-integrable model.
    """
    
    def __init__(self, model, z_tol=1e-6, max_iter=50):
        self.model = model
        self.z_tol = z_tol
        self.max_iter = max_iter
        
        # Cache for efficiency
        self._cache = {}
    
    def find_distance_from_redshift(self, target_z):
        """
        Newton solve for r(z) - find distance that gives target redshift.
        
        Parameters:
        -----------
        target_z : float
            Target redshift
            
        Returns:
        --------
        distance_Mpc : float
            Distance in Mpc that produces the target redshift
        """
        if target_z in self._cache:
            return self._cache[target_z]
        
        # Initial guess based on Hubble law
        H0_SI = (self.model.H0_kms_Mpc * 1000.0) / (3.0856775814913673e22)  # Mpc
        c = 299792458.0  # m/s
        D_guess = (target_z * c) / H0_SI
        
        D = D_guess
        for i in range(self.max_iter):
            z_current = self.model.z_of_distance_Mpc(D)
            error = z_current - target_z
            
            if abs(error) < self.z_tol:
                self._cache[target_z] = D
                return D
            
            # Numerical derivative for Newton step
            dz = max(1e-3 * D, 0.1)  # Ensure minimum step size
            z_plus = self.model.z_of_distance_Mpc(D + dz)
            z_minus = self.model.z_of_distance_Mpc(D - dz)
            dz_dD = (z_plus - z_minus) / (2 * dz)
            
            if abs(dz_dD) < 1e-10:
                break
                
            D = D - error / dz_dD
            D = max(D, 0.1)  # Prevent negative distances
        
        self._cache[target_z] = D
        return D
    
    def compute_distance_modulus(self, z_array):
        """
        Compute distance modulus μ = 5*log10(d_L) - 5 for luminosity distance d_L.
        
        In static universe: d_L = r(z) * (1+z)^2
        
        Parameters:
        -----------
        z_array : array-like
            Array of redshifts
            
        Returns:
        --------
        mu_array : np.ndarray
            Array of distance moduli
        """
        z_array = np.asarray(z_array)
        distances = np.array([self.find_distance_from_redshift(z) for z in z_array])
        d_L = distances * (1 + z_array)**2  # Luminosity distance
        return 5 * np.log10(d_L) - 5
    
    def compute_angular_diameter_distance(self, z_array):
        """
        Compute angular diameter distance d_A = r(z) / (1+z).
        
        Parameters:
        -----------
        z_array : array-like
            Array of redshifts
            
        Returns:
        --------
        d_A_array : np.ndarray
            Array of angular diameter distances
        """
        z_array = np.asarray(z_array)
        distances = np.array([self.find_distance_from_redshift(z) for z in z_array])
        return distances / (1 + z_array)
    
    def compute_ap_ratio(self, z_array):
        """
        Compute Alcock-Paczyński ratio F_AP = r(z) / (dr/dz).
        
        In static universe, this should be constant (isotropic).
        In expanding universe, this varies with z (anisotropic).
        
        Parameters:
        -----------
        z_array : array-like
            Array of redshifts
            
        Returns:
        --------
        f_ap_array : np.ndarray
            Array of AP ratios
        """
        z_array = np.asarray(z_array)
        distances = np.array([self.find_distance_from_redshift(z) for z in z_array])
        
        # Numerical derivative dr/dz
        dz = 1e-3
        distances_plus = np.array([self.find_distance_from_redshift(z + dz) for z in z_array])
        distances_minus = np.array([self.find_distance_from_redshift(z - dz) for z in z_array])
        dr_dz = (distances_plus - distances_minus) / (2 * dz)
        
        return distances / dr_dz
    
    def compute_redshift_drift(self, z_array, phi_dot=1e-18):
        """
        Compute redshift drift ż = ∂_t Φ along line of sight.
        
        This is a distinctive prediction of Weyl-integrable models.
        
        Parameters:
        -----------
        z_array : array-like
            Array of redshifts
        phi_dot : float
            Time derivative of Weyl potential (s^-1)
            
        Returns:
        --------
        z_dot_array : np.ndarray
            Array of redshift drift rates (s^-1)
        """
        z_array = np.asarray(z_array)
        distances = np.array([self.find_distance_from_redshift(z) for z in z_array])
        
        # Simple model: ż ∝ distance (for constant Φ̇)
        return phi_dot * distances * (3.0856775814913673e22) / (299792458.0)  # Convert Mpc to s

def create_weyl_inverter(kernel_params, model_params):
    """
    Convenience function to create a WeylDistanceInverter.
    
    Parameters:
    -----------
    kernel_params : dict
        Parameters for SigmaKernel (A, ell0_kpc, p, ncoh)
    model_params : dict
        Parameters for WeylModel (H0_kms_Mpc, alpha0_scale)
        
    Returns:
    --------
    inverter : WeylDistanceInverter
        Configured distance inverter
    """
    kernel = SigmaKernel(**kernel_params)
    model = WeylModel(kernel=kernel, **model_params)
    return WeylDistanceInverter(model)

# Example usage
if __name__ == "__main__":
    # Create inverter with default parameters
    kernel_params = {
        'A': 1.0,
        'ell0_kpc': 200.0,
        'p': 0.75,
        'ncoh': 0.5
    }
    
    model_params = {
        'H0_kms_Mpc': 70.0,
        'alpha0_scale': 0.95  # Optimized from previous analysis
    }
    
    inverter = create_weyl_inverter(kernel_params, model_params)
    
    # Test with some redshifts
    z_test = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    
    print("Weyl-Integrable Distance Inversion Test")
    print("="*50)
    
    for z in z_test:
        distance = inverter.find_distance_from_redshift(z)
        print(f"z = {z:.1f} → D = {distance:.2f} Mpc")
    
    print("\nDistance modulus:")
    mu = inverter.compute_distance_modulus(z_test)
    for i, z in enumerate(z_test):
        print(f"z = {z:.1f} → μ = {mu[i]:.2f}")
    
    print("\nAP ratio (should be constant for static universe):")
    f_ap = inverter.compute_ap_ratio(z_test)
    for i, z in enumerate(z_test):
        print(f"z = {z:.1f} → F_AP = {f_ap[i]:.3f}")
    
    print(f"\nMean AP ratio: {np.mean(f_ap):.3f} ± {np.std(f_ap):.3f}")
    print("(Should be constant for static universe)")








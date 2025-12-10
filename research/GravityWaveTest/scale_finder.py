"""
Comprehensive scale-finding tests for Σ-Gravity coherence length.
Tests dimensional combinations against SPARC and Gaia data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.stats import spearmanr, pearsonr
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict
import json
import os

# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458.0  # m/s
kpc_to_m = 3.086e19  # meters per kpc
Msun_to_kg = 1.989e30  # kg per solar mass

@dataclass
class ScaleHypothesis:
    """Container for a scale hypothesis."""
    name: str
    formula: Callable  # function(galaxy_params) -> ℓ₀ in kpc
    description: str
    expected_range: Tuple[float, float]  # (min, max) in kpc
    

@dataclass
class TestResult:
    """Container for test results."""
    hypothesis_name: str
    chi2: float
    dof: int
    bic: float
    aic: float
    ell0_median: float
    ell0_std: float
    correlation_with_fit: float
    scatter_dex: float
    params: Dict

class SPARCDataLoader:
    """Load and preprocess SPARC data for scale tests."""
    
    def __init__(self, sparc_dir: str = "data/sparc"):
        self.sparc_dir = sparc_dir
        self.galaxies = None
        
    def load(self) -> pd.DataFrame:
        """Load SPARC data with all necessary columns."""
        # Assuming you have a combined SPARC file
        # Adjust path to match your data structure
        df = pd.read_csv(f"{self.sparc_dir}/sparc_combined.csv")
        
        # Required columns:
        # - galaxy_name
        # - M_baryon (total baryonic mass, M_sun)
        # - M_stellar (stellar mass, M_sun)
        # - M_gas (gas mass, M_sun)
        # - v_flat (flat rotation velocity, km/s)
        # - R_disk (disk scale length, kpc)
        # - L_3.6 (3.6μm luminosity, L_sun)
        # - sigma_velocity (velocity dispersion, km/s)
        # - bulge_frac (bulge fraction)
        # - morphology_code
        
        self.galaxies = df
        return df

class ScaleTestSuite:
    """Test suite for finding the physical coherence scale."""
    
    def __init__(self, sparc_data: pd.DataFrame, 
                 fitted_ell0: float = 4.993,  # Your fitted value
                 fitted_A: float = 0.591):
        self.data = sparc_data
        self.fitted_ell0 = fitted_ell0
        self.fitted_A = fitted_A
        self.results = []
        
    def test_hypothesis(self, hypothesis: ScaleHypothesis, 
                       verbose: bool = True) -> TestResult:
        """Test a single scale hypothesis against data."""
        
        # Compute predicted ℓ₀ for each galaxy
        predicted_ell0 = []
        valid_indices = []
        
        for idx, row in self.data.iterrows():
            try:
                ell0_pred = hypothesis.formula(row)
                if np.isfinite(ell0_pred) and ell0_pred > 0:
                    predicted_ell0.append(ell0_pred)
                    valid_indices.append(idx)
            except Exception as e:
                if verbose:
                    print(f"Error computing {hypothesis.name} for {row.get('galaxy_name', idx)}: {e}")
                continue
        
        predicted_ell0 = np.array(predicted_ell0)
        valid_data = self.data.loc[valid_indices]
        
        if len(predicted_ell0) == 0:
            print(f"WARNING: No valid predictions for {hypothesis.name}")
            return None
        
        # Compute goodness of fit metrics
        # Here we compare against the fitted value
        # Better: use RAR scatter as the metric
        
        # Method 1: Direct comparison to fitted ℓ₀
        residuals = np.log10(predicted_ell0) - np.log10(self.fitted_ell0)
        scatter_dex = np.std(residuals)
        chi2_simple = np.sum(residuals**2)
        
        # Method 2: Check correlation with what would give best RAR fit
        # This requires re-fitting RAR for each galaxy - more complex
        
        # Method 3: Check Tully-Fisher consistency
        # v⁴ ∝ M_b requires ℓ ∝ M_b^0.5
        correlation = pearsonr(np.log10(predicted_ell0), 
                              np.log10(valid_data['M_baryon']))[0]
        
        # Compute information criteria
        n_params = 0  # Most hypotheses are 0-parameter (pure dimensional analysis)
        n_data = len(predicted_ell0)
        bic = chi2_simple + n_params * np.log(n_data)
        aic = chi2_simple + 2 * n_params
        
        result = TestResult(
            hypothesis_name=hypothesis.name,
            chi2=chi2_simple,
            dof=n_data - n_params,
            bic=bic,
            aic=aic,
            ell0_median=np.median(predicted_ell0),
            ell0_std=np.std(predicted_ell0),
            correlation_with_fit=correlation,
            scatter_dex=scatter_dex,
            params={}
        )
        
        self.results.append(result)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Hypothesis: {hypothesis.name}")
            print(f"Description: {hypothesis.description}")
            print(f"{'='*60}")
            print(f"Median ℓ₀: {result.ell0_median:.3f} kpc (fit: {self.fitted_ell0:.3f} kpc)")
            print(f"Std ℓ₀: {result.ell0_std:.3f} kpc")
            print(f"Scatter: {result.scatter_dex:.4f} dex")
            print(f"χ²: {result.chi2:.2f} (dof: {result.dof})")
            print(f"BIC: {result.bic:.2f}")
            print(f"Correlation with M_b: {correlation:.3f}")
            print(f"Valid galaxies: {n_data}/{len(self.data)}")
        
        return result
    
    def plot_hypothesis_results(self, hypothesis: ScaleHypothesis, 
                               result: TestResult,
                               save_path: str = None):
        """Generate diagnostic plots for a hypothesis."""
        
        # Compute predictions for plotting
        predicted_ell0 = []
        M_baryon = []
        v_flat = []
        R_disk = []
        
        for idx, row in self.data.iterrows():
            try:
                ell0_pred = hypothesis.formula(row)
                if np.isfinite(ell0_pred) and ell0_pred > 0:
                    predicted_ell0.append(ell0_pred)
                    M_baryon.append(row['M_baryon'])
                    v_flat.append(row['v_flat'])
                    R_disk.append(row['R_disk'])
            except:
                continue
        
        predicted_ell0 = np.array(predicted_ell0)
        M_baryon = np.array(M_baryon)
        v_flat = np.array(v_flat)
        R_disk = np.array(R_disk)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{hypothesis.name}\n{hypothesis.description}', fontsize=14)
        
        # Plot 1: Predicted ℓ₀ vs M_baryon
        ax = axes[0, 0]
        ax.scatter(M_baryon, predicted_ell0, alpha=0.6, s=20)
        ax.axhline(self.fitted_ell0, color='r', linestyle='--', 
                   label=f'Fitted: {self.fitted_ell0:.2f} kpc')
        ax.set_xlabel('M_baryon [M_sun]')
        ax.set_ylabel('Predicted ℓ₀ [kpc]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Predicted ℓ₀ vs v_flat
        ax = axes[0, 1]
        ax.scatter(v_flat, predicted_ell0, alpha=0.6, s=20)
        ax.axhline(self.fitted_ell0, color='r', linestyle='--')
        ax.set_xlabel('v_flat [km/s]')
        ax.set_ylabel('Predicted ℓ₀ [kpc]')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Predicted ℓ₀ vs R_disk
        ax = axes[0, 2]
        ax.scatter(R_disk, predicted_ell0, alpha=0.6, s=20)
        ax.axhline(self.fitted_ell0, color='r', linestyle='--')
        ax.set_xlabel('R_disk [kpc]')
        ax.set_ylabel('Predicted ℓ₀ [kpc]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Histogram of predicted values
        ax = axes[1, 0]
        ax.hist(predicted_ell0, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(self.fitted_ell0, color='r', linestyle='--', 
                   label=f'Fitted: {self.fitted_ell0:.2f}')
        ax.axvline(result.ell0_median, color='b', linestyle='--',
                   label=f'Median: {result.ell0_median:.2f}')
        ax.set_xlabel('Predicted ℓ₀ [kpc]')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Residuals vs M_baryon
        ax = axes[1, 1]
        residuals = np.log10(predicted_ell0) - np.log10(self.fitted_ell0)
        ax.scatter(M_baryon, residuals, alpha=0.6, s=20)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel('M_baryon [M_sun]')
        ax.set_ylabel('log₁₀(ℓ_pred / ℓ_fit)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Scatter: {result.scatter_dex:.4f} dex')
        
        # Plot 6: Tully-Fisher check
        ax = axes[1, 2]
        # If ℓ ∝ M^0.5, then v⁴ ∝ M
        # v² = αGM/ℓ, so v⁴ = (αGM)²/ℓ²
        # If ℓ² ∝ M, then v⁴ ∝ M
        ax.scatter(M_baryon, v_flat**4, alpha=0.6, s=20, label='Data')
        ax.set_xlabel('M_baryon [M_sun]')
        ax.set_ylabel('v_flat⁴ [(km/s)⁴]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Overplot expected relation if ℓ ∝ M^0.5
        x_theory = np.logspace(np.log10(M_baryon.min()), np.log10(M_baryon.max()), 100)
        y_theory = x_theory * (v_flat**4 / M_baryon).mean()  # Normalize
        ax.plot(x_theory, y_theory, 'r--', label='v⁴ ∝ M (if ℓ ∝ M^0.5)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

# ============================================================================
# HYPOTHESIS LIBRARY
# ============================================================================

def create_hypothesis_library() -> List[ScaleHypothesis]:
    """Create library of scale hypotheses to test."""
    
    hypotheses = []
    
    # ========================================================================
    # Category 1: Simple density scales (baseline - we know these fail)
    # ========================================================================
    
    def virial_density_scale(row):
        """ℓ = c/√(Gρ_virial)"""
        rho_virial = 200 * 1e2  # kg/m³, virial overdensity
        ell0_m = c / np.sqrt(G * rho_virial)
        return ell0_m / kpc_to_m
    
    hypotheses.append(ScaleHypothesis(
        name="virial_density",
        formula=virial_density_scale,
        description="ℓ = c/√(Gρ_virial), single density scale",
        expected_range=(1000, 2000)
    ))
    
    # ========================================================================
    # Category 2: Tully-Fisher inspired (from document 1)
    # ========================================================================
    
    def tully_fisher_scale(row, alpha=1.0):
        """ℓ = α(GM_b/v²_∞)
        
        From v_∞² = α(GM_b/λ_g) with flat curves.
        This predicts ℓ ∝ M_b/v² ∝ M_b^0.5 if v⁴ ∝ M_b.
        """
        M_b = row['M_baryon'] * Msun_to_kg
        v_flat = row['v_flat'] * 1e3  # km/s to m/s
        
        ell0_m = alpha * G * M_b / (v_flat**2)
        return ell0_m / kpc_to_m
    
    hypotheses.append(ScaleHypothesis(
        name="tully_fisher_direct",
        formula=lambda row: tully_fisher_scale(row, alpha=1.0),
        description="ℓ = GM_b/v²_flat (direct Tully-Fisher)",
        expected_range=(1, 10)
    ))
    
    # ========================================================================
    # Category 3: Multi-scale combinations
    # ========================================================================
    
    def disk_scale_height(row):
        """ℓ ∝ h_disk ∼ σ²/(πGΣ)
        
        Disk scale height from hydrostatic equilibrium.
        This is the natural scale for 2-D disk dynamics.
        """
        R_disk = row['R_disk'] * kpc_to_m
        sigma_v = row.get('sigma_velocity', 30) * 1e3  # km/s to m/s, default 30 km/s
        M_b = row['M_baryon'] * Msun_to_kg
        
        # Surface density Σ ~ M/(πR²)
        Sigma = M_b / (np.pi * R_disk**2)
        
        # Scale height h ~ σ²/(πGΣ)
        h = sigma_v**2 / (np.pi * G * Sigma)
        
        return h / kpc_to_m
    
    hypotheses.append(ScaleHypothesis(
        name="disk_scale_height",
        formula=disk_scale_height,
        description="ℓ ~ σ²/(πGΣ) (vertical scale height)",
        expected_range=(0.1, 2)
    ))
    
    def disk_crossing_time_scale(row):
        """ℓ ~ σ_v * t_cross where t_cross ~ R_disk/v_circ
        
        Characteristic distance a velocity perturbation travels
        in one disk crossing time.
        """
        R_disk = row['R_disk'] * kpc_to_m
        v_circ = row['v_flat'] * 1e3
        sigma_v = row.get('sigma_velocity', 30) * 1e3
        
        t_cross = R_disk / v_circ
        ell0 = sigma_v * t_cross
        
        return ell0 / kpc_to_m
    
    hypotheses.append(ScaleHypothesis(
        name="crossing_time_scale",
        formula=disk_crossing_time_scale,
        description="ℓ ~ σ_v * (R_disk/v_circ) (crossing time scale)",
        expected_range=(0.5, 5)
    ))
    
    def geometric_mean_scales(row):
        """ℓ ~ √(R_disk * h_disk)
        
        Geometric mean of horizontal and vertical scales.
        """
        R_disk = row['R_disk'] * kpc_to_m
        sigma_v = row.get('sigma_velocity', 30) * 1e3
        M_b = row['M_baryon'] * Msun_to_kg
        
        Sigma = M_b / (np.pi * R_disk**2)
        h = sigma_v**2 / (np.pi * G * Sigma)
        
        ell0 = np.sqrt(R_disk * h)
        
        return ell0 / kpc_to_m
    
    hypotheses.append(ScaleHypothesis(
        name="geometric_mean_RxH",
        formula=geometric_mean_scales,
        description="ℓ ~ √(R_disk × h_disk)",
        expected_range=(1, 10)
    ))
    
    # ========================================================================
    # Category 4: Dynamical time scales
    # ========================================================================
    
    def orbital_period_scale(row):
        """ℓ ~ v_circ * P_orb where P_orb = 2πR/v_circ
        
        Distance traveled in one orbital period.
        """
        R_disk = row['R_disk'] * kpc_to_m
        v_circ = row['v_flat'] * 1e3
        
        P_orb = 2 * np.pi * R_disk / v_circ
        ell0 = v_circ * P_orb  # This is just 2πR
        
        return ell0 / kpc_to_m
    
    hypotheses.append(ScaleHypothesis(
        name="orbital_period",
        formula=orbital_period_scale,
        description="ℓ ~ 2πR_disk (one orbital period)",
        expected_range=(5, 50)
    ))
    
    def jeans_length(row):
        """ℓ ~ σ/√(Gρ) (Jeans length)
        
        Scale above which gravity dominates pressure.
        """
        R_disk = row['R_disk'] * kpc_to_m
        sigma_v = row.get('sigma_velocity', 30) * 1e3
        M_b = row['M_baryon'] * Msun_to_kg
        
        # Average density ~ M/(2πR²h), approximate h ~ 0.1R
        h = 0.1 * R_disk
        rho = M_b / (2 * np.pi * R_disk**2 * h)
        
        lambda_J = sigma_v / np.sqrt(G * rho)
        
        return lambda_J / kpc_to_m
    
    hypotheses.append(ScaleHypothesis(
        name="jeans_length",
        formula=jeans_length,
        description="ℓ ~ σ/√(Gρ) (Jeans length)",
        expected_range=(1, 20)
    ))
    
    # ========================================================================
    # Category 5: Gravitational radius scales
    # ========================================================================
    
    def gravitational_radius(row):
        """ℓ ~ GM/v²
        
        Scale at which v² ~ GM/R (characteristic gravitational scale).
        """
        M_b = row['M_baryon'] * Msun_to_kg
        v = row['v_flat'] * 1e3
        
        R_g = G * M_b / (v**2)
        
        return R_g / kpc_to_m
    
    hypotheses.append(ScaleHypothesis(
        name="gravitational_radius",
        formula=gravitational_radius,
        description="ℓ ~ GM_b/v² (gravitational scale)",
        expected_range=(1, 20)
    ))
    
    def schwarzschild_times_beta(row):
        """ℓ ~ β * (GM/c²)
        
        Schwarzschild radius scaled by dimensionless β.
        For M ~ 10^10 M_sun, GM/c² ~ 15 km, so need β ~ 3×10^8 to get kpc.
        """
        M_b = row['M_baryon'] * Msun_to_kg
        
        r_s = G * M_b / c**2
        beta = 3e8  # Scaling factor to get kpc range
        
        return beta * r_s / kpc_to_m
    
    hypotheses.append(ScaleHypothesis(
        name="scaled_schwarzschild",
        formula=schwarzschild_times_beta,
        description="ℓ ~ β(GM/c²) with β ~ 3×10⁸",
        expected_range=(1, 10)
    ))
    
    # ========================================================================
    # Category 7: Hybrid scales (most promising)
    # ========================================================================
    
    def hybrid_mass_velocity_scale(row):
        """ℓ ~ (GM/v²)^α * R_disk^(1-α)
        
        Interpolation between gravitational scale and disk scale.
        Try α = 0.5 (geometric mean).
        """
        M_b = row['M_baryon'] * Msun_to_kg
        v = row['v_flat'] * 1e3
        R_disk = row['R_disk'] * kpc_to_m
        
        alpha = 0.5
        R_g = G * M_b / (v**2)
        
        ell0 = (R_g**alpha) * (R_disk**(1-alpha))
        
        return ell0 / kpc_to_m
    
    hypotheses.append(ScaleHypothesis(
        name="hybrid_GM_Rdisk",
        formula=hybrid_mass_velocity_scale,
        description="ℓ ~ (GM/v²)^0.5 × R_disk^0.5",
        expected_range=(1, 15)
    ))
    
    def velocity_dispersion_radius(row):
        """ℓ ~ R_disk * (σ/v_circ)
        
        Disk scale modulated by velocity dispersion ratio.
        """
        R_disk = row['R_disk']  # Already in kpc
        sigma_v = row.get('sigma_velocity', 30)
        v_circ = row['v_flat']
        
        ell0 = R_disk * (sigma_v / v_circ)
        
        return ell0
    
    hypotheses.append(ScaleHypothesis(
        name="dispersion_modulated_radius",
        formula=velocity_dispersion_radius,
        description="ℓ ~ R_disk × (σ_v/v_circ)",
        expected_range=(0.5, 5)
    ))
    
    # ========================================================================
    # Category 8: Empirical power-law fits
    # ========================================================================
    
    def power_law_fit(row, alpha_M=0.5, alpha_v=-2.0, alpha_R=0.0):
        """ℓ ~ M_b^α_M * v^α_v * R^α_R
        
        General power law - will optimize exponents.
        """
        M_b = row['M_baryon']  # M_sun
        v = row['v_flat']  # km/s
        R = row['R_disk']  # kpc
        
        # Normalize to typical values to avoid overflow
        M_norm = M_b / 1e10
        v_norm = v / 200
        R_norm = R / 5
        
        ell0 = (M_norm**alpha_M) * (v_norm**alpha_v) * (R_norm**alpha_R)
        
        # Scale to get into kpc range
        ell0 *= 5.0  # Typical scale
        
        return ell0
    
    hypotheses.append(ScaleHypothesis(
        name="power_law_Mb0.5_v-2",
        formula=lambda row: power_law_fit(row, alpha_M=0.5, alpha_v=-2.0, alpha_R=0.0),
        description="ℓ ~ M_b^0.5 × v^-2 (Tully-Fisher)",
        expected_range=(1, 15)
    ))
    
    hypotheses.append(ScaleHypothesis(
        name="power_law_Mb0.3_v-1_R0.3",
        formula=lambda row: power_law_fit(row, alpha_M=0.3, alpha_v=-1.0, alpha_R=0.3),
        description="ℓ ~ M_b^0.3 × v^-1 × R^0.3",
        expected_range=(1, 15)
    ))
    
    return hypotheses

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_scale_finder_tests(sparc_csv_path: str, 
                          output_dir: str = "GravityWaveTest/scale_tests"):
    """Run complete scale-finding test suite."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("ΣGRAVITY SCALE FINDER - Comprehensive Test Suite")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading SPARC data...")
    sparc_data = pd.read_csv(sparc_csv_path)
    print(f"Loaded {len(sparc_data)} galaxies")
    
    # Initialize test suite
    print("\n[2/4] Initializing test suite...")
    suite = ScaleTestSuite(sparc_data, fitted_ell0=4.993, fitted_A=0.591)
    
    # Create hypothesis library
    print("\n[3/4] Creating hypothesis library...")
    hypotheses = create_hypothesis_library()
    print(f"Created {len(hypotheses)} hypotheses to test")
    
    # Run tests
    print("\n[4/4] Running tests...")
    print("="*80)
    
    results = []
    for i, hyp in enumerate(hypotheses):
        print(f"\n[Test {i+1}/{len(hypotheses)}]")
        result = suite.test_hypothesis(hyp, verbose=True)
        
        if result is not None:
            results.append(result)
            
            # Generate plots
            plot_path = f"{output_dir}/{hyp.name}_diagnostic.png"
            suite.plot_hypothesis_results(hyp, result, save_path=plot_path)
    
    # Summarize results
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    # Sort by scatter_dex (lower is better)
    results.sort(key=lambda r: r.scatter_dex)
    
    print(f"\n{'Rank':<5} {'Hypothesis':<35} {'Median ℓ₀':<12} {'Scatter':<10} {'BIC':<10}")
    print("-"*80)
    for i, r in enumerate(results[:10]):  # Top 10
        print(f"{i+1:<5} {r.hypothesis_name:<35} {r.ell0_median:>8.3f} kpc  {r.scatter_dex:>8.4f}  {r.bic:>9.1f}")
    
    # Save results to JSON
    results_dict = {
        'fitted_ell0': 4.993,
        'fitted_A': 0.591,
        'n_galaxies': len(sparc_data),
        'results': [
            {
                'rank': i+1,
                'name': r.hypothesis_name,
                'ell0_median': r.ell0_median,
                'ell0_std': r.ell0_std,
                'scatter_dex': r.scatter_dex,
                'chi2': r.chi2,
                'bic': r.bic,
                'aic': r.aic,
                'correlation_with_Mb': r.correlation_with_fit
            }
            for i, r in enumerate(results)
        ]
    }
    
    with open(f"{output_dir}/scale_test_results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/scale_test_results.json")
    print(f"Diagnostic plots saved to {output_dir}/*.png")
    
    return results

if __name__ == "__main__":
    # Example usage
    run_scale_finder_tests(
        sparc_csv_path="data/sparc/sparc_combined.csv",
        output_dir="GravityWaveTest/scale_tests"
    )


"""
Optimize power-law exponents for ℓ₀ ~ M^α × v^β × R^γ
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from typing import Tuple
import json
import os

class PowerLawOptimizer:
    """Find best-fit power law for coherence length."""
    
    def __init__(self, sparc_data: pd.DataFrame, target_ell0: float = 4.993):
        self.data = sparc_data
        self.target = target_ell0
        
        # Extract and normalize data
        self.M_b = sparc_data['M_baryon'].values / 1e10  # Normalize to 10^10 M_sun
        self.v = sparc_data['v_flat'].values / 200  # Normalize to 200 km/s
        self.R = sparc_data['R_disk'].values / 5  # Normalize to 5 kpc
        
    def predict_ell0(self, params: np.ndarray) -> np.ndarray:
        """Predict ℓ₀ from power law.
        
        params = [alpha_M, alpha_v, alpha_R, scale]
        ℓ₀ = scale × M^alpha_M × v^alpha_v × R^alpha_R
        """
        alpha_M, alpha_v, alpha_R, scale = params
        
        ell0 = scale * (self.M_b**alpha_M) * (self.v**alpha_v) * (self.R**alpha_R)
        
        return ell0
    
    def objective_scatter(self, params: np.ndarray) -> float:
        """Minimize scatter around target value."""
        ell0_pred = self.predict_ell0(params)
        
        # Remove invalid predictions
        mask = np.isfinite(ell0_pred) & (ell0_pred > 0)
        ell0_pred = ell0_pred[mask]
        
        if len(ell0_pred) < 10:
            return 1e10  # Penalty for bad params
        
        # Log-space residuals
        residuals = np.log10(ell0_pred) - np.log10(self.target)
        scatter = np.std(residuals)
        
        return scatter
    
    def objective_median(self, params: np.ndarray) -> float:
        """Minimize deviation of median from target."""
        ell0_pred = self.predict_ell0(params)
        
        mask = np.isfinite(ell0_pred) & (ell0_pred > 0)
        ell0_pred = ell0_pred[mask]
        
        if len(ell0_pred) < 10:
            return 1e10
        
        median_pred = np.median(ell0_pred)
        deviation = abs(np.log10(median_pred) - np.log10(self.target))
        
        # Also penalize large scatter
        scatter = np.std(np.log10(ell0_pred) - np.log10(self.target))
        
        return deviation + 0.5 * scatter
    
    def optimize(self, method: str = 'differential_evolution') -> Tuple[np.ndarray, dict]:
        """Run optimization."""
        
        # Bounds: exponents in [-3, 3], scale in [0.1, 100]
        bounds = [(-3, 3), (-3, 3), (-3, 3), (0.1, 100)]
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective_median,
                bounds=bounds,
                seed=42,
                maxiter=1000,
                atol=1e-6,
                workers=-1
            )
        else:
            # Multi-start local optimization
            best_result = None
            best_score = np.inf
            
            for i in range(20):
                x0 = [
                    np.random.uniform(-2, 2),  # alpha_M
                    np.random.uniform(-3, 0),  # alpha_v (expect negative)
                    np.random.uniform(-1, 2),  # alpha_R
                    np.random.uniform(1, 10)   # scale
                ]
                
                res = minimize(
                    self.objective_median,
                    x0=x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res
            
            result = best_result
        
        # Compute diagnostics
        alpha_M, alpha_v, alpha_R, scale = result.x
        ell0_pred = self.predict_ell0(result.x)
        mask = np.isfinite(ell0_pred) & (ell0_pred > 0)
        ell0_pred = ell0_pred[mask]
        
        diagnostics = {
            'alpha_M': alpha_M,
            'alpha_v': alpha_v,
            'alpha_R': alpha_R,
            'scale': scale,
            'median_ell0': np.median(ell0_pred),
            'scatter_dex': np.std(np.log10(ell0_pred) - np.log10(self.target)),
            'n_valid': len(ell0_pred),
            'objective': result.fun
        }
        
        return result.x, diagnostics
    
    def plot_results(self, params: np.ndarray, save_path: str = None):
        """Visualize optimization results."""
        
        alpha_M, alpha_v, alpha_R, scale = params
        ell0_pred = self.predict_ell0(params)
        
        # Get original (unnormalized) values
        M_b_orig = self.M_b * 1e10
        v_orig = self.v * 200
        R_orig = self.R * 5
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Optimized Power Law: ℓ₀ = {scale:.3f} × M^{alpha_M:.3f} × v^{alpha_v:.3f} × R^{alpha_R:.3f}', 
                     fontsize=14)
        
        # Plot 1: ℓ₀ vs M_b
        ax = axes[0, 0]
        scatter = ax.scatter(M_b_orig, ell0_pred, c=v_orig, cmap='viridis', alpha=0.6, s=30)
        ax.axhline(self.target, color='r', linestyle='--', label=f'Target: {self.target:.2f} kpc')
        ax.set_xlabel('M_baryon [M_sun]')
        ax.set_ylabel('Predicted ℓ₀ [kpc]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='v [km/s]')
        
        # Plot 2: ℓ₀ vs v
        ax = axes[0, 1]
        scatter = ax.scatter(v_orig, ell0_pred, c=M_b_orig, cmap='plasma', alpha=0.6, s=30)
        ax.axhline(self.target, color='r', linestyle='--')
        ax.set_xlabel('v_flat [km/s]')
        ax.set_ylabel('Predicted ℓ₀ [kpc]')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='M_b [M_sun]')
        
        # Plot 3: ℓ₀ vs R
        ax = axes[0, 2]
        scatter = ax.scatter(R_orig, ell0_pred, c=M_b_orig, cmap='plasma', alpha=0.6, s=30)
        ax.axhline(self.target, color='r', linestyle='--')
        ax.set_xlabel('R_disk [kpc]')
        ax.set_ylabel('Predicted ℓ₀ [kpc]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='M_b [M_sun]')
        
        # Plot 4: Residuals vs M_b
        ax = axes[1, 0]
        residuals = np.log10(ell0_pred) - np.log10(self.target)
        ax.scatter(M_b_orig, residuals, alpha=0.6, s=20)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel('M_baryon [M_sun]')
        ax.set_ylabel('log₁₀(ℓ_pred / ℓ_target)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Scatter: {np.std(residuals):.4f} dex')
        
        # Plot 5: Histogram of predictions
        ax = axes[1, 1]
        ax.hist(ell0_pred, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(self.target, color='r', linestyle='--', linewidth=2, label='Target')
        ax.axvline(np.median(ell0_pred), color='b', linestyle='--', linewidth=2, label='Median')
        ax.set_xlabel('Predicted ℓ₀ [kpc]')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: QQ plot
        ax = axes[1, 2]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normality Check)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

def run_power_law_optimization(sparc_csv_path: str, 
                               output_dir: str = "GravityWaveTest/power_law_fits"):
    """Run power-law optimization."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("POWER-LAW OPTIMIZATION FOR ℓ₀")
    print("="*80)
    
    # Load data
    print("\nLoading SPARC data...")
    sparc_data = pd.read_csv(sparc_csv_path)
    print(f"Loaded {len(sparc_data)} galaxies")
    
    # Initialize optimizer
    optimizer = PowerLawOptimizer(sparc_data, target_ell0=4.993)
    
    # Run optimization
    print("\nRunning global optimization (this may take 1-2 minutes)...")
    params, diagnostics = optimizer.optimize(method='differential_evolution')
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"\nOptimized formula:")
    print(f"  ℓ₀ = {diagnostics['scale']:.4f} × M_b^{diagnostics['alpha_M']:.4f} × v^{diagnostics['alpha_v']:.4f} × R^{diagnostics['alpha_R']:.4f}")
    print(f"\nDiagnostics:")
    print(f"  Target ℓ₀: 4.993 kpc")
    print(f"  Median predicted: {diagnostics['median_ell0']:.4f} kpc")
    print(f"  Scatter: {diagnostics['scatter_dex']:.5f} dex")
    print(f"  Valid galaxies: {diagnostics['n_valid']}/{len(sparc_data)}")
    print(f"  Objective value: {diagnostics['objective']:.6f}")
    
    # Interpret exponents
    print(f"\nPhysical interpretation:")
    if abs(diagnostics['alpha_M'] - 0.5) < 0.1:
        print(f"  α_M ≈ 0.5 → consistent with Tully-Fisher (v⁴ ∝ M)")
    if abs(diagnostics['alpha_v'] + 2.0) < 0.2:
        print(f"  α_v ≈ -2 → consistent with ℓ ~ GM/v²")
    if abs(diagnostics['alpha_R']) < 0.1:
        print(f"  α_R ≈ 0 → scale independent of R_disk")
    
    # Generate plots
    print("\nGenerating diagnostic plots...")
    optimizer.plot_results(params, save_path=f"{output_dir}/optimized_power_law.png")
    
    # Save results
    results = {
        'target_ell0_kpc': 4.993,
        'optimized_params': {
            'alpha_M': float(diagnostics['alpha_M']),
            'alpha_v': float(diagnostics['alpha_v']),
            'alpha_R': float(diagnostics['alpha_R']),
            'scale': float(diagnostics['scale'])
        },
        'diagnostics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in diagnostics.items()}
    }
    
    with open(f"{output_dir}/optimized_params.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/optimized_params.json")
    
    return params, diagnostics

if __name__ == "__main__":
    run_power_law_optimization(
        sparc_csv_path="data/sparc/sparc_combined.csv",
        output_dir="GravityWaveTest/power_law_fits"
    )


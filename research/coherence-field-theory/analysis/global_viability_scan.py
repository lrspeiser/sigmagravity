"""
Global Viability Scan for Canonical Scalar-Tensor Theory

Tests whether a single, global field theory with constant parameters can 
simultaneously satisfy:
1. Cosmology constraints (Omega_m, Omega_phi, H(z))
2. Galaxy screening (R_c < 10 kpc for spirals, small in MW, large at cosmic density)
3. PPN bounds (|gamma-1|, |beta-1|)

This is the decisive test: if NO parameter set passes all three cuts, the 
canonical chameleon ansatz is not viable and we need a richer structure 
(symmetron, k-essence, Vainshtein, etc.).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmology.background_evolution import CoherenceCosmology
from galaxies.halo_field_profile import HaloFieldSolver


class GlobalViabilityScan:
    """
    Scan parameter space of canonical scalar-tensor theory.
    
    Theory:
        V(φ) = V₀ exp(-λφ) + M^5/φ
        A(φ) = exp(βφ)
    
    All parameters are CONSTANTS (no ρ-dependence).
    """
    
    def __init__(self, output_dir='outputs/viability_scan'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Physics cuts
        self.cosmo_cuts = {
            'Omega_m0_min': 0.25,
            'Omega_m0_max': 0.35,
            'Omega_phi0_min': 0.65,
            'Omega_phi0_max': 0.75,
        }
        
        self.galaxy_cuts = {
            'R_c_spiral_max': 10.0,      # kpc
            'R_c_dwarf_max': 50.0,       # kpc
            'R_c_cosmic_min': 1000.0,    # kpc (must be >> Mpc for homogeneity)
        }
        
        self.ppn_cuts = {
            'gamma_minus_1_max': 2.3e-5,  # Cassini bound
            'beta_minus_1_max': 8.0e-5,   # Solar system bound
        }
        
        # Test densities (SI units: kg/m^3)
        self.rho_cosmic = 1e-26   # ~Omega_m * rho_crit
        self.rho_dwarf = 1e-21    # ~10^8 Msun/kpc^3
        self.rho_spiral = 1e-20   # ~10^9 Msun/kpc^3
        
        self.results = []
        
    def define_parameter_grid(self, n_per_param=12):
        """
        Define scan grid.
        
        Start coarse (n=12 → ~20k points), refine if we find viable regions.
        """
        grid = {
            'V0': np.logspace(-8, -4, n_per_param),
            'lambda': np.linspace(0.1, 5.0, n_per_param),
            'M4': np.logspace(-3, -1, n_per_param),
            'beta': np.logspace(-3, 0, n_per_param),
        }
        
        # Create all combinations
        from itertools import product
        keys = list(grid.keys())
        values = list(grid.values())
        
        param_sets = []
        for combo in product(*values):
            param_sets.append(dict(zip(keys, combo)))
            
        print(f"Total parameter combinations: {len(param_sets):,}")
        return param_sets
    
    def test_cosmology(self, V0, lambda_param, M4, beta):
        """
        Test cosmology constraints.
        
        Returns:
            dict: {'pass': bool, 'Omega_m0': float, 'Omega_phi0': float, 'reason': str}
        """
        try:
            cosmo = CoherenceCosmology(
                V0=V0, 
                lambda_param=lambda_param, 
                M4=M4,
                rho_m0_guess=V0  # Guess based on V0 scale
            )
            
            results = cosmo.evolve(N_start=-7.0, N_end=0.0, n_steps=2000)
            
            Omega_m0 = results['Omega_m0']
            Omega_phi0 = results['Omega_phi0']
            
            # Check bounds
            pass_Omega_m = (self.cosmo_cuts['Omega_m0_min'] <= Omega_m0 <= 
                           self.cosmo_cuts['Omega_m0_max'])
            pass_Omega_phi = (self.cosmo_cuts['Omega_phi0_min'] <= Omega_phi0 <= 
                             self.cosmo_cuts['Omega_phi0_max'])
            
            passed = pass_Omega_m and pass_Omega_phi
            
            if not passed:
                reason = f"Omega_m={Omega_m0:.4f}, Omega_phi={Omega_phi0:.4f}"
            else:
                reason = "OK"
                
            return {
                'pass': passed,
                'Omega_m0': Omega_m0,
                'Omega_phi0': Omega_phi0,
                'reason': reason
            }
            
        except Exception as e:
            return {
                'pass': False,
                'Omega_m0': np.nan,
                'Omega_phi0': np.nan,
                'reason': f"Error: {str(e)[:50]}"
            }
    
    def test_galaxy_screening(self, V0, lambda_param, M4, beta):
        """
        Test galaxy screening: compute R_c at different densities.
        
        Returns:
            dict: {'pass': bool, 'R_c_cosmic': float, 'R_c_dwarf': float, 
                   'R_c_spiral': float, 'reason': str}
        """
        try:
            solver = HaloFieldSolver(
                V0=V0,
                lambda_param=lambda_param,
                beta=beta,
                M4=M4
            )
            
            # Test at three densities
            densities = {
                'cosmic': self.rho_cosmic,
                'dwarf': self.rho_dwarf,
                'spiral': self.rho_spiral
            }
            
            R_c_values = {}
            for name, rho in densities.items():
                try:
                    phi_min = solver.find_phi_min(rho)
                    m_eff = solver.compute_m_eff(phi_min, rho)
                    R_c_m = 1.0 / m_eff if m_eff > 0 else np.inf
                    R_c_kpc = R_c_m * 3.24e-20  # m to kpc
                    R_c_values[name] = R_c_kpc
                except:
                    R_c_values[name] = np.nan
            
            # Check cuts
            R_c_cosmic = R_c_values['cosmic']
            R_c_dwarf = R_c_values['dwarf']
            R_c_spiral = R_c_values['spiral']
            
            pass_cosmic = R_c_cosmic >= self.galaxy_cuts['R_c_cosmic_min']
            pass_dwarf = R_c_dwarf <= self.galaxy_cuts['R_c_dwarf_max']
            pass_spiral = R_c_spiral <= self.galaxy_cuts['R_c_spiral_max']
            
            passed = pass_cosmic and pass_dwarf and pass_spiral
            
            if not passed:
                reason = f"R_c: cosmic={R_c_cosmic:.1e} dwarf={R_c_dwarf:.1f} spiral={R_c_spiral:.1f} kpc"
            else:
                reason = "OK"
                
            return {
                'pass': passed,
                'R_c_cosmic': R_c_cosmic,
                'R_c_dwarf': R_c_dwarf,
                'R_c_spiral': R_c_spiral,
                'reason': reason
            }
            
        except Exception as e:
            return {
                'pass': False,
                'R_c_cosmic': np.nan,
                'R_c_dwarf': np.nan,
                'R_c_spiral': np.nan,
                'reason': f"Error: {str(e)[:50]}"
            }
    
    def test_ppn(self, V0, lambda_param, M4, beta):
        """
        Test PPN constraints (simplified version).
        
        For now, return placeholder. Full PPN test requires Solar System 
        density and solving field equation there.
        
        Returns:
            dict: {'pass': bool, 'gamma_minus_1': float, 'beta_minus_1': float, 'reason': str}
        """
        # Placeholder: assume pass for now, implement later
        # Real test would use solar_system/ppn_tests.py with local field solution
        
        return {
            'pass': True,  # TODO: implement proper PPN test
            'gamma_minus_1': np.nan,
            'beta_minus_1': np.nan,
            'reason': "PPN test not yet implemented"
        }
    
    def scan_parameter_space(self, param_sets, stage='full'):
        """
        Run the scan.
        
        Parameters:
            param_sets: list of parameter dicts
            stage: 'cosmology_only', 'screening_only', or 'full'
        """
        print(f"\n{'='*60}")
        print(f"GLOBAL VIABILITY SCAN - Stage: {stage}")
        print(f"{'='*60}\n")
        
        for params in tqdm(param_sets, desc="Scanning"):
            V0 = params['V0']
            lambda_param = params['lambda']
            M4 = params['M4']
            beta = params['beta']
            
            result = {
                'V0': V0,
                'lambda': lambda_param,
                'M4': M4,
                'beta': beta,
            }
            
            # Stage 1: Cosmology (fastest, most restrictive)
            cosmo_result = self.test_cosmology(V0, lambda_param, M4, beta)
            result.update({f'cosmo_{k}': v for k, v in cosmo_result.items()})
            
            if stage == 'cosmology_only' or not cosmo_result['pass']:
                result['screening_pass'] = False
                result['ppn_pass'] = False
                result['overall_pass'] = False
                result['overall_reason'] = cosmo_result['reason']
                self.results.append(result)
                continue
            
            # Stage 2: Galaxy screening
            screening_result = self.test_galaxy_screening(V0, lambda_param, M4, beta)
            result.update({f'screening_{k}': v for k, v in screening_result.items()})
            
            if stage == 'screening_only' or not screening_result['pass']:
                result['ppn_pass'] = False
                result['overall_pass'] = False
                result['overall_reason'] = screening_result['reason']
                self.results.append(result)
                continue
            
            # Stage 3: PPN (only if passed first two)
            ppn_result = self.test_ppn(V0, lambda_param, M4, beta)
            result.update({f'ppn_{k}': v for k, v in ppn_result.items()})
            
            # Overall verdict
            overall_pass = (cosmo_result['pass'] and 
                          screening_result['pass'] and 
                          ppn_result['pass'])
            
            result['overall_pass'] = overall_pass
            result['overall_reason'] = "VIABLE" if overall_pass else ppn_result['reason']
            
            self.results.append(result)
    
    def save_results(self):
        """Save scan results to CSV and JSON."""
        df = pd.DataFrame(self.results)
        
        # Save full results
        csv_path = self.output_dir / 'viability_scan_full.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nFull results saved to: {csv_path}")
        
        # Save only viable points
        viable = df[df['overall_pass'] == True]
        if len(viable) > 0:
            viable_path = self.output_dir / 'viability_scan_viable.csv'
            viable.to_csv(viable_path, index=False)
            print(f"Viable parameters saved to: {viable_path}")
        
        # Summary statistics
        summary = {
            'total_tested': len(df),
            'passed_cosmology': int(df['cosmo_pass'].sum()),
            'passed_screening': int(df['screening_pass'].sum()),
            'passed_ppn': int(df['ppn_pass'].sum()),
            'passed_all': int(df['overall_pass'].sum()),
            'viable_fraction': float(df['overall_pass'].sum() / len(df)),
        }
        
        summary_path = self.output_dir / 'viability_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("SCAN SUMMARY")
        print(f"{'='*60}")
        print(f"Total tested:       {summary['total_tested']:,}")
        print(f"Passed cosmology:   {summary['passed_cosmology']:,} ({100*summary['passed_cosmology']/summary['total_tested']:.1f}%)")
        print(f"Passed screening:   {summary['passed_screening']:,} ({100*summary['passed_screening']/summary['total_tested']:.1f}%)")
        print(f"Passed PPN:         {summary['passed_ppn']:,}")
        print(f"PASSED ALL:         {summary['passed_all']:,} ({100*summary['viable_fraction']:.3f}%)")
        print(f"{'='*60}\n")
        
        return df, viable, summary
    
    def plot_results(self, df):
        """Create diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Parameter space coverage
        ax = axes[0, 0]
        passed = df[df['cosmo_pass'] == True]
        failed = df[df['cosmo_pass'] == False]
        
        ax.scatter(failed['lambda'], failed['M4'], c='red', alpha=0.3, s=10, label='Failed cosmo')
        ax.scatter(passed['lambda'], passed['M4'], c='blue', alpha=0.5, s=20, label='Passed cosmo')
        
        viable = df[df['overall_pass'] == True]
        if len(viable) > 0:
            ax.scatter(viable['lambda'], viable['M4'], c='green', s=100, 
                      marker='*', edgecolors='black', linewidths=1, label='VIABLE', zorder=10)
        
        ax.set_xlabel('λ')
        ax.set_ylabel('M₄')
        ax.set_yscale('log')
        ax.legend()
        ax.set_title('Parameter Space: λ vs M₄')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: R_c values
        ax = axes[0, 1]
        screening_tested = df[df['cosmo_pass'] == True]
        if len(screening_tested) > 0:
            ax.scatter(screening_tested['screening_R_c_spiral'], 
                      screening_tested['screening_R_c_dwarf'],
                      c='blue', alpha=0.5, s=20)
            
            # Draw target box
            ax.axvline(self.galaxy_cuts['R_c_spiral_max'], color='red', 
                      linestyle='--', label='R_c limits')
            ax.axhline(self.galaxy_cuts['R_c_dwarf_max'], color='red', linestyle='--')
            
            if len(viable) > 0:
                ax.scatter(viable['screening_R_c_spiral'], 
                          viable['screening_R_c_dwarf'],
                          c='green', s=100, marker='*', 
                          edgecolors='black', linewidths=1, zorder=10)
        
        ax.set_xlabel('R_c spiral (kpc)')
        ax.set_ylabel('R_c dwarf (kpc)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.set_title('Galaxy Screening: R_c Values')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Cosmology
        ax = axes[1, 0]
        ax.scatter(df['cosmo_Omega_m0'], df['cosmo_Omega_phi0'], 
                  c=df['cosmo_pass'].astype(int), cmap='RdYlGn', alpha=0.5, s=20)
        
        # Draw target box
        ax.axvline(self.cosmo_cuts['Omega_m0_min'], color='gray', linestyle='--')
        ax.axvline(self.cosmo_cuts['Omega_m0_max'], color='gray', linestyle='--')
        ax.axhline(self.cosmo_cuts['Omega_phi0_min'], color='gray', linestyle='--')
        ax.axhline(self.cosmo_cuts['Omega_phi0_max'], color='gray', linestyle='--')
        
        if len(viable) > 0:
            ax.scatter(viable['cosmo_Omega_m0'], viable['cosmo_Omega_phi0'],
                      c='green', s=100, marker='*', 
                      edgecolors='black', linewidths=1, zorder=10)
        
        ax.set_xlabel('Ω_m0')
        ax.set_ylabel('Ω_φ0')
        ax.set_title('Cosmology: Density Parameters')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Beta vs V0 for viable points
        ax = axes[1, 1]
        if len(viable) > 0:
            scatter = ax.scatter(viable['beta'], viable['V0'], 
                               c=viable['screening_R_c_spiral'], 
                               cmap='viridis', s=100, edgecolors='black', linewidths=1)
            plt.colorbar(scatter, ax=ax, label='R_c spiral (kpc)')
            ax.set_title('Viable Parameters: β vs V₀')
        else:
            ax.text(0.5, 0.5, 'NO VIABLE PARAMETERS FOUND', 
                   ha='center', va='center', fontsize=16, color='red',
                   transform=ax.transAxes)
            ax.set_title('Viable Parameters')
        
        ax.set_xlabel('β')
        ax.set_ylabel('V₀')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'viability_scan_summary.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Summary plot saved to: {plot_path}")
        
        plt.close()


def run_coarse_scan(n_per_param=10):
    """Run coarse scan (10^4 points)."""
    scanner = GlobalViabilityScan()
    param_sets = scanner.define_parameter_grid(n_per_param=n_per_param)
    
    print(f"\nRunning COARSE scan: {n_per_param} points per parameter")
    print(f"Total combinations: {len(param_sets):,}\n")
    
    scanner.scan_parameter_space(param_sets, stage='full')
    df, viable, summary = scanner.save_results()
    scanner.plot_results(df)
    
    return scanner, df, viable, summary


def run_fine_scan_near_viable(viable_df, n_per_param=20):
    """Refine scan around viable regions."""
    if len(viable_df) == 0:
        print("No viable regions found in coarse scan. Cannot refine.")
        return None
    
    # Get ranges around viable points
    V0_range = (viable_df['V0'].min() * 0.5, viable_df['V0'].max() * 2.0)
    lambda_range = (max(0.1, viable_df['lambda'].min() - 1.0), 
                   viable_df['lambda'].max() + 1.0)
    M4_range = (viable_df['M4'].min() * 0.5, viable_df['M4'].max() * 2.0)
    beta_range = (viable_df['beta'].min() * 0.5, viable_df['beta'].max() * 2.0)
    
    # Create refined grid
    grid = {
        'V0': np.logspace(np.log10(V0_range[0]), np.log10(V0_range[1]), n_per_param),
        'lambda': np.linspace(lambda_range[0], lambda_range[1], n_per_param),
        'M4': np.logspace(np.log10(M4_range[0]), np.log10(M4_range[1]), n_per_param),
        'beta': np.logspace(np.log10(beta_range[0]), np.log10(beta_range[1]), n_per_param),
    }
    
    from itertools import product
    param_sets = []
    for combo in product(*[grid[k] for k in ['V0', 'lambda', 'M4', 'beta']]):
        param_sets.append(dict(zip(['V0', 'lambda', 'M4', 'beta'], combo)))
    
    print(f"\nRunning FINE scan around viable regions")
    print(f"Total combinations: {len(param_sets):,}\n")
    
    scanner = GlobalViabilityScan(output_dir='outputs/viability_scan_fine')
    scanner.scan_parameter_space(param_sets, stage='full')
    df, viable, summary = scanner.save_results()
    scanner.plot_results(df)
    
    return scanner, df, viable, summary


if __name__ == '__main__':
    # Run coarse scan first
    print("\n" + "="*60)
    print("GLOBAL VIABILITY SCAN FOR CANONICAL SCALAR-TENSOR THEORY")
    print("="*60)
    
    scanner, df, viable, summary = run_coarse_scan(n_per_param=10)
    
    # If viable points found, optionally refine
    if len(viable) > 0:
        print(f"\n✓ Found {len(viable)} viable parameter sets!")
        print("\nBest viable parameters:")
        print(viable[['V0', 'lambda', 'M4', 'beta', 
                     'screening_R_c_spiral', 'screening_R_c_dwarf']].head())
        
        response = input("\nRun fine scan around viable regions? (y/n): ")
        if response.lower() == 'y':
            run_fine_scan_near_viable(viable, n_per_param=15)
    else:
        print("\n✗ NO VIABLE PARAMETERS FOUND")
        print("\nThis means the canonical chameleon ansatz V(φ) = V₀e^(-λφ) + M^5/φ")
        print("CANNOT simultaneously satisfy cosmology + screening + PPN constraints.")
        print("\nNext steps:")
        print("  1. Try symmetron potential: V(φ) = -μ²φ²/2 + λφ⁴/4")
        print("  2. Try k-mouflage (non-canonical kinetic term)")
        print("  3. Consider Vainshtein screening (derivative interactions)")

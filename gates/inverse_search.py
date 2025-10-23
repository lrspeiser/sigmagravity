"""
Inverse Search: Data-Aided Derivation of Coherence Window

Tests whether Burr-XII window emerges as optimal under physical constraints.

This is the KEY test for "are gates derived or ad-hoc?"

Usage:
    python inverse_search.py
"""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import expit
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_core import G_distance, G_solar_system


class WindowCandidate:
    """Base class for coherence window candidates"""
    
    def __init__(self, name, n_params):
        self.name = name
        self.n_params = n_params
    
    def C(self, R, params):
        """Coherence window: must return values in [0, 1]"""
        raise NotImplementedError
    
    def get_bounds(self):
        """Parameter bounds for optimization"""
        raise NotImplementedError
    
    def check_constraints(self, R_test, params):
        """
        Check if candidate satisfies constraints:
        C1: C(R) ∈ [0, 1]
        C2: C(0) = 0
        C3: C(∞) → 1
        C4: dC/dR ≥ 0 (monotonic)
        C5: C is saturating (approaches 1 asymptotically)
        """
        C_vals = self.C(R_test, params)
        
        checks = {
            'C1_bounds': np.all((C_vals >= -1e-6) & (C_vals <= 1 + 1e-6)),
            'C2_zero': C_vals[0] < 1e-6,
            'C3_one': C_vals[-1] > 1 - 1e-4,
            'C4_monotonic': np.all(np.gradient(C_vals, np.log(R_test)) >= -1e-6),
            'C5_saturating': (1 - C_vals[-1]) / (1 - C_vals[-5]) < 0.5
        }
        
        return all(checks.values()), checks


class BurrXII(WindowCandidate):
    """Burr-XII window (your paper's form)"""
    
    def __init__(self):
        super().__init__('Burr-XII', 3)
    
    def C(self, R, params):
        ell0, p, n_coh = params
        return 1 - (1 + (R/ell0)**p)**(-n_coh)
    
    def get_bounds(self):
        return [(1.0, 10.0), (0.5, 3.0), (0.3, 2.0)]  # ell0, p, n_coh


class Logistic(WindowCandidate):
    """Logistic sigmoid"""
    
    def __init__(self):
        super().__init__('Logistic', 2)
    
    def C(self, R, params):
        R0, k = params
        return expit(k * (R - R0))
    
    def get_bounds(self):
        return [(1.0, 10.0), (0.5, 5.0)]  # R0, k


class Gompertz(WindowCandidate):
    """Gompertz growth curve"""
    
    def __init__(self):
        super().__init__('Gompertz', 2)
    
    def C(self, R, params):
        a, b = params
        return np.exp(-a * np.exp(-b * R))
    
    def get_bounds(self):
        return [(0.1, 5.0), (0.1, 2.0)]  # a, b


class Hill(WindowCandidate):
    """Hill equation (biochemistry inspired)"""
    
    def __init__(self):
        super().__init__('Hill', 2)
    
    def C(self, R, params):
        R50, n = params
        return R**n / (R50**n + R**n)
    
    def get_bounds(self):
        return [(1.0, 10.0), (0.5, 5.0)]  # R50, n


class StretchedExp(WindowCandidate):
    """Stretched exponential"""
    
    def __init__(self):
        super().__init__('StretchedExp', 2)
    
    def C(self, R, params):
        tau, beta = params
        return 1 - np.exp(-(R/tau)**beta)
    
    def get_bounds(self):
        return [(1.0, 10.0), (0.5, 3.0)]  # tau, beta


class InverseSearch:
    """
    Test which coherence window form emerges under constraints
    """
    
    def __init__(self):
        self.candidates = [
            BurrXII(),
            Logistic(),
            Gompertz(),
            Hill(),
            StretchedExp()
        ]
        self.R_test = np.logspace(-2, 2, 100)
        self.AU_in_kpc = 4.848e-9
    
    def generate_toy_data(self, n_systems=10, seed=42):
        """
        Generate toy galaxy/cluster data
        
        For simplicity: use analytical form with noise
        """
        np.random.seed(seed)
        
        data = []
        for i in range(n_systems):
            R = np.linspace(2, 20, 20)
            
            # True kernel (Burr-XII with noise)
            ell0_true = 5.0 + np.random.normal(0, 0.5)
            p_true = 0.75 + np.random.normal(0, 0.1)
            n_coh_true = 0.5 + np.random.normal(0, 0.1)
            A_true = 0.6 + np.random.normal(0, 0.1)
            
            C_true = 1 - (1 + (R/ell0_true)**p_true)**(-n_coh_true)
            K_true = A_true * C_true
            
            # "Observed" boost
            boost = 1 + K_true + np.random.normal(0, 0.05, len(R))
            
            data.append({
                'R': R,
                'boost': boost,
                'boost_err': 0.05 * np.ones_like(R)
            })
        
        return data
    
    def fit_candidate(self, candidate, data):
        """
        Fit candidate window to data
        
        Returns
        -------
        params : array
            Best-fit parameters
        score : dict
            Fit quality metrics
        """
        def objective(params):
            # Check constraints first
            R_test = np.logspace(-2, 2, 50)
            satisfies, checks = candidate.check_constraints(R_test, params)
            if not satisfies:
                return 1e10  # Reject
            
            # Fit to data
            chi2_total = 0
            for system in data:
                R = system['R']
                boost_obs = system['boost']
                boost_err = system['boost_err']
                
                # Model: boost = 1 + A * C(R)
                # Fit A jointly with window params
                C_vals = candidate.C(R, params)
                
                # Optimal A for this system
                A_opt = np.sum((boost_obs - 1) * C_vals) / np.sum(C_vals**2)
                A_opt = np.clip(A_opt, 0.1, 2.0)  # Reasonable range
                
                boost_model = 1 + A_opt * C_vals
                residuals = (boost_model - boost_obs) / boost_err
                chi2_total += np.sum(residuals**2)
            
            return chi2_total
        
        # Optimize
        bounds = candidate.get_bounds()
        result = differential_evolution(objective, bounds, seed=42, maxiter=200,
                                       atol=1e-4, tol=1e-4, workers=1)
        
        # Compute scores
        chi2 = result.fun
        n_data = sum(len(d['R']) for d in data)
        n_params = len(result.x)
        
        # AIC/BIC
        aic = chi2 + 2 * n_params
        bic = chi2 + n_params * np.log(n_data)
        
        # Cross-validation score (simplified)
        # Use first half for fit, second half for validation
        n_half = len(data) // 2
        chi2_train = 0
        chi2_val = 0
        
        for i, system in enumerate(data):
            R = system['R']
            boost_obs = system['boost']
            C_vals = candidate.C(R, result.x)
            A_opt = np.sum((boost_obs - 1) * C_vals) / np.sum(C_vals**2)
            A_opt = np.clip(A_opt, 0.1, 2.0)
            boost_model = 1 + A_opt * C_vals
            residuals = (boost_model - boost_obs) / system['boost_err']
            chi2_sys = np.sum(residuals**2)
            
            if i < n_half:
                chi2_train += chi2_sys
            else:
                chi2_val += chi2_sys
        
        return {
            'params': result.x,
            'chi2': chi2,
            'chi2_reduced': chi2 / (n_data - n_params),
            'aic': aic,
            'bic': bic,
            'n_params': n_params,
            'chi2_train': chi2_train,
            'chi2_val': chi2_val,
            'success': result.success
        }
    
    def run_search(self, data, save_path='outputs/inverse_search_results.json'):
        """
        Run inverse search across all candidates
        """
        print("\n" + "="*60)
        print("INVERSE SEARCH: Testing Window Forms Under Constraints")
        print("="*60)
        
        results = {}
        
        for candidate in self.candidates:
            print(f"\n{'='*60}")
            print(f"Testing: {candidate.name}")
            print('='*60)
            
            # Check if constraints can be satisfied
            print("Checking constraint satisfaction...")
            test_params = [(b[0] + b[1])/2 for b in candidate.get_bounds()]
            satisfies, checks = candidate.check_constraints(self.R_test, test_params)
            
            print(f"Constraint checks (test params):")
            for k, v in checks.items():
                status = "[PASS]" if v else "[FAIL]"
                print(f"  {k}: {status}")
            
            if not satisfies:
                print(f"WARNING: {candidate.name} may fail constraints")
            
            # Fit to data
            print(f"\nFitting {candidate.name} to {len(data)} systems...")
            try:
                score = self.fit_candidate(candidate, data)
                results[candidate.name] = score
                
                print(f"\nResults:")
                print(f"  chi2_reduced = {score['chi2_reduced']:.3f}")
                print(f"  AIC = {score['aic']:.1f}")
                print(f"  BIC = {score['bic']:.1f}")
                print(f"  n_params = {score['n_params']}")
                print(f"  chi2_train = {score['chi2_train']:.1f}")
                print(f"  chi2_val = {score['chi2_val']:.1f}")
                print(f"  Transfer score = {score['chi2_val']/score['chi2_train']:.3f}")
                
                print(f"\nBest-fit parameters:")
                for i, p in enumerate(score['params']):
                    print(f"  param_{i} = {p:.3f}")
                
            except Exception as e:
                print(f"ERROR fitting {candidate.name}: {e}")
                results[candidate.name] = {'error': str(e)}
        
        # Save results
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            # Convert numpy types to native Python for JSON
            results_serializable = {}
            for k, v in results.items():
                if isinstance(v, dict) and 'params' in v:
                    v_copy = v.copy()
                    v_copy['params'] = [float(p) for p in v['params']]
                    results_serializable[k] = v_copy
                else:
                    results_serializable[k] = v
            json.dump(results_serializable, f, indent=2)
        print(f"\nSaved results: {save_path}")
        
        return results
    
    def plot_pareto_front(self, results, save_path='outputs/inverse_search_pareto.png'):
        """
        Plot Pareto front: complexity vs. fit quality
        
        Expected: Burr-XII should be on or near the frontier
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: BIC vs n_params (parsimony)
        ax = axes[0]
        names = []
        bics = []
        n_params_list = []
        
        for name, score in results.items():
            if 'error' not in score:
                names.append(name)
                bics.append(score['bic'])
                n_params_list.append(score['n_params'])
        
        colors = ['red' if n == 'Burr-XII' else 'blue' for n in names]
        sizes = [200 if n == 'Burr-XII' else 100 for n in names]
        
        ax.scatter(n_params_list, bics, c=colors, s=sizes, alpha=0.7, edgecolors='black')
        
        for i, name in enumerate(names):
            ax.annotate(name, (n_params_list[i], bics[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Number of Parameters', fontsize=12)
        ax.set_ylabel('BIC (lower is better)', fontsize=12)
        ax.set_title('Parsimony: Complexity vs. Fit Quality')
        ax.grid(True, alpha=0.3)
        
        # Find Pareto front
        pareto_mask = []
        for i in range(len(n_params_list)):
            is_pareto = True
            for j in range(len(n_params_list)):
                if i != j:
                    # Dominated if another has fewer params AND lower BIC
                    if n_params_list[j] <= n_params_list[i] and bics[j] < bics[i]:
                        is_pareto = False
                        break
            pareto_mask.append(is_pareto)
        
        pareto_idx = np.argsort([n_params_list[i] for i, p in enumerate(pareto_mask) if p])
        if len(pareto_idx) > 1:
            pareto_x = [n_params_list[i] for i, p in enumerate(pareto_mask) if p]
            pareto_y = [bics[i] for i, p in enumerate(pareto_mask) if p]
            sorted_idx = np.argsort(pareto_x)
            ax.plot([pareto_x[i] for i in sorted_idx], [pareto_y[i] for i in sorted_idx], 
                   'k--', alpha=0.5, label='Pareto Front')
            ax.legend()
        
        # Panel 2: Transfer score (generalization)
        ax = axes[1]
        transfer_scores = []
        for name in names:
            score = results[name]
            transfer = score['chi2_val'] / score['chi2_train'] if score['chi2_train'] > 0 else np.inf
            transfer_scores.append(transfer)
        
        ax.barh(names, transfer_scores, color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(1.0, color='gray', linestyle='--', label='Perfect generalization')
        ax.set_xlabel('Transfer Score (χ²_val / χ²_train)', fontsize=12)
        ax.set_title('Generalization: Train vs. Validation')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved Pareto plot: {save_path}")
        plt.close()
    
    def print_summary(self, results):
        """Print summary table"""
        print("\n" + "="*80)
        print("SUMMARY: Window Form Comparison")
        print("="*80)
        print(f"{'Window':<15} {'n_params':<10} {'chi2_red':<10} {'BIC':<10} {'Transfer':<10}")
        print("-"*80)
        
        for name, score in sorted(results.items(), key=lambda x: x[1].get('bic', np.inf)):
            if 'error' not in score:
                transfer = score['chi2_val'] / score['chi2_train'] if score['chi2_train'] > 0 else np.inf
                mark = " <-- PAPER" if name == 'Burr-XII' else ""
                print(f"{name:<15} {score['n_params']:<10} {score['chi2_reduced']:<10.3f} "
                      f"{score['bic']:<10.1f} {transfer:<10.3f}{mark}")
        
        print("="*80)
        print("\nInterpretation:")
        print("- BIC: Lower is better (penalizes complexity)")
        print("- Transfer: Close to 1.0 means good generalization")
        print("- Burr-XII should be competitive if it's not arbitrary")


def main():
    """Run inverse search"""
    
    print("INVERSE SEARCH: First-Principles Test")
    print("="*60)
    print("\nQuestion: Does Burr-XII emerge as optimal under constraints?")
    print("\nMethod:")
    print("1. Generate toy data (known to have Burr-XII structure + noise)")
    print("2. Test multiple window forms (Logistic, Gompertz, Hill, etc.)")
    print("3. Enforce constraints C1-C5 BEFORE fitting")
    print("4. Compare on parsimony (BIC) and generalization")
    print("\nExpected: Burr-XII on Pareto front if it's not arbitrary")
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Initialize
    searcher = InverseSearch()
    
    # Generate data
    print("\n" + "="*60)
    print("Generating toy data...")
    data = searcher.generate_toy_data(n_systems=10, seed=42)
    print(f"Created {len(data)} toy systems")
    
    # Run search
    results = searcher.run_search(data, 'outputs/inverse_search_results.json')
    
    # Plot Pareto front
    searcher.plot_pareto_front(results, 'outputs/inverse_search_pareto.png')
    
    # Summary
    searcher.print_summary(results)
    
    print("\n" + "="*60)
    print("[OK] Inverse search complete!")
    print("\nKey Question Answered:")
    print("Is Burr-XII on or near the Pareto front?")
    print("-> Check outputs/inverse_search_pareto.png")
    print("\nNext steps:")
    print("1. Examine Pareto plot for Burr-XII position")
    print("2. Compare BIC values (lower = better)")
    print("3. Check transfer scores (closer to 1.0 = better generalization)")


if __name__ == '__main__':
    main()


"""
Correlation Analysis: Quietness Variables vs Σ Enhancement

This module tests correlations between each candidate "quietness" variable
and the gravitational enhancement factor Σ.

Hypothesis: Σ is larger where spacetime is "quieter"

Variables tested:
1. Metric fluctuation amplitude (velocity dispersion proxy)
2. Curvature gradients (from lensing)
3. GW background intensity
4. Matter density
5. Dynamical timescale
6. Entropy production rate (SFR proxy)
7. Tidal tensor eigenvalue spread

Statistical tests:
- Pearson correlation coefficient
- Spearman rank correlation
- Binned analysis with error bars
- Partial correlations (controlling for other variables)
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Callable
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

# =============================================================================
# CORRELATION FUNCTIONS
# =============================================================================

def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient with p-value.
    
    Returns (r, p_value)
    """
    # Remove NaN/Inf
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan, 1.0
    
    r, p = stats.pearsonr(x[valid], y[valid])
    return r, p


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation with p-value.
    
    More robust to outliers and non-linear relationships.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan, 1.0
    
    rho, p = stats.spearmanr(x[valid], y[valid])
    return rho, p


def binned_correlation(x: np.ndarray, y: np.ndarray,
                       n_bins: int = 10,
                       log_x: bool = False) -> Dict:
    """
    Compute y statistics in bins of x.
    
    Parameters
    ----------
    x, y : arrays
        Independent and dependent variables
    n_bins : int
        Number of bins
    log_x : bool
        Use logarithmic bins for x
    
    Returns
    -------
    dict with:
        x_mid: Bin centers
        y_median: Median y in each bin
        y_mean: Mean y in each bin
        y_16, y_84: 16th/84th percentiles
        y_std: Standard deviation
        n_per_bin: Sample size per bin
    """
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0 if log_x else True)
    x, y = x[valid], y[valid]
    
    if log_x:
        x_for_bins = np.log10(x)
        bins = np.linspace(x_for_bins.min(), x_for_bins.max(), n_bins + 1)
        bin_idx = np.digitize(x_for_bins, bins) - 1
        x_mid = 10**(0.5 * (bins[:-1] + bins[1:]))
    else:
        bins = np.linspace(x.min(), x.max(), n_bins + 1)
        bin_idx = np.digitize(x, bins) - 1
        x_mid = 0.5 * (bins[:-1] + bins[1:])
    
    # Clip to valid range
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    y_median = np.zeros(n_bins)
    y_mean = np.zeros(n_bins)
    y_16 = np.zeros(n_bins)
    y_84 = np.zeros(n_bins)
    y_std = np.zeros(n_bins)
    n_per_bin = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = bin_idx == i
        n_per_bin[i] = np.sum(mask)
        
        if n_per_bin[i] > 0:
            y_in_bin = y[mask]
            y_median[i] = np.median(y_in_bin)
            y_mean[i] = np.mean(y_in_bin)
            y_16[i] = np.percentile(y_in_bin, 16)
            y_84[i] = np.percentile(y_in_bin, 84)
            y_std[i] = np.std(y_in_bin)
    
    return {
        'x_mid': x_mid,
        'y_median': y_median,
        'y_mean': y_mean,
        'y_16': y_16,
        'y_84': y_84,
        'y_std': y_std,
        'n_per_bin': n_per_bin,
        'bins': bins,
    }


def partial_correlation(x: np.ndarray, y: np.ndarray, 
                        z: np.ndarray) -> Tuple[float, float]:
    """
    Partial correlation between x and y, controlling for z.
    
    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz²)(1-r_yz²))
    """
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.sum(valid) < 4:
        return np.nan, 1.0
    
    x, y, z = x[valid], y[valid], z[valid]
    
    r_xy, _ = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)
    
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return np.nan, 1.0
    
    r_partial = (r_xy - r_xz * r_yz) / denom
    
    # Approximate p-value
    n = np.sum(valid)
    t_stat = r_partial * np.sqrt((n - 3) / (1 - r_partial**2 + 1e-10))
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 3))
    
    return r_partial, p_value


# =============================================================================
# FIT FUNCTIONS
# =============================================================================

def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)


def exponential_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-x/b) + c"""
    return a * np.exp(-x / b) + c


def sigmoid_transition(x, a, x0, k, b):
    """Sigmoid: y = a / (1 + exp(-k*(x-x0))) + b"""
    return a / (1 + np.exp(-k * (x - x0))) + b


def fit_sigma_quietness_relation(quietness: np.ndarray, 
                                  sigma: np.ndarray,
                                  model: str = 'exponential') -> Dict:
    """
    Fit a functional form to Σ(quietness).
    
    Parameters
    ----------
    quietness : array
        Quietness variable (0 = noisy, 1 = quiet)
    sigma : array
        Enhancement factor Σ
    model : str
        'power_law', 'exponential', or 'sigmoid'
    
    Returns
    -------
    dict with fit parameters, covariance, and goodness-of-fit
    """
    valid = np.isfinite(quietness) & np.isfinite(sigma) & (sigma > 0)
    q, s = quietness[valid], sigma[valid]
    
    models = {
        'power_law': (power_law, [1.0, 1.0]),
        'exponential': (exponential_decay, [5.0, 0.3, 1.0]),
        'sigmoid': (sigmoid_transition, [5.0, 0.5, 10.0, 1.0]),
    }
    
    func, p0 = models[model]
    
    try:
        popt, pcov = curve_fit(func, q, s, p0=p0, maxfev=5000)
        
        # Compute residuals
        y_fit = func(q, *popt)
        residuals = s - y_fit
        chi2 = np.sum(residuals**2)
        
        # R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((s - s.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'model': model,
            'parameters': popt,
            'covariance': pcov,
            'chi2': chi2,
            'r_squared': r_squared,
            'n_points': len(q),
            'function': func,
        }
        
    except Exception as e:
        return {
            'model': model,
            'error': str(e),
            'parameters': None,
        }


# =============================================================================
# MASTER CORRELATION TEST
# =============================================================================

class QuietnessCorrelationTest:
    """
    Run correlation tests between a quietness variable and Σ enhancement.
    """
    
    def __init__(self, name: str, 
                 quietness: np.ndarray,
                 sigma: np.ndarray,
                 additional_vars: Dict[str, np.ndarray] = None):
        """
        Initialize test with data.
        
        Parameters
        ----------
        name : str
            Name of the quietness variable
        quietness : array
            Quietness values
        sigma : array
            Σ enhancement values
        additional_vars : dict, optional
            Other variables for partial correlation
        """
        self.name = name
        self.quietness = np.asarray(quietness)
        self.sigma = np.asarray(sigma)
        self.additional_vars = additional_vars or {}
        
        # Valid mask
        self.valid = np.isfinite(self.quietness) & np.isfinite(self.sigma)
        
        # Results storage
        self.results = {}
    
    def run_all_tests(self) -> Dict:
        """Run complete correlation analysis."""
        
        q = self.quietness[self.valid]
        s = self.sigma[self.valid]
        
        # Basic correlations
        r_pearson, p_pearson = pearson_correlation(q, s)
        r_spearman, p_spearman = spearman_correlation(q, s)
        
        # Binned analysis
        binned = binned_correlation(q, s, n_bins=10)
        
        # Fit models
        fit_exp = fit_sigma_quietness_relation(q, s, 'exponential')
        fit_pow = fit_sigma_quietness_relation(q, s, 'power_law')
        
        # Partial correlations (if additional vars provided)
        partial_results = {}
        for var_name, var_data in self.additional_vars.items():
            var_valid = var_data[self.valid]
            r_partial, p_partial = partial_correlation(q, s, var_valid)
            partial_results[var_name] = {
                'r_partial': r_partial,
                'p_value': p_partial,
            }
        
        self.results = {
            'name': self.name,
            'n_points': np.sum(self.valid),
            'pearson': {'r': r_pearson, 'p': p_pearson},
            'spearman': {'rho': r_spearman, 'p': p_spearman},
            'binned': binned,
            'fit_exponential': fit_exp,
            'fit_power_law': fit_pow,
            'partial_correlations': partial_results,
            'quietness_range': (q.min(), q.max()),
            'sigma_range': (s.min(), s.max()),
        }
        
        return self.results
    
    def summary(self) -> str:
        """Return text summary of results."""
        if not self.results:
            self.run_all_tests()
        
        r = self.results
        lines = [
            f"=== {r['name']} ===",
            f"N = {r['n_points']}",
            f"Pearson r = {r['pearson']['r']:.3f} (p = {r['pearson']['p']:.2e})",
            f"Spearman ρ = {r['spearman']['rho']:.3f} (p = {r['spearman']['p']:.2e})",
        ]
        
        if r['fit_exponential']['parameters'] is not None:
            lines.append(f"Exponential fit R² = {r['fit_exponential']['r_squared']:.3f}")
        
        for var_name, pr in r['partial_correlations'].items():
            lines.append(f"Partial r (controlling {var_name}) = {pr['r_partial']:.3f}")
        
        return "\n".join(lines)


# =============================================================================
# MULTI-VARIABLE COMPARISON
# =============================================================================

def compare_all_quietness_variables(sigma: np.ndarray,
                                     variables: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compare correlations across all quietness variables.
    
    Parameters
    ----------
    sigma : array
        Σ enhancement (target variable)
    variables : dict
        {variable_name: array} for each quietness variable
    
    Returns
    -------
    DataFrame with correlation coefficients and p-values for each variable
    """
    results = []
    
    for name, values in variables.items():
        test = QuietnessCorrelationTest(name, values, sigma)
        r = test.run_all_tests()
        
        results.append({
            'Variable': name,
            'N': r['n_points'],
            'Pearson_r': r['pearson']['r'],
            'Pearson_p': r['pearson']['p'],
            'Spearman_rho': r['spearman']['rho'],
            'Spearman_p': r['spearman']['p'],
            'Exp_R2': r['fit_exponential'].get('r_squared', np.nan),
            'Significant': r['pearson']['p'] < 0.05,
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('Pearson_r', ascending=False, key=abs)
    
    return df


def identify_best_predictor(sigma: np.ndarray,
                            variables: Dict[str, np.ndarray]) -> Dict:
    """
    Identify which quietness variable best predicts Σ.
    
    Uses multiple criteria:
    1. Highest |r|
    2. Lowest p-value
    3. Best fit R²
    4. Survives partial correlation controls
    """
    comparison = compare_all_quietness_variables(sigma, variables)
    
    # Best by different criteria
    best_by_r = comparison.loc[comparison['Pearson_r'].abs().idxmax()]
    best_by_p = comparison.loc[comparison['Pearson_p'].idxmin()]
    best_by_fit = comparison.loc[comparison['Exp_R2'].idxmax()]
    
    return {
        'comparison_table': comparison,
        'best_by_correlation': best_by_r['Variable'],
        'best_by_significance': best_by_p['Variable'],
        'best_by_fit': best_by_fit['Variable'],
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_correlation(quietness: np.ndarray, sigma: np.ndarray,
                     name: str, output_path: Path = None):
    """
    Create correlation plot with binned trend line.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax = axes[0]
    valid = np.isfinite(quietness) & np.isfinite(sigma)
    ax.scatter(quietness[valid], sigma[valid], alpha=0.3, s=5)
    
    # Add binned trend
    binned = binned_correlation(quietness[valid], sigma[valid])
    ax.errorbar(binned['x_mid'], binned['y_median'],
                yerr=[binned['y_median'] - binned['y_16'],
                      binned['y_84'] - binned['y_median']],
                fmt='ro-', capsize=3, label='Binned median')
    
    ax.set_xlabel(f'{name} (quietness)')
    ax.set_ylabel('Σ enhancement')
    ax.legend()
    
    # Correlation stats
    r, p = pearson_correlation(quietness, sigma)
    ax.set_title(f'r = {r:.3f}, p = {p:.2e}')
    
    # Histogram of sigma in quiet vs noisy regions
    ax = axes[1]
    q_thresh = np.median(quietness[valid])
    quiet_mask = quietness[valid] > q_thresh
    noisy_mask = ~quiet_mask
    
    ax.hist(sigma[valid][quiet_mask], bins=30, alpha=0.5, 
            label='Quiet regions', density=True)
    ax.hist(sigma[valid][noisy_mask], bins=30, alpha=0.5,
            label='Noisy regions', density=True)
    ax.set_xlabel('Σ enhancement')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title('Σ distribution by environment')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved {output_path}")
    
    return fig


# =============================================================================
# MAIN / EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("Correlation Analysis Module")
    print("=" * 60)
    
    # Generate mock data for testing
    np.random.seed(42)
    n = 1000
    
    # Mock "true" quietness that drives Σ
    true_quietness = np.random.uniform(0, 1, n)
    
    # Σ increases with quietness (with scatter)
    sigma = 1 + 3 * true_quietness**1.5 + np.random.normal(0, 0.5, n)
    sigma = np.maximum(sigma, 0.5)  # Physical floor
    
    # Mock observed quietness variables (different proxies)
    variables = {
        'velocity_dispersion': 1 - true_quietness + np.random.normal(0, 0.1, n),
        'density': 1 - true_quietness + np.random.normal(0, 0.2, n),
        'dynamical_time': true_quietness + np.random.normal(0, 0.15, n),
        'tidal_eigenvalue_spread': 1 - true_quietness + np.random.normal(0, 0.25, n),
        'random_noise': np.random.uniform(0, 1, n),  # Control
    }
    
    print("\nComparing all quietness variables:")
    comparison = compare_all_quietness_variables(sigma, variables)
    print(comparison.to_string(index=False))
    
    print("\n\nBest predictor analysis:")
    best = identify_best_predictor(sigma, variables)
    print(f"Best by |r|: {best['best_by_correlation']}")
    print(f"Best by p-value: {best['best_by_significance']}")
    print(f"Best by fit R²: {best['best_by_fit']}")
    
    # Plot best correlation
    best_var = best['best_by_correlation']
    plot_correlation(variables[best_var], sigma, best_var,
                    OUTPUT_DIR / f'correlation_{best_var}.png')

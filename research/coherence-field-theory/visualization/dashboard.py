"""
Visualization dashboard for coherence field theory results.

Create comprehensive plots showing model performance across all scales.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz


class CoherenceFieldDashboard:
    """
    Create multi-panel visualization of coherence field theory results.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        self.results = {}
        
    def add_cosmology_results(self, z, H_obs, H_model, dL_obs, dL_model, 
                             H_err=None, dL_err=None):
        """
        Add cosmology comparison results.
        
        Parameters:
        -----------
        z : array
            Redshifts
        H_obs, H_model : arrays
            Observed and model H(z)
        dL_obs, dL_model : arrays
            Observed and model d_L(z)
        H_err, dL_err : arrays
            Observational uncertainties
        """
        self.results['cosmology'] = {
            'z': z,
            'H_obs': H_obs,
            'H_model': H_model,
            'H_err': H_err,
            'dL_obs': dL_obs,
            'dL_model': dL_model,
            'dL_err': dL_err
        }
        
    def add_galaxy_results(self, name, r, v_obs, v_model, v_baryon, v_err=None):
        """
        Add galaxy rotation curve results.
        
        Parameters:
        -----------
        name : str
            Galaxy name
        r : array
            Radii
        v_obs, v_model, v_baryon : arrays
            Observed, total model, and baryon-only velocities
        v_err : array
            Uncertainties
        """
        if 'galaxies' not in self.results:
            self.results['galaxies'] = []
        
        self.results['galaxies'].append({
            'name': name,
            'r': r,
            'v_obs': v_obs,
            'v_model': v_model,
            'v_baryon': v_baryon,
            'v_err': v_err
        })
    
    def add_cluster_results(self, name, R, Sigma_obs, Sigma_model, 
                           Sigma_NFW, Sigma_coh, Sigma_err=None):
        """
        Add cluster lensing results.
        
        Parameters:
        -----------
        name : str
            Cluster name
        R : array
            Projected radii
        Sigma_obs, Sigma_model, Sigma_NFW, Sigma_coh : arrays
            Surface densities
        Sigma_err : array
            Uncertainties
        """
        if 'clusters' not in self.results:
            self.results['clusters'] = []
        
        self.results['clusters'].append({
            'name': name,
            'R': R,
            'Sigma_obs': Sigma_obs,
            'Sigma_model': Sigma_model,
            'Sigma_NFW': Sigma_NFW,
            'Sigma_coh': Sigma_coh,
            'Sigma_err': Sigma_err
        })
    
    def create_full_dashboard(self, savefig='coherence_field_dashboard.png'):
        """
        Create comprehensive multi-panel dashboard.
        
        Parameters:
        -----------
        savefig : str
            Filename to save figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Cosmology
        if 'cosmology' in self.results:
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_cosmology(ax1, ax2, ax3)
        
        # Row 2: Galaxies
        if 'galaxies' in self.results:
            n_gal = min(3, len(self.results['galaxies']))
            for i in range(n_gal):
                ax = fig.add_subplot(gs[1, i])
                self._plot_galaxy(ax, i)
        
        # Row 3: Clusters
        if 'clusters' in self.results:
            n_cluster = min(3, len(self.results['clusters']))
            for i in range(n_cluster):
                ax = fig.add_subplot(gs[2, i])
                self._plot_cluster(ax, i)
        
        plt.suptitle('Coherence Field Theory: Multi-Scale Validation', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved: {savefig}")
        plt.show()
    
    def _plot_cosmology(self, ax1, ax2, ax3):
        """Plot cosmology panels."""
        data = self.results['cosmology']
        z = data['z']
        
        # Panel 1: H(z)
        if data['H_obs'] is not None:
            if data['H_err'] is not None:
                ax1.errorbar(z, data['H_obs'], yerr=data['H_err'], 
                           fmt='o', label='Observed', alpha=0.6, markersize=4)
            else:
                ax1.plot(z, data['H_obs'], 'o', label='Observed', alpha=0.6)
        
        if data['H_model'] is not None:
            ax1.plot(z, data['H_model'], '-', label='Coherence model', linewidth=2)
        
        ax1.set_xlabel('Redshift z', fontsize=10)
        ax1.set_ylabel('H(z) / H₀', fontsize=10)
        ax1.set_title('Hubble Parameter', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        
        # Panel 2: d_L(z)
        if data['dL_err'] is not None:
            ax2.errorbar(z, data['dL_obs'], yerr=data['dL_err'], 
                       fmt='o', label='Observed', alpha=0.6, markersize=4)
        else:
            ax2.plot(z, data['dL_obs'], 'o', label='Observed', alpha=0.6)
        
        ax2.plot(z, data['dL_model'], '-', label='Coherence model', linewidth=2)
        ax2.set_xlabel('Redshift z', fontsize=10)
        ax2.set_ylabel('Luminosity Distance', fontsize=10)
        ax2.set_title('Distance-Redshift Relation', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        
        # Panel 3: Residuals
        residual = (data['dL_obs'] - data['dL_model']) / data['dL_obs'] * 100
        ax3.plot(z, residual, 'o-', markersize=4)
        ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Redshift z', fontsize=10)
        ax3.set_ylabel('Residual (%)', fontsize=10)
        ax3.set_title('Model Residuals', fontsize=11, fontweight='bold')
        ax3.grid(alpha=0.3)
    
    def _plot_galaxy(self, ax, index):
        """Plot galaxy rotation curve."""
        gal = self.results['galaxies'][index]
        
        if gal['v_err'] is not None:
            ax.errorbar(gal['r'], gal['v_obs'], yerr=gal['v_err'], 
                       fmt='o', label='Observed', alpha=0.6, markersize=4, capsize=3)
        else:
            ax.plot(gal['r'], gal['v_obs'], 'o', label='Observed', alpha=0.6)
        
        ax.plot(gal['r'], gal['v_baryon'], '--', label='Baryons only', 
               linewidth=2, alpha=0.7)
        ax.plot(gal['r'], gal['v_model'], '-', label='With coherence field', 
               linewidth=2.5)
        
        ax.set_xlabel('Radius (kpc)', fontsize=10)
        ax.set_ylabel('v_circ (km/s)', fontsize=10)
        ax.set_title(f"Galaxy: {gal['name']}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Compute chi-squared
        if gal['v_err'] is not None:
            chi2 = np.sum(((gal['v_obs'] - gal['v_model']) / gal['v_err'])**2)
            chi2_red = chi2 / (len(gal['r']) - 2)
            ax.text(0.05, 0.95, f"$\\chi^2_{{red}}$ = {chi2_red:.2f}", 
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_cluster(self, ax, index):
        """Plot cluster lensing profile."""
        cluster = self.results['clusters'][index]
        
        if cluster['Sigma_err'] is not None:
            ax.errorbar(cluster['R'], cluster['Sigma_obs'], yerr=cluster['Sigma_err'],
                       fmt='o', label='Observed', alpha=0.6, markersize=4, capsize=3)
        else:
            ax.loglog(cluster['R'], cluster['Sigma_obs'], 'o', label='Observed', alpha=0.6)
        
        ax.loglog(cluster['R'], cluster['Sigma_NFW'], '--', label='NFW only',
                 linewidth=2, alpha=0.7)
        ax.loglog(cluster['R'], cluster['Sigma_model'], '-', 
                 label='NFW + coherence', linewidth=2.5)
        
        ax.set_xlabel('Projected Radius (kpc)', fontsize=10)
        ax.set_ylabel('Σ (M☉/kpc²)', fontsize=10)
        ax.set_title(f"Cluster: {cluster['name']}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')


def example_dashboard():
    """Create example dashboard."""
    print("=" * 70)
    print("Creating Example Dashboard")
    print("=" * 70)
    
    dashboard = CoherenceFieldDashboard()
    
    # Add cosmology
    z = np.linspace(0.1, 2.0, 40)
    H_obs = np.sqrt(0.3 * (1+z)**3 + 0.7) * (1 + 0.02 * np.random.randn(len(z)))
    H_model = np.sqrt(0.3 * (1+z)**3 + 0.7)
    
    from scipy.integrate import trapz
    dL_obs = []
    dL_model = []
    for zi in z:
        zs = np.linspace(0, zi, 500)
        chi = trapz(1.0 / np.sqrt(0.3 * (1+zs)**3 + 0.7), zs)
        dL = (1 + zi) * chi
        dL_model.append(dL)
        dL_obs.append(dL * (1 + 0.03 * np.random.randn()))
    
    dashboard.add_cosmology_results(z, H_obs, H_model, 
                                   np.array(dL_obs), np.array(dL_model),
                                   H_err=H_obs*0.05, dL_err=np.array(dL_model)*0.03)
    
    # Add galaxies
    for i in range(3):
        r = np.linspace(1, 25, 30)
        v_baryon = 120 * np.sqrt(r / (r + 3))
        v_model = np.sqrt(v_baryon**2 + 100**2 * (1 - np.exp(-r/5)))
        v_obs = v_model + np.random.normal(0, 5, len(r))
        
        dashboard.add_galaxy_results(f'Galaxy{i+1}', r, v_obs, v_model, v_baryon, 
                                    v_err=np.ones_like(r)*5)
    
    # Add clusters
    for i in range(3):
        R = np.logspace(1.5, 3, 25)
        Sigma_NFW = 1e10 / R**2
        Sigma_coh = 5e9 / (R + 100)
        Sigma_model = Sigma_NFW + Sigma_coh
        Sigma_obs = Sigma_model * (1 + 0.1 * np.random.randn(len(R)))
        
        dashboard.add_cluster_results(f'Cluster{i+1}', R, Sigma_obs, Sigma_model,
                                     Sigma_NFW, Sigma_coh, 
                                     Sigma_err=Sigma_model*0.1)
    
    # Create dashboard
    dashboard.create_full_dashboard(savefig='example_dashboard.png')
    
    print("\n" + "=" * 70)
    print("Dashboard created successfully!")
    print("=" * 70)


if __name__ == '__main__':
    example_dashboard()


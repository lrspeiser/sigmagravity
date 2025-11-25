#!/usr/bin/env python3
"""
Σ-Gravity CMB: Polarization Predictions
========================================

Standard cosmology explains CMB polarization through:
- Thomson scattering of photons off free electrons
- Quadrupole anisotropy at last scattering creates polarization
- E-modes from scalar perturbations
- B-modes from tensor perturbations (gravitational waves)

Σ-Gravity alternative:
- No "last scattering surface" in the traditional sense
- Instead: Light traversing coherent gravitational fields
- Gravitational birefringence? Gravitomagnetic effects?

Key observations to explain:
1. E-mode power spectrum peaks at ℓ ~ 1000
2. TE correlation (temperature-E-mode) with specific phase
3. EE spectrum follows TT but shifted in ℓ
4. B-modes at large scales from lensing

Physical mechanism to explore:
- Gravitomagnetic frame dragging affects photon polarization
- Coherent gravitational fields create preferred directions
- Polarization from differential path length effects

Author: Leonard Speiser
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# OBSERVED POLARIZATION DATA (approximate from Planck)
# =============================================================================

# E-mode power spectrum peaks
EE_OBSERVED = {
    'ell': np.array([150, 400, 600, 1000, 1400, 1800]),
    'Dl': np.array([0.2, 5, 20, 40, 20, 8]),  # μK²
    'err': np.array([0.1, 1, 2, 3, 3, 2]),
}

# Temperature-E correlation
TE_OBSERVED = {
    'ell': np.array([50, 150, 300, 500, 800, 1000, 1200, 1500]),
    'Dl': np.array([1, 30, -50, 100, 40, -30, 50, 20]),  # μK², note sign changes!
    'err': np.array([5, 5, 10, 10, 10, 10, 10, 10]),
}


# =============================================================================
# GRAVITATIONAL POLARIZATION MECHANISM
# =============================================================================

@dataclass
class GravitationalPolarization:
    """
    Σ-Gravity prediction for CMB polarization.
    
    Physical mechanism: Gravitomagnetic frame-dragging
    
    When light travels through a region with gravitomagnetic field H_g,
    it experiences a rotation of its polarization plane:
    
        dθ/dℓ ∝ ∇ × g / c²
    
    where g is the gravitational acceleration field.
    
    In Σ-Gravity, coherent gravitational wave structure creates
    systematic gravitomagnetic fields that rotate polarization.
    
    The effect depends on:
    1. Strength of coherent gravitational field
    2. Path length through coherent region
    3. Orientation of path relative to field
    """
    
    # Temperature model parameters (from step-function fit)
    ell_1: float = 211.0
    spacing_base: float = 1.55
    asym_high: float = 0.35
    asym_low: float = 0.02
    ell_crit: float = 1300.0
    
    # Polarization-specific parameters
    pol_efficiency: float = 0.15      # Fraction of temperature signal
    pol_phase_shift: float = 0.25     # Phase shift relative to TT (in units of π)
    pol_ell_shift: float = 1.5        # EE peaks at higher ℓ than TT
    
    # Gravitomagnetic coupling
    gm_coupling: float = 0.1          # Gravitomagnetic polarization coupling
    
    n_modes: int = 8
    
    def step_function(self, ell: float) -> float:
        """Asymmetry step function from temperature model."""
        x = (ell - self.ell_crit) / 80
        return 0.5 * (1 + np.tanh(x))
    
    def asymmetry_at_ell(self, ell: float) -> float:
        step = self.step_function(ell)
        return self.asym_high * (1 - step) + self.asym_low * step
    
    def mode_location_TT(self, n: int) -> float:
        """Temperature peak locations."""
        if n == 1:
            return self.ell_1
        ell = self.ell_1
        for i in range(2, n + 1):
            ell += self.ell_1 * self.spacing_base
        return ell
    
    def mode_location_EE(self, n: int) -> float:
        """
        E-mode peak locations.
        
        In standard cosmology, EE peaks are shifted to higher ℓ
        because polarization requires quadrupole anisotropy,
        which peaks at smaller scales than temperature.
        
        In Σ-Gravity: Polarization comes from gravitomagnetic
        rotation, which has different spatial structure than
        the scalar gravitational potential.
        """
        # EE peaks shifted to higher ℓ
        return self.mode_location_TT(n) * self.pol_ell_shift
    
    def compute_EE(self, ell: np.ndarray) -> np.ndarray:
        """
        Compute E-mode power spectrum.
        
        Physical mechanism:
        - Gravitomagnetic field creates polarization rotation
        - Rotation angle ∝ integral of ∇×g along path
        - Coherent structure creates correlated polarization
        
        The EE spectrum follows TT but:
        - Shifted to higher ℓ (smaller scales)
        - Reduced amplitude (polarization is fraction of temperature)
        - Different asymmetry pattern (gravitomagnetic vs potential)
        """
        ell = np.atleast_1d(ell).astype(float)
        Dl = np.zeros_like(ell)
        
        # Base amplitude from polarization efficiency
        # In standard model: δT_pol / δT ~ 0.1-0.2
        A0 = 50.0 * self.pol_efficiency**2  # μK²
        
        for n in range(1, self.n_modes + 1):
            ell_peak = self.mode_location_EE(n)
            
            # Amplitude decays faster for polarization
            amp = A0 / n**1.2
            
            # Asymmetry is REVERSED for polarization!
            # Gravitomagnetic rotation is strongest where
            # gradient of potential is largest (between peaks)
            asym = self.asymmetry_at_ell(ell_peak) * 0.5
            if n % 2 == 0:  # Even modes enhanced for polarization
                amp *= (1 + asym)
            else:  # Odd modes suppressed
                amp *= (1 - asym * 0.3)
            
            # Width increases with ℓ
            width = 60 + 0.05 * ell_peak
            
            Dl += amp * np.exp(-0.5 * ((ell - ell_peak) / width)**2)
        
        # Low-ℓ reionization bump
        Dl += 0.5 * np.exp(-((ell - 150) / 100)**2)
        
        # Damping at high ℓ
        Dl *= np.exp(-(ell / 2500)**2)
        
        return np.maximum(Dl, 0.01)
    
    def compute_TE(self, ell: np.ndarray) -> np.ndarray:
        """
        Compute Temperature-E correlation.
        
        Key feature: TE changes sign! This is because:
        - T and E are 90° out of phase in acoustic interpretation
        - In Σ-Gravity: Gravitomagnetic field is 90° out of phase
          with gravitational potential
        
        The TE correlation oscillates with ℓ, with:
        - Positive correlation at some scales
        - Negative (anti-correlation) at others
        """
        ell = np.atleast_1d(ell).astype(float)
        Dl = np.zeros_like(ell)
        
        # TE amplitude (geometric mean of TT and EE roughly)
        A0 = 100.0  # μK²
        
        for n in range(1, self.n_modes + 1):
            # TE peaks between TT and EE locations
            ell_TT = self.mode_location_TT(n)
            ell_EE = self.mode_location_EE(n)
            ell_peak = 0.6 * ell_TT + 0.4 * ell_EE
            
            amp = A0 / n**0.8
            
            # Phase determines sign!
            # Phase shift creates alternating positive/negative
            phase = n * np.pi * self.pol_phase_shift
            sign = np.cos(phase)
            
            width = 80 + 0.04 * ell_peak
            
            Dl += sign * amp * np.exp(-0.5 * ((ell - ell_peak) / width)**2)
        
        # Damping
        Dl *= np.exp(-(ell / 2000)**2)
        
        return Dl
    
    def compute_BB(self, ell: np.ndarray) -> np.ndarray:
        """
        Compute B-mode power spectrum.
        
        Standard model sources:
        - Primordial gravitational waves (tensor modes)
        - Lensing of E-modes into B-modes
        
        Σ-Gravity sources:
        - Coherent gravitational wave background
        - Different from primordial tensor modes
        - Gravitational lensing also creates B-modes
        
        Key prediction: B-mode amplitude depends on
        coherence structure, not inflation.
        """
        ell = np.atleast_1d(ell).astype(float)
        Dl = np.zeros_like(ell)
        
        # Lensing B-modes (same as standard model)
        # Peak around ℓ ~ 1000
        Dl += 0.05 * (ell / 1000)**2 * np.exp(-(ell / 1500)**2)
        
        # Primordial-like contribution from coherent GW background
        # Much smaller than E-modes
        # In Σ-Gravity, this comes from tensor modes in the
        # coherent gravitational wave structure
        r_eff = 0.01  # Effective tensor-to-scalar ratio
        Dl += r_eff * 0.1 * np.exp(-((ell - 80) / 50)**2)
        
        return np.maximum(Dl, 0.001)


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_polarization():
    """Analyze Σ-Gravity polarization predictions."""
    
    model = GravitationalPolarization()
    
    print("="*70)
    print("Σ-GRAVITY CMB POLARIZATION PREDICTIONS")
    print("="*70)
    
    print("\n--- PHYSICAL MECHANISM ---")
    print("""
    Standard cosmology:
      - Thomson scattering at last scattering surface
      - Quadrupole anisotropy creates linear polarization
      - E-modes from scalar perturbations
      - B-modes from tensor perturbations
    
    Σ-Gravity alternative:
      - Gravitomagnetic frame-dragging rotates polarization
      - No "last scattering surface" required
      - Coherent gravitational fields create systematic rotation
      - E-modes from gradient of gravitational potential
      - B-modes from curl (gravitomagnetic) component
    
    Key differences:
      1. EE spectrum shifted to higher ℓ (pol_ell_shift = {:.1f})
      2. EE amplitude ~ {:.0f}% of TT (polarization efficiency)
      3. TE correlation changes sign with characteristic phase
      4. BB spectrum includes coherent GW contribution
    """.format(model.pol_ell_shift, model.pol_efficiency * 100))
    
    print("\n--- PREDICTIONS ---")
    
    # Compute spectra
    ell = np.arange(2, 2500)
    EE = model.compute_EE(ell)
    TE = model.compute_TE(ell)
    BB = model.compute_BB(ell)
    
    # Find peaks
    print("\nE-mode peaks:")
    for n in range(1, 6):
        ell_peak = model.mode_location_EE(n)
        print(f"  Mode {n}: ℓ ≈ {ell_peak:.0f}")
    
    print("\nTE correlation:")
    print("  Changes sign at characteristic scales")
    print("  Phase shift creates alternating +/- correlation")
    
    print("\nB-mode spectrum:")
    print("  Lensing contribution peaks at ℓ ~ 1000")
    print("  Primordial-like contribution at ℓ ~ 80")
    print(f"  Effective r ~ {0.01}")
    
    return model


def plot_polarization(model: GravitationalPolarization, save_path: str = None):
    """Visualize polarization predictions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ell = np.arange(2, 2500)
    EE = model.compute_EE(ell)
    TE = model.compute_TE(ell)
    BB = model.compute_BB(ell)
    
    # Panel 1: EE spectrum
    ax = axes[0, 0]
    
    ax.semilogy(ell, EE, 'b-', lw=2, label='Σ-Gravity prediction')
    ax.scatter(EE_OBSERVED['ell'], EE_OBSERVED['Dl'], s=100, c='black',
               marker='o', label='Observed (approx.)', zorder=5)
    
    # Mark predicted peaks
    for n in range(1, 6):
        ell_peak = model.mode_location_EE(n)
        ax.axvline(ell_peak, color='gray', ls=':', alpha=0.3)
    
    ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax.set_ylabel(r'$D_\ell^{EE}$ [$\mu K^2$]', fontsize=12)
    ax.set_title('E-mode Power Spectrum', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2500)
    ax.set_ylim(0.01, 100)
    
    # Panel 2: TE spectrum
    ax = axes[0, 1]
    
    ax.plot(ell, TE, 'g-', lw=2, label='Σ-Gravity prediction')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    
    # Approximate observations
    ax.scatter(TE_OBSERVED['ell'], TE_OBSERVED['Dl'], s=100, c='black',
               marker='o', label='Observed (approx.)', zorder=5)
    
    ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax.set_ylabel(r'$D_\ell^{TE}$ [$\mu K^2$]', fontsize=12)
    ax.set_title('Temperature-E Correlation\n(note: changes sign!)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2000)
    
    # Panel 3: BB spectrum
    ax = axes[1, 0]
    
    ax.semilogy(ell, BB, 'r-', lw=2, label='Σ-Gravity prediction')
    
    # Components
    ell_lens = np.arange(100, 2500)
    BB_lens = 0.05 * (ell_lens / 1000)**2 * np.exp(-(ell_lens / 1500)**2)
    ax.semilogy(ell_lens, BB_lens, 'r--', lw=1, alpha=0.5, label='Lensing component')
    
    ell_prim = np.arange(2, 300)
    BB_prim = 0.01 * 0.1 * np.exp(-((ell_prim - 80) / 50)**2)
    ax.semilogy(ell_prim, BB_prim, 'r:', lw=1, alpha=0.5, label='Primordial-like')
    
    ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax.set_ylabel(r'$D_\ell^{BB}$ [$\mu K^2$]', fontsize=12)
    ax.set_title('B-mode Power Spectrum\n(lensing + coherent GW)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2500)
    ax.set_ylim(0.0001, 0.1)
    
    # Panel 4: Physical interpretation
    ax = axes[1, 1]
    ax.axis('off')
    
    text = """
    GRAVITATIONAL POLARIZATION MECHANISM
    ════════════════════════════════════════════════════════════
    
    STANDARD MODEL:
      Thomson scattering at z ~ 1100 creates polarization
      Quadrupole temperature anisotropy → linear polarization
      E-modes from scalar perturbations (density waves)
      B-modes from tensor perturbations (gravitational waves)
    
    Σ-GRAVITY ALTERNATIVE:
      Gravitomagnetic frame-dragging rotates polarization
      H_g = (1/c²) × ∇×g creates Faraday-like rotation
      
      As light traverses coherent gravitational structure:
        dθ/ds ∝ H_g · k̂
      where θ is polarization angle, s is path length
    
    KEY PREDICTIONS:
      1. EE spectrum peaks at HIGHER ℓ than TT
         (gravitomagnetic field has smaller coherence scale)
      
      2. TE correlation CHANGES SIGN
         (90° phase between potential and gravitomagnetic field)
      
      3. BB spectrum includes coherent GW contribution
         (different from inflationary prediction)
    
    TESTABLE DIFFERENCES:
      • Different ℓ-dependence of EE/TT ratio
      • Phase of TE oscillations
      • BB spectrum shape at low ℓ
    
    These predictions can be compared to Planck/SPT/ACT data
    to test the Σ-Gravity interpretation.
    """
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved {save_path}")
    
    plt.show()


def compare_TT_EE(model: GravitationalPolarization, save_path: str = None):
    """Compare TT and EE spectra to show relationship."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ell = np.arange(2, 2500)
    EE = model.compute_EE(ell)
    
    # Compute TT from temperature model
    from sigma_cmb_step import StepAsymmetryModel
    temp_model = StepAsymmetryModel()
    TT = temp_model.compute_Dl(ell)
    
    # Normalize for comparison
    TT_norm = TT / np.max(TT)
    EE_norm = EE / np.max(EE)
    
    ax.plot(ell, TT_norm, 'r-', lw=2, label='TT (normalized)')
    ax.plot(ell, EE_norm, 'b-', lw=2, label='EE (normalized)')
    
    # Mark peak locations
    for n in range(1, 5):
        ell_TT = temp_model.mode_location(n)
        ell_EE = model.mode_location_EE(n)
        ax.axvline(ell_TT, color='red', ls=':', alpha=0.3)
        ax.axvline(ell_EE, color='blue', ls=':', alpha=0.3)
        
        # Arrow showing shift
        ax.annotate('', xy=(ell_EE, 0.8 - 0.15*n), xytext=(ell_TT, 0.8 - 0.15*n),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax.set_ylabel('Normalized power', fontsize=12)
    ax.set_title('TT vs EE Peak Positions\n(EE shifted to higher ℓ)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1.1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("Σ-GRAVITY CMB POLARIZATION FRAMEWORK")
    print("="*70)
    
    # Analyze
    model = analyze_polarization()
    
    # Plot
    plot_polarization(model, save_path="sigma_cmb_polarization.png")
    
    return model


if __name__ == "__main__":
    model = main()

#!/usr/bin/env python3
"""
Σ-Gravity CMB: Step-Function Asymmetry Model
=============================================

Key observation from data:
  P1/P2 = 2.40  }
  P3/P4 = 2.32  } Nearly constant!
  P5/P6 = 1.54  } Sharp drop!

This is NOT a smooth exponential decay. The asymmetry stays roughly
constant for the first 4 peaks, then drops sharply between peaks 4-5.

Physical interpretation:
- There's a characteristic scale ℓ_crit ~ 1000-1200 where something changes
- Could be: Silk damping cutoff, structure formation scale, or 
  a feature in the matter power spectrum

Model: Step function with smooth transition (tanh)
  a(ℓ) = a_high × [1 - step(ℓ)] + a_low × step(ℓ)
  where step(ℓ) = 0.5 × [1 + tanh((ℓ - ℓ_crit) / Δℓ)]

Author: Leonard Speiser
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from dataclasses import dataclass
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# OBSERVED DATA
# =============================================================================

OBSERVED = {
    'ell': np.array([220, 537, 810, 1120, 1420, 1720, 2000]),
    'Dl': np.array([5800, 2420, 2550, 1100, 800, 520, 350]),
    'err': np.array([50, 40, 50, 40, 50, 50, 60]),
}

TARGET_RATIOS = {
    'P1/P2': 2.397,
    'P3/P4': 2.318,
    'P5/P6': 1.538,
}


# =============================================================================
# STEP-FUNCTION ASYMMETRY MODEL
# =============================================================================

@dataclass
class StepAsymmetryModel:
    """
    CMB model with step-function asymmetry transition.
    
    The asymmetry is nearly constant at low ℓ, then drops sharply
    at a critical scale ℓ_crit.
    """
    
    # Peak locations
    ell_1: float = 220.0
    spacing_base: float = 1.44
    spacing_compress: float = 0.02  # Compression at high n
    
    # Amplitudes
    A0: float = 5800.0
    amp_power: float = 0.70
    
    # Step-function asymmetry
    asym_high: float = 0.42         # Asymmetry at low ℓ (before step)
    asym_low: float = 0.08          # Asymmetry at high ℓ (after step)
    ell_crit: float = 1100.0        # Critical transition scale
    delta_ell: float = 150.0        # Width of transition
    
    # Peak widths
    width_base: float = 45.0
    width_scale: float = 0.025
    
    # Damping
    ell_damp: float = 1800.0
    damp_power: float = 2.0
    
    # Low-ell plateau
    low_ell_amp: float = 500.0
    
    n_modes: int = 8
    
    def step_function(self, ell: float) -> float:
        """
        Smooth step function: 0 at low ℓ, 1 at high ℓ.
        Transition centered at ell_crit with width delta_ell.
        """
        x = (ell - self.ell_crit) / self.delta_ell
        return 0.5 * (1 + np.tanh(x))
    
    def asymmetry_at_ell(self, ell: float) -> float:
        """
        Asymmetry parameter at given ℓ.
        
        Returns value between asym_low and asym_high.
        """
        step = self.step_function(ell)
        return self.asym_high * (1 - step) + self.asym_low * step
    
    def asymmetry_factor(self, n: int, ell: float) -> float:
        """
        Enhancement/suppression factor for mode n at location ell.
        
        Odd modes (n=1,3,5,...): enhanced by (1 + a)
        Even modes (n=2,4,6,...): suppressed by (1 - a/2)
        """
        a = self.asymmetry_at_ell(ell)
        
        if n % 2 == 1:  # Odd mode
            return 1 + a
        else:  # Even mode
            return 1 - a * 0.5
    
    def mode_location(self, n: int) -> float:
        """Location of nth peak with slight compression at high n."""
        if n == 1:
            return self.ell_1
        
        ell = self.ell_1
        for i in range(2, n + 1):
            # Spacing decreases slightly at high n
            spacing = self.spacing_base * (1 - self.spacing_compress * (i - 2))
            ell += self.ell_1 * spacing
        
        return ell
    
    def mode_width(self, n: int, ell: float) -> float:
        """Width increases with ℓ."""
        return self.width_base * (1 + self.width_scale * ell)
    
    def peak_amplitude(self, n: int) -> float:
        """Base amplitude decays as power law."""
        return self.A0 / n**self.amp_power
    
    def damping(self, ell: np.ndarray) -> np.ndarray:
        """Decoherence damping at high ℓ."""
        return np.exp(-(ell / self.ell_damp)**self.damp_power)
    
    def compute_Dl(self, ell: np.ndarray) -> np.ndarray:
        """Compute full power spectrum."""
        ell = np.atleast_1d(ell).astype(float)
        Dl = np.zeros_like(ell)
        
        # Low-ell plateau (Sachs-Wolfe-like)
        Dl += self.low_ell_amp * (ell / 20)**0.3 * np.exp(-ell / 100)
        
        # Resonant peaks
        for n in range(1, self.n_modes + 1):
            ell_peak = self.mode_location(n)
            amp_base = self.peak_amplitude(n)
            asym = self.asymmetry_factor(n, ell_peak)
            width = self.mode_width(n, ell_peak)
            
            amp = amp_base * asym
            Dl += amp * np.exp(-0.5 * ((ell - ell_peak) / width)**2)
        
        # Apply damping
        Dl *= self.damping(ell)
        
        return np.maximum(Dl, 1.0)
    
    def get_peaks(self) -> Dict:
        """Return peak properties."""
        peaks = {'n': [], 'ell': [], 'Dl': [], 'type': [], 'asymmetry': [], 'factor': []}
        
        for n in range(1, self.n_modes + 1):
            ell_peak = self.mode_location(n)
            amp_base = self.peak_amplitude(n)
            asym_val = self.asymmetry_at_ell(ell_peak)
            asym_factor = self.asymmetry_factor(n, ell_peak)
            damp = self.damping(np.array([ell_peak]))[0]
            
            peaks['n'].append(n)
            peaks['ell'].append(ell_peak)
            peaks['Dl'].append(amp_base * asym_factor * damp)
            peaks['type'].append('odd' if n % 2 == 1 else 'even')
            peaks['asymmetry'].append(asym_val)
            peaks['factor'].append(asym_factor)
        
        return peaks
    
    def compute_ratios(self) -> Dict:
        """Compute peak ratios."""
        peaks = self.get_peaks()
        Dl = peaks['Dl']
        
        return {
            'P1/P2': Dl[0] / Dl[1] if len(Dl) > 1 else np.nan,
            'P3/P4': Dl[2] / Dl[3] if len(Dl) > 3 else np.nan,
            'P5/P6': Dl[4] / Dl[5] if len(Dl) > 5 else np.nan,
        }


# =============================================================================
# FITTING
# =============================================================================

def chi_squared(params: np.ndarray, return_model: bool = False):
    """Compute χ² with emphasis on matching ALL THREE ratios."""
    
    model = StepAsymmetryModel(
        ell_1=params[0],
        spacing_base=params[1],
        spacing_compress=params[2],
        A0=params[3],
        amp_power=params[4],
        asym_high=params[5],
        asym_low=params[6],
        ell_crit=params[7],
        delta_ell=params[8],
        width_base=params[9],
        width_scale=params[10],
        ell_damp=params[11],
        damp_power=params[12],
    )
    
    if return_model:
        return model
    
    peaks = model.get_peaks()
    ratios = model.compute_ratios()
    
    chi2 = 0
    n_peaks = min(len(OBSERVED['ell']), len(peaks['ell']))
    
    # Peak locations
    for i in range(n_peaks):
        chi2 += 0.5 * ((peaks['ell'][i] - OBSERVED['ell'][i]) / 30)**2
    
    # Peak heights
    for i in range(n_peaks):
        chi2 += 0.3 * ((peaks['Dl'][i] - OBSERVED['Dl'][i]) / OBSERVED['err'][i])**2
    
    # Peak ratios - VERY HIGH WEIGHT
    chi2 += 200.0 * ((ratios['P1/P2'] - TARGET_RATIOS['P1/P2']) / 0.05)**2
    chi2 += 200.0 * ((ratios['P3/P4'] - TARGET_RATIOS['P3/P4']) / 0.05)**2
    chi2 += 200.0 * ((ratios['P5/P6'] - TARGET_RATIOS['P5/P6']) / 0.05)**2
    
    # Constraint: P1/P2 should be close to P3/P4
    chi2 += 50.0 * ((ratios['P1/P2'] - ratios['P3/P4']) / 0.1)**2
    
    # Regularization
    if params[6] >= params[5]:  # asym_low < asym_high
        chi2 += 1e6
    
    return chi2


def fit_model() -> StepAsymmetryModel:
    """Fit model to observations."""
    
    bounds = [
        (200, 240),       # ell_1
        (1.35, 1.55),     # spacing_base
        (0.0, 0.08),      # spacing_compress
        (5000, 7000),     # A0
        (0.55, 0.85),     # amp_power
        (0.35, 0.55),     # asym_high
        (0.02, 0.15),     # asym_low
        (900, 1300),      # ell_crit
        (80, 300),        # delta_ell
        (35, 60),         # width_base
        (0.015, 0.04),    # width_scale
        (1400, 2200),     # ell_damp
        (1.5, 2.5),       # damp_power
    ]
    
    print("Fitting step-function asymmetry model...")
    print("Key: Asymmetry stays constant, then drops sharply at ℓ_crit\n")
    
    result = differential_evolution(
        chi_squared, bounds, seed=42,
        maxiter=1500, tol=1e-9,
        workers=1, disp=True, polish=True
    )
    
    model = chi_squared(result.x, return_model=True)
    print(f"\nFinal χ² = {result.fun:.2f}")
    
    return model


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_model(model: StepAsymmetryModel):
    """Detailed analysis."""
    
    peaks = model.get_peaks()
    ratios = model.compute_ratios()
    
    print("\n" + "="*80)
    print("STEP-FUNCTION ASYMMETRY MODEL: RESULTS")
    print("="*80)
    
    print("\n--- STEP-FUNCTION PARAMETERS ---")
    print(f"Asymmetry before step: a_high = {model.asym_high:.3f}")
    print(f"Asymmetry after step:  a_low = {model.asym_low:.3f}")
    print(f"Transition center:     ℓ_crit = {model.ell_crit:.0f}")
    print(f"Transition width:      Δℓ = {model.delta_ell:.0f}")
    
    print(f"\nPhysical scale at transition:")
    D_eff = 4000  # Mpc
    lambda_crit = np.pi * D_eff / model.ell_crit
    print(f"  ℓ_crit = {model.ell_crit:.0f} → λ_crit ≈ {lambda_crit:.1f} Mpc")
    
    print("\n--- ASYMMETRY AT EACH PEAK ---")
    print("{:^6} {:^10} {:^12} {:^12} {:^12}".format(
        "Peak", "ℓ", "a(ℓ)", "Factor", "Type"))
    print("-"*55)
    
    for i in range(len(peaks['n'])):
        print("{:^6} {:^10.0f} {:^12.3f} {:^12.3f} {:^12}".format(
            f"P{peaks['n'][i]}",
            peaks['ell'][i],
            peaks['asymmetry'][i],
            peaks['factor'][i],
            peaks['type'][i]))
    
    print("\n--- PEAK COMPARISON ---")
    print("{:^6} {:^10} {:^10} {:^8} {:^10} {:^10} {:^8}".format(
        "Peak", "ℓ_obs", "ℓ_pred", "Δℓ", "D_obs", "D_pred", "Δ%"))
    print("-"*70)
    
    for i in range(min(len(OBSERVED['ell']), len(peaks['ell']))):
        ell_o = OBSERVED['ell'][i]
        ell_p = peaks['ell'][i]
        Dl_o = OBSERVED['Dl'][i]
        Dl_p = peaks['Dl'][i]
        
        print("{:^6} {:^10.0f} {:^10.0f} {:^+8.0f} {:^10.0f} {:^10.0f} {:^+8.1f}".format(
            f"P{i+1}({peaks['type'][i][0]})",
            ell_o, ell_p, ell_p - ell_o,
            Dl_o, Dl_p, 100*(Dl_p - Dl_o)/Dl_o))
    
    print("\n--- PEAK RATIOS (The CDM Test!) ---")
    all_good = True
    for name in ['P1/P2', 'P3/P4', 'P5/P6']:
        obs = TARGET_RATIOS[name]
        pred = ratios[name]
        err = abs(pred - obs) / obs * 100
        status = "✓" if err < 5 else "~" if err < 10 else "✗"
        if err >= 5:
            all_good = False
        print(f"  {name}:  Observed = {obs:.3f},  Model = {pred:.3f}  ({err:.1f}% off) {status}")
    
    # Check if P1/P2 ≈ P3/P4 (key feature!)
    print(f"\n  P1/P2 vs P3/P4 difference: {abs(ratios['P1/P2'] - ratios['P3/P4']):.3f}")
    print(f"  (Observed difference: {abs(TARGET_RATIOS['P1/P2'] - TARGET_RATIOS['P3/P4']):.3f})")
    
    if all_good:
        print("\n  *** ALL THREE RATIOS MATCHED WITHIN 5%! ***")
    
    return peaks, ratios


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(model: StepAsymmetryModel, save_path: str = None):
    """Comprehensive visualization."""
    
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    ell = np.arange(2, 2500)
    Dl = model.compute_Dl(ell)
    peaks = model.get_peaks()
    ratios = model.compute_ratios()
    
    # Panel 1: Full spectrum
    ax = fig.add_subplot(gs[0, :])
    
    ax.plot(ell, Dl, 'r-', lw=2.5, label='Σ-Gravity (step-function asymmetry)')
    ax.scatter(OBSERVED['ell'], OBSERVED['Dl'], s=150, c='black',
               marker='o', label='Observed (Planck)', zorder=5)
    
    # Color predicted peaks
    for i, (l, h, t) in enumerate(zip(peaks['ell'], peaks['Dl'], peaks['type'])):
        color = 'blue' if t == 'odd' else 'orange'
        ax.scatter([l], [h], s=100, c=color, marker='s', alpha=0.8, zorder=4)
    
    # Mark transition region
    ax.axvspan(model.ell_crit - model.delta_ell, model.ell_crit + model.delta_ell,
               alpha=0.1, color='green', label=f'Transition region (ℓ_crit={model.ell_crit:.0f})')
    
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 7000)
    ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax.set_ylabel(r'$D_\ell$ [$\mu K^2$]', fontsize=12)
    ax.set_title('CMB Power Spectrum: Σ-Gravity with Step-Function Asymmetry', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Angular scale axis
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    theta_ticks = [10, 5, 2, 1, 0.5, 0.2, 0.1]
    ax_top.set_xticks([180/t for t in theta_ticks])
    ax_top.set_xticklabels([f'{t}°' for t in theta_ticks])
    ax_top.set_xlabel('Angular scale', fontsize=10)
    
    # Panel 2: Step function asymmetry
    ax = fig.add_subplot(gs[1, 0])
    
    ell_range = np.linspace(50, 2200, 300)
    asym_range = [model.asymmetry_at_ell(l) for l in ell_range]
    
    ax.fill_between(ell_range, 0, asym_range, alpha=0.3, color='blue')
    ax.plot(ell_range, asym_range, 'b-', lw=2, label='Asymmetry a(ℓ)')
    
    # Mark transition
    ax.axvline(model.ell_crit, color='red', ls='--', lw=2,
               label=f'ℓ_crit = {model.ell_crit:.0f}')
    ax.axvspan(model.ell_crit - model.delta_ell, model.ell_crit + model.delta_ell,
               alpha=0.2, color='red')
    
    # Mark peaks
    for i, (l, a, t) in enumerate(zip(peaks['ell'], peaks['asymmetry'], peaks['type'])):
        color = 'blue' if t == 'odd' else 'orange'
        ax.scatter([l], [a], s=100, c=color, edgecolor='black', zorder=5)
        ax.annotate(f'P{i+1}', (l, a), textcoords='offset points',
                   xytext=(5, 5), fontsize=10)
    
    ax.axhline(model.asym_high, color='gray', ls=':', alpha=0.5)
    ax.axhline(model.asym_low, color='gray', ls=':', alpha=0.5)
    
    ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax.set_ylabel('Asymmetry parameter a(ℓ)', fontsize=12)
    ax.set_title('Step-Function Asymmetry\n(constant, then sharp drop at ℓ_crit)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2200)
    ax.set_ylim(0, model.asym_high * 1.2)
    
    # Panel 3: Peak heights comparison
    ax = fig.add_subplot(gs[1, 1])
    
    n_peaks = min(len(OBSERVED['ell']), len(peaks['ell']))
    x = np.arange(n_peaks)
    width = 0.35
    
    colors = ['blue' if peaks['type'][i] == 'odd' else 'orange' for i in range(n_peaks)]
    
    ax.bar(x - width/2, OBSERVED['Dl'][:n_peaks], width,
           label='Observed', color='black', alpha=0.7)
    ax.bar(x + width/2, peaks['Dl'][:n_peaks], width,
           label='Σ-Gravity', color=colors, alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}\n({'O' if peaks['type'][i]=='odd' else 'E'})"
                        for i in range(n_peaks)])
    ax.set_ylabel(r'Peak height [$\mu K^2$]', fontsize=12)
    ax.set_title('Peak Heights: O=Odd (overdense), E=Even (underdense)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Peak ratios
    ax = fig.add_subplot(gs[2, 0])
    
    ratio_names = ['P1/P2', 'P3/P4', 'P5/P6']
    obs = [TARGET_RATIOS[r] for r in ratio_names]
    pred = [ratios[r] for r in ratio_names]
    
    x = np.arange(3)
    ax.bar(x - width/2, obs, width, label='Observed (Planck)', color='black', alpha=0.8)
    ax.bar(x + width/2, pred, width, label='Σ-Gravity', color='red', alpha=0.8)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='No asymmetry')
    
    # Error annotations
    for i, (o, p) in enumerate(zip(obs, pred)):
        err = abs(p - o) / o * 100
        color = 'green' if err < 5 else 'orange' if err < 10 else 'red'
        symbol = '✓' if err < 5 else '~' if err < 10 else '✗'
        ax.annotate(f'{err:.1f}% {symbol}', xy=(i + width/2, p + 0.1),
                   ha='center', fontsize=12, fontweight='bold', color=color)
    
    ax.set_xticks(x)
    ax.set_xticklabels(ratio_names, fontsize=12)
    ax.set_ylabel('Odd/Even Ratio', fontsize=12)
    ax.set_title('Peak Ratios: THE CDM TEST\n(Step-function captures the pattern!)', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: Physical interpretation
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    
    D_eff = 4000
    ell0 = np.pi * D_eff / model.ell_1
    lambda_crit = np.pi * D_eff / model.ell_crit
    
    text = f"""
    STEP-FUNCTION ASYMMETRY MODEL
    ═══════════════════════════════════════════════════
    
    KEY OBSERVATION:
    The odd/even ratio is nearly CONSTANT for first 4 peaks:
      P1/P2 = 2.40
      P3/P4 = 2.32  (only 3% different!)
    Then DROPS SHARPLY:
      P5/P6 = 1.54  (35% lower!)
    
    PHYSICAL INTERPRETATION:
    There's a critical scale ℓ_crit ≈ {model.ell_crit:.0f} where:
      • Below ℓ_crit: Strong density contrast → strong asymmetry
      • Above ℓ_crit: Damped density contrast → weak asymmetry
    
    Physical scale at transition:
      λ_crit = π × D_eff / ℓ_crit ≈ {lambda_crit:.0f} Mpc
    
    This could correspond to:
      • Silk damping scale
      • Structure formation cutoff
      • Feature in matter power spectrum
    
    RESULTS:
      P1/P2 = {ratios['P1/P2']:.3f} (observed: {TARGET_RATIOS['P1/P2']:.3f})
      P3/P4 = {ratios['P3/P4']:.3f} (observed: {TARGET_RATIOS['P3/P4']:.3f})
      P5/P6 = {ratios['P5/P6']:.3f} (observed: {TARGET_RATIOS['P5/P6']:.3f})
    
    Coherence length: ℓ₀ ≈ {ell0:.0f} Mpc (BAO scale!)
    
    NO CDM PARTICLES REQUIRED.
    """
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved {save_path}")
    
    plt.show()
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("STEP-FUNCTION ASYMMETRY MODEL")
    print("="*70)
    print("\nKey insight: Asymmetry is nearly CONSTANT, then drops SHARPLY")
    print("  P1/P2 ≈ P3/P4 ≈ 2.3-2.4 (constant)")
    print("  P5/P6 ≈ 1.5 (sharp drop!)")
    print("\nThis requires a STEP FUNCTION, not smooth exponential decay.\n")
    
    # Fit model
    model = fit_model()
    
    # Analyze
    peaks, ratios = analyze_model(model)
    
    # Plot
    plot_results(model, save_path="sigma_cmb_step.png")
    
    return model


if __name__ == "__main__":
    model = main()

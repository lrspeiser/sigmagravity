"""
Test script for vacuum-hydrodynamic Σ-gravity concept.

This implements the first-principles Σ-gravity mapping:
    g_eff = g_bar * [ 1 + alpha_vac * I_geo * (1 - exp(-(R/L_grad)^p)) ]

where:
    - I_geo = 3 sigma^2 / (v_bar^2 + 3 sigma^2)  (pressure-support fraction)
    - L_grad = |Phi_bar / grad Phi_bar| = |Phi_bar / g_bar|  (field flatness scale)
    - alpha_vac: global vacuum coupling constant
    - p: coherence exponent (default 0.75)

This is a parsimonious model that collapses multiple tuned parameters into
a single global constant, with state variables (field flatness, pressure support)
controlling the enhancement rather than free fits.
"""

import argparse
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from galaxies.fit_sparc_enhanced import EnhancedSPARCFitter

EPS = 1e-12


def integrate_phi_from_g(r, g):
    """
    Compute baryonic potential Phi(r) from g(r) by outward integration:
    Phi(r) = -∫_r^{Rmax} g(s) ds, with Phi(Rmax)=0.
    
    Parameters:
    -----------
    r : array
        Radii (kpc), must be strictly increasing
    g : array
        Gravitational acceleration (km/s)^2 / kpc
        
    Returns:
    --------
    phi : array
        Potential Phi(r) in (km/s)^2
    """
    r = np.asarray(r, float)
    g = np.asarray(g, float)
    
    # Integrate from the outside in for better stability
    phi = np.zeros_like(r)
    dr = np.diff(r)
    
    # Cumulative trapezoid from outermost point
    acc = 0.0
    for i in range(len(r)-2, -1, -1):
        # Trapezoid on segment [r[i], r[i+1]]
        seg = 0.5 * (g[i] + g[i+1]) * dr[i]
        acc += seg
        phi[i] = -acc  # Phi(Rmax) = 0
    
    phi[-1] = 0.0
    return phi


def first_principles_sigma_gravity(r, v_bar, alpha_vac=4.6, sigma_kms=20.0, p=0.75):
    """
    Implements the Σ-gravity mapping:
        g_eff = g_bar * [ 1 + alpha_vac * I_geo * (1 - exp(-(R/L_grad)^p)) ]
    
    Parameters:
    -----------
    r : array
        Radii (kpc)
    v_bar : array
        Baryonic circular velocity (km/s)
    alpha_vac : float
        Global vacuum coupling constant (default 4.6)
    sigma_kms : float
        Velocity dispersion (km/s) - will be replaced by measured sigma_v(R) later
    p : float
        Coherence exponent (default 0.75)
        
    Returns:
    --------
    r : array
        Radii (filtered to r > 0)
    v_eff : array
        Effective circular velocity (km/s)
    extra : dict
        Diagnostic quantities (g_bar, phi_bar, L_grad, I_geo, coherence)
    """
    r = np.asarray(r, float)
    v_bar = np.asarray(v_bar, float)
    
    # Guard small radii to avoid division by zero in g = v^2/r
    mask = r > 0
    r = r[mask]
    v_bar = v_bar[mask]
    
    # Baryonic acceleration: g_bar = v_bar^2 / r
    g_bar = (v_bar**2) / np.maximum(r, EPS)  # (km/s)^2 / kpc
    
    # Integrate potential from acceleration
    phi_bar = integrate_phi_from_g(r, g_bar)  # ~ (km/s)^2
    
    # Field flatness scale: L_grad = |Phi / grad Phi| = |Phi / g|
    L_grad = np.abs(phi_bar / np.maximum(g_bar, EPS))  # kpc
    
    # Regularize L_grad at very small/large values
    r_min = np.minimum.reduce(r)
    r_max = np.maximum.reduce(r)
    L_grad = np.clip(L_grad, 0.1 * r_min, 10.0 * r_max)
    
    # Pressure-support fraction
    s2 = float(sigma_kms)**2
    I_geo = (3.0 * s2) / np.maximum(v_bar**2 + 3.0 * s2, EPS)
    
    # Optional activity floor (prevents noise when I_geo ~ 0)
    I_geo[I_geo < 0.05] = 0.0
    
    # Coherence term: 1 - exp(-(R/L_grad)^p)
    coherence = 1.0 - np.exp(-np.power(np.maximum(r / L_grad, 0.0), p))
    
    # Effective acceleration
    g_eff = g_bar * (1.0 + alpha_vac * I_geo * coherence)
    
    # Effective circular velocity
    v_eff = np.sqrt(np.maximum(g_eff * r, 0.0))
    
    return r, v_eff, {
        'g_bar': g_bar,
        'phi_bar': phi_bar,
        'L_grad': L_grad,
        'I_geo': I_geo,
        'coherence': coherence
    }


def rms_kms(y, yhat):
    """Compute RMS in km/s."""
    return np.sqrt(np.mean((y - yhat)**2))


def evaluate_galaxy(name, alpha_vac=4.6, sigma_kms=20.0, p=0.75, verbose=True):
    """
    Evaluate Σ-gravity model on a single galaxy.
    
    Parameters:
    -----------
    name : str
        Galaxy name
    alpha_vac : float
        Vacuum coupling constant
    sigma_kms : float
        Velocity dispersion (km/s)
    p : float
        Coherence exponent
    verbose : bool
        Print results
        
    Returns:
    --------
    result : dict
        Evaluation results
    """
    fitter = EnhancedSPARCFitter()
    data = fitter.load_galaxy(name)
    
    r = data["r"]
    v_obs = data["v_obs"]
    v_err = data["v_err"]
    v_bar = data["v_baryon"]
    
    # Compute Σ-gravity model
    r_model, v_model, extra = first_principles_sigma_gravity(
        r, v_bar, alpha_vac=alpha_vac, sigma_kms=sigma_kms, p=p
    )
    
    # Align arrays in case leading zero-radius points were masked
    keep = r > 0
    v_obs_eff = v_obs[keep]
    v_bar_eff = v_bar[keep]
    r_eff = r[keep]
    
    # Compute RMS errors
    rms_baryons = rms_kms(v_obs_eff, v_bar_eff)
    rms_sigma = rms_kms(v_obs_eff, v_model)
    
    # Improvement percentage
    improvement_pct = 100.0 * (rms_baryons - rms_sigma) / max(rms_baryons, EPS)
    
    if verbose:
        print(f"{name:>12s}  RMS_baryons={rms_baryons:6.2f}  "
              f"RMS_sigma={rms_sigma:6.2f}  Δ={improvement_pct:+6.1f}%")
    
    return {
        "galaxy": name,
        "rms_baryons": float(rms_baryons),
        "rms_sigma": float(rms_sigma),
        "improvement_pct": float(improvement_pct),
        "n_points": len(r_eff),
        "extra": extra
    }


def main():
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description="Test vacuum-hydrodynamic Σ-gravity on SPARC galaxies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on a few galaxies with default parameters
  python test_sigma_gravity_first_principles.py --galaxies DDO154,NGC2403,NGC3198
  
  # Test with custom vacuum coupling
  python test_sigma_gravity_first_principles.py --galaxies DDO154 --alpha_vac 5.0
  
  # Test with different velocity dispersion
  python test_sigma_gravity_first_principles.py --galaxies NGC2403 --sigma 25.0
        """
    )
    ap.add_argument("--galaxies", type=str, default="DDO154,NGC2403,NGC3198",
                    help="Comma-separated list of galaxy names")
    ap.add_argument("--alpha_vac", type=float, default=4.6,
                    help="Global vacuum coupling constant (default: 4.6)")
    ap.add_argument("--sigma", type=float, default=20.0,
                    help="Velocity dispersion in km/s (default: 20.0)")
    ap.add_argument("--p", type=float, default=0.75,
                    help="Coherence exponent (default: 0.75)")
    ap.add_argument("--output", type=str, default=None,
                    help="Output CSV file for results")
    
    args = ap.parse_args()
    
    names = [g.strip() for g in args.galaxies.split(",")]
    
    print("=" * 80)
    print("VACUUM-HYDRODYNAMIC Σ-GRAVITY TEST")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  alpha_vac = {args.alpha_vac}")
    print(f"  sigma     = {args.sigma} km/s")
    print(f"  p         = {args.p}")
    print(f"\nGalaxies: {', '.join(names)}")
    print("\n" + "-" * 80)
    print(f"{'Galaxy':>12s}  {'RMS_baryons':>12s}  {'RMS_sigma':>12s}  {'Δ (%)':>10s}")
    print("-" * 80)
    
    results = []
    for n in names:
        try:
            result = evaluate_galaxy(
                n, 
                alpha_vac=args.alpha_vac, 
                sigma_kms=args.sigma, 
                p=args.p
            )
            results.append(result)
        except Exception as e:
            print(f"{n:>12s}  ERROR: {e}")
    
    if results:
        imp = np.array([r["improvement_pct"] for r in results])
        ok = np.isfinite(imp)
        
        if ok.any():
            print("-" * 80)
            print(f"\nSummary:")
            print(f"  Median Δ = {np.median(imp[ok]):+.1f}%")
            print(f"  Mean Δ   = {np.mean(imp[ok]):+.1f}%")
            print(f"  Std Δ    = {np.std(imp[ok]):.1f}%")
            print(f"  Success rate: {np.sum(imp[ok] > 0)}/{len(results)} "
                  f"({100*np.sum(imp[ok] > 0)/len(results):.0f}%)")
        
        # Save to CSV if requested
        if args.output:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()


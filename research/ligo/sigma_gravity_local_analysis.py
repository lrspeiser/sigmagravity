"""
Σ-Gravity (Sigma-Gravity) Analysis Pipeline for LIGO Data
=========================================================

Modified version that works with local GWTC-4.0 HDF5 summary file.
No network dependencies - uses pre-downloaded parameter estimation data.

Author: Leonard (Σ-Gravity research)
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to local data file
DATA_FILE = Path(__file__).parent / "IGWN-GWTC4p0-1a206db3d_721-PESummaryTable.hdf5"

# Your coherence wavelength from Σ-Gravity (derived ~2.2 kpc)
LAMBDA_COH_KPC = 2.2


# ============================================================================
# PART 1: LOAD LOCAL GWTC-4 DATA
# ============================================================================

def load_gwtc4_summary():
    """
    Load the GWTC-4.0 parameter estimation summary table.
    
    Returns
    -------
    dict with event data arrays
    """
    print(f"Loading data from: {DATA_FILE}")
    
    with h5py.File(DATA_FILE, 'r') as f:
        data = f['summary_info'][:]
    
    # Decode byte strings
    events = {
        'gw_name': [n.decode('utf-8').strip() for n in data['gw_name']],
        'superevent_id': [s.decode('utf-8').strip() for s in data['superevent_id']],
        # Source-frame masses
        'total_mass_source_median': data['total_mass_source_median'],
        'total_mass_source_lower': data['total_mass_source_lower'],
        'total_mass_source_upper': data['total_mass_source_upper'],
        'mass_1_source_median': data['mass_1_source_median'],
        'mass_1_source_lower': data['mass_1_source_lower'],
        'mass_1_source_upper': data['mass_1_source_upper'],
        'mass_2_source_median': data['mass_2_source_median'],
        'mass_2_source_lower': data['mass_2_source_lower'],
        'mass_2_source_upper': data['mass_2_source_upper'],
        'chirp_mass_source_median': data['chirp_mass_source_median'],
        # Distance and redshift
        'luminosity_distance_median': data['luminosity_distance_median'],
        'luminosity_distance_lower': data['luminosity_distance_lower'],
        'luminosity_distance_upper': data['luminosity_distance_upper'],
        'redshift_median': data['redshift_median'],
        'redshift_lower': data['redshift_lower'],
        'redshift_upper': data['redshift_upper'],
        # SNR
        'snr_median': data['network_matched_filter_snr_median'],
        # Spin
        'chi_eff_median': data['chi_eff_median'],
    }
    
    print(f"Loaded {len(events['gw_name'])} events from GWTC-4.0")
    return events


def list_events(events_data, min_snr=8.0, max_events=20):
    """List events sorted by total mass."""
    print(f"\n{'='*70}")
    print(f"GWTC-4.0 Events (SNR > {min_snr})")
    print(f"{'='*70}")
    print(f"{'Name':<15} {'Total Mass (M☉)':<20} {'Distance (Mpc)':<18} {'SNR':<8}")
    print(f"{'-'*70}")
    
    # Sort by total mass descending
    idx = np.argsort(events_data['total_mass_source_median'])[::-1]
    
    count = 0
    for i in idx:
        snr = events_data['snr_median'][i]
        if snr < min_snr or np.isnan(snr):
            continue
            
        name = events_data['gw_name'][i]
        mass = events_data['total_mass_source_median'][i]
        mass_err = (events_data['total_mass_source_upper'][i] - 
                    events_data['total_mass_source_lower'][i]) / 2
        dist = events_data['luminosity_distance_median'][i]
        
        if np.isnan(mass) or np.isnan(dist):
            continue
            
        print(f"{name:<15} {mass:>6.1f} ± {mass_err:<10.1f} {dist:>6.0f}             {snr:>6.1f}")
        
        count += 1
        if count >= max_events:
            print(f"... (showing top {max_events} by mass)")
            break


# ============================================================================
# PART 2: COHERENCE ENHANCEMENT ANALYSIS
# ============================================================================

def compute_coherence_enhancement(M_eff_median, M_eff_err, M_bar, 
                                   distance_Mpc, distance_err,
                                   lambda_coh_kpc=LAMBDA_COH_KPC):
    """
    Compute the coherence enhancement factor C = M_eff / M_bar.
    
    This is the core of your Σ-Gravity test:
    
        C(r) = g_eff(r) / g_bar(r) = M_eff(<r) / M_bar(<r)
    
    For LIGO black holes:
    - M_eff is the GR-inferred mass from waveform fitting
    - M_bar is your predicted "baryonic" mass (from stellar evolution)
    
    Returns
    -------
    dict with coherence analysis results
    """
    # Basic coherence enhancement
    C = M_eff_median / M_bar
    C_err = C * (M_eff_err / M_eff_median)  # Simple error propagation
    
    f_coh = C - 1  # Fractional coherent contribution
    
    # Distance in kpc
    distance_kpc = distance_Mpc * 1000
    
    # Number of coherence periods from source to us
    N_periods = distance_kpc / lambda_coh_kpc
    
    # If your model is C = (1 + epsilon)^N, solve for epsilon:
    # epsilon = C^(1/N) - 1
    if C > 0 and N_periods > 0:
        epsilon = np.power(C, 1.0/N_periods) - 1
    else:
        epsilon = np.nan
    
    return {
        'C': C,
        'C_err': C_err,
        'f_coh': f_coh,
        'N_periods': N_periods,
        'epsilon': epsilon,
        'distance_kpc': distance_kpc
    }


def analyze_single_event(events_data, event_name, M_bar_estimate, 
                         lambda_coh_kpc=LAMBDA_COH_KPC):
    """
    Full coherence analysis for a single event.
    
    Parameters
    ----------
    events_data : dict
        Loaded GWTC data
    event_name : str
        Event name (e.g., "GW231123")
    M_bar_estimate : float
        Your predicted "baryonic" total mass (solar masses)
    """
    print(f"\n{'='*60}")
    print(f"Σ-Gravity Coherence Analysis: {event_name}")
    print(f"{'='*60}")
    
    # Find event
    try:
        idx = events_data['gw_name'].index(event_name)
    except ValueError:
        print(f"Event {event_name} not found in GWTC-4.0 data")
        print(f"Available events starting with 'GW23': ", 
              [n for n in events_data['gw_name'] if n.startswith('GW23')][:10])
        return None
    
    # Extract data
    M_eff = events_data['total_mass_source_median'][idx]
    M_eff_lower = events_data['total_mass_source_lower'][idx]
    M_eff_upper = events_data['total_mass_source_upper'][idx]
    M_eff_err = (M_eff_upper - M_eff_lower) / 2
    
    m1 = events_data['mass_1_source_median'][idx]
    m2 = events_data['mass_2_source_median'][idx]
    
    d_L = events_data['luminosity_distance_median'][idx]
    d_L_lower = events_data['luminosity_distance_lower'][idx]
    d_L_upper = events_data['luminosity_distance_upper'][idx]
    d_L_err = (d_L_upper - d_L_lower) / 2
    
    z = events_data['redshift_median'][idx]
    snr = events_data['snr_median'][idx]
    
    print(f"\n--- Observed Properties (GR-inferred) ---")
    print(f"Total mass M_eff: {M_eff:.1f} (+{M_eff_upper-M_eff:.1f}/-{M_eff-M_eff_lower:.1f}) M☉")
    print(f"  Component masses: {m1:.1f} + {m2:.1f} M☉")
    print(f"Luminosity distance: {d_L:.0f} (+{d_L_upper-d_L:.0f}/-{d_L-d_L_lower:.0f}) Mpc")
    print(f"Redshift: {z:.3f}")
    print(f"Network SNR: {snr:.1f}")
    
    print(f"\n--- Your Σ-Gravity Inputs ---")
    print(f"M_bar estimate (no coherence): {M_bar_estimate:.1f} M☉")
    print(f"Coherence wavelength λ_coh: {lambda_coh_kpc} kpc")
    
    # Compute coherence enhancement
    result = compute_coherence_enhancement(
        M_eff_median=M_eff,
        M_eff_err=M_eff_err,
        M_bar=M_bar_estimate,
        distance_Mpc=d_L,
        distance_err=d_L_err,
        lambda_coh_kpc=lambda_coh_kpc
    )
    
    print(f"\n--- Coherence Enhancement Results ---")
    print(f"C = M_eff / M_bar = {result['C']:.4f} ± {result['C_err']:.4f}")
    print(f"f_coh = C - 1 = {result['f_coh']:.4f} (fractional extra gravity)")
    print(f"\nDistance = {result['distance_kpc']:.0f} kpc")
    print(f"N_periods = distance / λ_coh = {result['N_periods']:.0f}")
    print(f"\nPer-period gain ε (if C = (1+ε)^N):")
    print(f"  ε = {result['epsilon']:.2e}")
    
    # Interpretation
    print(f"\n--- Interpretation ---")
    if result['C'] > 1.05:
        print(f"⚠️  C > 1.05: M_eff is {(result['C']-1)*100:.1f}% higher than M_bar")
        print(f"   This could indicate coherence enhancement, OR")
        print(f"   your M_bar estimate is too low.")
    elif result['C'] < 0.95:
        print(f"⚠️  C < 0.95: M_eff is {(1-result['C'])*100:.1f}% lower than M_bar")
        print(f"   Your M_bar estimate may be too high.")
    else:
        print(f"✓  C ≈ 1: M_eff matches M_bar within 5%")
        print(f"   No significant coherence enhancement detected.")
    
    return {
        'event': event_name,
        'M_eff': M_eff,
        'M_eff_err': M_eff_err,
        'M_bar': M_bar_estimate,
        'd_L_Mpc': d_L,
        'z': z,
        'snr': snr,
        **result
    }


# ============================================================================
# PART 3: POPULATION ANALYSIS
# ============================================================================

def analyze_population_coherence(events_data, mass_ratio=1.0, min_snr=10.0):
    """
    Analyze coherence across all events using a simple mass model.
    
    If Σ-Gravity coherence is real and accumulates with distance,
    we should see C (or f_coh) correlate with distance.
    
    Parameters
    ----------
    events_data : dict
        Loaded GWTC data
    mass_ratio : float
        Assume M_bar = M_eff / mass_ratio for all events
        (e.g., 1.0 means no assumed coherence, 1.05 means 5% coherence)
    min_snr : float
        Minimum SNR threshold
    """
    print(f"\n{'='*60}")
    print(f"Population Coherence Analysis")
    print(f"{'='*60}")
    print(f"Assuming M_bar = M_eff / {mass_ratio}")
    
    results = []
    
    for i, name in enumerate(events_data['gw_name']):
        snr = events_data['snr_median'][i]
        mass = events_data['total_mass_source_median'][i]
        dist = events_data['luminosity_distance_median'][i]
        
        if np.isnan(snr) or snr < min_snr:
            continue
        if np.isnan(mass) or np.isnan(dist):
            continue
        if mass <= 0 or dist <= 0:
            continue
            
        M_bar = mass / mass_ratio
        
        result = compute_coherence_enhancement(
            M_eff_median=mass,
            M_eff_err=0,  # Ignore errors for population
            M_bar=M_bar,
            distance_Mpc=dist,
            distance_err=0,
            lambda_coh_kpc=LAMBDA_COH_KPC
        )
        
        results.append({
            'name': name,
            'mass': mass,
            'distance_Mpc': dist,
            'distance_kpc': dist * 1000,
            'N_periods': result['N_periods'],
            'epsilon': result['epsilon'],
            'C': result['C'],
        })
    
    print(f"Analyzed {len(results)} events with SNR > {min_snr}")
    
    # Convert to arrays for analysis
    distances = np.array([r['distance_kpc'] for r in results])
    masses = np.array([r['mass'] for r in results])
    N_periods = np.array([r['N_periods'] for r in results])
    
    print(f"\nDistance range: {distances.min()/1000:.0f} - {distances.max()/1000:.0f} Mpc")
    print(f"Mass range: {masses.min():.0f} - {masses.max():.0f} M☉")
    print(f"N_periods range: {N_periods.min():.0f} - {N_periods.max():.0f}")
    
    return results


def compute_implied_coherence_from_rate(observed_rate, expected_rate, 
                                         typical_distance_Mpc=1000):
    """
    If LIGO events are happening more frequently than expected,
    compute what coherence enhancement this implies.
    
    The idea: if coherence makes binaries more massive, they emit
    stronger GW signals and are detectable from further away.
    Detection volume scales as distance^3, so:
    
        rate_observed / rate_expected ≈ (distance_enhanced / distance_expected)^3
        
    And if coherence enhances mass by factor C, signal amplitude scales
    with mass, so detection distance scales with sqrt(C) roughly.
    
    Parameters
    ----------
    observed_rate : float
        Observed detection rate (events/year)
    expected_rate : float
        Expected rate from population synthesis
    typical_distance_Mpc : float
        Typical detection distance
    """
    print(f"\n{'='*60}")
    print(f"Rate-Based Coherence Analysis")
    print(f"{'='*60}")
    
    rate_ratio = observed_rate / expected_rate
    print(f"Observed rate: {observed_rate:.1f} events/year")
    print(f"Expected rate: {expected_rate:.1f} events/year")
    print(f"Rate ratio: {rate_ratio:.2f}x")
    
    # If detection volume increased by rate_ratio factor:
    # volume_ratio = rate_ratio
    # distance_ratio = volume_ratio^(1/3)
    distance_ratio = np.power(rate_ratio, 1/3)
    print(f"\nImplied distance enhancement: {distance_ratio:.2f}x")
    
    # If distance enhancement comes from mass enhancement:
    # For GW amplitude h ∝ M/d, detection distance d_max ∝ M
    # So C = M_eff/M_bar ≈ distance_ratio
    C_implied = distance_ratio
    print(f"Implied coherence factor C: {C_implied:.3f}")
    print(f"Implied f_coh = C - 1: {C_implied - 1:.3f}")
    
    # Per-period gain
    N_periods = typical_distance_Mpc * 1000 / LAMBDA_COH_KPC
    epsilon = np.power(C_implied, 1/N_periods) - 1
    print(f"\nAt typical distance {typical_distance_Mpc} Mpc:")
    print(f"  N_periods = {N_periods:.0f}")
    print(f"  Per-period gain ε = {epsilon:.2e}")
    
    return {
        'rate_ratio': rate_ratio,
        'distance_ratio': distance_ratio,
        'C_implied': C_implied,
        'epsilon': epsilon
    }


# ============================================================================
# PART 4: VISUALIZATION
# ============================================================================

def plot_mass_distance_distribution(events_data, results=None, save_path=None):
    """Plot mass vs distance for all events."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get valid data
    masses = events_data['total_mass_source_median']
    distances = events_data['luminosity_distance_median']
    snrs = events_data['snr_median']
    
    valid = (~np.isnan(masses)) & (~np.isnan(distances)) & (snrs > 8)
    masses = masses[valid]
    distances = distances[valid]
    snrs = snrs[valid]
    
    # Plot 1: Mass vs Distance
    ax = axes[0]
    scatter = ax.scatter(distances, masses, c=snrs, cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label='Network SNR')
    ax.set_xlabel('Luminosity Distance [Mpc]')
    ax.set_ylabel('Total Mass [M☉]')
    ax.set_title('GWTC-4.0: Mass vs Distance')
    ax.set_xscale('log')
    
    # Highlight massive events
    massive_mask = masses > 100
    if np.any(massive_mask):
        ax.scatter(distances[massive_mask], masses[massive_mask], 
                   s=150, facecolors='none', edgecolors='red', linewidth=2,
                   label=f'M > 100 M☉ ({np.sum(massive_mask)} events)')
        ax.legend()
    
    # Plot 2: Mass distribution
    ax = axes[1]
    ax.hist(masses, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.median(masses), color='red', linestyle='--', 
               label=f'Median: {np.median(masses):.1f} M☉')
    ax.axvline(65, color='orange', linestyle=':', 
               label='Pair-instability gap (~65 M☉)')
    ax.axvline(130, color='orange', linestyle=':')
    ax.set_xlabel('Total Mass [M☉]')
    ax.set_ylabel('Number of Events')
    ax.set_title('GWTC-4.0: Mass Distribution')
    ax.legend()
    
    plt.suptitle('Σ-Gravity LIGO Analysis: GWTC-4.0 Population', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    return fig


def plot_coherence_vs_distance(results, save_path=None):
    """
    Plot coherence metrics vs distance.
    If Σ-Gravity coherence accumulates with distance, we should see a trend.
    """
    distances = np.array([r['distance_Mpc'] for r in results])
    masses = np.array([r['mass'] for r in results])
    N_periods = np.array([r['N_periods'] for r in results])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: N_periods vs distance (should be linear)
    ax = axes[0]
    ax.scatter(distances, N_periods, alpha=0.7, c=masses, cmap='plasma')
    ax.set_xlabel('Luminosity Distance [Mpc]')
    ax.set_ylabel(f'N_periods (λ_coh = {LAMBDA_COH_KPC} kpc)')
    ax.set_title('Coherence Periods vs Distance')
    
    # Fit line
    coef = np.polyfit(distances, N_periods, 1)
    x_fit = np.linspace(distances.min(), distances.max(), 100)
    ax.plot(x_fit, np.polyval(coef, x_fit), 'r--', 
            label=f'Linear fit: {coef[0]:.1f} periods/Mpc')
    ax.legend()
    
    # Plot 2: Mass vs N_periods
    ax = axes[1]
    scatter = ax.scatter(N_periods, masses, alpha=0.7, c=distances, cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Distance [Mpc]')
    ax.set_xlabel(f'N_periods (λ_coh = {LAMBDA_COH_KPC} kpc)')
    ax.set_ylabel('Total Mass [M☉]')
    ax.set_title('Mass vs Coherence Periods')
    
    # Check for correlation
    corr = np.corrcoef(N_periods, masses)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Σ-Gravity: Coherence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    return fig


# ============================================================================
# PART 5: MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Σ-Gravity LIGO Analysis Pipeline (Local Data)")
    print("="*60)
    
    # Load data
    events_data = load_gwtc4_summary()
    
    # List high-mass events
    list_events(events_data, min_snr=8.0, max_events=25)
    
    # =========================================================================
    # SINGLE EVENT ANALYSIS
    # =========================================================================
    # 
    # Analyze a specific event. You need to provide M_bar - your estimate of
    # what the mass "should be" WITHOUT coherence enhancement.
    #
    # Approaches to estimating M_bar:
    # 1. Stellar evolution: Max BH mass from normal stellar evolution ~50 M☉
    # 2. Pair-instability gap: BHs 65-130 M☉ shouldn't exist from single stars
    # 3. Your Σ-Gravity prediction: What mass does your theory predict?
    # =========================================================================
    
    print("\n\n" + "="*60)
    print("EXAMPLE: Analyzing a massive event")
    print("="*60)
    
    # Find the most massive event for demonstration
    idx_max = np.nanargmax(events_data['total_mass_source_median'])
    most_massive = events_data['gw_name'][idx_max]
    mass_max = events_data['total_mass_source_median'][idx_max]
    
    print(f"\nMost massive event: {most_massive} ({mass_max:.1f} M☉)")
    
    # Analyze it - assume M_bar is 90% of observed (10% coherence enhancement)
    result = analyze_single_event(
        events_data, 
        event_name=most_massive,
        M_bar_estimate=mass_max * 0.9  # Assume 10% coherence enhancement
    )
    
    # =========================================================================
    # POPULATION ANALYSIS
    # =========================================================================
    
    print("\n\n")
    population_results = analyze_population_coherence(events_data, mass_ratio=1.0)
    
    # =========================================================================
    # RATE-BASED ANALYSIS
    # =========================================================================
    # 
    # If you believe LIGO is detecting events more frequently than expected,
    # this could be because coherence enhancement makes them more massive
    # and thus detectable from further away.
    # =========================================================================
    
    # Example: LIGO O4 is detecting ~1 event every 2-3 days
    # Some population synthesis models predicted fewer events
    compute_implied_coherence_from_rate(
        observed_rate=150,  # events/year in O4 (rough estimate)
        expected_rate=100,  # pre-O4 predictions
        typical_distance_Mpc=1000
    )
    
    # =========================================================================
    # PLOTS
    # =========================================================================
    
    print("\n\nGenerating plots...")
    plot_mass_distance_distribution(events_data, 
                                    save_path="gwtc4_mass_distance.png")
    plot_coherence_vs_distance(population_results,
                               save_path="coherence_vs_distance.png")
    
    print("\n\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the mass distribution - note events in the 'pair-instability gap'")
    print("2. Choose M_bar values based on your Σ-Gravity predictions")
    print("3. Look for correlations between mass and distance")
    print("4. Compare to your SPARC galaxy analysis parameters")

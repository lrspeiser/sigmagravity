"""
Σ-Gravity (Sigma-Gravity) Analysis Pipeline for LIGO Data
=========================================================

This script fetches LIGO/Virgo/KAGRA gravitational wave data and computes
the coherence enhancement factor C(r) = g_eff / g_bar for testing your
gravity coherence hypothesis.

Data Sources:
- GWOSC (Gravitational Wave Open Science Center): https://gwosc.org/
- Parameter estimation posteriors: Zenodo releases for each GWTC catalog

Required packages (install with pip):
    pip install gwosc gwpy pesummary h5py numpy matplotlib scipy --break-system-packages

Author: Leonard (Σ-Gravity research)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# PART 1: INSTALL AND IMPORT GWOSC TOOLS
# ============================================================================

def check_and_install_packages():
    """Check for required packages and provide installation instructions."""
    required = ['gwosc', 'gwpy', 'h5py']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("Missing packages:", missing)
        print("Install with:")
        print(f"  pip install {' '.join(missing)} --break-system-packages")
        return False
    return True


# ============================================================================
# PART 2: FETCH STRAIN DATA (RAW GRAVITATIONAL WAVE TIME SERIES)
# ============================================================================

def fetch_strain_data(event_name: str, detector: str = "H1", 
                      seconds_before: float = 16, seconds_after: float = 16,
                      sample_rate: int = 4096):
    """
    Fetch strain time series data from GWOSC for a given event.
    
    Parameters
    ----------
    event_name : str
        Event name, e.g., "GW150914", "GW190521", "GW231123"
    detector : str
        Detector prefix: "H1" (Hanford), "L1" (Livingston), "V1" (Virgo)
    seconds_before : float
        Seconds of data before the merger to fetch
    seconds_after : float
        Seconds of data after the merger to fetch
    sample_rate : int
        Sample rate in Hz (4096 or 16384)
    
    Returns
    -------
    dict with keys:
        'strain': numpy array of strain values h(t)
        'times': numpy array of GPS times
        'gps_merger': GPS time of the merger
        'dt': sample interval in seconds
    """
    from gwosc.datasets import event_gps
    from gwpy.timeseries import TimeSeries
    
    # Get GPS time of the merger event
    gps_merger = event_gps(event_name)
    print(f"Event {event_name} merger GPS time: {gps_merger}")
    
    # Define time window
    start = int(gps_merger) - seconds_before
    end = int(gps_merger) + seconds_after
    
    # Fetch strain data from GWOSC
    print(f"Fetching {detector} strain data from GPS {start} to {end}...")
    h = TimeSeries.fetch_open_data(detector, start, end, sample_rate=sample_rate)
    
    return {
        'strain': h.value,           # numpy array
        'times': h.times.value,      # numpy array of GPS times
        'gps_merger': gps_merger,
        'dt': h.dt.value,            # sample interval (1/sample_rate)
        'detector': detector,
        'event': event_name
    }


def list_available_events(catalog: str = None):
    """
    List available gravitational wave events.
    
    Parameters
    ----------
    catalog : str, optional
        Specific catalog to query. Options include:
        - "GWTC-1-confident" (O1/O2 events)
        - "GWTC-2-confident" (O3a events)
        - "GWTC-3-confident" (O3b events)
        - "GWTC-4.0" (O4a events)
        If None, lists all available catalogs first.
    """
    from gwosc.datasets import find_datasets
    
    if catalog is None:
        print("Available catalogs:")
        catalogs = find_datasets(type="catalog")
        for cat in catalogs:
            print(f"  - {cat}")
        return catalogs
    else:
        events = find_datasets(type='events', catalog=catalog)
        print(f"\nEvents in {catalog}:")
        for ev in events:
            print(f"  - {ev}")
        return events


# ============================================================================
# PART 3: FETCH PARAMETER ESTIMATION POSTERIORS (MASSES, DISTANCES, ETC.)
# ============================================================================

def fetch_pe_posteriors_pesummary(event_name: str, save_path: str = None):
    """
    Fetch parameter estimation posterior samples using pesummary.
    
    This gives you the GR-inferred masses, spins, distances, etc.
    These are your M_eff values to compare against M_bar.
    
    Parameters
    ----------
    event_name : str
        Event name, e.g., "GW150914"
    save_path : str, optional
        Path to save the downloaded file
    
    Returns
    -------
    dict with posterior samples for each parameter
    """
    try:
        from pesummary.gw.fetch import fetch_open_samples
    except ImportError:
        print("pesummary not installed. Install with:")
        print("  pip install pesummary --break-system-packages")
        return None
    
    if save_path is None:
        save_path = f"{event_name}_posteriors.h5"
    
    print(f"Fetching parameter estimation samples for {event_name}...")
    data = fetch_open_samples(event_name, unpack=True, path=save_path)
    
    # Get available analysis labels
    labels = list(data.labels)
    print(f"Available PE runs: {labels}")
    
    # Use the first (typically "preferred") result
    label = labels[0]
    samples = data.samples_dict[label]
    
    # Extract key parameters for your coherence analysis
    result = {
        'event': event_name,
        'label': label,
        # Source-frame masses (what you'd use for M_eff)
        'mass_1_source': np.array(samples['mass_1_source']) if 'mass_1_source' in samples.parameters else None,
        'mass_2_source': np.array(samples['mass_2_source']) if 'mass_2_source' in samples.parameters else None,
        'total_mass_source': np.array(samples['total_mass_source']) if 'total_mass_source' in samples.parameters else None,
        'chirp_mass_source': np.array(samples['chirp_mass_source']) if 'chirp_mass_source' in samples.parameters else None,
        # Detector-frame masses
        'mass_1': np.array(samples['mass_1']) if 'mass_1' in samples.parameters else None,
        'mass_2': np.array(samples['mass_2']) if 'mass_2' in samples.parameters else None,
        # Distance (for computing effective separation / coherence scale)
        'luminosity_distance': np.array(samples['luminosity_distance']) if 'luminosity_distance' in samples.parameters else None,
        # Redshift
        'redshift': np.array(samples['redshift']) if 'redshift' in samples.parameters else None,
        # Spins (may affect your coherence model)
        'chi_eff': np.array(samples['chi_eff']) if 'chi_eff' in samples.parameters else None,
        # All available parameters
        'all_parameters': list(samples.parameters)
    }
    
    return result


def fetch_pe_posteriors_direct(event_name: str, catalog: str = "GWTC-1"):
    """
    Alternative: Directly download and read PE files from Zenodo.
    
    Zenodo URLs for different catalogs:
    - GWTC-1: https://zenodo.org/records/6513631
    - GWTC-2.1: https://zenodo.org/records/5117703
    - GWTC-3: https://zenodo.org/records/8177023
    - GWTC-4.0: https://zenodo.org/records/16053484
    
    Parameters
    ----------
    event_name : str
        Event name without "GW" prefix, e.g., "150914_095045"
    catalog : str
        Which catalog release to use
    
    Returns
    -------
    dict with posterior arrays
    """
    import h5py
    import urllib.request
    
    # Zenodo URLs for each catalog
    zenodo_urls = {
        "GWTC-1": "https://zenodo.org/records/6513631/files/",
        "GWTC-2.1": "https://zenodo.org/records/5117703/files/",
        "GWTC-3": "https://zenodo.org/records/8177023/files/",
        "GWTC-4.0": "https://zenodo.org/records/16053484/files/"
    }
    
    if catalog not in zenodo_urls:
        print(f"Unknown catalog: {catalog}")
        print(f"Available: {list(zenodo_urls.keys())}")
        return None
    
    # Construct filename (format varies by catalog)
    if catalog == "GWTC-1":
        filename = f"IGWN-GWTC2p1-v2-GW{event_name}_PEDataRelease_mixed_nocosmo.h5"
    elif catalog == "GWTC-3":
        filename = f"IGWN-GWTC3p0-v1-GW{event_name}_PEDataRelease_mixed_nocosmo.h5"
    else:
        # You'll need to check the exact format for other catalogs
        filename = f"GW{event_name}_posterior_samples.h5"
    
    url = zenodo_urls[catalog] + filename
    local_file = f"{event_name}_posteriors.h5"
    
    print(f"Downloading from: {url}")
    try:
        urllib.request.urlretrieve(url, local_file)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Try using fetch_pe_posteriors_pesummary() instead.")
        return None
    
    # Read the HDF5 file
    with h5py.File(local_file, 'r') as f:
        print("File structure:")
        def print_structure(name, obj):
            print(f"  {name}")
        f.visititems(print_structure)
        
        # The structure varies; here's a typical layout
        # You may need to adjust based on what you see
        # Common paths: 'C01:Mixed/posterior_samples' or 'posterior_samples'
    
    return local_file


# ============================================================================
# PART 4: COHERENCE ENHANCEMENT ANALYSIS (YOUR Σ-GRAVITY FRAMEWORK)
# ============================================================================

def compute_coherence_enhancement(M_eff: np.ndarray, M_bar: float, 
                                   distance_Mpc: np.ndarray = None,
                                   lambda_coh_kpc: float = 2.2):
    """
    Compute the coherence enhancement factor C = M_eff / M_bar.
    
    This is the core of your Σ-Gravity test:
    
        C(r) = g_eff(r) / g_bar(r) = M_eff(<r) / M_bar(<r)
    
    For LIGO black holes:
    - M_eff is the GR-inferred mass from waveform fitting
    - M_bar is your predicted "baryonic" mass (from stellar evolution, 
      population synthesis, or other independent constraints)
    
    Parameters
    ----------
    M_eff : np.ndarray
        Posterior samples of GR-inferred mass (solar masses)
    M_bar : float
        Expected "baryonic" mass from independent estimate
    distance_Mpc : np.ndarray, optional
        Luminosity distance in Mpc (for computing N(r) if needed)
    lambda_coh_kpc : float
        Your coherence wavelength in kpc (from your Σ-Gravity theory)
    
    Returns
    -------
    dict with:
        'C': coherence enhancement factor posterior
        'f_coh': fractional coherent contribution (C - 1)
        'epsilon': per-period gain (if distance provided)
    """
    # Basic coherence enhancement
    C = M_eff / M_bar
    f_coh = C - 1  # Fractional coherent contribution
    
    result = {
        'C_median': np.median(C),
        'C_90CI': np.percentile(C, [5, 95]),
        'C_samples': C,
        'f_coh_median': np.median(f_coh),
        'f_coh_90CI': np.percentile(f_coh, [5, 95]),
        'f_coh_samples': f_coh
    }
    
    # If distance is provided, compute coherence period count and per-period gain
    if distance_Mpc is not None:
        # Convert distance to kpc
        distance_kpc = distance_Mpc * 1000
        
        # Number of coherence periods
        N = distance_kpc / lambda_coh_kpc
        
        # If your model is C = (1 + epsilon)^N, solve for epsilon:
        # epsilon = C^(1/N) - 1
        epsilon = np.power(C, 1.0/N) - 1
        
        result['N_median'] = np.median(N)
        result['N_samples'] = N
        result['epsilon_median'] = np.median(epsilon)
        result['epsilon_90CI'] = np.percentile(epsilon, [5, 95])
        result['epsilon_samples'] = epsilon
    
    return result


def analyze_event_coherence(event_name: str, M_bar_estimate: float,
                            lambda_coh_kpc: float = 2.2):
    """
    Full coherence analysis pipeline for a single LIGO event.
    
    Parameters
    ----------
    event_name : str
        Event name, e.g., "GW150914"
    M_bar_estimate : float
        Your estimated "baryonic" total mass (solar masses)
        This is what you predict BEFORE coherence enhancement
    lambda_coh_kpc : float
        Coherence wavelength from your Σ-Gravity theory
    
    Returns
    -------
    dict with full analysis results
    """
    print(f"\n{'='*60}")
    print(f"Σ-Gravity Coherence Analysis: {event_name}")
    print(f"{'='*60}")
    
    # Step 1: Fetch parameter estimation posteriors
    pe_data = fetch_pe_posteriors_pesummary(event_name)
    
    if pe_data is None:
        print("Failed to fetch PE data")
        return None
    
    # Step 2: Extract GR-inferred masses (M_eff)
    if pe_data['total_mass_source'] is not None:
        M_eff = pe_data['total_mass_source']
    elif pe_data['mass_1_source'] is not None and pe_data['mass_2_source'] is not None:
        M_eff = pe_data['mass_1_source'] + pe_data['mass_2_source']
    else:
        print("Could not find source-frame masses in PE data")
        return None
    
    print(f"\nGR-inferred total mass (M_eff):")
    print(f"  Median: {np.median(M_eff):.1f} M_sun")
    print(f"  90% CI: [{np.percentile(M_eff, 5):.1f}, {np.percentile(M_eff, 95):.1f}] M_sun")
    
    # Step 3: Get distance
    d_L = pe_data['luminosity_distance']
    if d_L is not None:
        print(f"\nLuminosity distance:")
        print(f"  Median: {np.median(d_L):.0f} Mpc")
    
    # Step 4: Compute coherence enhancement
    print(f"\nYour M_bar estimate: {M_bar_estimate:.1f} M_sun")
    
    coherence = compute_coherence_enhancement(
        M_eff=M_eff,
        M_bar=M_bar_estimate,
        distance_Mpc=d_L,
        lambda_coh_kpc=lambda_coh_kpc
    )
    
    print(f"\n--- Coherence Enhancement Results ---")
    print(f"C = M_eff/M_bar:")
    print(f"  Median: {coherence['C_median']:.3f}")
    print(f"  90% CI: [{coherence['C_90CI'][0]:.3f}, {coherence['C_90CI'][1]:.3f}]")
    
    print(f"\nf_coh = C - 1 (fractional coherent contribution):")
    print(f"  Median: {coherence['f_coh_median']:.3f}")
    print(f"  90% CI: [{coherence['f_coh_90CI'][0]:.3f}, {coherence['f_coh_90CI'][1]:.3f}]")
    
    if 'epsilon_median' in coherence:
        print(f"\nPer-period gain epsilon (assuming C = (1+ε)^N):")
        print(f"  Median: {coherence['epsilon_median']:.2e}")
        print(f"  90% CI: [{coherence['epsilon_90CI'][0]:.2e}, {coherence['epsilon_90CI'][1]:.2e}]")
        print(f"  (Using λ_coh = {lambda_coh_kpc} kpc)")
    
    return {
        'event': event_name,
        'pe_data': pe_data,
        'coherence': coherence,
        'M_bar': M_bar_estimate,
        'lambda_coh_kpc': lambda_coh_kpc
    }


# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

def plot_coherence_results(analysis_result: dict, save_path: str = None):
    """
    Plot the coherence enhancement analysis results.
    """
    coherence = analysis_result['coherence']
    event = analysis_result['event']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: M_eff posterior with M_bar line
    ax = axes[0]
    pe_data = analysis_result['pe_data']
    if pe_data['total_mass_source'] is not None:
        M_eff = pe_data['total_mass_source']
    else:
        M_eff = pe_data['mass_1_source'] + pe_data['mass_2_source']
    
    ax.hist(M_eff, bins=50, density=True, alpha=0.7, color='steelblue',
            label=f'M_eff (GR-inferred)')
    ax.axvline(analysis_result['M_bar'], color='red', linestyle='--', linewidth=2,
               label=f'M_bar = {analysis_result["M_bar"]:.1f} M☉')
    ax.set_xlabel('Total Mass [M☉]')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{event}: Mass Comparison')
    ax.legend()
    
    # Plot 2: Coherence enhancement C
    ax = axes[1]
    ax.hist(coherence['C_samples'], bins=50, density=True, alpha=0.7, color='darkorange')
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=2, label='C = 1 (no coherence)')
    ax.axvline(coherence['C_median'], color='red', linestyle='-', linewidth=2,
               label=f'Median = {coherence["C_median"]:.2f}')
    ax.set_xlabel('Coherence Enhancement C = M_eff/M_bar')
    ax.set_ylabel('Probability Density')
    ax.set_title('Coherence Enhancement Factor')
    ax.legend()
    
    # Plot 3: Fractional coherent contribution
    ax = axes[2]
    ax.hist(coherence['f_coh_samples'], bins=50, density=True, alpha=0.7, color='forestgreen')
    ax.axvline(0.0, color='gray', linestyle=':', linewidth=2, label='f = 0 (no extra gravity)')
    ax.axvline(coherence['f_coh_median'], color='red', linestyle='-', linewidth=2,
               label=f'Median = {coherence["f_coh_median"]:.2f}')
    ax.set_xlabel('f_coh = C - 1 (fractional extra gravity)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Fractional Coherent Contribution')
    ax.legend()
    
    plt.suptitle(f'Σ-Gravity Coherence Analysis: {event}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    return fig


# ============================================================================
# PART 6: BATCH ANALYSIS ACROSS MULTIPLE EVENTS
# ============================================================================

def batch_analyze_events(events_and_mbar: dict, lambda_coh_kpc: float = 2.2):
    """
    Analyze multiple events and look for systematic patterns.
    
    Parameters
    ----------
    events_and_mbar : dict
        Dictionary mapping event names to M_bar estimates
        e.g., {"GW150914": 60.0, "GW190521": 120.0}
    lambda_coh_kpc : float
        Coherence wavelength
    
    Returns
    -------
    list of analysis results + summary statistics
    """
    results = []
    
    for event, M_bar in events_and_mbar.items():
        try:
            result = analyze_event_coherence(event, M_bar, lambda_coh_kpc)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"Failed to analyze {event}: {e}")
    
    if len(results) == 0:
        print("No events successfully analyzed")
        return None
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    C_medians = [r['coherence']['C_median'] for r in results]
    f_coh_medians = [r['coherence']['f_coh_median'] for r in results]
    
    print(f"\nAnalyzed {len(results)} events:")
    for r in results:
        print(f"  {r['event']}: C = {r['coherence']['C_median']:.3f}, "
              f"f_coh = {r['coherence']['f_coh_median']:.3f}")
    
    print(f"\nOverall statistics:")
    print(f"  Mean C: {np.mean(C_medians):.3f} ± {np.std(C_medians):.3f}")
    print(f"  Mean f_coh: {np.mean(f_coh_medians):.3f} ± {np.std(f_coh_medians):.3f}")
    
    return results


# ============================================================================
# PART 7: MAIN EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Σ-Gravity LIGO Analysis Pipeline")
    print("="*60)
    
    # Check packages are installed
    if not check_and_install_packages():
        print("\nPlease install required packages first.")
        exit(1)
    
    # Example: List available events
    print("\n--- Available Catalogs ---")
    list_available_events()
    
    # Example: Analyze GW150914 (the first detection)
    # 
    # IMPORTANT: You need to provide M_bar - your estimate of what the
    # mass "should be" without coherence enhancement.
    #
    # For GW150914:
    # - GR-inferred total mass: ~65 M_sun
    # - If you think coherence adds ~10%, M_bar would be ~59 M_sun
    # - Adjust based on your Σ-Gravity predictions!
    
    print("\n--- Single Event Analysis ---")
    
    # =========================================================================
    # MODIFY THIS SECTION FOR YOUR ANALYSIS:
    # =========================================================================
    
    # Event to analyze
    event_name = "GW150914"
    
    # Your M_bar estimate (what mass you predict WITHOUT coherence)
    # This is the key theoretical input from your Σ-Gravity model
    M_bar_estimate = 60.0  # M_sun - ADJUST THIS BASED ON YOUR THEORY
    
    # Your coherence wavelength from Σ-Gravity (you derived ~2.2 kpc)
    lambda_coh = 2.2  # kpc
    
    # =========================================================================
    
    try:
        result = analyze_event_coherence(
            event_name=event_name,
            M_bar_estimate=M_bar_estimate,
            lambda_coh_kpc=lambda_coh
        )
        
        if result is not None:
            plot_coherence_results(result, save_path=f"{event_name}_coherence.png")
    
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("\nThis might be due to network issues or missing packages.")
        print("Try running the individual functions to debug.")
    
    # Example: Batch analysis
    # Uncomment to analyze multiple events:
    """
    events_to_analyze = {
        "GW150914": 60.0,   # Adjust M_bar estimates for each event
        "GW190521": 130.0,  # based on your Σ-Gravity predictions
        "GW190814": 25.0,
    }
    batch_results = batch_analyze_events(events_to_analyze, lambda_coh_kpc=2.2)
    """

"""
Real Spacetime Curvature Data Sources
=====================================

Loaders for actual observational/simulated curvature data:

1. SXS Catalog - Numerical relativity binary black hole simulations
2. LIGO/GWOSC - Gravitational wave strain data
3. Einstein Toolkit - Open-source NR framework outputs

Data Sources:
- SXS: https://data.black-holes.org/waveforms/catalog.html
- GWOSC: https://gwosc.org/
- NANOGrav: https://nanograv.org/science/data

Requirements:
    pip install requests h5py numpy
    # For LIGO data:
    pip install gwosc
"""

import numpy as np
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

# Cache directory for downloaded data
CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ============================================
# SXS NUMERICAL RELATIVITY CATALOG
# ============================================

SXS_CATALOG_URL = "https://data.black-holes.org/waveforms/catalog.json"
SXS_DATA_URL = "https://data.black-holes.org/waveforms"

def fetch_sxs_catalog() -> Dict:
    """
    Fetch the SXS waveform catalog metadata.
    
    Returns dict with 'simulations' key containing available simulations.
    """
    cache_file = CACHE_DIR / "sxs_catalog.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
            # Handle both raw and wrapped formats
            if 'simulations' in data:
                return data['simulations']
            return data
    
    print("Fetching SXS catalog...")
    try:
        response = requests.get(SXS_CATALOG_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Cache the full response
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        # Extract simulations
        simulations = data.get('simulations', data)
        print(f"  Found {len(simulations)} simulations")
        return simulations
    except Exception as e:
        print(f"  Error fetching catalog: {e}")
        return {}


def list_sxs_simulations(max_show: int = 20) -> None:
    """Print available SXS simulations."""
    catalog = fetch_sxs_catalog()
    
    # Handle different catalog formats
    if isinstance(catalog, dict):
        # New format: dict with simulation names as keys
        sim_list = list(catalog.items())[:max_show]
        print(f"\nSXS Catalog ({len(catalog)} simulations)")
        print("=" * 60)
        for name, sim in sim_list:
            q = sim.get('initial_mass_ratio', '?')
            print(f"  {name}: q={q}")
        if len(catalog) > max_show:
            print(f"  ... and {len(catalog) - max_show} more")
    elif isinstance(catalog, list):
        # Old format: list of dicts
        print(f"\nSXS Catalog ({len(catalog)} simulations)")
        print("=" * 60)
        for i, sim in enumerate(catalog[:max_show]):
            name = sim.get('name', 'Unknown')
            q = sim.get('initial_mass_ratio', '?')
            chi1 = sim.get('initial_dimensionless_spin1', [0,0,0])
            chi2 = sim.get('initial_dimensionless_spin2', [0,0,0])
            print(f"  {name}: q={q}, χ₁={chi1[2] if chi1 else 0:.2f}, χ₂={chi2[2] if chi2 else 0:.2f}")
        if len(catalog) > max_show:
            print(f"  ... and {len(catalog) - max_show} more")
    else:
        print(f"\nSXS Catalog format unknown: {type(catalog)}")


def download_sxs_waveform(sim_name: str = "SXS:BBH:0001") -> Optional[Dict]:
    """
    Download waveform data from an SXS simulation.
    
    The waveform encodes the metric perturbation h_μν at infinity.
    
    Args:
        sim_name: Simulation identifier (e.g., "SXS:BBH:0001")
    
    Returns:
        Dictionary with time, h_plus, h_cross arrays
    """
    # Convert name format
    sim_id = sim_name.replace("SXS:BBH:", "").replace("SXS:", "")
    
    cache_file = CACHE_DIR / f"sxs_{sim_id}_waveform.npz"
    
    if cache_file.exists():
        print(f"Loading cached waveform for {sim_name}...")
        data = np.load(cache_file)
        return {k: data[k] for k in data.files}
    
    # Try to download from SXS
    # Note: Full data requires HDF5 files which are large
    # The public catalog has pre-extracted waveforms
    
    print(f"Downloading waveform for {sim_name}...")
    print("  Note: Full NR data requires HDF5 download (~GB)")
    print("  Using catalog metadata for now...")
    
    # For demo, return None - full implementation would download HDF5
    return None


# ============================================
# LIGO GRAVITATIONAL WAVE DATA
# ============================================

def check_gwosc_available() -> bool:
    """Check if gwosc package is installed."""
    try:
        import gwosc
        return True
    except ImportError:
        return False


def list_gw_events() -> List[str]:
    """
    List available gravitational wave events.
    
    Requires: pip install gwosc
    """
    if not check_gwosc_available():
        print("GWOSC package not installed. Run: pip install gwosc")
        return []
    
    from gwosc.datasets import find_datasets
    
    events = find_datasets(type='event')
    return events


def fetch_gw_strain(event: str = "GW150914", 
                    detector: str = "H1",
                    duration: float = 32.0) -> Optional[Dict[str, np.ndarray]]:
    """
    Fetch gravitational wave strain data for an event.
    
    The strain h(t) is the metric perturbation: g_μν = η_μν + h_μν
    
    Args:
        event: Event name (e.g., "GW150914", "GW170817")
        detector: Detector ("H1" = Hanford, "L1" = Livingston, "V1" = Virgo)
        duration: Duration in seconds around merger
    
    Returns:
        Dictionary with 'time', 'strain', 'sample_rate'
    
    Requires: pip install gwosc
    
    Data source: https://gwosc.org/
    """
    if not check_gwosc_available():
        print("GWOSC package not installed. Run: pip install gwosc")
        return None
    
    from gwosc.datasets import event_gps
    from gwosc import datasets
    
    try:
        # Get GPS time of event
        gps = event_gps(event)
        print(f"Event {event} at GPS time {gps}")
        
        # Try to load strain data
        # This requires the full gwosc package with data access
        from gwosc.locate import get_urls
        
        urls = get_urls(detector, gps - duration/2, gps + duration/2)
        print(f"  Data URLs: {urls[:2]}...")
        
        # For full implementation, would download and parse the data
        # Return format info for now
        return {
            'event': event,
            'detector': detector,
            'gps_time': gps,
            'urls': urls,
            'note': 'Use gwpy to load: TimeSeries.fetch_open_data(detector, start, end)'
        }
        
    except Exception as e:
        print(f"  Error fetching {event}: {e}")
        return None


def fetch_gw_strain_simple(event: str = "GW150914") -> Optional[Dict]:
    """
    Simplified strain data fetch using direct URL.
    
    Returns pre-processed strain data for the event.
    """
    # GWOSC provides direct download links for tutorial data
    tutorial_url = f"https://gwosc.org/eventapi/json/GWTC-1-confident/"
    
    try:
        print(f"Fetching GWTC catalog for {event}...")
        response = requests.get(tutorial_url, timeout=30)
        response.raise_for_status()
        catalog = response.json()
        
        # Find our event
        events = catalog.get('events', {})
        if event in events:
            event_data = events[event]
            print(f"  Found {event}")
            print(f"  GPS: {event_data.get('GPS', 'unknown')}")
            print(f"  Mass1: {event_data.get('mass_1_source', 'unknown')} M☉")
            print(f"  Mass2: {event_data.get('mass_2_source', 'unknown')} M☉")
            return event_data
        else:
            print(f"  Event {event} not found in GWTC-1")
            return None
            
    except Exception as e:
        print(f"  Error: {e}")
        return None


# ============================================
# GRAVITATIONAL WAVE → CURVATURE
# ============================================

def strain_to_riemann_estimate(h_plus: np.ndarray, 
                                h_cross: np.ndarray,
                                time: np.ndarray,
                                distance: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Estimate Riemann tensor components from GW strain.
    
    In the transverse-traceless gauge, the metric perturbation is:
    h_μν = [0   0    0    0  ]
           [0   h_+  h_×  0  ]
           [0   h_×  -h_+ 0  ]
           [0   0    0    0  ]
    
    The Riemann tensor for GWs:
    R_i0j0 = -½ ∂²h_ij/∂t²
    
    Args:
        h_plus: Plus polarization strain
        h_cross: Cross polarization strain  
        time: Time array
        distance: Distance to source (for scaling)
    
    Returns:
        Dictionary with estimated Riemann components
    """
    dt = time[1] - time[0]
    
    # Second time derivative
    h_plus_ddot = np.gradient(np.gradient(h_plus, dt), dt)
    h_cross_ddot = np.gradient(np.gradient(h_cross, dt), dt)
    
    # Riemann components (in TT gauge, propagating along z)
    # R_x0x0 = -½ ∂²h_+/∂t²
    # R_y0y0 = +½ ∂²h_+/∂t²
    # R_x0y0 = -½ ∂²h_×/∂t²
    
    R_x0x0 = -0.5 * h_plus_ddot
    R_y0y0 = 0.5 * h_plus_ddot
    R_x0y0 = -0.5 * h_cross_ddot
    
    return {
        'time': time,
        'R_x0x0': R_x0x0,
        'R_y0y0': R_y0y0,
        'R_x0y0': R_x0y0,
        'h_plus': h_plus,
        'h_cross': h_cross,
    }


# ============================================
# SYNTHETIC GW DATA (for testing)
# ============================================

def generate_inspiral_waveform(m1: float = 30.0, 
                                m2: float = 30.0,
                                distance_mpc: float = 400.0,
                                f_low: float = 20.0,
                                sample_rate: float = 4096.0,
                                duration: float = 8.0) -> Dict[str, np.ndarray]:
    """
    Generate a simple inspiral gravitational waveform.
    
    Uses the leading-order quadrupole formula (Newtonian chirp).
    
    Args:
        m1, m2: Component masses in solar masses
        distance_mpc: Distance in megaparsecs
        f_low: Starting frequency in Hz
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    
    Returns:
        Dictionary with time, h_plus, h_cross
    """
    # Constants
    G = 6.674e-11
    c = 3e8
    M_sun = 1.989e30
    Mpc = 3.086e22
    
    # Convert to SI
    m1_kg = m1 * M_sun
    m2_kg = m2 * M_sun
    d = distance_mpc * Mpc
    
    # Chirp mass
    M_chirp = (m1_kg * m2_kg)**(3/5) / (m1_kg + m2_kg)**(1/5)
    
    # Time array (counting backwards from merger)
    dt = 1.0 / sample_rate
    t = np.arange(-duration, 0, dt)
    
    # Time to coalescence (from frequency evolution)
    # f(t) = (1/8π) × (5/256)^(3/8) × (GM_c/c³)^(-5/8) × τ^(-3/8)
    # where τ = t_c - t
    
    tau = -t + 0.01  # Time to merger (avoid division by zero)
    
    # Frequency evolution
    prefactor = (5 / (256 * tau))**(3/8)
    f_factor = (c**3 / (G * M_chirp))**(5/8) / (8 * np.pi)
    f_gw = prefactor * f_factor
    
    # Clip to physical range
    f_gw = np.clip(f_gw, f_low, sample_rate/2)
    
    # Phase
    phi = 2 * np.pi * np.cumsum(f_gw) * dt
    
    # Amplitude
    # h ~ (GM_c/c²d) × (πGM_c f/c³)^(2/3)
    amp_factor = (G * M_chirp / (c**2 * d)) 
    amp_factor *= (np.pi * G * M_chirp * f_gw / c**3)**(2/3)
    
    # Waveform
    h_plus = amp_factor * np.cos(phi)
    h_cross = amp_factor * np.sin(phi)
    
    # Mask out post-merger (simple ringdown would go here)
    mask = t < -0.01
    h_plus[~mask] *= np.exp(-100 * (t[~mask] + 0.01))
    h_cross[~mask] *= np.exp(-100 * (t[~mask] + 0.01))
    
    return {
        'time': t,
        'h_plus': h_plus,
        'h_cross': h_cross,
        'frequency': f_gw,
        'parameters': {
            'm1': m1,
            'm2': m2,
            'M_chirp': M_chirp / M_sun,
            'distance_mpc': distance_mpc,
        }
    }


# ============================================
# MAIN: DEMONSTRATE DATA ACCESS
# ============================================

def main():
    print("=" * 60)
    print("REAL CURVATURE DATA SOURCES")
    print("=" * 60)
    
    # 1. SXS Catalog
    print("\n--- SXS Numerical Relativity Catalog ---")
    list_sxs_simulations(max_show=5)
    
    # 2. LIGO Events
    print("\n--- LIGO Gravitational Wave Events ---")
    if check_gwosc_available():
        events = list_gw_events()
        print(f"  Found {len(events)} events")
        print(f"  Examples: {events[:5]}")
    else:
        print("  Install gwosc: pip install gwosc")
    
    # 3. Fetch GW150914 metadata
    print("\n--- GW150914 Event Data ---")
    gw_data = fetch_gw_strain_simple("GW150914")
    
    # 4. Generate synthetic waveform
    print("\n--- Synthetic Inspiral Waveform ---")
    waveform = generate_inspiral_waveform(m1=36, m2=29, distance_mpc=410)
    print(f"  Duration: {waveform['time'][-1] - waveform['time'][0]:.2f} s")
    print(f"  Peak strain: {np.max(np.abs(waveform['h_plus'])):.2e}")
    print(f"  Chirp mass: {waveform['parameters']['M_chirp']:.1f} M☉")
    
    # 5. Convert to Riemann estimate
    print("\n--- Riemann Tensor from GW Strain ---")
    riemann = strain_to_riemann_estimate(
        waveform['h_plus'],
        waveform['h_cross'],
        waveform['time']
    )
    print(f"  Peak |R_x0x0|: {np.max(np.abs(riemann['R_x0x0'])):.2e} (1/s²)")
    print(f"  This is spacetime curvature from gravitational waves!")
    
    print("\n" + "=" * 60)
    print("To use real LIGO data:")
    print("  pip install gwosc gwpy")
    print("  from gwpy.timeseries import TimeSeries")
    print("  strain = TimeSeries.fetch_open_data('H1', start, end)")
    print("=" * 60)


if __name__ == "__main__":
    main()

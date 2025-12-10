"""
Morphology-based gates for time-coherence kernel.
Suppresses enhancement for galaxies with bars, warps, or large bulges.
"""

import numpy as np

def morphology_gate(galaxy_meta: dict) -> float:
    """
    Compute morphology gate factor G_morph in [0, 1].
    
    Parameters:
    -----------
    galaxy_meta : dict
        Galaxy metadata from SPARC summary CSV
        
    Returns:
    --------
    G_morph : float
        Gate factor (1.0 = no suppression, 0.0 = full suppression)
    """
    gate = 1.0
    
    # Strong bar flag
    bar_flag = galaxy_meta.get("bar_flag", 0)
    if bar_flag == 1:
        gate *= 0.5  # Suppress by 50% for barred galaxies
    
    # Warp flag
    warp_flag = galaxy_meta.get("warp_flag", 0)
    if warp_flag == 1:
        gate *= 0.7  # Suppress by 30% for warped galaxies
    
    # Large bulge fraction
    bulge_frac = galaxy_meta.get("bulge_frac", 0.0)
    if bulge_frac > 0.4:
        gate *= 0.6  # Suppress by 40% for bulge-dominated galaxies
    elif bulge_frac > 0.2:
        gate *= 0.8  # Suppress by 20% for moderate bulges
    
    # Low inclination (face-on) - less reliable rotation curves
    inclination = galaxy_meta.get("inclination", 90.0)
    if inclination < 30.0:
        gate *= 0.7  # Suppress by 30% for face-on galaxies
    
    return gate

def apply_morphology_gate(K_raw: np.ndarray, galaxy_meta: dict) -> np.ndarray:
    """
    Apply morphology gate to raw kernel.
    
    Parameters:
    -----------
    K_raw : np.ndarray
        Raw kernel from time-coherence calculation
    galaxy_meta : dict
        Galaxy metadata
        
    Returns:
    --------
    K : np.ndarray
        Gated kernel
    """
    import numpy as np
    G_morph = morphology_gate(galaxy_meta)
    return K_raw * G_morph


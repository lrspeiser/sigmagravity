"""
Σ-Gravity Gate Modeling and Validation Package

This package provides tools to derive, test, and validate gate functions
that suppress coherence where needed (solar system, bulges, bars) while
allowing it in extended, low-acceleration environments.

Quick Start
-----------
>>> from gate_core import G_distance, K_sigma_gravity
>>> import numpy as np
>>> R = np.logspace(-1, 2, 100)
>>> g_bar = 1e-10 / R**2
>>> K = K_sigma_gravity(R, g_bar, A=0.6, ell0=5.0,
...                     gate_type='distance',
...                     gate_params={'R_min': 1.0, 'alpha': 2.0, 'beta': 1.0})

Modules
-------
gate_core : Core gate functions (distance, acceleration, exponential, unified)
gate_modeling : Visualization and parameter exploration
gate_fitting_tool : Fit gates to rotation curve data
tests/ : Invariant tests and validation

Documentation
-------------
README.md : Package overview
gate_quick_reference.md : Quick formulas and usage
gate_mathematical_framework.md : Complete theory
"""

__version__ = '1.0.0'
__author__ = 'Σ-Gravity Team'

# Core functions
from .gate_core import (
    G_distance,
    G_acceleration,
    G_bulge_exponential,
    G_unified,
    G_solar_system,
    C_burr_XII,
    K_sigma_gravity,
    check_gate_properties
)

__all__ = [
    'G_distance',
    'G_acceleration',
    'G_bulge_exponential',
    'G_unified',
    'G_solar_system',
    'C_burr_XII',
    'K_sigma_gravity',
    'check_gate_properties'
]


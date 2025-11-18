"""
Create fiducial unified kernel parameters JSON file.
"""

from pathlib import Path
from unified_kernel import UnifiedKernelParams
from f_missing_mass_model import FMissingParams

# Create fiducial parameters based on current best fits
uk = UnifiedKernelParams(
    use_rough=True,
    use_f_missing=True,
    f_amp=1.0,
    extra_amp=0.25,  # Start conservative
    f_missing=FMissingParams(
        A0=10.02,
        sigma_ref=14.8,
        R_ref=12.3,
        a_sigma=0.10,
        a_Rd=0.31,
        F_max=5.0,
        use_sigma_gate=True,
        sigma_gate_ref=25.0,
        gamma_sigma=1.0,
        gate_floor=0.1,
    ),
    use_morph_gate=False,  # Start with morphology gate off
    morph_floor=0.2,
)

# Save to JSON
project_root = Path(__file__).parent.parent
out_path = project_root / "time-coherence" / "unified_kernel_fiducial.json"
uk.to_json(out_path)
print(f"Created fiducial parameters at: {out_path}")

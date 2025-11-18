"""
Update test scripts to use fiducial parameters by default.
"""

import json
from pathlib import Path
import re

def load_fiducial():
    """Load fiducial parameters."""
    fiducial_path = Path("time-coherence/time_coherence_fiducial.json")
    if not fiducial_path.exists():
        raise FileNotFoundError(f"Fiducial parameters not found: {fiducial_path}")
    
    with open(fiducial_path, "r") as f:
        return json.load(f)

def update_test_script(script_path: Path, fiducial: dict):
    """Update a test script to use fiducial parameters."""
    if not script_path.exists():
        print(f"Warning: {script_path} not found, skipping")
        return False
    
    content = script_path.read_text(encoding='utf-8')
    original_content = content
    
    # Replace default parameter values with fiducial values
    replacements = {
        r'alpha_length\s*=\s*0\.037': f'alpha_length={fiducial["alpha_length"]}',
        r'beta_sigma\s*=\s*1\.5': f'beta_sigma={fiducial["beta_sigma"]}',
        r'backreaction_cap\s*=\s*None': f'backreaction_cap={fiducial["backreaction_cap"]}',
        r'A_global\s*=\s*1\.0': f'A_global={fiducial["A_global"]}',
        r'p\s*=\s*0\.757': f'p={fiducial["p"]}',
        r'n_coh\s*=\s*0\.5': f'n_coh={fiducial["n_coh"]}',
    }
    
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)
    
    # Add fiducial parameter loading at the top if not present
    if "fiducial" not in content.lower() and "time_coherence_fiducial" not in content:
        # Add import and load after imports
        import_section = re.search(r'(from __future__ import.*?\n|import.*?\n)+', content)
        if import_section:
            insert_pos = import_section.end()
            fiducial_load = f'''
# Load fiducial parameters
_fiducial_path = Path(__file__).parent / "time_coherence_fiducial.json"
if _fiducial_path.exists():
    with open(_fiducial_path, "r") as f:
        _fiducial = json.load(f)
else:
    _fiducial = {{"alpha_length": 0.037, "beta_sigma": 1.5, "backreaction_cap": 10.0, "A_global": 1.0, "p": 0.757, "n_coh": 0.5}}

'''
            content = content[:insert_pos] + fiducial_load + content[insert_pos:]
    
    if content != original_content:
        script_path.write_text(content, encoding='utf-8')
        return True
    return False

def main():
    """Update all test scripts."""
    fiducial = load_fiducial()
    
    scripts = [
        Path("time-coherence/test_mw_coherence.py"),
        Path("time-coherence/test_sparc_coherence.py"),
        Path("time-coherence/test_cluster_coherence.py"),
        Path("time-coherence/test_fitted_params.py"),
    ]
    
    print("=" * 80)
    print("UPDATING TEST SCRIPTS TO USE FIDUCIAL PARAMETERS")
    print("=" * 80)
    
    updated = []
    for script_path in scripts:
        if update_test_script(script_path, fiducial):
            updated.append(script_path.name)
            print(f"  Updated: {script_path.name}")
        else:
            print(f"  No changes: {script_path.name}")
    
    print(f"\nUpdated {len(updated)} scripts")
    print(f"\nFiducial parameters:")
    for key, value in fiducial.items():
        if key not in ["source", "notes", "performance"]:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()


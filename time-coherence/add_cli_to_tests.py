"""
Add --params-json and --out-* arguments to test scripts.
"""

import re
from pathlib import Path

def add_cli_to_script(script_path: Path):
    """Add CLI argument parsing to a test script."""
    content = script_path.read_text(encoding='utf-8')
    original = content
    
    # Check if argparse is already imported
    if 'import argparse' in content:
        return False
    
    # Add argparse import after other imports
    import_match = re.search(r'(from __future__ import.*?\n|import.*?\n)+', content)
    if import_match:
        insert_pos = import_match.end()
        argparse_import = "import argparse\n"
        content = content[:insert_pos] + argparse_import + content[insert_pos:]
    
    # Find main() function and add argument parsing at the start
    main_match = re.search(r'def main\(\):', content)
    if main_match:
        # Find the first line after def main():
        start_pos = main_match.end()
        next_line = content.find('\n', start_pos) + 1
        
        # Add argument parser
        parser_code = '''
    parser = argparse.ArgumentParser(description='Test time-coherence kernel')
    parser.add_argument('--params-json', type=str, 
                       default='time-coherence/time_coherence_fiducial.json',
                       help='Path to parameters JSON file')
    parser.add_argument('--out-json', type=str, 
                       default='time-coherence/mw_coherence_test.json',
                       help='Path to output JSON file')
    args = parser.parse_args()
    
    # Load parameters
    params_path = Path(args.params_json)
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = json.load(f)
    else:
        print(f"Warning: {params_path} not found, using defaults")
        params = _fiducial
    
'''
        content = content[:next_line] + parser_code + content[next_line:]
    
    if content != original:
        script_path.write_text(content, encoding='utf-8')
        return True
    return False

# This is a helper script - actual implementation will be done directly
print("This script shows the pattern. Implementing directly in test scripts...")


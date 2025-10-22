#!/usr/bin/env python3
"""Test 2: Pandoc + LaTeX"""
import subprocess
from pathlib import Path

# Check if pandoc is available
try:
    result = subprocess.run(['pandoc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Pandoc found: {result.stdout.split()[1]}")
        
        # Generate PDF with pandoc (uses LaTeX backend)
        subprocess.run([
            'pandoc',
            'test_section_25.md',
            '-o', 'test_pandoc.pdf',
            '--pdf-engine=pdflatex',
            '-V', 'geometry:margin=20mm',
            '-V', 'fontsize=12pt',
            '-V', 'mainfont=Times New Roman'
        ], check=True)
        
        print("✓ test_pandoc.pdf generated")
    else:
        print("✗ Pandoc not working")
except FileNotFoundError:
    print("✗ Pandoc not installed")
    print("  Install with: choco install pandoc (or download from pandoc.org)")
except Exception as e:
    print(f"✗ Error: {e}")

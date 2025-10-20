#!/usr/bin/env python3
"""
make_pdf_latex.py — Generate publication-quality PDF using Pandoc + LaTeX

Advantages over Chrome/MathJax:
- No line breaks in formulas
- Publication-quality math rendering
- Proper hyphenation and spacing
- Standard for academic papers

Usage:
  python scripts/make_pdf_latex.py [--md README.md] [--out sigmagravity_paper.pdf]
"""
import argparse
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if pandoc and pdflatex are available"""
    missing = []
    
    # Check pandoc
    try:
        result = subprocess.run(['pandoc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"[OK] {version}")
        else:
            missing.append('pandoc')
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing.append('pandoc')
    
    # Check pdflatex (from MiKTeX)
    try:
        result = subprocess.run(['pdflatex', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"[OK] {version}")
        else:
            missing.append('pdflatex (MiKTeX)')
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing.append('pdflatex (MiKTeX)')
    
    if missing:
        print(f"\n[ERROR] Missing dependencies: {', '.join(missing)}")
        print("\nTo install:")
        print("  1. Run: .\\install_latex_pdf.ps1")
        print("  2. Or install manually:")
        print("     - Pandoc: https://pandoc.org/installing.html")
        print("     - MiKTeX: https://miktex.org/download")
        print("\nAfter installation, close and reopen PowerShell.")
        return False
    
    return True

def generate_pdf_latex(md_path: Path, pdf_path: Path):
    """Generate PDF using Pandoc + LaTeX backend"""
    
    print(f"\nGenerating PDF from {md_path}...")
    print(f"Output: {pdf_path}")
    print("\nThis may take 1-2 minutes for first run (LaTeX packages download)...")
    
    # Pandoc options for academic papers
    cmd = [
        'pandoc',
        str(md_path),
        '-o', str(pdf_path),
        '--pdf-engine=xelatex',  # XeLaTeX handles Unicode (Σ, †, ℓ, etc.)
        '-V', 'geometry:margin=20mm',
        '-V', 'fontsize=11pt',
    ]
    
    try:
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"\n[SUCCESS] PDF generated: {size_mb:.1f} MB")
            print(f"   {pdf_path}")
            return True
        else:
            print(f"\n[ERROR] Error generating PDF:")
            print(result.stderr)
            
            # Common error hints
            if 'pdflatex' in result.stderr.lower():
                print("\nHint: pdflatex not found. Install MiKTeX.")
            elif 'unicode' in result.stderr.lower():
                print("\nHint: Unicode issue. Try adding --pdf-engine=xelatex")
            
            return False
            
    except subprocess.TimeoutExpired:
        print("\n[ERROR] Timeout: PDF generation took > 5 minutes")
        print("   This usually means LaTeX is downloading packages.")
        print("   Try running again.")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality PDF using Pandoc + LaTeX'
    )
    parser.add_argument('--md', default='README.md', 
                       help='Input Markdown file (default: README.md)')
    parser.add_argument('--out', default='sigmagravity_paper.pdf',
                       help='Output PDF file (default: sigmagravity_paper.pdf)')
    
    args = parser.parse_args()
    
    md_path = Path(args.md)
    pdf_path = Path(args.out)
    
    # Validate input
    if not md_path.exists():
        print(f"[ERROR] Input file not found: {md_path}")
        sys.exit(1)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Generate PDF
    success = generate_pdf_latex(md_path, pdf_path)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

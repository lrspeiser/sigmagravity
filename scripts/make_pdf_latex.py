#!/usr/bin/env python3
"""
make_pdf_latex.py — Generate publication-quality PDF using Pandoc + LaTeX

Advantages over Chrome/MathJax:
- No line breaks in formulas
- Publication-quality math rendering
- Proper hyphenation and spacing
- Standard for academic papers

Usage:
  python scripts/make_pdf_latex.py [--md README.md] [--out docs/sigmagravity_paper.pdf]
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
        print("  1. Run: .\\docs\\install_latex_pdf.ps1")
        print("  2. Or install manually:")
        print("     - Pandoc: https://pandoc.org/installing.html")
        print("     - MiKTeX: https://miktex.org/download")
        print("\nAfter installation, close and reopen PowerShell.")
        return False
    
    return True

def generate_pdf_latex(md_path: Path, pdf_path: Path, two_column: bool = False, journal_style: bool = False):
    """Generate PDF using Pandoc + LaTeX backend
    
    Args:
        md_path: Input markdown file
        pdf_path: Output PDF file
        two_column: Use two-column layout (like PRD)
        journal_style: Use journal-like formatting (smaller margins, etc.)
    """
    
    print(f"\nGenerating PDF from {md_path}...")
    print(f"Output: {pdf_path}")
    if two_column:
        print("Layout: Two-column (journal style)")
    if journal_style:
        print("Style: Physical Review D-like formatting")
    print("\nThis may take 1-2 minutes for first run (LaTeX packages download)...")
    
    # LaTeX header for journal-style formatting
    if journal_style:
        latex_header = r'''
\usepackage{float}
% Allow floats to break if needed
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.85}
% Tighter paragraph spacing
\setlength{\parskip}{0.5em}
'''
    else:
        latex_header = r'''
\usepackage{float}
% Allow floats to break if needed
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.85}
'''
    
    # Write header to temp file
    header_file = pdf_path.parent / '_latex_header.tex'
    header_file.write_text(latex_header, encoding='utf-8')
    
    # Pandoc options
    cmd = [
        'pandoc',
        str(md_path),
        '-o', str(pdf_path),
        '--pdf-engine=xelatex',  # XeLaTeX handles Unicode (Σ, †, ℓ, etc.)
        '-H', str(header_file),  # Include header for image sizing
    ]
    
    # Geometry and layout options
    if journal_style:
        cmd.extend([
            '-V', 'geometry:top=18mm,left=18mm,right=18mm,bottom=25mm',
            '-V', 'fontsize=10pt',
        ])
    else:
        cmd.extend([
            '-V', 'geometry:top=20mm,left=20mm,right=20mm,bottom=30mm',
            '-V', 'fontsize=11pt',
        ])
    
    # Two-column layout
    if two_column:
        cmd.extend([
            '-V', 'classoption=twocolumn',
        ])
    
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
    finally:
        # Clean up temp header file
        if header_file.exists():
            header_file.unlink()

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality PDF using Pandoc + LaTeX'
    )
    parser.add_argument('--md', default='README.md', 
                       help='Input Markdown file (default: README.md)')
    parser.add_argument('--out', default='docs/sigmagravity_paper.pdf',
                       help='Output PDF file (default: docs/sigmagravity_paper.pdf)')
    parser.add_argument('--two-column', action='store_true',
                       help='Use two-column layout (like Physical Review D)')
    parser.add_argument('--journal', action='store_true',
                       help='Use journal-style formatting (smaller margins, tighter spacing)')
    
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
    success = generate_pdf_latex(md_path, pdf_path, 
                                  two_column=args.two_column,
                                  journal_style=args.journal)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

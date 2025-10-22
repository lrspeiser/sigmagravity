# Documentation & Build Artifacts

This folder contains all documentation, build scripts, and generated files related to PDF generation and LaTeX compilation.

## Contents

### PDF Generation Files
- **sigmagravity_paper.pdf** — Generated PDF of the main paper (from README.md)
- **sigmagravity_paper.tex** — LaTeX source (generated from README.md)
- **sigmagravity_paper.aux**, **.log**, **.out** — LaTeX compilation artifacts
- **paper_formatted.html** — HTML version with formatted figures

### Build Scripts & Tools
- **rebuild_pdf.bat** — Quick rebuild script (Windows batch file)
- **install_latex_pdf.ps1** — PowerShell script to install Pandoc + MiKTeX

### Documentation
- **PDF_GENERATION.md** — Complete guide to the PDF generation workflow
- **PDF_GENERATION_GUIDE.md** — Comparison of different PDF generation methods
- **PDF_QUALITY_ASSESSMENT.md** — Quality metrics and assessment
- **INSTALL_LATEX_MANUAL.md** — Manual installation instructions for LaTeX tools

### Project Documentation
- **LATEX_STATUS.md** — Status of LaTeX compilation
- **NUMBERING_AUDIT.md** — Audit of equation/figure numbering
- **NUMBERING_FIXES_COMPLETE.md** — Documentation of numbering fixes
- **THEORY_ENHANCEMENTS_SUMMARY.md** — Summary of theoretical enhancements
- **WARP.md** — Documentation about warp/curvature analysis

## Quick Start

### Generate PDF from README.md

**Option 1: Run the batch file**
```bash
cd docs
rebuild_pdf.bat
```

**Option 2: Use Python directly**
```bash
# From project root
python scripts/make_pdf_latex.py
```

**Option 3: Alternative method (Chrome/MathJax)**
```bash
python scripts/make_pdf.py
```

## Output Locations

All generated files are output to this `docs/` folder to keep the project root clean.

- PDF: `docs/sigmagravity_paper.pdf`
- LaTeX: `docs/sigmagravity_paper.tex`
- HTML: `docs/paper_formatted.html`

## Source File

The source file for all PDFs is the main **`README.md`** in the project root, which serves as both:
1. The academic paper content
2. The GitHub repository landing page

This ensures a single source of truth for the paper content.


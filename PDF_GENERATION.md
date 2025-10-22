# PDF Generation Workflow

This repository automatically converts `README.md` to a professional LaTeX-formatted PDF paper.

## Quick Start

### Option 1: Windows Batch Script
```bash
rebuild_pdf.bat
```
or simply double-click `rebuild_pdf.bat`

### Option 2: Python Script Directly
```bash
python scripts/md_to_latex.py
```

## What It Does

The conversion script (`scripts/md_to_latex.py`) performs the following:

1. **Reads** `README.md` (full Markdown source)
2. **Converts** Markdown to LaTeX:
   - Sections → `\section{}`, `\subsection{}`, `\subsubsection{}`
   - Math `$...$` → preserved as-is
   - Block equations `$$...$$` → `\begin{equation}...\end{equation}`
   - Tables `| ... |` → `\begin{tabular}...\end{tabular}`
   - Lists → `\begin{itemize}` or `\begin{enumerate}`
   - Bold/italic → `\textbf{}` / `\textit{}`
   - Images → `\includegraphics{}`
   - Links → `\href{}{}`
3. **Writes** `sigmagravity_paper.tex` (complete LaTeX document)
4. **Compiles** with `pdflatex` (2 passes for cross-references)
5. **Produces** `sigmagravity_paper.pdf`

## Output

- **sigmagravity_paper.tex** — Generated LaTeX source (can be edited manually if needed)
- **sigmagravity_paper.pdf** — Final PDF document (~2.2 MB, full paper with figures)
- **sigmagravity_paper.log** — LaTeX compilation log (for debugging)

## Requirements

- **Python 3.8+**
- **pdflatex** (MiKTeX, TeX Live, or similar LaTeX distribution)
- **Figures** must exist at the paths referenced in `README.md` (e.g., `figures/rc_gallery.png`)

## Customization

### LaTeX Preamble

Edit `scripts/md_to_latex.py` lines 19-69 to customize:
- Page margins: `\usepackage[margin=20mm]{geometry}`
- Fonts: Add `\usepackage{times}` or similar
- Unicode characters: Add `\DeclareUnicodeCharacter{}`

### Manual LaTeX Edits

You can edit `sigmagravity_paper.tex` directly after generation, but changes will be overwritten on next rebuild. For persistent changes, either:
1. Modify `README.md` (preferred — single source of truth)
2. Modify the conversion logic in `scripts/md_to_latex.py`

## Workflow Integration

### After Every README Update

```bash
# Edit README.md
# ...

# Regenerate PDF
rebuild_pdf.bat

# Commit both
git add README.md sigmagravity_paper.pdf sigmagravity_paper.tex
git commit -m "Update paper content"
git push origin main
```

### CI/CD (Optional)

You can automate PDF generation in GitHub Actions:

```yaml
name: Generate PDF
on:
  push:
    paths:
      - 'README.md'
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: xu-cheng/latex-action@v2
        with:
          root_file: sigmagravity_paper.tex
          pre_compile: python scripts/md_to_latex.py
      - uses: actions/upload-artifact@v3
        with:
          name: sigmagravity_paper
          path: sigmagravity_paper.pdf
```

## Troubleshooting

### "pdflatex not found"
Install a LaTeX distribution:
- **Windows**: [MiKTeX](https://miktex.org/download)
- **macOS**: [MacTeX](https://www.tug.org/mactex/)
- **Linux**: `sudo apt-get install texlive-full`

### "File ended while scanning use of..."
Indicates unmatched braces or special characters. Check:
- Underscores outside math mode (should be `\_`)
- Ampersands (should be `\&`)
- Percent signs (should be `\%`)

The script automatically escapes these, but manual edits to `.tex` may introduce issues.

### Missing figures
Ensure all image paths in `README.md` exist:
```bash
ls figures/*.png
```

If images are missing, either:
1. Generate them (run relevant Python scripts)
2. Comment out the `![...]()` lines in `README.md`

### Unicode errors
The script handles common Unicode characters (Σ, ℓ, ±, etc.). If you encounter new ones:
1. Find the Unicode code point (e.g., U+03A3)
2. Add `\DeclareUnicodeCharacter{03A3}{...}` in the preamble

## Architecture

```
README.md  (source)
    ↓
scripts/md_to_latex.py  (parser + LaTeX generator)
    ↓
sigmagravity_paper.tex  (intermediate)
    ↓
pdflatex  (compiler)
    ↓
sigmagravity_paper.pdf  (final output)
```

**Single source of truth**: `README.md`  
**Reproducible**: Re-run `rebuild_pdf.bat` anytime

## Version History

- **2025-01-21**: Initial automated PDF pipeline
  - Full README.md → LaTeX conversion
  - Tables, equations, figures, sections preserved
  - 2-pass compilation for cross-references
  - Batch script for Windows

---

**Maintained by**: Leonard Speiser  
**Repository**: sigmagravity  
**Status**: Production-ready

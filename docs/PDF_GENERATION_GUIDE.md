# PDF Generation Methods Comparison

## Problem: Formulas Breaking Across Lines

Your current method (Chrome + MathJax) sometimes wraps long formulas incorrectly, causing them to break mid-equation.

---

## Solution: Pandoc + LaTeX

### Why LaTeX?
- **Gold standard** for academic papers
- **Native math rendering** — no line breaks in formulas
- Used by arXiv, most journals, and publishers
- Proper hyphenation, spacing, and typography
- Handles complex equations correctly

### Installation:

#### Option 1: Automated (Recommended)
```powershell
# Run the installer script
.\docs\install_latex_pdf.ps1
```

#### Option 2: Manual

**Step 1 — Install Pandoc:**
```powershell
# Download installer from: https://pandoc.org/installing.html
# Or use Windows installer:
# https://github.com/jgm/pandoc/releases/latest
```

**Step 2 — Install MiKTeX (LaTeX):**
```powershell
# Download from: https://miktex.org/download
# Choose: "Basic MiKTeX Installer" (64-bit)
# Size: ~200 MB download, ~1 GB installed
# Time: 5-10 minutes
```

**Step 3 — Restart PowerShell** to update PATH

---

## Usage:

### Generate PDF with LaTeX:
```powershell
python scripts/make_pdf_latex.py
```

Output: `docs/sigmagravity_paper.pdf` (publication-quality)

### Compare with current method:
```powershell
# Current method (Chrome + MathJax)
python scripts/make_pdf.py

# New method (Pandoc + LaTeX)
python scripts/make_pdf_latex.py --out docs/sigmagravity_paper_latex.pdf
```

Then open both PDFs and compare formula rendering.

---

## Comparison Table:

| Feature | Chrome + MathJax | Pandoc + LaTeX |
|---------|------------------|----------------|
| **Install size** | 0 MB (Chrome already installed) | ~1 GB (MiKTeX) |
| **Generation time** | ~15 seconds | ~1-2 minutes (first run), ~30s after |
| **Formula quality** | Good, but can wrap incorrectly | **Perfect** ✓ |
| **Line breaks in math** | ❌ Sometimes happens | ✓ Never happens |
| **Typography** | Web-style | **Academic paper style** ✓ |
| **Standard for journals** | ❌ No | ✓ **Yes** |
| **arXiv submission** | ❌ Can't submit HTML | ✓ Can submit .tex source |

---

## What Will Change:

### Before (Chrome + MathJax):
```
Some formulas wrap like this: K(R) = A₀ (g†/g_bar
(R))^p C(R; ℓ₀, p, n_coh) G_bulge G_shear
```
**Problem**: Formula broken mid-equation

### After (Pandoc + LaTeX):
```
K(R) = A₀ (g†/g_bar(R))^p C(R; ℓ₀, p, n_coh) G_bulge G_shear
```
**Fixed**: Formula stays on one line or breaks intelligently at operators

---

## Troubleshooting:

### "pdflatex not found"
- MiKTeX not installed or not in PATH
- Solution: Restart PowerShell after installing MiKTeX

### "Timeout: PDF generation took > 5 minutes"
- MiKTeX is downloading packages on first run
- Solution: Wait, then run again. Subsequent runs will be fast.

### "Unicode character ... could not be printed"
- Some special characters not in default LaTeX fonts
- Solution: Edit `scripts/make_pdf_latex.py` line 71:
  ```python
  '--pdf-engine=xelatex',  # Change from pdflatex
  ```

### Formula still wraps
- Equation is truly too long for page width
- Solution: Rewrite equation as multi-line with aligned breaks:
  ```latex
  \begin{align}
  K(R) &= A₀ (g†/g_bar(R))^p \\
       &\quad \times C(R; ℓ₀, p, n_coh) G_bulge G_shear
  \end{align}
  ```

---

## Recommendation:

### For Draft/Review:
✓ Use **Chrome + MathJax** (current method)
- Fast
- No installation
- Good enough for internal review

### For Publication/Submission:
✓ Use **Pandoc + LaTeX** (new method)
- Install once
- Use for final PDF
- Required for journal/arXiv submission anyway

---

## Next Steps:

1. **Install** Pandoc + MiKTeX (one-time, 10 minutes)
   ```powershell
   .\docs\install_latex_pdf.ps1
   ```

2. **Test** on your document:
   ```powershell
   python scripts/make_pdf_latex.py
   ```

3. **Compare** PDFs:
   - Open `docs/sigmagravity_paper.pdf` (current method)
   - Generate `docs/sigmagravity_paper_latex.pdf` (new method)
   - Check formula rendering quality

4. **Decide**:
   - If LaTeX looks better → update your workflow
   - If Chrome is fine → keep current method

---

## File Locations:

- **Installer**: `docs/install_latex_pdf.ps1`
- **LaTeX script**: `scripts/make_pdf_latex.py`
- **Current script**: `scripts/make_pdf.py` (unchanged)
- **This guide**: `docs/PDF_GENERATION_GUIDE.md`

---

## One-Line Summary:

**Install Pandoc + MiKTeX (~10 min) → Run `python scripts/make_pdf_latex.py` → Get publication-quality PDF with perfect math rendering.**

# PDF Quality Assessment — Section 2.5

## Current Method: Chrome + MathJax

**File generated**: `test_section25_current_method.pdf` (122 KB)

### What to check manually:

Open the PDF and inspect these specific formulas from section 2.5:

#### 1. Display equation (line 8-9):
```
g_model(R) = g_bar(R)[1 + K(R)]
```
**Look for**:
- Subscripts "model" and "bar" should be clearly smaller than main text
- Should NOT be bolded
- Parentheses should be properly sized

#### 2. Complex display equation (line 14):
```
K(R) = A_0 (g†/g_bar(R))^p  C(R; ℓ_0, p, n_coh)  G_bulge  G_shear  G_bar
```
**Look for**:
- Dagger symbol (†) should render as superscript, not as regular text
- ℓ₀ (script-ell with subscript zero) should be distinct from l or ł
- Subscripts: bar, bulge, shear should be clear
- Semicolon spacing in C(R; ...)

#### 3. Inline math (line 17):
```
(A_0, p) ... (ℓ_0, n_coh) ... (G_·) ... (K→0 as R→0)
```
**Look for**:
- Subscripts/superscripts in inline text
- Arrow (→) symbol
- Centered dot (·) in G_·
- Should integrate smoothly with surrounding text

#### 4. Parameter list (line 19):
```
ℓ_0=4.993 kpc, β_bulge=1.759, α_shear=0.149, γ_bar=1.932, A_0=0.591, p=0.757, n_coh=0.5
```
**Look for**:
- Greek letters (β, α, γ) should be italic and distinct
- Subscripts (bulge, shear, bar, coh) clear
- ℓ₀ consistent with earlier usage

#### 5. Exponent notation (line 21):
```
margin ≥ 10^13
```
**Look for**:
- Greater-than-or-equal (≥) symbol clear
- Superscript 13 positioned correctly

---

## Common rendering problems to watch for:

### ❌ Bad:
- **Bold formulas**: Variables appearing in bold when they should be italic
- **Missing symbols**: † renders as "dagger", ℓ renders as "l"
- **Subscript collisions**: "g_bar" looks like "g" followed by "_bar" in regular font
- **Spacing issues**: No space between terms or excessive spacing
- **Size inconsistency**: Inline math larger/smaller than display math

### ✓ Good:
- **Italic variables**: All single-letter variables (g, R, K, p) in math italic
- **Clear hierarchy**: Display equations stand out, inline math integrates
- **Symbol fidelity**: Special chars (→, ≥, ·, †, ℓ) render correctly
- **Consistent sizing**: Sub/superscripts proportional across document
- **Professional spacing**: Appropriate gaps between operators and operands

---

## Alternative methods to try:

### Option 1: Pandoc + LaTeX (BEST for math)
**Pro**: Native LaTeX rendering, publication-quality math  
**Con**: Requires MiKTeX or TeX Live installed (~3 GB)

**To test**:
1. Install Pandoc: https://pandoc.org/installing.html
2. Install MiKTeX: https://miktex.org/download (or TeX Live)
3. Run:
   ```
   pandoc test_section_25.md -o test_pandoc.pdf --pdf-engine=pdflatex
   ```

### Option 2: KaTeX instead of MathJax
**Pro**: Faster rendering, potentially cleaner output  
**Con**: Slightly different syntax support

**To test**: Modify `scripts/make_pdf.py` line 146:
```python
# Change from:
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
# To:
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16/dist/contrib/auto-render.min.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "$", right: "$", display: false}
      ]
    });
  });
</script>
```

### Option 3: Increase MathJax quality settings
**Pro**: Uses existing setup, just tweaks config  
**Con**: May increase file size

**To test**: Modify `scripts/make_pdf.py` line 143-145:
```python
window.MathJax = { 
  tex: { 
    inlineMath: [['$','$']], 
    displayMath: [['$$','$$']]
  },
  svg: { 
    fontCache: 'global',
    scale: 1.2  // Increase for sharper rendering
  },
  chtml: {
    scale: 1.1,
    minScale: 0.5
  }
};
```

---

## Recommendation workflow:

1. **Check current PDF** (`test_section25_current_method.pdf`)
   - If math looks good → done, use current method
   - If math has issues → try alternatives below

2. **If symbols/spacing are bad**:
   - Try Option 2 (KaTeX) or Option 3 (MathJax settings)
   - Regenerate and compare

3. **If still not publication-quality**:
   - Install Pandoc + LaTeX (Option 1)
   - This is the gold standard for academic papers

4. **Once you find the best method**:
   - Update `scripts/make_pdf.py` with the winning configuration
   - Regenerate full README.md → sigmagravity_paper.pdf
   - Commit with note about rendering improvements

---

## Decision criteria:

| Quality level | Action |
|---|---|
| **Symbols render correctly, readable** | ✓ Keep current method |
| **Minor issues (spacing, sizing)** | Try Option 2 or 3 first |
| **Major issues (missing symbols, bold)** | Install Pandoc + LaTeX (Option 1) |
| **Need arXiv submission** | Must use LaTeX source anyway |

---

**Next step**: Open `test_section25_current_method.pdf` and check against the criteria above. Report what you see.

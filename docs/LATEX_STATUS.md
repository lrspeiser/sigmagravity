# LaTeX PDF Generation Status

## Current Issue

Pandoc + LaTeX installation is complete, but there's a parsing error in the Markdown:

```
! Missing $ inserted.
l.455 ...\\frac{\\Sigma_{\\rm bar}(R)}{\\Sigma_{\\rm crit}}
```

**Root cause:** Line 167 in README.md has an inline equation that Pandoc is misinterpreting.

---

## Current Working Solution

**Use Chrome + MathJax method:**
```powershell
python scripts/make_pdf.py
```

**Output:** `sigmagravity_paper.pdf` (2.7 MB) ✓

**Quality:** Good enough for review. Formulas occasionally wrap but are readable.

---

## To Fix LaTeX Method (Future)

### Option 1: Quick Fix in README
Change line 167 from inline to display math:

**Current:**
```markdown
κ_eff(R) = \\frac{\\Sigma_{\\rm bar}(R)}{\\Sigma_{\\rm crit}}\\,[1+K_{\\rm cl}(R)],\\quad K_{\\rm cl}(R)=A_c\\,C(R;\\,\\ell_0,p,n_{\\rm coh}).
```

**Fixed:**
```markdown
$$
κ_eff(R) = \\frac{\\Sigma_{\\rm bar}(R)}{\\Sigma_{\\rm crit}}\\,[1+K_{\\rm cl}(R)],\\quad K_{\\rm cl}(R)=A_c\\,C(R;\\,\\ell_0,p,n_{\\rm coh}).
$$
```

### Option 2: Pre-process Markdown
Create a script that fixes inline math before passing to Pandoc.

### Option 3: Convert to Native LaTeX
For journal submission, you'll need to write in `.tex` format anyway. Pandoc can help:
```powershell
pandoc README.md -o paper.tex --pdf-engine=xelatex
# Then manually fix the .tex file and compile
xelatex paper.tex
```

---

## Recommendation

### For Now:
✓ **Use current PDF** (`sigmagravity_paper.pdf` from Chrome + MathJax)
- It works
- Formulas are mostly fine
- Good for internal review

### For Submission:
When ready to submit to journal/arXiv:
1. Start with your Markdown
2. Convert to LaTeX: `pandoc README.md -o paper.tex`
3. Manually fix any issues in the `.tex` file
4. Compile with `xelatex paper.tex`
5. Submit the `.tex` source + PDF

---

## What You Have Installed

✓ Pandoc 3.1.11  
✓ MiKTeX 24.1 (XeLaTeX)

These will be useful for final paper preparation, even if the automated script needs work.

---

## Bottom Line

**Your PDF is ready:** `sigmagravity_paper.pdf` (2.7 MB, Chrome + MathJax method)

The LaTeX method requires some Markdown cleanup to work perfectly, but you have the tools installed for when you need them.

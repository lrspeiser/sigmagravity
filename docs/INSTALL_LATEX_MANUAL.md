# Manual Installation: Pandoc + MiKTeX

## Current Status
✅ **PDF generated using Chrome + MathJax** → `sigmagravity_paper.pdf` (2.7 MB)

For better formula rendering (no line breaks), install Pandoc + LaTeX:

---

## Step-by-Step Installation

### Step 1: Install Pandoc (~5 minutes)

1. **Download installer:**
   - Go to: https://github.com/jgm/pandoc/releases/latest
   - Download: `pandoc-3.x.x-windows-x86_64.msi` (latest version)
   - Size: ~50 MB

2. **Run installer:**
   - Double-click the .msi file
   - Accept defaults
   - Click "Install"

3. **Verify:**
   ```powershell
   pandoc --version
   ```
   Should show version number

---

### Step 2: Install MiKTeX (~10 minutes)

1. **Download installer:**
   - Go to: https://miktex.org/download
   - Click: "Download Basic MiKTeX Installer" (64-bit)
   - Size: ~200 MB download, ~1 GB installed

2. **Run installer:**
   - Double-click `basic-miktex-xx.x-x64.exe`
   - Choose: "Install MiKTeX for anyone using this computer" (recommended)
   - Accept defaults
   - **Important:** Check "Always install missing packages on-the-fly"
   - Click "Start"
   - Wait 5-10 minutes

3. **Verify:**
   ```powershell
   pdflatex --version
   ```
   Should show MiKTeX version

---

### Step 3: Restart PowerShell

Close and reopen PowerShell to update PATH.

---

### Step 4: Generate PDF with LaTeX

```powershell
python scripts/make_pdf_latex.py
```

**First run:** 1-2 minutes (downloads LaTeX packages)  
**Subsequent runs:** ~30 seconds

**Output:** `sigmagravity_paper.pdf` with perfect formula rendering ✓

---

## Comparison Test

Generate both versions and compare:

```powershell
# Current method (fast, some formula wrapping)
python scripts/make_pdf.py --out paper_chrome.pdf

# LaTeX method (slower, perfect formulas)
python scripts/make_pdf_latex.py --out paper_latex.pdf
```

Open both PDFs side-by-side and check formula quality in Section 2.

---

## If Installation Fails

### Option 1: Use Current Method
The Chrome + MathJax method (`scripts/make_pdf.py`) works fine for drafts.

### Option 2: Install Chocolatey First
Makes future installs easier:

```powershell
# Run as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Then:
```powershell
choco install pandoc miktex -y
```

---

## Why Bother with LaTeX?

| Issue | Chrome + MathJax | Pandoc + LaTeX |
|-------|------------------|----------------|
| Formulas break mid-line | Sometimes ❌ | Never ✓ |
| Installation | None | ~1 GB |
| Speed | Fast (15s) | Slow first time (2 min) |
| **For journal submission** | Can't use | **Required** ✓ |
| **For arXiv** | Can't use | **Required** ✓ |

**Bottom line:** You'll need LaTeX for publication anyway. Install now or later.

---

## Current PDF Generated ✓

`sigmagravity_paper.pdf` (2.7 MB) was generated successfully using the Chrome + MathJax method.

To upgrade to LaTeX rendering:
1. Follow steps above
2. Run `python scripts/make_pdf_latex.py`
3. Compare the two PDFs

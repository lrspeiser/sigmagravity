#!/usr/bin/env python3
"""
Test all PDF generators on section 2.5
Compare rendering quality of math formulas
"""
import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("PDF Generator Comparison Test — Section 2.5")
print("=" * 70)
print()

tests = [
    ("1. Chrome + MathJax (current)", "test_pdf_chrome.py", "test_chrome.pdf"),
    ("2. Pandoc + LaTeX", "test_pdf_pandoc.py", "test_pandoc.pdf"),
    ("3. WeasyPrint", "test_pdf_weasyprint.py", "test_weasyprint.pdf"),
]

results = []

for name, script, pdf_file in tests:
    print(f"\n{name}")
    print("-" * 70)
    
    # Clean up old PDF if exists
    if Path(pdf_file).exists():
        Path(pdf_file).unlink()
    
    # Run test
    try:
        result = subprocess.run([sys.executable, script], 
                              capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print(f"  stderr: {result.stderr[:200]}")
        
        # Check if PDF was created
        if Path(pdf_file).exists():
            size_kb = Path(pdf_file).stat().st_size / 1024
            results.append((name, pdf_file, size_kb, "✓ SUCCESS"))
        else:
            results.append((name, pdf_file, 0, "✗ FAILED (no output)"))
    except subprocess.TimeoutExpired:
        results.append((name, pdf_file, 0, "✗ TIMEOUT"))
    except Exception as e:
        results.append((name, pdf_file, 0, f"✗ ERROR: {e}"))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Generator':<30} {'Output':<25} {'Size (KB)':<12} {'Status'}")
print("-" * 70)
for name, pdf, size, status in results:
    print(f"{name:<30} {pdf:<25} {size:>8.1f}     {status}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

successful = [r for r in results if "SUCCESS" in r[3]]
if successful:
    # Sort by size (smaller = better compression, but check quality manually)
    successful.sort(key=lambda x: x[2])
    print(f"\n✓ {len(successful)} generator(s) succeeded.")
    print("\nReview the PDFs manually to check math rendering quality:")
    for name, pdf, size, _ in successful:
        print(f"  - {pdf:25} ({size:>6.1f} KB)")
    print("\nLook for:")
    print("  • Crisp subscripts/superscripts (g_bar, g_model, n_coh)")
    print("  • Proper symbol spacing (\\dagger, \\ell_0)")
    print("  • No bold artifacts or missing characters")
    print("  • Display equations centered and well-spaced")
else:
    print("\n✗ All generators failed.")
    print("Check error messages above.")

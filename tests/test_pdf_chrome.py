#!/usr/bin/env python3
"""Test 1: Chrome + MathJax (current method)"""
import subprocess
from pathlib import Path
import markdown

CSS = r"""
@page { margin: 20mm; }
body { font-family: 'Times New Roman', serif; font-size: 12pt; line-height: 1.6; max-width: 180mm; margin: 20mm auto; }
h1, h3 { margin: 16pt 0 8pt; }
p { margin: 0 0 12pt 0; }
"""

md_text = Path('test_section_25.md').read_text(encoding='utf-8')
html_body = markdown.markdown(md_text, extensions=['extra'])

html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>{CSS}</style>
  <script>
    window.MathJax = {{ tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }} }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>{html_body}</body>
</html>
"""

Path('test_chrome.html').write_text(html, encoding='utf-8')

# Find Chrome/Edge
browser = None
for p in [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]:
    if Path(p).exists():
        browser = p
        break

if browser:
    url = Path('test_chrome.html').resolve().as_uri()
    subprocess.run([browser, '--headless=new', '--disable-gpu', '--virtual-time-budget=15000', 
                   f'--print-to-pdf=test_chrome.pdf', url])
    print("✓ test_chrome.pdf generated")
else:
    print("✗ Chrome/Edge not found")

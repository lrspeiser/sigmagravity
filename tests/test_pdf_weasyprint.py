#!/usr/bin/env python3
"""Test 3: WeasyPrint (CSS-based PDF)"""
from pathlib import Path
import markdown

CSS = """
@page { margin: 20mm; size: A4; }
body { font-family: 'Times New Roman', serif; font-size: 12pt; line-height: 1.6; }
h1, h3 { margin: 16pt 0 8pt; }
p { margin: 0 0 12pt 0; }
.math { font-style: italic; }
"""

md_text = Path('test_section_25.md').read_text(encoding='utf-8')
html_body = markdown.markdown(md_text, extensions=['extra'])

html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>{CSS}</style>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>{html_body}</body>
</html>
"""

try:
    from weasyprint import HTML
    HTML(string=html).write_pdf('test_weasyprint.pdf')
    print("✓ test_weasyprint.pdf generated")
except ImportError:
    print("✗ WeasyPrint not installed")
    print("  Install with: pip install weasyprint")
except Exception as e:
    print(f"✗ Error: {e}")

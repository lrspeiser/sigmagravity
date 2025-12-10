#!/usr/bin/env python3
"""
make_pdf.py — Build a two‑column, figure‑captioned PDF of paper.md

- Rewrites Markdown image + italic caption lines into HTML <figure><img><figcaption>
- Wraps content in a two‑column print CSS layout (journal‑style)
- Uses Chrome/Edge headless to print to PDF

Usage:
  python many_path_model/paper_release/scripts/make_pdf.py \
    --md many_path_model/paper_release/paper.md \
    --out many_path_model/paper_release/paper_formatted.pdf
"""
import argparse
import os
import re
import subprocess
from pathlib import Path

try:
    import markdown
except Exception as e:
    raise SystemExit("Please install markdown: python -m pip install --user markdown")

CSS = r"""
@page { margin: 18mm 16mm 20mm 16mm; }
* { box-sizing: border-box; }
html, body { height: 100%; }
body { font-family: 'Times New Roman', serif; font-size: 10pt; line-height: 1.22; color: #111; }
#titlepage { column-count: 1; margin-bottom: 8mm; }
#titlepage h1 { font-size: 18pt; margin: 0 0 4pt 0; }
#titlepage .meta { font-size: 10pt; color: #444; margin-bottom: 6pt; }
#abstract { font-size: 10pt; margin-top: 6pt; }
#content { column-count: 2; column-gap: 8mm; }
figure.fullwidth { column-span: all; }
h1 { font-size: 14pt; margin: 10pt 0 6pt; break-after: avoid; }
h2 { font-size: 12pt; margin: 8pt 0 4pt; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 6pt 0 3pt; break-after: avoid; }
p { margin: 0 0 5pt 0; }
ul, ol { margin: 0 0 6pt 18pt; }
code, pre { font-family: 'Courier New', monospace; font-size: 9pt; }
pre { background: #f7f7f7; border: 1px solid #ddd; padding: 6pt; border-radius: 3px; }
pre code { background: transparent; }
figure { break-inside: avoid; margin: 4pt 0 8pt 0; }
figure img { width: 100%; height: auto; display: block; margin: 0 auto; }
figcaption { font-size: 8.5pt; color: #444; margin-top: 3pt; }
.table { break-inside: avoid; margin: 6pt 0; }
hr { border: 0; border-top: 1px solid #ccc; margin: 8pt 0; }
.small { font-size: 9pt; color: #555; }
"""

IMG_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^\)]+)\)\s*(\{(?P<attr>[^\}]*)\})?\s*$")
ITALIC_RE = re.compile(r"^\*(?P<cap>[^*]+)\*\s*$")


def md_with_figures(md_text: str, base_dir: Path) -> str:
    """Rewrite Markdown image+caption blocks into HTML <figure>...</figure> before HTML conversion.
    Supports attribute list on images, e.g. ![alt](img.png){.fullwidth} to span two columns.
    """
    lines = md_text.splitlines()
    out_lines = []
    i = 0
    while i < len(lines):
        m = IMG_RE.match(lines[i].strip())
        if m:
            alt = m.group('alt').strip()
            src = m.group('src').strip()
            attrs = (m.group('attr') or '').strip()
            classes = []
            if attrs:
                # crude parse: tokens split by space, class tokens like .fullwidth
                for tok in attrs.split():
                    if tok.startswith('.'): classes.append(tok[1:])
            cap = None
            j = i + 1
            # Skip blank lines to find caption
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                c = ITALIC_RE.match(lines[j].strip())
                if c:
                    cap = c.group('cap').strip()
            # Normalize relative path (not used directly in HTML src to keep relative refs intact)
            _ = (base_dir / src).as_posix()
            cls_attr = f" class=\"{' '.join(classes)}\"" if classes else ""
            fig_html = (
                f"<figure{cls_attr}>\n  <img src=\"{src}\" alt=\"{alt}\"/>\n" +
                (f"  <figcaption>{cap}</figcaption>\n" if cap else "") +
                "</figure>"
            )
            out_lines.append(fig_html)
            i = j + (1 if cap is not None else 0)
            continue
        else:
            out_lines.append(lines[i])
            i += 1
    return "\n".join(out_lines)


def build_html(md_path: Path) -> Path:
    raw = md_path.read_text(encoding='utf-8')
    # Extract title and author (first heading line and following non-empty line)
    title = None; author = None
    lines = raw.splitlines()
    k = 0
    while k < len(lines) and lines[k].strip() == "":
        k += 1
    if k < len(lines) and lines[k].lstrip().startswith('# '):
        title = lines[k].lstrip()[2:].strip()
        k += 1
        # next non-empty, non-heading becomes author
        while k < len(lines) and lines[k].strip() == "":
            k += 1
        if k < len(lines) and not lines[k].lstrip().startswith('#'):
            author = lines[k].strip()
            # remove these lines from body
            lines[k] = ""
        # also remove title line from body
        lines_wo_title = lines[:]
        # find original title line index again
        # simple: drop the first non-empty heading occurrence
        rebuilt = []
        dropped_title = False
        for L in lines_wo_title:
            if not dropped_title and L.lstrip().startswith('# '):
                dropped_title = True
                continue
            rebuilt.append(L)
        raw = "\n".join(rebuilt)
    # Preprocess to wrap figures (and support {.fullwidth})
    pre = md_with_figures(raw, md_path.parent)
    html_body = markdown.markdown(pre, extensions=['extra','sane_lists','tables','toc','attr_list'])
    # Build HTML skeleton with titlepage and MathJax for formulas
    title_html = ""
    if title:
        title_html = f"<div id=\"titlepage\">\n  <h1>{title}</h1>\n" + (f"  <div class=\"meta\">{author}</div>\n" if author else "") + "</div>\n"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\"/>
  <style>
{CSS}
  </style>
  <script>
    window.MathJax = {{ tex: {{ inlineMath: [['$','$'], ['\\(','\\)']], displayMath: [['$$','$$'], ['\\[','\\]']] }} }};
  </script>
  <script id=\"MathJax-script\" async src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js\"></script>
  <title>Σ‑Gravity Paper</title>
</head>
<body>
{title_html}
<div id=\"content\">
{html_body}
</div>
</body>
</html>
"""
    out_html = md_path.parent / 'paper_formatted.html'
    out_html.write_text(html, encoding='utf-8')
    return out_html


def find_browser() -> str:
    candidates = [
        r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
        r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError("Chrome/Edge not found. Please install Chrome or Edge.")


def print_pdf(html_path: Path, pdf_path: Path):
    browser = find_browser()
    url = html_path.resolve().as_uri()
    pdf_abs = str(pdf_path.resolve())
    # Give MathJax a moment with virtual time budget using --virtual-time-budget (ms)
    cmd = [browser, '--headless=new', '--disable-gpu', '--virtual-time-budget=5000', f"--print-to-pdf={pdf_abs}", url]
    # For legacy Edge, headless flag may be '--headless' instead
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        cmd[1] = '--headless'
        subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=str(Path(__file__).resolve().parents[1] / 'paper.md'))
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'paper_formatted.pdf'))
    args = ap.parse_args()
    md_path = Path(args.md)
    pdf_path = Path(args.out)

    html_path = build_html(md_path)
    print_pdf(html_path, pdf_path)
    print(str(pdf_path))

if __name__ == '__main__':
    main()

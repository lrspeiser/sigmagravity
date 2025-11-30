#!/usr/bin/env python3
"""
md_to_latex.py — Convert README.md to sigmagravity_paper.tex

Usage:
  python scripts/md_to_latex.py

Reads README.md and generates a complete LaTeX document at sigmagravity_paper.tex
Then compiles it to PDF using pdflatex.
"""
import re
import subprocess
from pathlib import Path

def convert_markdown_to_latex(md_content):
    """Convert markdown content to LaTeX, preserving equations and structure."""
    
    # Start with preamble
    latex = r"""\documentclass[11pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage[margin=20mm]{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{longtable}
\usepackage{ragged2e}  % Better paragraph formatting

% Line breaking settings
\sloppy  % Allow more flexible line breaking
\hyphenpenalty=1000  % Reduce hyphenation
\exhyphenpenalty=1000  % Reduce hyphenation in compound words
\doublehyphendemerits=10000  % Penalize consecutive hyphenated lines

% Math line breaking
\allowdisplaybreaks  % Allow page breaks in display math
\binoppenalty=1000   % Penalty for breaking after binary operators
\relpenalty=1000     % Penalty for breaking after relations

% Unicode support for special characters
\DeclareUnicodeCharacter{03A3}{$\Sigma$}  % Σ
\DeclareUnicodeCharacter{2020}{$\dagger$} % †
\DeclareUnicodeCharacter{2113}{$\ell$}    % ℓ
\DeclareUnicodeCharacter{2192}{$\to$}     % →
\DeclareUnicodeCharacter{2261}{$\equiv$}  % ≡
\DeclareUnicodeCharacter{00D7}{$\times$}  % ×
\DeclareUnicodeCharacter{2211}{$\sum$}    % ∑
\DeclareUnicodeCharacter{220F}{$\prod$}   % ∏
\DeclareUnicodeCharacter{2032}{$'$}       % ′
\DeclareUnicodeCharacter{2264}{$\leq$}    % ≤
\DeclareUnicodeCharacter{2265}{$\geq$}    % ≥
\DeclareUnicodeCharacter{226B}{$\gg$}     % ≫
\DeclareUnicodeCharacter{226A}{$\ll$}     % ≪
\DeclareUnicodeCharacter{27E8}{$\langle$} % ⟨
\DeclareUnicodeCharacter{27E9}{$\rangle$} % ⟩
\DeclareUnicodeCharacter{2013}{--}        % en-dash
\DeclareUnicodeCharacter{2014}{---}       % em-dash
\DeclareUnicodeCharacter{2212}{$-$}       % minus
\DeclareUnicodeCharacter{2260}{$\neq$}    % ≠
\DeclareUnicodeCharacter{2248}{$\approx$} % ≈
\DeclareUnicodeCharacter{00B1}{$\pm$}     % ±

% Hyperref setup
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue,
}

% Custom commands for better line breaking
\newcommand{\allowbreakmath}[1]{$#1$\allowbreak}
\newcommand{\breakablemath}[1]{$#1$\allowbreak}

% Custom commands
\newcommand{\Sigmagrav}{$\Sigma$-Gravity}
\newcommand{\ellzero}{\ell_0}

\begin{document}

"""
    
    lines = md_content.split('\n')
    i = 0
    in_equation = False
    in_table = False
    table_buffer = []
    
    while i < len(lines):
        line = lines[i]
        
        # Skip yaml frontmatter if present
        if i == 0 and line.strip() == '---':
            i += 1
            while i < len(lines) and lines[i].strip() != '---':
                i += 1
            i += 1
            continue
        
        # Title (first # heading becomes \title)
        if i < 10 and line.startswith('# ') and '\\title' not in latex:
            title = line[2:].strip()
            # Convert markdown bold and math in title
            title = convert_inline_formatting(title)
            latex += f"\\title{{{title}}}\n\n"
            latex += "\\author{Leonard Speiser}\n\n"
            latex += "\\date{}\n\n"  # Empty date - common for preprints
            latex += "\\maketitle\n\n"
            i += 1
            # Skip author/date lines if they immediately follow the title
            while i < len(lines) and (lines[i].strip().startswith('**Authors:**') or 
                                       lines[i].strip().startswith('**Date:**') or
                                       lines[i].strip() == ''):
                i += 1
            continue
        
        # Abstract
        if line.strip().startswith('## Abstract') or line.strip().startswith('**Abstract**'):
            latex += "\\begin{abstract}\n"
            i += 1
            # Collect abstract content until next section as one paragraph
            while i < len(lines) and not lines[i].strip().startswith('##'):
                # Skip horizontal rules that are just separators after abstract
                if lines[i].strip() in ['---', '***', '___']:
                    i += 1
                    continue
                if lines[i].strip():
                    latex += convert_inline_formatting(lines[i]) + " "
                i += 1
            latex += "\n\\end{abstract}\n\n"
            continue
        
        # Section headers
        if line.startswith('## '):
            section_title = line[3:].strip()
            # Strip leading hierarchical numbers like "1.", "2.6.", "11.2.3." from section titles
            section_title = re.sub(r'^\d+(\.\d+)*\.?\s*', '', section_title)
            section_title = convert_inline_formatting(section_title)
            latex += f"\\section{{{section_title}}}\n\n"
            i += 1
            continue
        
        if line.startswith('### '):
            subsection_title = line[4:].strip()
            # Strip leading hierarchical numbers from subsection titles
            subsection_title = re.sub(r'^\d+(\.\d+)*\.?\s*', '', subsection_title)
            subsection_title = convert_inline_formatting(subsection_title)
            latex += f"\\subsection{{{subsection_title}}}\n\n"
            i += 1
            continue
        
        if line.startswith('#### '):
            subsubsection_title = line[5:].strip()
            # Strip leading hierarchical numbers from subsubsection titles
            subsubsection_title = re.sub(r'^\d+(\.\d+)*\.?\s*', '', subsubsection_title)
            subsubsection_title = convert_inline_formatting(subsubsection_title)
            latex += f"\\subsubsection{{{subsubsection_title}}}\n\n"
            i += 1
            continue
        
        # Block equations $$...$$
        stripped = line.strip()
        # Single-line block equation: $$content$$
        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
            equation_content = stripped[2:-2].strip()
            latex += "\\begin{equation}\n"
            latex += equation_content + "\n"
            latex += "\\end{equation}\n\n"
            i += 1
            continue
        
        # Multi-line block equation start/end: $$ on its own line
        if stripped == '$$':
            if not in_equation:
                latex += "\\begin{equation}\n"
                in_equation = True
            else:
                latex += "\\end{equation}\n\n"
                in_equation = False
            i += 1
            continue
        
        # Inside multi-line equation
        if in_equation:
            latex += line + "\n"
            i += 1
            continue
        
        # Tables
        if '|' in line and not line.strip().startswith('<!--'):
            if not in_table:
                in_table = True
                table_buffer = [line]
            else:
                table_buffer.append(line)
            i += 1
            # Check if table continues
            if i >= len(lines) or '|' not in lines[i]:
                # End of table, convert it
                latex += convert_table(table_buffer)
                in_table = False
                table_buffer = []
            continue
        
        # Lists
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            latex += "\\begin{itemize}\n"
            while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                item = lines[i].strip()[2:]
                item = convert_inline_formatting(item)
                latex += f"\\item {item}\n"
                i += 1
            latex += "\\end{itemize}\n\n"
            continue
        
        # Numbered lists
        if re.match(r'^\d+[\.\)]\s', line.strip()):
            latex += "\\begin{enumerate}\n"
            while i < len(lines) and re.match(r'^\d+[\.\)]\s', lines[i].strip()):
                item = re.sub(r'^\d+[\.\)]\s', '', lines[i].strip())
                item = convert_inline_formatting(item)
                latex += f"\\item {item}\n"
                i += 1
            latex += "\\end{enumerate}\n\n"
            continue
        
        # Block quotes
        if line.strip().startswith('>'):
            latex += "\\begin{quote}\n"
            while i < len(lines) and lines[i].strip().startswith('>'):
                quote_line = lines[i].strip()[1:].strip()
                quote_line = convert_inline_formatting(quote_line)
                latex += quote_line + "\n\n"
                i += 1
            latex += "\\end{quote}\n\n"
            continue
        
        # Images ![alt](path)
        if line.strip().startswith('!['):
            match = re.match(r'!\[(.*?)\]\((.*?)\)', line.strip())
            if match:
                caption = match.group(1)
                path = match.group(2)
                # Fix path - images are relative to project root, but LaTeX compiles from docs/
                if not path.startswith('http') and not path.startswith('/') and not path.startswith('..'):
                    path = '../' + path
                latex += "\\begin{figure}[h]\n"
                latex += "\\centering\n"
                latex += f"\\includegraphics[width=0.8\\textwidth]{{{path}}}\n"
                latex += f"\\caption{{{caption}}}\n"
                latex += "\\end{figure}\n\n"
            i += 1
            continue
        
        # Horizontal rules
        if line.strip() in ['---', '***', '___']:
            latex += "\\medskip\\hrule\\medskip\n\n"
            i += 1
            continue
        
        # Standalone equations (detect LaTeX equations not wrapped in $$ or $)
        # These are lines with LaTeX commands but no markdown formatting
        stripped = line.strip()
        if stripped and not stripped.startswith('$') and not stripped.startswith('\\begin'):
            # Skip lines that already have inline math delimiters - they're regular text, not standalone equations
            if '$' not in stripped:
                # Check if line looks like an equation (has LaTeX commands or math-like structure)
                has_latex_commands = any(cmd in stripped for cmd in ['\\,', '\\;', '\\rm', '\\dagger', '\\ell', '\\Sigma', '\\quad'])
                has_equation_pattern = '=' in stripped and ('(' in stripped or '_' in stripped)
                # Exclude markdown links [text](url) but allow equation brackets
                is_markdown_link = '](' in stripped or stripped.startswith('**') or stripped.startswith('*Figure')
                
                if has_latex_commands or (has_equation_pattern and not is_markdown_link and 'http' not in stripped):
                    # This is likely a standalone equation - wrap it in display math
                    latex += "\\[\n" + stripped + "\n\\]\n\n"
                    i += 1
                    continue
        
        # Regular paragraphs
        if line.strip():
            converted = convert_inline_formatting(line)
            latex += converted + "\n\n"
        else:
            latex += "\n"
        
        i += 1
    
    latex += "\\end{document}\n"
    return latex


def convert_inline_formatting(text):
    """Convert markdown inline formatting to LaTeX."""
    # Protect math mode content first
    math_blocks = []
    def save_math(match):
        math_blocks.append(match.group(0))
        return f"MATHBLOCK{len(math_blocks)-1}MATHBLOCK"
    
    # Save inline math $...$
    text = re.sub(r'\$[^$]+\$', save_math, text)
    
    # Escape special LaTeX characters outside math
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')  # Escape underscores outside math/formatting
    
    # Now handle markdown formatting
    # Bold **text** (do this before italic to avoid conflicts)
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    
    # Italic *text* (single asterisk)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', text)
    
    # Code `code`
    text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)
    
    # Links [text](url)
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'\\href{\2}{\1}', text)
    
    # Restore math blocks and add line breaks for long expressions
    for i, math in enumerate(math_blocks):
        # Add \allowbreak after long math expressions
        if len(math) > 30:  # Long math expressions
            math = math.replace('$', '$\\allowbreak ')
        text = text.replace(f"MATHBLOCK{i}MATHBLOCK", math)
    
    return text


def convert_table(table_lines):
    """Convert markdown table to LaTeX tabular."""
    if len(table_lines) < 2:
        return ""
    
    # Parse header
    header = [cell.strip() for cell in table_lines[0].split('|') if cell.strip()]
    num_cols = len(header)
    
    # Skip separator line
    # Build table
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\begin{tabular}{" + "l" * num_cols + "}\n"
    latex += "\\toprule\n"
    
    # Header row
    header_latex = ' & '.join([convert_inline_formatting(h) for h in header])
    latex += header_latex + " \\\\\n"
    latex += "\\midrule\n"
    
    # Data rows (skip first 2 lines: header and separator)
    for line in table_lines[2:]:
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if cells:
            row_latex = ' & '.join([convert_inline_formatting(c) for c in cells])
            latex += row_latex + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n\n"
    
    return latex


def main():
    repo_root = Path(__file__).parent.parent
    readme_path = repo_root / "README.md"
    docs_path = repo_root / "docs"
    tex_path = docs_path / "sigmagravity_paper.tex"
    pdf_path = docs_path / "sigmagravity_paper.pdf"
    
    print(f"Reading {readme_path}...")
    md_content = readme_path.read_text(encoding='utf-8')
    
    print("Converting to LaTeX...")
    latex_content = convert_markdown_to_latex(md_content)
    
    print(f"Writing {tex_path}...")
    tex_path.write_text(latex_content, encoding='utf-8')
    
    print(f"Compiling to PDF...")
    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', str(tex_path)],
        cwd=docs_path,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # Check if PDF was created regardless of return code
    if pdf_path.exists():
        print(f"[OK] PDF generated: {pdf_path}")
        print(f"  Size: {pdf_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"[FAILED] PDF generation failed")
        if result.stdout:
            try:
                print("STDOUT:", result.stdout[-500:])
            except UnicodeEncodeError:
                print("STDOUT: [Unicode content - check LaTeX log]")
        if result.stderr:
            try:
                print("STDERR:", result.stderr[-500:])
            except UnicodeEncodeError:
                print("STDERR: [Unicode content - check LaTeX log]")
        return 1
    
    # Run pdflatex again for cross-references
    print("Running pdflatex second pass for cross-references...")
    subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', str(tex_path)],
        cwd=docs_path,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    print("[OK] Complete")
    return 0


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""md_to_latex.py — Generate APS/PRD (REVTeX) journal TeX+PDF from README.md

Submission layout requirement:
- Full-width abstract, two-column body (REVTeX `reprint` does this)

Outputs:
- docs/sigmagravity_paper.tex
- docs/sigmagravity_paper.pdf

Also runs a coverage check to ensure README prose is not dropped.

Usage:
  python scripts/md_to_latex.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _strip_heading_numbering(title: str) -> str:
    title = re.sub(r"^[IVXLCDM]+\.\s*", "", title)  # I.
    title = re.sub(r"^[A-Z]\.\s*", "", title)  # A.
    title = re.sub(r"^\d+(\.\d+)*\.?\s*", "", title)  # 2.6.
    return title.strip()


def _extract_abstract(md_lines: list[str]) -> tuple[str, list[str]]:
    out: list[str] = []
    parts: list[str] = []
    i = 0
    while i < len(md_lines):
        if md_lines[i].strip().startswith("## Abstract") or md_lines[i].strip().startswith("**Abstract**"):
            i += 1
            while i < len(md_lines) and not md_lines[i].strip().startswith("## "):
                s = md_lines[i].strip()
                if s in ("---", "***", "___"):
                    i += 1
                    continue
                if s:
                    parts.append(s)
                i += 1
            while i < len(md_lines) and md_lines[i].strip() in ("", "---", "***", "___"):
                i += 1
            continue
        out.append(md_lines[i])
        i += 1

    return " ".join(parts).strip(), out


def _extract_readme_frontmatter(md_lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """
    Extract a lightweight frontmatter block from the top of README.md (before the first horizontal rule).

    Expected README pattern:
    - '# Title'
    - optional '**Pre-print**'
    - '**Author Name**<sup>1,*</sup>'
    - '<sup>1</sup>Affiliation, City, State, Country'
    - '<sup>*</sup>Contact author: email | ORCID: [id](url)'
    - '---'
    """
    meta: dict[str, str] = {}

    i = 0
    while i < len(md_lines) and not md_lines[i].strip():
        i += 1

    if i < len(md_lines) and md_lines[i].startswith("# "):
        meta["title"] = md_lines[i][2:].strip()
        i += 1

    while i < len(md_lines):
        s = md_lines[i].strip()
        if s in ("---", "***", "___"):
            i += 1
            break

        if s.lower() in ("**pre-print**", "pre-print"):
            meta["preprint"] = "true"

        m_author = re.match(r"^\*\*(.+?)\*\*", s)
        if m_author and "author" not in meta:
            cand = m_author.group(1).strip()
            if cand.casefold() not in ("pre-print", "preprint"):
                meta["author"] = cand

        m_aff = re.match(r"^<\s*sup\s*>\s*1\s*<\s*/\s*sup\s*>(.+)$", s, flags=re.IGNORECASE)
        if m_aff and "affiliation" not in meta:
            meta["affiliation"] = m_aff.group(1).strip()

        if "contact author:" in s.lower():
            m_email = re.search(r"contact author:\s*([^\s|]+@[^\s|]+)", s, flags=re.IGNORECASE)
            if m_email:
                meta["email"] = m_email.group(1).strip()

            m_orcid = re.search(r"ORCID:\s*\[([^\]]+)\]\(([^)]+)\)", s, flags=re.IGNORECASE)
            if m_orcid:
                meta["orcid_id"] = m_orcid.group(1).strip()
                meta["orcid_url"] = m_orcid.group(2).strip()

        i += 1

    return meta, md_lines[i:]


def convert_inline_formatting(text: str) -> str:
    # Inline HTML used in README
    text = re.sub(r"<\s*sup\s*>(.*?)<\s*/\s*sup\s*>", r"\\textsuperscript{\1}", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+?>", "", text)

    # Protect inline math
    math_blocks: list[str] = []

    def save_math(m: re.Match[str]) -> str:
        math_blocks.append(m.group(0))
        return f"MATHBLOCK{len(math_blocks)-1}MATHBLOCK"

    text = re.sub(r"\$[^$]+\$", save_math, text)

    # Normalize common unicode outside math
    replacements = {
        "—": "---",
        "–": "--",
        "−": "-",
        "→": r"$\to$",
        "×": r"$\times$",
        "≈": r"$\approx$",
        "±": r"$\pm$",
        "≪": r"$\ll$",
        "≫": r"$\gg$",
        "≠": r"$\neq$",
        "ℓ": r"$\ell$",
        "†": r"$\dagger$",
        "☉": r"$\odot$",
        "μ": r"$\mu$",
        "²": r"\textsuperscript{2}",
        "°": r"$^\circ$",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Escape special chars outside math
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("#", "\\#")
    text = text.replace("_", "\\_")

    # Markdown formatting
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", text)
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\\href{\2}{\1}", text)

    # Restore math blocks
    for i, math in enumerate(math_blocks):
        if len(math) > 30:
            math = math.replace("$", "$\\allowbreak ")
        text = text.replace(f"MATHBLOCK{i}MATHBLOCK", math)

    return text


def _table_sep_line(s: str) -> bool:
    s = s.strip()
    return bool(re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", s))


def convert_table(table_lines: list[str], *, width_ref: str) -> str:
    header = [c.strip() for c in table_lines[0].split("|") if c.strip()]
    ncol = len(header)
    if ncol == 0:
        return ""

    # Use a standard tabular. For narrow tables, wrap the last column to prevent overflow.
    # For wide tables (many columns), wrapping a column tends to blow up the layout; keep them numeric.
    if ncol >= 5:
        colspec = "l" * ncol
    elif ncol >= 2:
        colspec = ("l" * (ncol - 1)) + "p{0.48\\linewidth}"
    else:
        colspec = "l"

    out = f"\\begin{{tabular}}{{{colspec}}}\n"
    out += "\\toprule\n"
    out += " & ".join(convert_inline_formatting(h) for h in header) + " \\\\\n"
    out += "\\midrule\n"

    for line in table_lines[2:]:
        if not line.strip():
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if not cells:
            continue
        out += " & ".join(convert_inline_formatting(c) for c in cells) + " \\\\\n"

    out += "\\bottomrule\n"
    out += "\\end{tabular}\n"
    return out


def _latex_token_string(latex: str) -> str:
    s = re.sub(r"%.*", "", latex)
    s = s.replace("Σ", " sigma ").replace("σ", " sigma ")
    s = s.replace("Λ", " lambda ").replace("λ", " lambda ")
    s = s.replace("μ", " mu ").replace("°", " deg ")
    s = s.replace("\\\\", " ")
    s = re.sub(r"\\[a-zA-Z@]+\\*?", " ", s)
    s = s.translate(str.maketrans({c: " " for c in "{}$[]"}))
    s = re.sub(r"\s+", " ", s).strip().casefold()
    toks = re.findall(r"[0-9]+(?:\.[0-9]+)?|\w+", s, flags=re.UNICODE)
    toks = [t for t in toks if not re.fullmatch(r"ref\d+", t)]
    return " ".join(toks)


def _validate_readme_coverage(md: str, latex: str) -> None:
    tex_tok = _latex_token_string(latex)

    in_code = False
    candidates: list[str] = []
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            if s:
                candidates.append(s)
            continue
        if not s or s in ("---", "***", "___"):
            continue
        if s.startswith("|"):
            continue
        if s.startswith("# "):
            continue
        if "<sup" in s.lower() or "contact author:" in s.lower() or "orcid:" in s.lower():
            continue
        if s.lower() in ("**pre-print**", "pre-print"):
            continue

        s = re.sub(r"^#{1,6}\s+", "", s)
        s = _strip_heading_numbering(s)
        if re.match(r"^figure\s+\d+\s*:\s*", s, flags=re.IGNORECASE):
            continue
        s = re.sub(r"^!\[([^\]]*)\]\([^\)]+\).*$", r"\1", s)
        s = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", s)
        s = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", "", s)
        s = s.replace("**", "").replace("`", "")
        if s.startswith("*") and s.endswith("*") and len(s) > 2:
            s = s[1:-1]
        s = re.sub(r"^\s*(FIG|Fig)\.\s*\d+\.\s*", "", s).strip()
        if "$" in s or "\\" in s:
            continue
        if sum(ch.isalpha() for ch in s) < 6:
            continue
        candidates.append(convert_inline_formatting(s).casefold())

    seen: set[str] = set()
    uniq = [x for x in candidates if not (x in seen or seen.add(x))]

    word_re = re.compile(r"[0-9]+(?:\.[0-9]+)?|\w+", re.UNICODE)
    missing: list[str] = []
    for s in uniq:
        toks = word_re.findall(_latex_token_string(s))
        if len(toks) < 8:
            continue
        needles = [" ".join(toks[:12]), " ".join(toks[:8]), " ".join(toks[:6])]
        if not any(n and n in tex_tok for n in needles):
            missing.append(s)

    if missing:
        sample = "\n".join(f"- {m}" for m in missing[:30])
        raise RuntimeError(
            f"README coverage check failed: {len(missing)} content lines not found in generated TeX.\n"
            f"First missing lines:\n{sample}\n"
        )


def convert_markdown_to_revtex(md_content: str) -> str:
    lines = md_content.splitlines()
    meta, lines = _extract_readme_frontmatter(lines)
    abstract, lines = _extract_abstract(lines)

    latex = r"""\documentclass[aps,prd,reprint,superscriptaddress,showpacs,floatfix,longbibliography]{revtex4-2}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\DeclareUnicodeCharacter{03A3}{\ensuremath{\Sigma}} % Σ
\DeclareUnicodeCharacter{03C3}{\ensuremath{\sigma}} % σ
\DeclareUnicodeCharacter{039B}{\ensuremath{\Lambda}} % Λ
\DeclareUnicodeCharacter{03BB}{\ensuremath{\lambda}} % λ
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage[protrusion=true,expansion=false]{microtype}

\graphicspath{{../figures/}}

\begin{document}

"""

    i = 0
    in_equation = False
    in_table = False
    table_buffer: list[str] = []

    bibkey_for_number: dict[str, str] = {}
    in_ack = False
    in_appendix = False
    pending_table_caption: str | None = None

    while i < len(lines):
        line = lines[i]

        # YAML frontmatter
        if i == 0 and line.strip() == "---":
            i += 1
            while i < len(lines) and lines[i].strip() != "---":
                i += 1
            i += 1
            continue

        # Title/author block (from README frontmatter; avoid hardcoding in the converter)
        if i == 0 and "\\title" not in latex:
            title = convert_inline_formatting(meta.get("title", "").strip())
            author = convert_inline_formatting(meta.get("author", "Leonard Speiser").strip())
            affiliation = convert_inline_formatting(meta.get("affiliation", "Horizon 3, Independent Research").strip())
            email = meta.get("email", "").strip()
            orcid_url = meta.get("orcid_url", "").strip()

            if meta.get("preprint") == "true":
                latex += f"\\title{{{title}\\\\\n\\textnormal{{\\textit{{Pre-print}}}}}}\n\n"
            else:
                latex += f"\\title{{{title}}}\n\n"

            latex += f"\\author{{{author}}}\n"
            if email:
                latex += f"\\email[Contact author: ]{{{email}}}\n"
            if orcid_url:
                latex += f"\\homepage[ORCID: ]{{{orcid_url}}}\n"
            latex += f"\\affiliation{{{affiliation}}}\n\n"
            latex += "\\date{\\today}\n\n"
            latex += "\\begin{abstract}\n" + convert_inline_formatting(abstract) + "\n\\end{abstract}\n\n"
            latex += "\\maketitle\n\n"

            i += 1
            continue

        # Acknowledgments
        if line.strip().startswith("## ") and _strip_heading_numbering(line.strip()[3:]).lower().startswith("acknowledg"):
            latex += "\\begin{acknowledgments}\n"
            in_ack = True
            i += 1
            continue

        # References
        if line.strip().startswith("## References"):
            if in_ack:
                latex += "\\end{acknowledgments}\n\n"
                in_ack = False
            latex += "\\begin{thebibliography}{99}\n\n"
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("## "):
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                m = re.match(r"^\[(\d+)\]\s*(.*)$", s)
                if m:
                    num = m.group(1)
                    bibkey = f"ref{num}"
                    bibkey_for_number[num] = bibkey
                    latex += f"\\bibitem{{{bibkey}}} {convert_inline_formatting(m.group(2))}\n\n"
                i += 1
            latex += "\\end{thebibliography}\n\n"
            continue

        # Table caption line
        cap_plain = line.strip().strip("*").strip()
        if re.match(r"^(Table)\s+[A-Za-z0-9IVXLCDM]+\s*:\s+.+", cap_plain):
            pending_table_caption = convert_inline_formatting(cap_plain)
            i += 1
            continue

        # Bold label immediately before a markdown table (treat as caption)
        # Example: "**Parameter accounting:**" followed by a |---| separator table.
        if line.strip().startswith("**") and line.strip().endswith("**"):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            k = j + 1
            while k < len(lines) and lines[k].strip() == "":
                k += 1
            if j < len(lines) and lines[j].lstrip().startswith("|") and k < len(lines) and _table_sep_line(lines[k]):
                label = line.strip().strip("*").strip()
                if label.endswith(":"):
                    label = label[:-1].strip()
                pending_table_caption = convert_inline_formatting(label)
                i += 1
                continue

        # Headings
        if line.startswith("## "):
            raw = line[3:].strip()
            clean = convert_inline_formatting(_strip_heading_numbering(raw))

            if raw.lower().startswith("appendix"):
                if not in_appendix:
                    latex += "\\appendix\n\n"
                    in_appendix = True
                latex += f"\\section{{{clean}}}\n\n"
                i += 1
                continue

            if in_ack:
                latex += "\\end{acknowledgments}\n\n"
                in_ack = False

            latex += f"\\section{{{clean}}}\n\n"
            i += 1
            continue

        if line.startswith("### "):
            latex += f"\\subsection{{{convert_inline_formatting(_strip_heading_numbering(line[4:].strip()))}}}\n\n"
            i += 1
            continue

        if line.startswith("#### "):
            latex += f"\\subsubsection{{{convert_inline_formatting(_strip_heading_numbering(line[5:].strip()))}}}\n\n"
            i += 1
            continue

        # Block equations
        stripped = line.strip()
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            latex += "\\begin{equation}\n" + stripped[2:-2].strip() + "\n\\end{equation}\n\n"
            i += 1
            continue

        if stripped == "$$":
            latex += "\\begin{equation}\n" if not in_equation else "\\end{equation}\n\n"
            in_equation = not in_equation
            i += 1
            continue

        if in_equation:
            latex += line + "\n"
            i += 1
            continue

        # Tables (robust parsing): detect header + separator, then consume following pipe rows.
        if not in_table and "|" in line and not line.strip().startswith("<!--"):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and _table_sep_line(lines[j]):
                in_table = True
                table_buffer = [line, lines[j]]
                i = j + 1
                continue

        if in_table:
            # Consume data rows
            if "|" in line and not line.strip().startswith("<!--") and not _table_sep_line(line):
                table_buffer.append(line)
                i += 1
                continue

            # Flush table when pipe rows end (or on blank/non-table line)
            k = i
            while k < len(lines) and lines[k].strip() == "":
                k += 1
            table_note = None
            note_line = lines[k].strip() if k < len(lines) else ""
            # Only treat fully-italic note lines like *...* as table notes.
            if note_line.startswith("*") and note_line.endswith("*") and not note_line.startswith("**"):
                raw_note = note_line.strip().lstrip("*").rstrip("*").strip()
                if raw_note and not raw_note.lower().startswith(("fig.", "figure")):
                    table_note = raw_note
                    i = k + 1

            ncol = len([c for c in table_buffer[0].split("|") if c.strip()])
            wide = ncol >= 5
            env = "table*" if wide else "table"
            width_ref = "\\textwidth" if wide else "\\columnwidth"

            latex += f"\\begin{{{env}}}[t]\n\\centering\n"
            if pending_table_caption:
                if table_note:
                    latex += f"\\caption{{{pending_table_caption} \\\\ \\textit{{{convert_inline_formatting(table_note)}}}}}\n"
                else:
                    latex += f"\\caption{{{pending_table_caption}}}\n"
            elif table_note:
                latex += f"\\caption{{\\textit{{{convert_inline_formatting(table_note)}}}}}\n"
            latex += ("\\scriptsize\n" if wide else "\\small\n")
            latex += convert_table(table_buffer, width_ref=width_ref)
            latex += f"\\end{{{env}}}\n\n"

            pending_table_caption = None
            in_table = False
            table_buffer = []
            continue

        # Lists
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            latex += "\\begin{itemize}\n"
            while i < len(lines) and (lines[i].strip().startswith("- ") or lines[i].strip().startswith("* ")):
                latex += f"\\item {convert_inline_formatting(lines[i].strip()[2:])}\n"
                i += 1
            latex += "\\end{itemize}\n\n"
            continue

        if re.match(r"^\d+[\.\)]\s", line.strip()):
            latex += "\\begin{enumerate}\n"
            while i < len(lines) and re.match(r"^\d+[\.\)]\s", lines[i].strip()):
                item = re.sub(r"^\d+[\.\)]\s", "", lines[i].strip())
                latex += f"\\item {convert_inline_formatting(item)}\n"
                i += 1
            latex += "\\end{enumerate}\n\n"
            continue

        # Quotes
        if line.strip().startswith(">"):
            latex += "\\begin{quote}\n"
            while i < len(lines) and lines[i].strip().startswith(">"):
                latex += convert_inline_formatting(lines[i].strip()[1:].strip()) + "\n\n"
                i += 1
            latex += "\\end{quote}\n\n"
            continue

        # Images (use italic FIG caption line)
        if line.strip().startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
            if m:
                alt_caption = m.group(1).strip()
                path = m.group(2).strip()
                if not path.startswith("http") and not path.startswith("/") and not path.startswith(".."):
                    path = "../" + path

                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                long_caption = None
                if j < len(lines):
                    mcap = re.match(r"^\*(.+)\*\s*$", lines[j].strip())
                    if mcap:
                        long_caption = mcap.group(1).strip()

                cap_for_layout = (alt_caption + " " + (long_caption or "")).lower()
                env = "figure*" if any(k in cap_for_layout for k in ("gallery", "cluster", "kappa", "profiles")) else "figure"
                pos = "t" if env == "figure*" else "htbp"
                width = "\\textwidth" if env == "figure*" else "\\columnwidth"

                latex += f"\\begin{{{env}}}[{pos}]\n\\centering\n"
                latex += f"\\includegraphics[width={width}]{{{path}}}\n"
                if long_caption:
                    cleaned = re.sub(r"^\s*(FIG|Fig)\.\s*\d+\.\s*", "", long_caption).strip()
                    latex += f"\\caption{{{convert_inline_formatting(cleaned)}}}\n"
                    i = j
                else:
                    latex += f"\\caption{{{convert_inline_formatting(alt_caption)}}}\n"
                latex += f"\\end{{{env}}}\n\n"
            i += 1
            continue

        # Horizontal rules
        if line.strip() in ("---", "***", "___"):
            latex += "\\medskip\\hrule\\medskip\n\n"
            i += 1
            continue

        # Paragraphs + numeric citations
        if line.strip():
            converted = convert_inline_formatting(line)

            def _cite_repl(m: re.Match[str]) -> str:
                nums = [x.strip() for x in m.group(1).split(",")]
                keys = [bibkey_for_number.get(n, f"ref{n}") for n in nums]
                return "\\cite{" + ",".join(keys) + "}"

            converted = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", _cite_repl, converted)
            latex += converted + "\n\n"
        else:
            latex += "\n"

        i += 1

    if in_ack:
        latex += "\\end{acknowledgments}\n\n"

    latex += "\\end{document}\n"
    return latex


def main() -> int:
    repo_root = Path(__file__).parent.parent
    readme_path = repo_root / "README.md"
    docs_path = repo_root / "docs"
    tex_path = docs_path / "sigmagravity_paper.tex"
    pdf_path = docs_path / "sigmagravity_paper.pdf"

    docs_path.mkdir(exist_ok=True)

    print(f"Reading {readme_path}...")
    md_content = readme_path.read_text(encoding="utf-8")

    print("Converting to REVTeX (PRD reprint)...")
    latex_content = convert_markdown_to_revtex(md_content)

    print("Validating README coverage...")
    _validate_readme_coverage(md_content, latex_content)

    print(f"Writing {tex_path}...")
    tex_path.write_text(latex_content, encoding="utf-8")

    print("Compiling to PDF (pdflatex, 2 passes)...")
    jobname = pdf_path.stem

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"-jobname={jobname}", str(tex_path)],
        cwd=docs_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    produced_pdf = docs_path / f"{jobname}.pdf"
    if not produced_pdf.exists():
        print("[FAILED] PDF generation failed")
        if result.stdout:
            print("STDOUT:", result.stdout[-2000:])
        if result.stderr:
            print("STDERR:", result.stderr[-2000:])
        return 1

    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"-jobname={jobname}", str(tex_path)],
        cwd=docs_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    print(f"[OK] Generated: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build Frontiers LaTeX sources from the frozen revision Markdown files.

The conversion is mechanical: Pandoc handles equations, lists, and long
tables; this script supplies the official Frontiers wrappers, bibliography
commands, citations, end matter, and publication figures. It does not execute
or refit any scientific analysis.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESUBMISSION = HERE.parent
ROOT = RESUBMISSION.parents[1]
TMP = ROOT / "tmp"
MAIN_MD = RESUBMISSION / "SigmaGravity_FINAL_Manuscript_2026-07-26.md"
SUPP_MD = RESUBMISSION / "SigmaGravity_Supplementary_2026-07-25.md"
MAIN_TEX = RESUBMISSION / "SigmaGravity_FINAL_Manuscript_2026-07-26.tex"
SUPP_TEX = RESUBMISSION / "SigmaGravity_Supplementary_2026-07-25.tex"


def pandoc_body(source: Path, destination: Path, *, listings: bool = False) -> str:
    cmd = [
        "pandoc",
        str(source),
        "-f",
        "markdown+tex_math_single_backslash",
        "-t",
        "latex",
        "--wrap=none",
        "--shift-heading-level-by=-1",
    ]
    if listings:
        cmd.append("--listings")
    cmd.extend(["-o", str(destination)])
    subprocess.run(cmd, cwd=ROOT, check=True)
    return destination.read_text(encoding="utf-8")


def approximate_word_count(markdown: str) -> int:
    # Frontiers counts the main body but excludes the abstract, section titles,
    # figure/table captions, acknowledgments, and references.
    start = markdown.index("## 1 Introduction")
    end = markdown.index("## Data availability statement")
    text = markdown[start:end]
    text = re.sub(r"^#{1,6}.*$", " ", text, flags=re.M)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.S)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    text = re.sub(r"[#|*_>`~]", " ", text)
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def latex_safe_unicode(text: str) -> str:
    replacements = {
        "Σ-Gravity": r"\(\Sigma\)-Gravity",
        "ΛCDM": r"\(\Lambda\)CDM",
        "Σ": r"\(\Sigma\)",
        "α": r"\(\alpha\)",
        "±": r"\(\pm\)",
        "−": "-",
        "≤": r"\(\leq\)",
        "≥": r"\(\geq\)",
        "×": r"\(\times\)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def replace_citations(text: str) -> str:
    replacements = {
        "(Planck Collaboration, 2020)": r"\citep{planck2020}",
        "(Milgrom, 1983; Sanders and McGaugh, 2002; Famaey and McGaugh, 2012)": (
            r"\citep{milgrom1983,sanders2002,famaey2012}"
        ),
        "(McGaugh et al., 2016)": r"\citep{mcgaugh2016}",
        "(Milgrom, 2010)": r"\citep{milgrom2010}",
        "(Lelli et al., 2016)": r"\citep{lelli2016}",
        "Fox et al.~(2022)": r"\citet{fox2022}",
        "Tian et al.~(2020)": r"\citet{tian2020}",
        "Bevacqua et al.~(2022)": r"\citet{bevacqua2022}",
        "Zhu et al.~(2023)": r"\citet{zhu2023}",
        "(Cesare et al., 2020, 2022; Sanna et al., 2023)": (
            r"\citep{cesare2020,cesare2022,sanna2023}"
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_main_body(body: str) -> tuple[str, str]:
    abstract_match = re.search(
        r"\\section\{Abstract\}\\label\{abstract\}\s*(.*?)\s*"
        r"\\section\{1 Introduction\}\\label\{introduction\}",
        body,
        flags=re.S,
    )
    if not abstract_match:
        raise RuntimeError("Could not locate converted abstract.")
    abstract = abstract_match.group(1).strip()

    main_start = abstract_match.end()
    main_end = body.index(r"\section{Data availability statement}")
    main = r"\section{Introduction}\label{introduction}" + "\n\n" + body[main_start:main_end]
    main = re.sub(r"\\section\{([2-5]) ([^}]+)\}", r"\\section{\2}", main)
    main = re.sub(r"\\subsection\{\d+\.\d+ ([^}]+)\}", r"\\subsection{\1}", main)

    main = replace_citations(main)
    abstract = latex_safe_unicode(abstract)
    main = latex_safe_unicode(main)

    figure_refs = {
        "Figure 1": r"Figure~\ref{fig:1}",
        "Figure 2": r"Figure~\ref{fig:2}",
        "Figure 3": r"Figure~\ref{fig:3}",
        "Figure 4": r"Figure~\ref{fig:4}",
        "Table 1": r"Table~\ref{tab:roles}",
    }
    for old, new in figure_refs.items():
        main = main.replace(old, new)
    return abstract, main.strip()


def main_tables() -> str:
    return r"""
\section*{Tables}

\begin{table}[h!]
\centering
\caption{Role of each dataset and analysis.}\label{tab:roles}
\scriptsize
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.25\textwidth}>{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}X}
\toprule
Dataset or analysis & Role & Independent unit & Determines a model parameter? \\
\midrule
SPARC disk sample & Retrospective evaluation & Galaxy & No per-galaxy parameter \\
SPARC photometric scale-length test & Secondary structural-hypothesis sensitivity & Galaxy & No; catalog values used without fitting \\
Fox clusters & Illustrative in-sample calibration & Cluster & Yes, one effective cluster amplitude \\
Repeated Fox splits & Calibration-stability diagnostic & Held-out cluster within Fox & Refits on each training subset \\
Non-overlapping Tian/CLASH profiles & No-refit external profile check & Cluster, with radii grouped & No \\
Matched MaNGA/JAM catalog & Secondary counterrotation diagnostic & Galaxy/control set & No \\
Numerical QUMOND disks & Algebraic-approximation diagnostic & Reconstructed galaxy model & No \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[h!]
\centering
\caption{Parameter and assumption accounting.}\label{tab:parameters}
\scriptsize
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.22\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}X}
\toprule
Quantity & Role & Status & Principal limitation \\
\midrule
\(g^\dagger\) & Acceleration scale in \(h(g_N)\) & Fixed model choice & BTFR constrains \(B^2g^\dagger\), not each factor \\
\(A_0=e^{1/(2\pi)}\) & Galaxy normalization before \(F(V_\Sigma)\) & Fixed model choice & Not uniquely derived from the data \\
\(\sigma=20\ {\rm km\,s^{-1}}\) & Regulator in \(F(V_\Sigma)\) & Fixed & Endogenous; sensitivity tested from 10--50 km s\(^{-1}\) \\
\(\Upsilon_{\rm disk},\Upsilon_{\rm bulge}\) & Stellar baryonic contributions & Fixed astrophysical assumptions & Change relative \(\Sigma\)/MOND performance \\
\(B_{\rm Fox}=8.446\) & Cluster response & Illustrative calibration & Does not transfer as a universal radial amplitude \\
\(0.4\times0.15M_{500}\) & Fox baryon mass inside 200 kpc & Approximation & Material amplitude sensitivity \\
Lensing closure & Relates response to lensing target & Assumed & Gravitational slip is undetermined \\
\(R_d\) & Photometric scale length & Tested only in a secondary no-refit window & Catalog assignments do not outperform permutation or fixed-median controls \\
\(L_0,n\) & System-scale path hypothesis & Not used in locked galaxy result & No unique operational 3D functional; not separately identified \\
\bottomrule
\end{tabularx}
\end{table}
""".strip()


def main_figures() -> str:
    captions = [
        (
            "figure_1_sparc_paired.pdf",
            r"""Locked SPARC comparison and nuisance sensitivity. Left: per-galaxy velocity RMS for \(\Sigma\)-Gravity and the tested MOND prescription, with SPARC quality classes shown separately. Center: distribution of the paired contrast \({\rm RMS}_\Sigma-{\rm RMS}_{\rm MOND}\); the mean is \(+0.309\ {\rm km\,s^{-1}}\) and its 95\% galaxy-bootstrap interval includes zero. Right: mean contrast for the 81 frozen nuisance combinations. Negative values favor \(\Sigma\)-Gravity and positive values favor MOND. Neither the primary paired interval nor this sensitivity grid establishes a statistically significant difference or equivalence.""",
        ),
        (
            "figure_2_cluster_roles.pdf",
            r"""Illustrative cluster calibration and no-refit radial evaluation. Left: predicted and observed 200-kpc aperture masses for the 42 Fox clusters under the simplified baryon proxy and calibrated \(B_{\rm Fox}\). This panel is an illustrative in-sample calibration, not validation. Right: predicted-to-observed acceleration ratios for 73 Tian/CLASH radial measurements in 17 clusters after the fixed name-normalization rule excludes MACS0416, MACS0717, and MACS1149, with \(B_{\rm Fox}\) frozen. Large markers show disjoint-sample radius-bin medians. The systematic increase with radius shows that the fixed Fox-calibrated amplitude does not transfer without bias under the present baryonic and lensing assumptions.""",
        ),
        (
            "figure_3_qumond_approximation.pdf",
            r"""Algebraic approximation error in representative axisymmetric disk reconstructions. Fractional acceleration difference between the algebraic relation in Equation (13) and the numerical fixed-\(B\) QUMOND solution for analytic reconstructions representative of F574-2, NGC3741, and UGC05716. The comparison quantifies a geometry-dependent approximation error; it is not a fit to the locked endogenous prescription or the full observed gas and bulge maps.""",
        ),
        (
            "figure_4_counterrotation_matched.pdf",
            r"""Matched counterrotation diagnostic. Left: absolute standardized mean differences before and after matching counterrotators to controls. All post-match values are below the stated 0.1 balance threshold. Right: matched difference in the JAM/NFW-derived \(f_{\rm DM}(<R_e)\), with matched-set bootstrap 95\% interval. The interval includes zero. Because the outcome is model derived, this is a secondary catalog diagnostic rather than a direct test of \(\Sigma\)-Gravity.""",
        ),
    ]
    blocks = [r"\section*{Figure captions}"]
    for index, (filename, caption) in enumerate(captions, start=1):
        blocks.append(
            "\n".join(
                [
                    r"\begin{figure}[h!]",
                    r"\begin{center}",
                    rf"\includegraphics[width=0.98\textwidth]{{figures/{filename}}}",
                    r"\end{center}",
                    rf"\caption{{{caption}}}\label{{fig:{index}}}",
                    r"\end{figure}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_main() -> None:
    TMP.mkdir(exist_ok=True)
    converted = pandoc_body(MAIN_MD, TMP / "SigmaGravity_main_shift.tex")
    abstract, main = clean_main_body(converted)
    word_count = approximate_word_count(MAIN_MD.read_text(encoding="utf-8"))

    header = rf"""\documentclass[utf8]{{FrontiersinHarvard}}
\usepackage{{url,hyperref,lineno,microtype,subcaption}}
\usepackage[onehalfspacing]{{setspace}}
\usepackage{{amsmath,amssymb,booktabs,tabularx,array,graphicx}}
\linenumbers
\providecommand{{\tightlist}}{{}}

\def\keyFont{{\fontsize{{8}}{{11}}\helveticabold}}
\def\firstAuthorLast{{Speiser}}
\def\Authors{{Leonard Speiser\,$^{{1,*}}$}}
\def\Address{{$^{{1}}$Horizon 3, Independent Research, Los Altos, CA, United States}}
\def\corrAuthor{{Leonard Speiser}}
\def\corrEmail{{leonard@horizon3.net}}

\begin{{document}}
\onecolumn
\firstpage{{1}}
\title[$\Sigma$-Gravity]{{$\Sigma$-Gravity: A Coherence-Motivated Empirical Response Tested in Galaxies and Clusters}}
\author[\firstAuthorLast]{{\Authors}}
\address{{}}
\correspondance{{}}
\extraAuth{{}}
\maketitle

\begin{{center}}
\small Manuscript ID 1866133 \quad Word count: {word_count} \quad Figures: 4 \quad Tables: 2
\end{{center}}

\begin{{abstract}}
\section{{}}
{abstract}

\tiny
\keyFont{{\section{{Keywords:}} galaxy rotation curves, radial acceleration relation, modified gravity, dark matter, galaxy clusters, gravitational lensing, SPARC, phenomenology}}
\end{{abstract}}
"""

    endmatter = r"""
\section*{Data Availability Statement}

SPARC data are publicly available at \href{http://astroweb.cwru.edu/SPARC/}{astroweb.cwru.edu/SPARC}. The CLASH radial-acceleration catalog is available through VizieR as J/ApJ/896/70. The Fox cluster table used for calibration, frozen residuals, split definitions, matched samples, parameter diagnostics, and figure-generation code are provided in the accompanying repository and Supplementary Material.

\section*{Author Contributions}

LS conceived the study, developed the model, assembled the data and code, performed the analyses, interpreted the results, and wrote and revised the manuscript.

\section*{Funding}

The author declares that no financial support was received for the research, authorship, and/or publication of this article.

\section*{Conflict of Interest Statement}

The author declares that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

\section*{Acknowledgments}

The author thanks Emmanuel N. Saridakis, Rafael Ferraro, and Tiberiu Harko for earlier discussions concerning theoretical consistency and modified-gravity frameworks.

\section*{Supplemental Data}

The Supplementary Material contains the full statistical procedures, identifiability audit, frozen dataset roles, software-regression commands, and machine-readable output manifest.

\bibliographystyle{Frontiers-Harvard}
\bibliography{SigmaGravity_FINAL_Manuscript_2026-07-26}
"""

    content = "\n\n".join(
        [header.strip(), main, endmatter.strip(), main_tables(), main_figures(), r"\end{document}"]
    )
    MAIN_TEX.write_text(content + "\n", encoding="utf-8")


def build_supplement() -> None:
    converted = pandoc_body(
        SUPP_MD, TMP / "SigmaGravity_supp_shift.tex", listings=True
    )
    converted = latex_safe_unicode(converted)
    converted = converted.replace(
        r"\section{S1. Scope and reproducibility contract}",
        r"\section{Scope and reproducibility contract}",
    )
    converted = re.sub(
        r"\\section\{S(\d+)\. ([^}]+)\}",
        r"\\section{\2}",
        converted,
    )
    converted = re.sub(
        r"\\subsection\{S(\d+)\.(\d+) ([^}]+)\}",
        r"\\subsection{\3}",
        converted,
    )
    header = r"""\documentclass[utf8]{frontiers_suppmat}
\usepackage{url,hyperref,microtype}
\usepackage[onehalfspacing]{setspace}
\usepackage{amsmath,amssymb,booktabs,longtable,array,graphicx,listings,xcolor,calc}
\providecommand{\tightlist}{}
\newcommand{\real}[1]{#1}
\newcommand{\passthrough}[1]{#1}
\lstset{basicstyle=\ttfamily\small,breaklines=true,columns=fullflexible}

\begin{document}
\onecolumn
\firstpage{1}
\title[Supplementary Material]{{\helveticaitalic{Supplementary Material}}}
\maketitle
"""
    content = "\n\n".join([header.strip(), converted.strip(), r"\end{document}"])
    SUPP_TEX.write_text(content + "\n", encoding="utf-8")


def copy_frontiers_template_files() -> None:
    template = ROOT / "tmp" / "frontiers_latex_template_20260725"
    targets = [
        "FrontiersinHarvard.cls",
        "Frontiers-Harvard.bst",
        "frontiers_suppmat.cls",
        "logo1.pdf",
    ]
    for name in targets:
        source = template / name
        destination = RESUBMISSION / name
        if source.exists():
            shutil.copy2(source, destination)
        elif not destination.exists():
            raise FileNotFoundError(
                f"Frontiers template asset is missing from both {source} and {destination}"
            )
    # Tectonic/XeTeX does not include EPS graphics. The template bundle ships
    # an equivalent PDF logo, so point both copied classes at that asset.
    for class_name in ("FrontiersinHarvard.cls", "frontiers_suppmat.cls"):
        class_path = RESUBMISSION / class_name
        class_text = class_path.read_text(encoding="latin-1")
        class_path.write_text(
            class_text.replace("./logo1.eps", "./logo1.pdf"),
            encoding="latin-1",
        )


def main() -> None:
    build_main()
    build_supplement()
    copy_frontiers_template_files()
    print(f"Wrote {MAIN_TEX}")
    print(f"Wrote {SUPP_TEX}")


if __name__ == "__main__":
    main()

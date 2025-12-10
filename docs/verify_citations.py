#!/usr/bin/env python3
"""
Verify README.md citations against downloaded reference PDFs.
Extracts key information from each PDF and compares against citation claims.
"""

import pdfplumber
import os
from pathlib import Path

# Define the citations from README.md and what we need to verify
CITATIONS = {
    1: {
        "claim": "F. Zwicky, Helv. Phys. Acta 6, 110 (1933) - Original cluster observations showing missing mass",
        "pdf": "zwicky-redshift.pdf",
        "verify": ["Zwicky", "1933", "Helv", "cluster", "mass"]
    },
    2: {
        "claim": "Planck Collaboration, A&A 641, A6 (2020) - Dark matter ~27% of cosmic energy density",
        "pdf": "Planck2020_cosmological_params.pdf",
        "verify": ["Planck", "2020", "A&A", "641", "dark matter", "27%", "cosmological"]
    },
    3: {
        "claim": "M. Milgrom, Astrophys. J. 270, 365 (1983) - MOND introduction, a0 ~ 1.2×10^-10 m/s²",
        "pdf": "1983ApJ.pdf",
        "verify": ["Milgrom", "1983", "Astrophys", "270", "365", "acceleration"]
    },
    4: {
        "claim": "M. Milgrom, Astrophys. J. 270, 371 (1983) - MOND implications for galaxies",
        "pdf": "1983ApJ-milgrom-implications-for-galaxies.pdf",
        "verify": ["Milgrom", "1983", "Astrophys", "270", "371", "galaxies"]
    },
    5: {
        "claim": "S. S. McGaugh et al., Astrophys. J. Lett. 533, L99 (2000) - Baryonic Tully-Fisher relation",
        "pdf": "McGaugh2000_BTFR.pdf",
        "verify": ["McGaugh", "2000", "533", "Tully-Fisher", "baryonic"]
    },
    6: {
        "claim": "J. D. Bekenstein, Phys. Rev. D 70, 083509 (2004) - TeVeS relativistic MOND",
        "pdf": "Bekenstein2004_TeVeS.pdf",
        "verify": ["Bekenstein", "2004", "Phys. Rev", "70", "083509", "TeVeS"]
    },
    7: {
        "claim": "M. Milgrom, Phys. Rev. D 80, 123536 (2009) - BIMOND",
        "pdf": "Milgrom2009_BiMOND.pdf",
        "verify": ["Milgrom", "2009", "Phys. Rev", "80", "123536", "bimetric"]
    },
    8: {
        "claim": "R. H. Sanders and S. S. McGaugh, Annu. Rev. Astron. Astrophys. 40, 263 (2002) - MOND review, cluster challenges",
        "pdf": "Sanders2002_MOND_review.pdf",
        "verify": ["Sanders", "McGaugh", "2002", "Annu. Rev", "40", "263", "MOND"]
    },
    9: {
        "claim": "M. Milgrom, Phys. Rev. D 82, 043523 (2010) - QUMOND formulation",
        "pdf": "Milgrom2010_QUMOND.pdf",
        "verify": ["Milgrom", "2010", "Phys. Rev", "82", "043523", "QUMOND"]
    },
    10: {
        "claim": "R. Ferraro and F. Fiorini, Phys. Rev. D 75, 084031 (2007) - f(T) teleparallel gravity",
        "pdf": "Ferraro2007_teleparallel.pdf",
        "verify": ["Ferraro", "Fiorini", "2007", "Phys. Rev", "75", "084031", "teleparallel"]
    },
    11: {
        "claim": "S. Bahamonde et al., Rep. Prog. Phys. 86, 026901 (2023) - Teleparallel review",
        "pdf": "Bahamonde2023_teleparallel_review.pdf",
        "verify": ["Bahamonde", "2023", "Rep. Prog. Phys", "86", "026901", "teleparallel"]
    },
    12: {
        "claim": "E. P. Verlinde, SciPost Phys. 2, 016 (2017) - Emergent gravity",
        "pdf": "Verlinde2017_emergent_gravity.pdf",
        "verify": ["Verlinde", "2017", "SciPost", "emergent", "gravity"]
    },
    13: {
        "claim": "B. Bertotti, L. Iess, and P. Tortora, Nature 425, 374 (2003) - Cassini PPN bound |γ-1| < 2.3×10^-5",
        "pdf": "bertotti-cassini.pdf",
        "verify": ["Bertotti", "Iess", "Tortora", "2003", "Nature", "425", "374", "Cassini", "2.3"]
    },
    15: {
        "claim": "F. Lelli, S. S. McGaugh, and J. M. Schombert, Astron. J. 152, 157 (2016) - SPARC database",
        "pdf": "Lelli2016_SPARC.pdf",
        "verify": ["Lelli", "McGaugh", "Schombert", "2016", "Astron. J", "152", "157", "SPARC"]
    },
    16: {
        "claim": "A.-C. Eilers et al., Astrophys. J. 871, 120 (2019) - Milky Way rotation curve from Gaia",
        "pdf": "Eilers2019_MW.pdf",
        "verify": ["Eilers", "2019", "Astrophys. J", "871", "120", "Milky Way", "Gaia"]
    },
    17: {
        "claim": "C. Fox et al., Astrophys. J. 928, 87 (2022) - Strong-lensing clusters",
        "pdf": "Fox2022_clusters.pdf",
        "verify": ["Fox", "2022", "Astrophys. J", "928", "87", "lensing", "cluster"]
    },
}

def extract_pdf_text(pdf_path, max_pages=5):
    """Extract text from first few pages of PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for i, page in enumerate(pdf.pages[:max_pages]):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- PAGE {i+1} ---\n{page_text}"
            return text
    except Exception as e:
        return f"ERROR reading PDF: {str(e)}"

def verify_citation(ref_num, citation_info, refs_dir):
    """Verify a single citation against its PDF."""
    pdf_path = refs_dir / citation_info["pdf"]
    
    result = {
        "ref_num": ref_num,
        "claim": citation_info["claim"],
        "pdf_file": citation_info["pdf"],
        "pdf_exists": pdf_path.exists(),
        "found_terms": [],
        "missing_terms": [],
        "status": "UNKNOWN",
        "extracted_text": "",
        "key_excerpts": []
    }
    
    if not pdf_path.exists():
        result["status"] = "PDF NOT FOUND"
        return result
    
    text = extract_pdf_text(pdf_path)
    result["extracted_text"] = text[:3000]  # First 3000 chars for report
    
    # Check for verification terms
    text_lower = text.lower()
    for term in citation_info["verify"]:
        if term.lower() in text_lower:
            result["found_terms"].append(term)
        else:
            result["missing_terms"].append(term)
    
    # Determine status
    found_ratio = len(result["found_terms"]) / len(citation_info["verify"])
    if found_ratio >= 0.8:
        result["status"] = "VERIFIED"
    elif found_ratio >= 0.5:
        result["status"] = "LIKELY CORRECT"
    else:
        result["status"] = "NEEDS REVIEW"
    
    # Extract key excerpts (lines containing author names or key terms)
    lines = text.split('\n')
    key_terms = citation_info["verify"][:3]  # First 3 terms (usually author, year)
    for line in lines[:100]:  # First 100 lines
        line_lower = line.lower()
        for term in key_terms:
            if term.lower() in line_lower and len(line.strip()) > 10:
                if line.strip() not in result["key_excerpts"]:
                    result["key_excerpts"].append(line.strip())
                break
    
    return result

def generate_report(results):
    """Generate markdown verification report."""
    report = """# Citation Verification Report

This document verifies the citations in README.md against the downloaded reference PDFs in `docs/references/`.

## Summary

"""
    
    verified = sum(1 for r in results if r["status"] == "VERIFIED")
    likely = sum(1 for r in results if r["status"] == "LIKELY CORRECT")
    needs_review = sum(1 for r in results if r["status"] == "NEEDS REVIEW")
    not_found = sum(1 for r in results if r["status"] == "PDF NOT FOUND")
    
    report += f"- **VERIFIED**: {verified}\n"
    report += f"- **LIKELY CORRECT**: {likely}\n"
    report += f"- **NEEDS REVIEW**: {needs_review}\n"
    report += f"- **PDF NOT FOUND**: {not_found}\n\n"
    
    report += "---\n\n## Detailed Verification\n\n"
    
    for r in results:
        report += f"### Reference [{r['ref_num']}]\n\n"
        report += f"**README Claim:** {r['claim']}\n\n"
        report += f"**PDF File:** `{r['pdf_file']}`\n\n"
        report += f"**Status:** {r['status']}\n\n"
        
        if r["found_terms"]:
            report += f"**Found Terms:** {', '.join(r['found_terms'])}\n\n"
        if r["missing_terms"]:
            report += f"**Missing Terms:** {', '.join(r['missing_terms'])}\n\n"
        
        if r["key_excerpts"]:
            report += "**Key Excerpts from PDF:**\n\n"
            for excerpt in r["key_excerpts"][:5]:
                # Clean up the excerpt
                clean = excerpt.replace('\n', ' ').strip()
                if len(clean) > 200:
                    clean = clean[:200] + "..."
                report += f"> {clean}\n\n"
        
        if r["status"] == "NEEDS REVIEW":
            report += "**⚠️ Manual Review Recommended**\n\n"
            report += f"First 500 chars of PDF:\n```\n{r['extracted_text'][:500]}\n```\n\n"
        
        report += "---\n\n"
    
    return report

def main():
    refs_dir = Path("/Users/leonardspeiser/Projects/sigmagravity/docs/references")
    
    print("Verifying citations...")
    results = []
    
    for ref_num in sorted(CITATIONS.keys()):
        print(f"  Checking reference [{ref_num}]...")
        result = verify_citation(ref_num, CITATIONS[ref_num], refs_dir)
        results.append(result)
        print(f"    Status: {result['status']}")
    
    report = generate_report(results)
    
    output_path = Path("/Users/leonardspeiser/Projects/sigmagravity/docs/CITATION_VERIFICATION.md")
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport written to: {output_path}")
    print("\nSummary:")
    for r in results:
        emoji = "✅" if r["status"] == "VERIFIED" else "⚠️" if r["status"] == "LIKELY CORRECT" else "❌"
        print(f"  {emoji} [{r['ref_num']}] {r['status']}")

if __name__ == "__main__":
    main()


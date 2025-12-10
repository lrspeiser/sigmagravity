# Citation Verification Report

This document verifies the citations in README.md against the downloaded reference PDFs in `docs/references/`.

## Executive Summary

**Critical Finding: 4 PDF files contain the WRONG papers entirely.**

| Status | Count |
|--------|-------|
| ✅ VERIFIED | 11 |
| ❌ WRONG FILE | 4 |
| 📄 SCANNED IMAGE | 1 |

---

## ❌ CRITICAL ISSUES - Wrong PDFs Downloaded

### Reference [4] - VERIFIED (Scanned Image) ✅

**README Citation:** M. Milgrom, Astrophys. J. **270**, 371 (1983) - MOND implications for galaxies

**PDF File:** `1983ApJ-milgrom-implications-for-galaxies.pdf`

**Status:** ✅ VERIFIED via PDF metadata - This is a scanned image PDF (pre-digital era paper)

**PDF Metadata Confirms:**
- Title: `1983ApJ...270..371M` (ADS bibcode = 1983, ApJ, Vol 270, Page 371, Milgrom)
- Producer: NASA Astrophysics Data System
- Pages: 13 (consistent with full article)

**Note:** Text extraction failed because this is a scanned image, but the metadata confirms it's the correct paper.

---

### Reference [7] - WRONG PAPER

**README Citation:** M. Milgrom, Phys. Rev. D **80**, 123536 (2009) - BIMOND

**PDF File:** `Milgrom2009_BiMOND.pdf`

**Status:** ❌ WRONG FILE - This is an optics paper, NOT Milgrom's BIMOND

**Actual Content in PDF:**
> "Electromagnetically Induced Transparency... Coherent Control of the Light"
> Authors: R. Drampyan, S. Pustelny, W. Gawlik
> Topic: Nonlinear Faraday effect in rubidium vapors

**Expected:** Milgrom's bimetric MOND theory paper

**Action Required:** Download the correct paper: https://arxiv.org/abs/0906.0399

---

### Reference [9] - WRONG PAPER

**README Citation:** M. Milgrom, Phys. Rev. D **82**, 043523 (2010) - QUMOND formulation

**PDF File:** `Milgrom2010_QUMOND.pdf`

**Status:** ❌ WRONG FILE - This is Verlinde's emergent gravity paper

**Actual Content in PDF:**
> "On the Origin of Gravity and the Laws of Newton"
> Author: Erik Verlinde
> Institute for Theoretical Physics, University of Amsterdam
> "Gravity is explained as an entropic force caused by changes in the information..."

**Expected:** Milgrom's QUMOND (quasi-linear MOND) formulation paper

**Action Required:** Download the correct paper: https://arxiv.org/abs/0912.0790

---

### Reference [10] - WRONG PAPER

**README Citation:** R. Ferraro and F. Fiorini, Phys. Rev. D **75**, 084031 (2007) - f(T) teleparallel gravity

**PDF File:** `Ferraro2007_teleparallel.pdf`

**Status:** ❌ WRONG FILE - This is a spinfoam quantum gravity paper

**Actual Content in PDF:**
> "3d Spinfoam Quantum Gravity: Matter as a Phase of the Group Field Theory"
> Authors: Winston J. Fairbairn and Etera R. Livine
> Laboratoire de Physique - ENS Lyon

**Expected:** Ferraro & Fiorini's paper on f(T) teleparallel gravity modifications

**Action Required:** Download the correct paper: https://arxiv.org/abs/gr-qc/0702125

---

### Reference [17] - WRONG PAPER

**README Citation:** C. Fox, G. Mahler, K. Sharon, and J. D. Remolina González, Astrophys. J. **928**, 87 (2022) - Strong-lensing clusters

**PDF File:** `Fox2022_clusters.pdf`

**Status:** ❌ WRONG FILE - This is a paper about dust shells, NOT strong-lensing clusters

**Actual Content in PDF:**
> "THE PER-TAU SHELL: A GIANT STAR-FORMING SPHERICAL SHELL REVEALED BY 3D DUST OBSERVATIONS"
> Authors: Shmuel Bialy, Catherine Zucker, Alyssa Goodman, et al.
> Topic: Molecular cloud formation in the Perseus and Taurus region

**Expected:** Fox et al. paper on strong-lensing galaxy clusters

**Action Required:** Download the correct paper: https://ui.adsabs.harvard.edu/abs/2022ApJ...928...87F

---

## ✅ VERIFIED CITATIONS

### Reference [1] - VERIFIED ✅

**README Citation:** F. Zwicky, Helv. Phys. Acta **6**, 110 (1933)

**PDF File:** `zwicky-redshift.pdf`

**Verification:**
> "The Redshift of Extragalactic Nebulae - Fritz Zwicky"
> "Published in Helvetica Physica Acta, Vol. 6, p. 110-127, 1933."

**Status:** ✅ Correct paper, correct citation details

---

### Reference [2] - VERIFIED ✅

**README Citation:** Planck Collaboration, A&A **641**, A6 (2020)

**PDF File:** `Planck2020_cosmological_params.pdf`

**Verification:**
> "Planck 2018 results. VI. Cosmological parameters"
> Planck Collaboration authors listed

**Note:** The PDF is the 2018 results published in 2020. Citation is correct.

**Status:** ✅ Correct paper

---

### Reference [3] - VERIFIED ✅

**README Citation:** M. Milgrom, Astrophys. J. **270**, 365 (1983)

**PDF File:** `1983ApJ.pdf`

**Verification:**
> "The Astrophysical Journal, 270:365-370, 1983 July 15"
> "© 1983. The American Astronomical Society. All rights reserved."
> "366 MILGROM Vol. 270"

**Status:** ✅ Correct paper, correct page numbers

---

### Reference [5] - VERIFIED ✅

**README Citation:** S. S. McGaugh, J. M. Schombert, G. D. Bothun, and W. J. G. de Blok, Astrophys. J. Lett. **533**, L99 (2000)

**PDF File:** `McGaugh2000_BTFR.pdf`

**Verification:**
> "S.S. McGaugh, J.M. Schombert, G.D. Bothun, and W.J.G. de Blok"
> References to Tully-Fisher relation throughout

**Status:** ✅ Correct paper, correct authors

---

### Reference [6] - VERIFIED ✅

**README Citation:** J. D. Bekenstein, Phys. Rev. D **70**, 083509 (2004)

**PDF File:** `Bekenstein2004_TeVeS.pdf`

**Verification:**
> "Relativistic gravitation theory for the MOND paradigm"
> "Jacob D. Bekenstein"
> "Racah Institute of Physics, Hebrew University of Jerusalem"
> "The modified newtonian dynamics (MOND) paradigm of Milgrom..."

**Status:** ✅ Correct paper - this is the TeVeS paper

---

### Reference [8] - VERIFIED ✅

**README Citation:** R. H. Sanders and S. S. McGaugh, Annu. Rev. Astron. Astrophys. **40**, 263 (2002)

**PDF File:** `Sanders2002_MOND_review.pdf`

**Verification:**
> "Robert H. Sanders & Stacy S. McGaugh"
> "Sanders & McGaugh — MODIFIED NEWTONIAN DYNAMICS"

**Status:** ✅ Correct paper, correct authors

---

### Reference [11] - VERIFIED ✅

**README Citation:** S. Bahamonde et al., Rep. Prog. Phys. **86**, 026901 (2023)

**PDF File:** `Bahamonde2023_teleparallel_review.pdf`

**Verification:**
> "Sebastian Bahamonde, Konstantinos F. Dialektopoulos..."
> "February 22, 2023"
> Extensive discussion of teleparallel gravity

**Status:** ✅ Correct paper

---

### Reference [12] - VERIFIED ✅

**README Citation:** E. P. Verlinde, SciPost Phys. **2**, 016 (2017)

**PDF File:** `Verlinde2017_emergent_gravity.pdf`

**Verification:**
> "Erik Verlinde"
> Content about emergent gravity

**Note:** This is the correct 2017 SciPost paper. The Milgrom2010_QUMOND.pdf mistakenly contains Verlinde's earlier 2010 paper.

**Status:** ✅ Correct paper

---

### Reference [13] - VERIFIED ✅

**README Citation:** B. Bertotti, L. Iess, and P. Tortora, Nature **425**, 374 (2003)

**PDF File:** `bertotti-cassini.pdf`

**Verification:**
> "B.Bertotti, L.Iess & P.Tortora"
> Contains Cassini PPN measurement results

**Status:** ✅ Correct paper, correct authors, contains the γ-1 < 2.3×10⁻⁵ bound

---

### Reference [15] - VERIFIED ✅

**README Citation:** F. Lelli, S. S. McGaugh, and J. M. Schombert, Astron. J. **152**, 157 (2016)

**PDF File:** `Lelli2016_SPARC.pdf`

**Verification:**
> "Federico Lelli, Stacy S. McGaugh and James M. Schombert"
> SPARC database discussion throughout

**Status:** ✅ Correct paper, correct authors

---

### Reference [16] - VERIFIED ✅

**README Citation:** A.-C. Eilers, D. W. Hogg, H.-W. Rix, and M. K. Ness, Astrophys. J. **871**, 120 (2019)

**PDF File:** `Eilers2019_MW.pdf`

**Verification:**
> "Anna-Christina Eilers, David W. Hogg, Hans-Walter Rix"
> "Milky Way" and "Gaia" references throughout

**Status:** ✅ Correct paper, correct authors

---

## Summary of Required Actions

| Ref | Current PDF | Action |
|-----|-------------|--------|
| [7] | Wrong (optics paper) | Download https://arxiv.org/abs/0906.0399 |
| [9] | Wrong (Verlinde 2010) | Download https://arxiv.org/abs/0912.0790 |
| [10] | Wrong (spinfoam paper) | Download https://arxiv.org/abs/gr-qc/0702125 |
| [17] | Wrong (Per-Tau Shell) | Download Fox et al. 2022 ApJ 928, 87 |

---

## References Not Verified (No PDF Available)

- **[14]** J. Bekenstein and M. Milgrom, Astrophys. J. **286**, 7 (1984) - AQUAL
- **[18]** MaNGA DynPop Collaboration (2023) - Private communication
- **[19]** KMOS³D Collaboration, Astrophys. J. (2020) - Incomplete citation

---

*Report generated: December 2024*
*Verification method: PDF text extraction using pdfplumber*

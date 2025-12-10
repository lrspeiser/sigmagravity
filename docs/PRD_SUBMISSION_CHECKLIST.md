# Physical Review D Submission Checklist

## Author Information
- [x] **Author**: Leonard Speiser
- [x] **Affiliation**: Horizon 3, Independent Research
- [x] **Email**: leonard@horizon3.net
- [x] **ORCID**: 0009-0008-8797-2457

## Manuscript Files

### Required
- [x] `sigmagravity_revtex.tex` - Main manuscript in REVTeX 4.2 format
- [x] `cover_letter_PRD.tex` - Cover letter for submission
- [ ] Figures (PNG/PDF format, to be uploaded separately):
  - `figures/coherence_window.png` (Fig. 1)
  - `figures/h_function_comparison.png` (Fig. 2)
  - `figures/amplitude_comparison.png` (Fig. 3)
  - `figures/solar_system_safety.png` (Fig. 4)
  - `figures/rar_comparison.png` (Fig. 5)
  - `figures/rotation_curve_gallery.png` (Fig. 6)
  - `figures/mw_rotation_curve.png` (Fig. 7)
  - `figures/cluster_validation.png` (Fig. 8)
  - `figures/counter_rotation_test.png` (Fig. 9)

### Supplemental Material
- [x] `SUPPLEMENTARY_INFORMATION.md` - Extended derivations and validation
- [x] `sigmagravity_supplementary.pdf` - PDF version

## Pre-Submission Checks

### Content
- [x] Abstract is single paragraph, under 500 words
- [x] PACS codes included: 04.50.Kd, 98.80.-k, 95.35.+d, 98.62.Dm
- [x] All equations numbered
- [x] References include titles (PRD encouraged)
- [x] Data Availability Statement included
- [x] Acknowledgments section present

### Formatting
- [x] REVTeX 4.2 document class
- [x] Two-column format (`twocolumn` option)
- [x] `superscriptaddress` for affiliations
- [x] Contact email in footnote format

### Data & Code
- [x] GitHub repository: https://github.com/lrspeiser/SigmaGravity
- [x] All analysis scripts included
- [x] Instructions to reproduce figures
- [x] SPARC data processing code
- [x] Cluster validation code

## Submission Process

### Step 1: Prepare Files
```bash
# Generate PDFs for review
cd /Users/leonardspeiser/Projects/sigmagravity
python3 scripts/make_pdf_latex.py --md README.md --out docs/sigmagravity_paper.pdf
python3 scripts/make_pdf_latex.py --md SUPPLEMENTARY_INFORMATION.md --out docs/sigmagravity_supplementary.pdf
```

### Step 2: Compile REVTeX (optional, for local preview)
```bash
cd docs
pdflatex sigmagravity_revtex.tex
bibtex sigmagravity_revtex
pdflatex sigmagravity_revtex.tex
pdflatex sigmagravity_revtex.tex
```

### Step 3: Submit to APS
1. Go to: https://authors.aps.org/
2. Select "Physical Review D"
3. Upload:
   - `sigmagravity_revtex.tex` (main manuscript)
   - All figure files
   - Cover letter (paste text or upload PDF)
4. Complete author information form
5. Link ORCID: 0009-0008-8797-2457
6. Select article type: "Research Article"
7. Add PACS codes: 04.50.Kd, 98.80.-k, 95.35.+d, 98.62.Dm

### Step 4: After Submission
- Note the 7-digit Accession Code
- Check status at: https://authors.aps.org/Submissions/status/

## Suggested Referees
1. **Stacy McGaugh** (Case Western Reserve University) - SPARC data, MOND expert
2. **Benoit Famaey** (Observatoire astronomique de Strasbourg) - Modified gravity
3. **Kfir Blum** (Weizmann Institute) - Gravitational dynamics

## Key Selling Points for Cover Letter
1. **Unified treatment**: Single formula works for galaxies AND clusters
2. **MOND's cluster problem solved**: We predict cluster masses correctly (MOND fails by 3×)
3. **Falsifiable predictions confirmed**: Counter-rotation test passed (p < 0.01)
4. **Solar System safe**: Cassini bound satisfied by 4 orders of magnitude
5. **Open science**: All code and data publicly available

## Timeline
- [ ] Final review of manuscript
- [ ] Upload to arXiv (optional, recommended)
- [ ] Submit to Physical Review D
- [ ] Respond to referee reports (typically 2-4 weeks after submission)

---

*Last updated: December 2024*


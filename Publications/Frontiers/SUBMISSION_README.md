# Frontiers Resubmission Package

## Primary reviewer-continuity manuscript

- `Reviewer_Continuity/SigmaGravity_Resubmission_REVTeX_2026-07-25.pdf` — compiled 8-page revised manuscript
- `Reviewer_Continuity/SigmaGravity_Resubmission_REVTeX_2026-07-25.tex` — authoritative REVTeX source
- `Reviewer_Continuity/SigmaGravity_Resubmission_REVTeX_2026-07-25.bbl` — compiled numbered bibliography
- `Reviewer_Continuity/SigmaGravity_Resubmission_REVTeX_2026-07-25_Source.zip` — self-contained LaTeX source bundle
- `Reviewer_Continuity/figures/` — the four manuscript figures in vector PDF and 360-dpi PNG formats

This is the recommended manuscript for resubmission. It preserves the original title, two-column REVTeX appearance, US Letter page size, numbered citations, and original major-section sequence. The substantive changes are limited to the reviewer responses and the analyses required to support the revised wording.

## Optional Frontiers-template version

- `SigmaGravity_Resubmission_2026-07-25.pdf` — compiled 14-page manuscript with line numbers
- `SigmaGravity_Resubmission_2026-07-25.tex` — Frontiers Harvard LaTeX source
- `SigmaGravity_Resubmission_2026-07-25.bib` — bibliography
- `SigmaGravity_Resubmission_2026-07-25.md` — editable Markdown source
- `FrontiersinHarvard.cls`
- `Frontiers-Harvard.bst`
- `logo1.pdf`

## Supplementary Material

- `SigmaGravity_Supplementary_2026-07-25.pdf`
- `SigmaGravity_Supplementary_2026-07-25.tex`
- `SigmaGravity_Supplementary_2026-07-25.md`
- `frontiers_suppmat.cls`

## Reviewer responses

- `Response_to_Reviewer_1_2026-07-25.md`
- `Response_to_Reviewer_2_2026-07-25.md`

The response files are formatted for copying into the interactive review forum. Their page, section, equation, table, and figure references correspond to `Reviewer_Continuity/SigmaGravity_Resubmission_REVTeX_2026-07-25.pdf`.

## Figures

The `figures` folder contains four publication figures in vector PDF and 360-dpi PNG formats. Upload the vector PDFs as the primary figure files unless the submission portal requests raster files.

## Rebuild

From the repository root:

```powershell
python "Frontier Feedback/Resubmission/scripts/generate_revision_figures.py"
python "Frontier Feedback/Resubmission/scripts/build_frontiers_sources.py"
```

To compile the recommended reviewer-continuity manuscript, run Tectonic or another REVTeX-compatible LaTeX engine from the `Reviewer_Continuity` folder. The optional Frontiers-template source is generated separately by the commands above.

## Verification completed

- Reviewer-continuity manuscript, optional Frontiers-template manuscript, and Supplement compile successfully.
- All pages were rendered and visually inspected.
- The manuscript-facing test subset passed: 19 tests.
- No new cluster amplitude or replacement response formula was introduced.

## Author check before upload

Confirm the correspondence email and affiliation in the first-page proof. The package currently uses:

- Leonard Speiser
- Horizon 3, Independent Research, Los Altos, CA, United States
- `leonard@horizon3.net`
- ORCID `0009-0008-8797-2457`

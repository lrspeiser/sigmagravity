# Frontiers Resubmission Package

## Primary reviewer-continuity manuscript

- `Reviewer_Continuity/SigmaGravity_Resubmission_REVTeX_2026-07-25.pdf` — compiled revised manuscript in the original two-column format with continuous line numbers
- `Reviewer_Continuity/SigmaGravity_Resubmission_REVTeX_2026-07-25.tex` — authoritative line-numbered REVTeX source
- `Reviewer_Continuity/SigmaGravity_Resubmission_REVTeX_2026-07-25.bbl` — compiled numbered bibliography
- `Reviewer_Continuity/SigmaGravity_Resubmission_REVTeX_2026-07-25_Source.zip` — self-contained LaTeX source bundle
- `Reviewer_Continuity/figures/` — the four manuscript figures in vector PDF and 360-dpi PNG formats

This is the recommended manuscript for resubmission. It is a 10-page, 4,115-word proof that preserves the original two-column REVTeX appearance, US Letter page size, numbered citations, and major-section sequence, while adding continuous line numbers for peer review. The revision incorporates the analyses required by peer review while presenting the scientific results as a standalone paper.

## Optional Frontiers-template version

- `SigmaGravity_Resubmission_2026-07-25.pdf` — compiled 17-page manuscript with line numbers
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
python "Publications/Frontiers/scripts/generate_revision_figures.py"
python "Publications/Frontiers/scripts/run_sparc_scale_length_sensitivity.py"
python "Publications/Frontiers/scripts/build_frontiers_sources.py"
```

To compile the recommended reviewer-continuity manuscript, run Tectonic or another REVTeX-compatible LaTeX engine from the `Reviewer_Continuity` folder. The optional Frontiers-template source is generated separately by the commands above.

## Verification completed

- Reviewer-continuity manuscript, optional Frontiers-template manuscript, and Supplement compile successfully.
- All pages were rendered and visually inspected.
- The manuscript-facing test subset passed: 27 tests.
- The actual SPARC photometric scale lengths were evaluated in a no-refit candidate window with permutation and fixed-median controls. The candidate was not promoted to the canonical response.
- No new cluster amplitude or replacement cluster formula was introduced.
- The final feedback audit clarified the provenance and non-uniqueness of the empirical functions, defined “locked” as retrospective non-reoptimization, limited the auxiliary action to prescribed spatially constant `B`, made the Fox result visibly illustrative, quantified its disjoint-sample radial bias, and documented deduplicated no-replacement counterrotation matching.

The scale-length script, tests, and machine-readable outputs are in `scripts/` and `analysis/sparc_scale_length/` within this folder. The QUMOND, cluster, and original paired-SPARC audit packages used by the manuscript are preserved at `research/reviewer_derivation_audit/` and `research/sparc_statistical_validation/` in the repository root.

## Author check before upload

Confirm the correspondence email and affiliation in the first-page proof. The package currently uses:

- Leonard Speiser
- Horizon 3, Independent Research, Los Altos, CA, United States
- `leonard@horizon3.net`
- ORCID `0009-0008-8797-2457`

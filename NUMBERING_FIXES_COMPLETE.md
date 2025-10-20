# Section and Figure Numbering Audit — COMPLETE

**Date:** 2025-10-20  
**Commit:** 2696b09

---

## ✅ SECTION 2 FIXED

### Problems Found:
- **Duplicate 2.3** (lines 88 and 102) — two separate headings with same number
- **Out of order**: 2.9, 2.7, 2.5, 2.6, 2.8 appearing after 2.4

### Actions Taken:
1. **Merged duplicate §2.3** sections into one cohesive section (lines 88-114)
2. **Renumbered sequentially**:
   - 2.9 → **2.5** (Illustrative example)
   - 2.7 → **2.6** (What is derived vs calibrated)
   - 2.5 → **2.7** (Galaxy-scale kernel)
   - 2.6 → **2.8** (Cluster-scale kernel)
   - 2.8 → **2.9** (Safety)

### Current §2 Structure (✓ CORRECT):
```
2.1. Plain-language primer
2.2. Stationary-phase reduction
2.3. Coherence window and constants (merged)
2.4. Canonical kernel
2.5. Illustrative example
2.6. What is derived vs calibrated
2.7. Galaxy-scale kernel
2.8. Cluster-scale kernel
2.9. Safety
```

---

## ✅ FIGURES RENUMBERED

### Problems Found:
- **G-series**: Started at G2 (missing G1)
- **C-series**: Jumped from C2 to C4 (missing C3)

### Actions Taken:
1. G2 → **G1** (Rotation-curve gallery)
2. G3 → **G2** (RC residual histogram)
3. C4 → **C3** (Convergence panels)
4. C5 → **C4** (Deflection panels)

### Current Figure Scheme (✓ SEQUENTIAL):
```
G1, G2          (Galaxy figures)
H1, H2, H3      (Hierarchical/holdout cluster figures)
C1, C2, C3, C4  (Cluster figures)
MW-1 to MW-6    (Milky Way figures)
```

---

## ✅ CROSS-REFERENCES UPDATED

**Line 212**: Changed "§§2.5–2.6" → "§§2.7–2.8"  
(Reference to galaxy and cluster kernel sections)

---

## ✅ MAIN SECTIONS 3-13

**All verified correct and sequential**:
- §3 Data
- §4 Methods (4.1–4.4)
- §5 Results (5.1–5.4)
- §6 Discussion
- §7 Predictions
- §8 Cosmological Implications (8.1–8.5)
- §9 Reproducibility (9.0–9.7)
- §10 What changed
- §11 Planned analyses (11.1)
- §12a Figures
- §13 Conclusion

---

## ✅ TABLES

**All verified correct**:
- Table G1 (Galaxy RAR/BTFR metrics)
- Table C1 (Training clusters)
- Table C2 (Population posteriors)
- Table F1, F2, F3 (Appendix F sensitivity tables)

---

## Files Changed:
- `README.md`: Section and figure numbering corrected
- `sigmagravity_paper.pdf`: Regenerated (2.7 MB)
- `NUMBERING_AUDIT.md`: Initial audit report
- `NUMBERING_FIXES_COMPLETE.md`: This completion summary

---

## Git Status:
```
commit 2696b09
Date:   Sun Oct 20 09:51:21 2025

    Fix section and figure numbering: merge duplicate §2.3, 
    renumber §2.5-2.9 sequentially, close figure gaps 
    (G2→G1, G3→G2, C4→C3, C5→C4). All sections and figures 
    now sequential. Updated cross-reference in §4 to 
    corrected §§2.7-2.8. Regenerated PDF.
```

Pushed to: **GitHub main branch** ✓

---

## Verification Commands:

```powershell
# Check Section 2 numbering
rg -n "^### 2\.[0-9]" README.md

# Check figure numbering
rg -n "Figure [GHC][0-9]" README.md | Select-Object -First 20

# Check all section headings
rg -n "^#{1,3} [0-9]" README.md
```

---

## Status: ✅ COMPLETE

All section numbers are now **sequential and accurate**.  
All figure numbers are now **sequential with no gaps**.  
Cross-references have been **updated to match new numbering**.  
PDF has been **regenerated** with corrected structure.  
Changes **committed and pushed** to GitHub.

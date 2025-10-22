# README.md Section and Figure Numbering Audit

## SECTION NUMBERING ISSUES

### ❌ Section 2 subsections OUT OF ORDER:
- Line 62: **2.1** ✓
- Line 66: **2.2** ✓
- Line 88: **2.3** ✓ (first occurrence)
- Line 102: **2.3** ❌ **DUPLICATE** — should be renumbered or merged
- Line 118: **2.4** ✓
- Line 128: **2.9** ❌ **SKIPS 2.5-2.8** — should be **2.5**
- Line 139: **2.7** ❌ **OUT OF ORDER** — should be **2.6**
- Line 149: **2.5** ❌ **OUT OF ORDER** — should be **2.7**
- Line 165: **2.6** ❌ **OUT OF ORDER** — should be **2.8**
- Line 178: **2.8** ❌ **OUT OF ORDER** — should be **2.9**

### Correct Section 2 order should be:
1. **2.1** Plain-language primer ✓
2. **2.2** Stationary-phase reduction ✓
3. **2.3** Coherence window (MERGE two 2.3 sections) 
4. **2.4** Canonical kernel ✓
5. **2.5** Illustrative example (currently 2.9)
6. **2.6** What is derived vs calibrated (currently 2.7)
7. **2.7** Galaxy-scale kernel (currently 2.5)
8. **2.8** Cluster-scale kernel (currently 2.6)
9. **2.9** Safety (currently 2.8)

### ✓ Main sections 3-13 are CORRECT:
- 3. Data ✓
- 4. Methods & Validation ✓
  - 4.1-4.4 ✓
- 5. Results ✓
  - 5.1-5.4 ✓
- 6. Discussion ✓
- 7. Predictions & falsifiability ✓
- 8. Cosmological Implications ✓
  - 8.1-8.5 ✓
- 9. Reproducibility ✓
  - 9.0-9.7 ✓
- 10. What changed ✓
- 11. Planned analyses ✓
  - 11.1 ✓
- 12a. Figures ✓
- 13. Conclusion ✓

---

## FIGURE NUMBERING ISSUES

### Current figure scheme (INCONSISTENT):
- **G-series**: Galaxy figures (G2, G3) — ❌ Missing G1
- **H-series**: Hierarchical/holdout cluster figures (H1, H2, H3) ✓
- **C-series**: Cluster figures (C1, C2, C4, C5) — ❌ Missing C3
- **MW-series**: Milky Way figures (MW-1 to MW-6) ✓
- **F-series**: Appendix F tables (F1, F2, F3) ✓

### ❌ Problems:
1. **Missing G1** (galaxy figures start at G2)
2. **C3 missing** (jumps C2 → C4)
3. **Figure references in §5** don't match numbering style in §12a

### Sequential numbering recommendation:
Either:
- **Option A**: Use letter prefixes consistently (current style)
  - Fix: Add G1, insert C3, or renumber to close gaps
  
- **Option B**: Use pure sequential (1, 2, 3...)
  - More standard for academic papers
  - Easier to reference and maintain

---

## TABLE NUMBERING

### ✓ Tables are CORRECT:
- **G1**: Galaxy RAR/BTFR metrics ✓
- **C1**: Training clusters ✓
- **C2**: Population posteriors ✓
- **F1, F2, F3**: Appendix F sensitivity tables ✓

---

## RECOMMENDED FIXES

### Priority 1: Section 2 (CRITICAL)
```
Line 88:  ### 2.3. Coherence window and constants of the model
Line 102: ### 2.3. Coherence window and constants of the model  ← DELETE or MERGE into line 88

Line 128: ### 2.9. → ### 2.5.  Illustrative example
Line 139: ### 2.7. → ### 2.6.  What is derived vs calibrated
Line 149: ### 2.5. → ### 2.7.  Galaxy‑scale kernel
Line 165: ### 2.6. → ### 2.8.  Cluster‑scale kernel
Line 178: ### 2.8. → ### 2.9.  Safety
```

### Priority 2: Figures (MEDIUM)
Either:
- **Fix A**: Renumber to close gaps
  - G2→G1, G3→G2
  - C4→C3, C5→C4
  
- **Fix B**: Insert missing figures
  - Add Figure G1 (RAR main plot?)
  - Add Figure C3 (what should this be?)

### Priority 3: Cross-references (LOW)
Update any text references to §2.5, §2.6, §2.7, §2.8, §2.9 after renumbering.

---

## AUTOMATED FIX SCRIPT

Run after manual review to apply all fixes.

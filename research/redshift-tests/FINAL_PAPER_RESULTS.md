
# FINAL PAPER-READY RESULTS TABLE

## Model Comparison Results

| Model | Parameters | Chi2 | AIC | BIC | Delta AIC |
|-------|------------|------|-----|-----|-----------|
| TG-tau | H_Sigma = 72.00, alpha_SB = 1.200 | 871.83 | 875.83 | 886.70 | 59.21 |
| FRW | Om = 0.380, intercept = -0.0731 | 812.61 | 816.61 | 827.49 | 0.00 |

## Key Findings

1. **Fair Model Comparison**: FRW statistically preferred with Delta AIC = 59.21
2. **TG-tau Physical Consistency**: H_Sigma = 72.00, alpha_SB = 1.200
3. **Distance-Duality Prediction**: eta(z) = (1+z)^0.2
4. **Zero-Point Stability**: alpha_SB unchanged across anchoring methods
5. **Anisotropy**: North-South difference ~0.056 mag (not significant)

## Distance-Duality Testable Prediction

TG-tau predicts: eta(z) = (1+z)^0.2

- eta at z=1: 1.1487
- eta at z=2: 1.2457

This provides a clear, testable signature for future validation with BAO/cluster angular diameter distances.

## Reproducibility

All results reproducible with:
- `phase2_hardening.py`: Complete Phase-2 validation suite
- `phase2_key_fixes.py`: Key fixes implementation
- `complete_validation_suite.py`: All validation checks
- `final_referee_proof.py`: Final referee-proof validation

Entry points: `run_phase2_validation()`, `generate_parity_table()`, `run_final_validation()`

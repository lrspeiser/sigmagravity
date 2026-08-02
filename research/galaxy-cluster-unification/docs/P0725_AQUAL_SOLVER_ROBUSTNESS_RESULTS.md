# Universal AQUAL solver-robustness results (P0725)

Date: 2026-08-02

## Outcome

P0725 identified one universal numerical control that resolves both fine-grid
AQUAL failures from P0724: a formula-neutral unit-coefficient linearized warm
start followed by Picard iteration with damping `0.20`. DDO53 converged in 164
iterations and DDO101 in 163, both below the frozen 200-iteration limit and
with maximum normalized equation residual below `1e-8`.

The stage nevertheless has status **fail**, because the preregistered gate
requires at least two independently successful universal variants whose final
predictions agree. Only one of the six real-galaxy variants converged both
systems. P0725 therefore names `linearized_d020` as a credible candidate but
does not select a production default.

## Frozen protocol

The configuration was committed before the real-system matrix was opened in
`configs/p0725_aqual_solver_robustness.json`. Its SHA-256 is
`61373a8fc7e4b9eb47ec16703eeed5ec59af7f6f046b451d3ee117b75c3ae90a`.

- Systems: the exact P0724 `49 x 49 x 17` DDO53 and DDO101 volume bundles and
  observation targets.
- Physics: the AQUAL simple-mu equation, `G`, `a0`, coefficient floor, density,
  grid, spacing, boundary, and requested observable remained identical.
- Numerical matrix: zero or unit-coefficient linearized initialization crossed
  with fixed damping `0.70`, `0.35`, or `0.20`.
- Tolerances: relative update and normalized equation residual at most `1e-8`.
- Iterations: requested and executed maximum both exactly 200.
- Parameter policy: published fixed, zero per-object gravity parameters.
- Selection: convergence and cross-variant solution agreement only;
  observational fit could not select a numerical method.

All six variants also ran a discrete manufactured nonlinear field with known
truth. All converged, with relative field errors from `1.65e-9` to `1.08e-8`.
This confirms that each numerical control is executable; it does not imply
that each is robust for the harder degenerate AQUAL coefficient.

## Real-system results

| Initialization | Damping | DDO53 | DDO53 residual | DDO101 | DDO101 residual |
|---|---:|---|---:|---|---:|
| zero | 0.70 | failed at 200 | `3.925e-2` | failed at 200 | `1.549e-2` |
| zero | 0.35 | failed at 200 | `1.662e-2` | failed at 200 | `1.370e-2` |
| zero | 0.20 | failed at 200 | `4.896e-8` | failed at 200 | `9.611e-7` |
| linearized | 0.70 | failed at 200 | `4.703e-2` | failed at 200 | `1.715e-1` |
| linearized | 0.35 | failed at 200 | `1.563e-2` | failed at 200 | `1.510e-2` |
| linearized | 0.20 | **passed at 164** | `9.217e-9` | **passed at 163** | `9.486e-9` |

The successful DDO53 and DDO101 runs took 985.7 and 997.5 wall seconds in the
six-process matrix. Their maximum relative updates were `3.50e-10` and
`1.14e-9`, respectively. They produced scored circular-speed artifacts only
after both numerical tolerances passed.

## What the trajectories show

The original `0.70` setting and both `0.35` settings settle into oscillatory
behavior rather than a slow monotonic tail. Lowering damping to `0.20` changes
the behavior qualitatively: after a long transient, residuals decrease
monotonically. With zero initialization, the two final residuals are close but
still outside tolerance after 200 iterations. The linearized warm start removes
enough of that transient to reach the same strict root criterion in 163--164
iterations.

This supports a numerical explanation for the two P0724 failures. It does not
support changing AQUAL physics or fitting galaxy-specific controls. One
initialization and damping pair works for both inputs with the unchanged
equation.

## Disclosed iteration policy

The investigation also found that the earlier example manifest requested
20,000 iterations while the preview worker silently executed at most 200.
The worker now records requested and effective iteration limits and whether a
limit was adjusted; the manifest validator warns about the preview ceiling.
P0725 requests exactly 200 and verifies that exactly 200 is available, so no
hidden adjustment occurs.

This is an interim preview resource policy. A production API still needs a
separate, explicit resource-class contract rather than placing operational
ceilings inside a scientific model manifest.

## Required next stage

P0726 must independently solve the same two fields with a different generic
nonlinear algorithm, such as Newton--Krylov, Anderson-accelerated fixed point,
or a preregistered coefficient-continuation method. The AQUAL equation and all
inputs must remain locked. Acceptance requires:

1. both galaxies converge under one universal setting;
2. the independent solution and `linearized_d020` predictions agree within the
   frozen 1% normalized RMS gate;
3. both methods meet the same update and equation-residual tolerances; and
4. the manufactured nonlinear known-answer test remains accurate.

If no independent method agrees, `linearized_d020` remains a diagnostic
candidate only. If agreement passes, use the selected method to complete all
four P0724 fine-grid AQUAL sentinels before freezing a production grid.

## Reproduce

```powershell
python scripts/run_p0725_aqual_solver_robustness.py
```

The command exits nonzero because the cross-method selection gate failed. The
report, all real-system residual histories, known-answer results, and plot are
under `results/p0725_aqual_solver_robustness/`.

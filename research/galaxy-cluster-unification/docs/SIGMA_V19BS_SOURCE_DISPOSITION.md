# Sigma V19BS source disposition

V19BS freezes what happens after the observed I4/I5 source test before that
result exists. It prevents an unfavorable source result from being repaired by
opening lensing or inventing a convenient action.

If V19BQ fails, every action-placement class is excluded for this source route.
The next admissible evidence is a direct gas-velocity or independently clocked
merger measurement, or a preregistered independent merger sample evaluated
with the unchanged source definitions. Threshold relaxation, dropping a
cluster or robustness branch, combining failed features after inspection and
using halo/lensing targets to select a source are forbidden.

If V19BQ passes, it authorizes mathematical derivation—not observational
fitting—for the two time-even placements compatible with the snapshot source:

- P1, a constrained composite response with no free halo-shaped mode;
- P3, a degenerate or second-order pure-metric nonlinear vertex.

P2 causal dynamic memory is not authorized by I4/I5 alone because neither is a
directly clocked lag or time-odd current. P2 would require new temporal
evidence. P1 versus P3 must be decided by constraint closure, degrees of
freedom, conservation, Hamiltonian boundedness, hyperbolicity, stability and a
one-metric weak-field derivation. Galaxy or lensing performance cannot choose
the action.

I4 direction remains mandatory. I5 may rescue strength but never direction.
The disposition therefore preserves both scientifically valid terminal
outcomes:

```text
I4 direction AND (I4 amplitude OR I5 scalar)
    -> mathematical action comparison authorized

otherwise
    -> source route falsified; no action authorized
```

The current preflight opens no V19BQ result, lensing, halo, galaxy rotation,
action, gravity parameter, Solar-System result or holdout.

## Reproduction

```powershell
python scripts/check_sigma_v19bs_source_disposition_preflight.py
python -m pytest tests/test_sigma_v19bs_source_disposition.py -q
```

The frozen evidence is
`results/sigma_v19bs_source_disposition/preflight_report.json`.

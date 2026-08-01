# Transition-radius isolation check

The first galaxy-scaling run showed a strong held-out improvement, but its mass
amplitude exponent was near 0.54 while its surface-density transition exponent
was near zero. This locked follow-up distinguishes a mass-amplitude relation from
a genuinely galaxy-dependent transition radius.

Every model receives the same mass-scaled outer contribution:

```
v^2 = vbar^2 + V0^2 M^eta r^2/(r^2+rt^2).
```

Only the transition definition changes:

```
mass-amplitude-only control: rt = cR Rd
mass transition:             rt = cR Rd M^alpha
surface-density transition:  rt = cR Rd Sigma^beta
concentration transition:    rt = cR Rd C^gamma
```

The five whole-galaxy folds, fixed baryonic inputs, training-fold
normalizations, parameter bounds, and 100,000 paired bootstrap draws remain
frozen. A transition driver must reduce held-out RMSE by at least 5%, have at
least 0.95 bootstrap probability of improving chi squared, and keep a nonzero
same-sign exponent away from its bounds in every fold.

This is explicitly a post-primary isolation check. It does not revise the
already completed void result and introduces no new void parameter.

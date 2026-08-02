# P0673 compound-path activation preregistration

P0672 showed that the P0669 tensor term is geometrically active but only
perturbative. P0673 changes the mechanism, not its fitted strength. Let

\[
\epsilon=A_{\rm multipole}\,{a_0\over a_0+|g_N|}\,C_\perp,
\qquad N=\left({\ell\over L_c}\right)^2.
\]

Instead of multiplying `epsilon` by one saturating survival factor, P0673
treats a long coherent path as `N` repeated opportunities. The fraction that
survives unrouted is

\[
1-\sigma_{\rm compound}=(1-\epsilon)^N.
\]

This has useful exact limits: a co-centered radial system has zero multipole
amplitude and therefore zero response; a zero-length path has zero response;
high acceleration suppresses each elementary opportunity; and long cluster
paths can build a nonperturbative response without a fitted cluster amplitude.
The already frozen `a0`, `Lc=10 kpc`, and power `q=2` are unchanged.

Before scoring, P0673 requires the typical registered galaxy coefficient to
remain below `0.001`, the typical cluster coefficient to exceed `0.05`, at
least `50x` nominal domain separation and `20x` under every mass sensitivity,
and an RX J2129 coefficient above `0.05`. It must remain positive-definite,
add no constant or per-object setting, compute no new raw-lens score, and keep
P0633/P0640 sealed.

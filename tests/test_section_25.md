# Test: Section 2.5 — Galaxy-scale kernel

### 2.5. Galaxy‑scale kernel (RAR; rotation curves)

For circular motion in an axisymmetric disk,

$$
g_{\rm model}(R) = g_{\rm bar}(R)[1 + K(R)],
$$

with

$$
K(R) = A_0\, (g^{\dagger}/g_{\rm bar}(R))^p\; C(R;\,\ell_0, p, n_{\rm coh})\; G_{\rm bulge}\; G_{\rm shear}\; G_{\rm bar}.
$$

Here $g^{\dagger}$ is an acceleration scale; $(A_0,p)$ govern the path‑spectrum slope; $(\ell_0,n_{\rm coh})$ set coherence length and damping; the gates $(G_{\cdot})$ suppress coherence for bulges, shear and stellar bars. The kernel multiplies Newton by $(1+K)$, preserving the Newtonian limit $(K \to 0$ as $R \to 0)$.

Best‑fit hyperparameters from the SPARC analysis (166 galaxies, 80/20 split; validation suite pass): $\ell_0=4.993$ kpc, $\beta_{\rm bulge}=1.759$, $\alpha_{\rm shear}=0.149$, $\gamma_{\rm bar}=1.932$, $A_0=0.591$, $p=0.757$, $n_{\rm coh}=0.5$.

Result: hold‑out RAR scatter = 0.087 dex, bias $-0.078$ dex (after Newtonian‑limit bug fix and unit hygiene). Cassini‑class bounds are satisfied with margin $\geq 10^{13}$ by construction (hard saturation gates).

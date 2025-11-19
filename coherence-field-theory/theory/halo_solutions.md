# Halo Solutions from Scalar Field Theory

**Date**: November 19, 2025  
**Goal**: Derive galaxy halo profiles from scalar field equations (make halos predictive, not phenomenological)

---

## 1. Theoretical Foundation

### 1.1 Action and Field Equations

Start from the scalar-tensor action:

\[
S = \int d^4x \sqrt{-g}\left[\frac{M_{\rm Pl}^2}{2}R -\frac12(\nabla\phi)^2 - V(\phi) \right] + S_m[ A^2(\phi) g_{\mu\nu}, \psi_m]
\]

where:
- \(M_{\rm Pl} = (8\pi G)^{-1/2}\) is the reduced Planck mass
- \(V(\phi)\) is the scalar potential (same as cosmology)
- \(A(\phi)\) encodes the coupling to matter
- \(S_m\) is the matter action

### 1.2 Weak-Field, Spherically Symmetric Limit

For a static galaxy, assume:

1. **Metric** (Newtonian limit):
   \[
   ds^2 = -(1+2\Phi)dt^2 + (1-2\Phi) d\vec x^2
   \]
   where \(\Phi \ll 1\) is the Newtonian potential.

2. **Scalar field**:
   \[
   \phi = \phi(r)
   \]
   (static, spherically symmetric)

3. **Matter density**:
   \[
   \rho_b(r) = \rho_{\rm disk}(r) + \rho_{\rm bulge}(r) + \rho_{\rm gas}(r)
   \]
   (baryonic components)

### 1.3 Static Klein-Gordon Equation

In the weak-field limit, the scalar field equation becomes:

\[
\frac{1}{r^2}\frac{d}{dr}\left( r^2 \frac{d\phi}{dr}\right) = \frac{dV_{\rm eff}}{d\phi}
\]

where the **effective potential** is:

\[
V_{\rm eff}(\phi) = V(\phi) + \frac{d\ln A(\phi)}{d\phi} \rho_b(r)
\]

For exponential coupling:

\[
A(\phi) = e^{\beta \phi/M_{\rm Pl}}
\]

we get:

\[
\frac{d\ln A}{d\phi} = \frac{\beta}{M_{\rm Pl}}
\]

so:

\[
V_{\rm eff}(\phi) = V(\phi) + \frac{\beta}{M_{\rm Pl}} \rho_b(r)
\]

### 1.4 Scalar Potential

Use the same potential as cosmology (for consistency):

**Base form**:
\[
V(\phi) = V_0 e^{-\lambda \phi}
\]

**With chameleon screening** (needed for solar system):
\[
V(\phi) = V_0 e^{-\lambda \phi} + \frac{M^{4+n}}{\phi^n}
\]

For now, start with \(n=1\):
\[
V(\phi) = V_0 e^{-\lambda \phi} + \frac{M^5}{\phi}
\]

### 1.5 Effective Density

The scalar field contributes to gravity via its stress-energy tensor. In the weak-field limit, the **effective density** is:

\[
\rho_\phi(r) = \frac12\left(\frac{d\phi}{dr}\right)^2 + V(\phi)
\]

This is what should reproduce the pseudo-isothermal halo profile:
\[
\rho_{\rm halo}(r) = \frac{\rho_{c0}}{1 + (r/R_c)^2}
\]

### 1.6 Boundary Conditions

**At \(r = 0\)**:
- Regularity: \(\frac{d\phi}{dr}\big|_{r=0} = 0\)
- Central value: \(\phi(0) = \phi_0\) (free parameter)

**At large \(r \gg R_{\rm disk}\)**:
- Cosmological value: \(\phi(r) \to \phi_\infty\) (from cosmology)
- Vanishing gradient: \(\frac{d\phi}{dr}\big|_{r \to \infty} \to 0\)

**Practical integration**: Start from outside (\(r = r_{\rm max}\)) and integrate inward, adjusting \(\phi(r_{\rm max})\) until solution is regular at \(r = 0\).

---

## 2. Dimensionless Formulation

### 2.1 Rescaling

To make numerical integration stable:

1. **Radial coordinate**:
   \[
   x = \frac{r}{R_c}
   \]
   where \(R_c \sim R_{\rm disk}\) is a characteristic scale.

2. **Scalar field**:
   \[
   \psi = \frac{\phi - \phi_\infty}{\phi_*}
   \]
   where \(\phi_*\) is chosen so \(V(\phi)\) and kinetic terms are comparable.

3. **Potential**:
   \[
   U(\psi) = \frac{V(\phi) - V(\phi_\infty)}{V_*}
   \]
   where \(V_*\) is a characteristic energy scale.

### 2.2 Dimensionless Equation

After rescaling, the KG equation becomes:

\[
\frac{1}{x^2}\frac{d}{dx}\left( x^2 \frac{d\psi}{dx}\right) = \frac{\partial U_{\rm eff}}{\partial \psi}
\]

with:

\[
U_{\rm eff}(\psi) = U(\psi) + \alpha \tilde{\rho}_b(x)
\]

where:
- \(\alpha = \frac{\beta \rho_0}{M_{\rm Pl} V_*}\) (coupling strength × density / potential scale)
- \(\tilde{\rho}_b(x) = \rho_b(r)/\rho_0\) (dimensionless baryon density)
- \(\rho_0\) is a characteristic density (e.g., \(\rho_0 \sim M_{\rm disk}/R_{\rm disk}^3\))

### 2.3 Dimensionless Parameters

The solution depends on a small set of dimensionless combinations:

1. **Potential parameters**:
   - \(\lambda\) (from exponential)
   - \(M^5 / V_0\) (chameleon strength)

2. **Coupling**:
   - \(\beta\) (matter coupling)

3. **Galaxy properties**:
   - \(M_{\rm disk}/M_*\) (disk mass in units of \(M_* = \rho_0 R_c^3\))
   - \(R_{\rm disk}/R_c\) (disk scale)

4. **Boundary**:
   - \(\phi_\infty / \phi_*\) (cosmological field value)

---

## 3. Solution Strategy

### 3.1 Shooting Method

**Outward-inward integration**:

1. **Start at large radius** (\(x_{\rm max} \sim 10 \times R_{\rm disk}/R_c\)):
   - Set \(\psi(x_{\rm max}) = 0\) (cosmological value)
   - Set \(\frac{d\psi}{dx}\big|_{x_{\rm max}} = 0\) (vanishing gradient)

2. **Integrate inward** using RK4 or `scipy.integrate.odeint`:
   - Solve: \(\frac{d^2\psi}{dx^2} + \frac{2}{x}\frac{d\psi}{dx} = \frac{\partial U_{\rm eff}}{\partial \psi}\)

3. **Check regularity at \(x = 0\)**:
   - Adjust \(\phi(r_{\rm max})\) if needed until \(\frac{d\psi}{dx}\big|_{x=0} \approx 0\)

### 3.2 Alternative: Boundary Value Problem

Use `scipy.optimize.solve_bvp`:

- **Boundary conditions**:
  - At \(x = 0\): \(\frac{d\psi}{dx} = 0\)
  - At \(x = x_{\rm max}\): \(\psi = 0\), \(\frac{d\psi}{dx} = 0\)

- **Initial guess**: Linear interpolation between boundaries

---

## 4. Mapping to Halo Parameters

### 4.1 Extract Effective Density

For each solution \(\phi(r)\), compute:

\[
\rho_\phi(r) = \frac12\left(\frac{d\phi}{dr}\right)^2 + V(\phi)
\]

### 4.2 Fit Pseudo-Isothermal Profile

Fit \(\rho_\phi(r)\) over the range \(0.5 R_{\rm disk} < r < 5 R_{\rm disk}\) with:

\[
\rho_{\rm fit}(r) = \frac{\rho_{c0}}{1 + (r/R_c)^2}
\]

This gives predicted \((\rho_{c0}, R_c)\) from field parameters.

### 4.3 Compare to SPARC Fits

Compare predicted \((\rho_{c0}, R_c)\) to fitted values from `sparc_fit_summary.csv`:

- If a single set of \((V_0, \lambda, \beta, M, \ldots)\) reproduces multiple galaxies' halo parameters, the theory is **predictive**.

---

## 5. Connection to Cosmology

### 5.1 Cosmological Field Value

The boundary condition \(\phi_\infty\) comes from cosmology:

- Solve background evolution with `CoherenceCosmology`
- Extract \(\phi_0\) (today's value): \(\phi_\infty = \phi_0\)

### 5.2 Consistency Check

The same \((V_0, \lambda)\) used in:
- **Cosmology**: Reproduces \(H(z)\), \(d_L(z)\), \(\Omega_m\), \(\Omega_\phi\)
- **Galaxies**: Predicts \((\rho_{c0}, R_c)\) from field equations
- **Solar System**: (with screening) Passes PPN tests

This is the **unified field theory** goal.

---

## 6. Implementation Outline

### 6.1 HaloFieldSolver Class

```python
class HaloFieldSolver:
    def __init__(self, V0, lambda_param, beta, M4=None):
        """Initialize with field parameters."""
        
    def solve(self, rho_baryon, r_grid, phi_inf=None):
        """Solve φ(r) for given baryon profile.
        
        Returns:
            phi(r): scalar field profile
            dphi_dr(r): gradient
        """
        
    def effective_density(self, phi, dphi_dr):
        """Compute ρ_φ(r) = ½(∇φ)² + V(φ)."""
        
    def fit_halo_parameters(self, rho_phi, r_grid):
        """Fit pseudo-isothermal to ρ_φ(r).
        
        Returns:
            rho_c0, R_c: effective halo parameters
        """
```

### 6.2 Integration with Rotation Curves

1. **Set field parameters** (global): \(V_0, \lambda, \beta, M^4\)

2. **For each galaxy**:
   - Get baryon profile: \(\rho_b(r)\) from \((M_{\rm disk}, R_{\rm disk})\)
   - Solve \(\phi(r)\) → \(\rho_\phi(r)\)
   - Compute rotation curve: \(v_{\rm tot} = \sqrt{v_{\rm baryon}^2 + v_\phi^2}\)

3. **Fit only baryonic parameters**: \((M_{\rm disk}, R_{\rm disk})\) per galaxy

4. **Compare predicted vs fitted halos**: Check if \(\rho_{c0}, R_c\) match SPARC fits

---

## 7. References

- Chameleon screening: Khoury & Weltman (2004)
- Symmetron screening: Hinterbichler & Khoury (2010)
- Scalar-tensor gravity: Will (2014) "Theory and Experiment in Gravitational Physics"
- SPARC data: Lelli et al. (2016)

---

**Status**: Theoretical foundation complete  
**Next**: Implement `galaxies/halo_field_profile.py`


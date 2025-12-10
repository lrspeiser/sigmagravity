
import numpy as np
from sigma_cosmo import Cosmology, coherence_kernel, AofA, K_of_R_a, Rofk
from sigma_cosmo import mu_of_k_a, growth_D_of_a, endpoint_grz, los_isw_redshift

cosmo = Cosmology(H0_kms_Mpc=70.0, Omega_b0=0.048, Omega_r0=8.6e-5, Omega_L0=0.70)

ell0 = 200e3 * 3.085677581e16
p, ncoh = 0.75, 0.5
A0 = 4.6
a_t = 0.01
s = 2.0

k_vals = np.logspace(-5, -1, 5) / (3.085677581e22)
for k in k_vals:
    mu = mu_of_k_a(k, 1.0, ell0, p, ncoh, A0=A0, a_t=a_t, s=s, A_form="logistic", two_pi=True)
    print(f"k={k:.3e}  mu(k, a=1)={mu:.4f}")

a_grid, D, f = growth_D_of_a(cosmo, k=k_vals[2], ell0=ell0, p=p, ncoh=ncoh, A0=A0, a_t=a_t, s=s)
print(f"Growth D(a) at a=1: {D[-1]:.3f}, f(a=1)={f[-1]:.3f}")

a_path = np.linspace(0.1, 1.0, 1001)
A_callable = lambda a: AofA(a, A0=A0, a_t=a_t, s=s, form='logistic')
c = 299792458.0
phi_bar = lambda a: 5e-5 * c**2
z_los = los_isw_redshift(a_path, A_callable, phi_bar, two_potentials_equal=True)
print(f"Toy LOS ISW-like z = {z_los:.3e}  ->  c*z ≈ {z_los*3e5:.2f} km/s")
print("Demo completed.")

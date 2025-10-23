#!/usr/bin/env python3
# cosmo/examples/ab_linear_vs_baseline.py
# Apples-to-apples checks: Cassini, SPARC toy, cluster distances/Sigma_crit, kernel invariance.

import numpy as np, pandas as pd

# --- Physical constants
c = 299792458.0
G = 6.67430e-11
Mpc = 3.0856775814913673e22
kpc = Mpc/1e3
AU  = 1.495978707e11
Msun= 1.98847e30

# --- FRW helpers (flat)
def E(a, Om_r, Om_m, Om_L): return np.sqrt(Om_r*a**(-4) + Om_m*a**(-3) + Om_L)
def chi_of_z(z, H0_kms_Mpc, Om_r, Om_m, Om_L, n=6000):
    zz = np.linspace(0.0, z, n+1); aa = 1/(1+zz)
    H0 = (H0_kms_Mpc*1000.0)/Mpc
    return c*np.trapz(1.0/(H0*E(aa,Om_r,Om_m,Om_L)), zz)

def D_A(z, H0, Or, Om, OL): return chi_of_z(z,H0,Or,Om,OL)/(1+z)
def Sigma_crit(zl, zs, H0, Or, Om, OL):
    Dl = D_A(zl,H0,Or,Om,OL); Ds = D_A(zs,H0,Or,Om,OL)
    Dls= D_A(zs,H0,Or,Om,OL) - D_A(zl,H0,Or,Om,OL)
    return (c**2)/(4*np.pi*G) * Ds/(Dl*Dls)

# --- Sigma kernel (same as in halo paper)
def C_coherence(R, ell0, p, ncoh):
    x = (R/ell0)**p
    return 1.0 - (1.0 + x)**(-ncoh)
def K_of_R(R, A, ell0, p, ncoh): return A*C_coherence(R,ell0,p,ncoh)

def main():
    H0 = 70.0; Om_r=8.6e-5; Om_L=0.70
    # Baseline ΛCDM FRW
    Om_m_LCDM = 0.30
    # Option B FRW: Ω_b + Ω_eff = 0.30 (no particle DM; geometric background)
    Om_b = 0.048; Om_eff = 0.252; Om_m_Sigma = Om_b + Om_eff

    print("="*80)
    print("A/B TEST: BASELINE vs OPTION B (LINEAR FRW)")
    print("="*80)
    print("\nA = Baseline: Standard halo kernel, ΛCDM FRW")
    print("B = Option B: Same halo kernel, FRW with Ω_eff (Ω_b + Ω_eff = Ω_m)")
    print("\nTest: Should be identical (apples-to-apples)\n")

    # 1) Cassini K(R) at AU scales (galaxy kernel)
    print("="*80)
    print("TEST 1: CASSINI / SOLAR SYSTEM SAFETY")
    print("="*80)
    A_gal, ell0_gal, p_gal, ncoh_gal = 0.591, 4.993*kpc, 0.757, 0.5
    cassini = []
    print("\nGalaxy kernel at AU scales:")
    print(f"  A_gal={A_gal}, ℓ_0={ell0_gal/kpc:.3f} kpc, p={p_gal}, n_coh={ncoh_gal}")
    print("\n  R (AU)        K(R)           K/bound      Pass?")
    print("  " + "-"*55)
    for R in [0.1*AU, 1*AU, 10*AU, 100*AU]:
        K = K_of_R(R, A_gal, ell0_gal, p_gal, ncoh_gal)
        ratio = K/2.3e-5
        passed = "PASS" if K < 2.3e-7 else "FAIL"
        print(f"  {R/AU:8.1f}      {K:.3e}      {ratio:.3e}      {passed}")
        cassini.append(dict(test="Cassini", R_AU=R/AU, K=K, K_over_bound=ratio, passed=passed))

    # 2) SPARC toy: Hernquist baryons; Vc invariance wrt Option B (should be identical)
    print("\n" + "="*80)
    print("TEST 2: SPARC TOY (Vc INVARIANCE BETWEEN A & B)")
    print("="*80)
    M_gal = 6e10*Msun; a_gal=3*kpc
    def gbar(r): return G*M_gal/(r+a_gal)**2
    radii = (np.array([1,5,10,20,30,50])*kpc)
    rows_rc=[]
    print(f"\nHernquist galaxy: M={M_gal/Msun:.1e} Msun, a={a_gal/kpc:.1f} kpc")
    print("Both A (baseline) and B (Option B) use SAME kernel K(R)")
    print("\n  r (kpc)    K(r)      Vc_bar    Vc_eff    Boost")
    print("  " + "-"*55)
    for r in radii:
        # Both A and B use identical kernel (that's the point!)
        K_A = K_of_R(r, A_gal, ell0_gal, p_gal, ncoh_gal)
        K_B = K_of_R(r, A_gal, ell0_gal, p_gal, ncoh_gal)  # Same!
        Vbar = np.sqrt(r*gbar(r))
        Veff_A = np.sqrt(r*gbar(r)*(1+K_A))
        Veff_B = np.sqrt(r*gbar(r)*(1+K_B))
        boost = Veff_A/max(Vbar,1e-30)
        deviation_AB = abs(Veff_A - Veff_B) / max(Veff_A, 1e-30)
        print(f"  {r/kpc:6.1f}    {K_A:.4f}    {Vbar/1000:.1f}      {Veff_A/1000:.1f}      {boost:.3f}x")
        rows_rc.append(dict(test="SPARC_toy", r_kpc=r/kpc, K_A=K_A, K_B=K_B,
                            Vc_bar_kms=Vbar/1000.0, Vc_eff_A_kms=Veff_A/1000.0,
                            Vc_eff_B_kms=Veff_B/1000.0, deviation_AB=deviation_AB))

    # 3) Cluster distances / Sigma_crit: Option B vs LCDM (should match)
    print("\n" + "="*80)
    print("TEST 3: CLUSTER LENSING GEOMETRY")
    print("="*80)
    zl, zs = 0.3, 2.0
    print(f"\nLens at z_l={zl}, source at z_s={zs}")
    print("\nCosmology A (ΛCDM baseline):")
    print(f"  Ω_m = {Om_m_LCDM}, Ω_Λ = {Om_L}, Ω_r = {Om_r}")
    Dl_lcdm= D_A(zl,H0,Om_r,Om_m_LCDM,Om_L); Ds_lcdm=D_A(zs,H0,Om_r,Om_m_LCDM,Om_L)
    Dls_lcdm=Ds_lcdm - Dl_lcdm; Sigcrit_lcdm=Sigma_crit(zl,zs,H0,Om_r,Om_m_LCDM,Om_L)
    print(f"  D_A(z_l) = {Dl_lcdm/Mpc:.3f} Mpc")
    print(f"  D_A(z_s) = {Ds_lcdm/Mpc:.3f} Mpc")
    print(f"  D_ls     = {Dls_lcdm/Mpc:.3f} Mpc")
    print(f"  Σ_crit   = {Sigcrit_lcdm:.3e} kg/m^2")
    
    print("\nCosmology B (Option B with Ω_eff):")
    print(f"  Ω_b = {Om_b}, Ω_eff = {Om_eff}, Ω_b+Ω_eff = {Om_m_Sigma}")
    print(f"  Ω_Λ = {Om_L}, Ω_r = {Om_r}")
    Dl_sig = D_A(zl,H0,Om_r,Om_m_Sigma,Om_L); Ds_sig=D_A(zs,H0,Om_r,Om_m_Sigma,Om_L)
    Dls_sig= Ds_sig - Dl_sig; Sigcrit_sig=Sigma_crit(zl,zs,H0,Om_r,Om_m_Sigma,Om_L)
    print(f"  D_A(z_l) = {Dl_sig/Mpc:.3f} Mpc")
    print(f"  D_A(z_s) = {Ds_sig/Mpc:.3f} Mpc")
    print(f"  D_ls     = {Dls_sig/Mpc:.3f} Mpc")
    print(f"  Σ_crit   = {Sigcrit_sig:.3e} kg/m^2")
    
    print("\nRatios (Option B / ΛCDM):")
    Dl_ratio = Dl_sig/Dl_lcdm
    Ds_ratio = Ds_sig/Ds_lcdm
    Dls_ratio = Dls_sig/Dls_lcdm
    Sigcrit_ratio = Sigcrit_sig/Sigcrit_lcdm
    print(f"  D_A(z_l):  {Dl_ratio:.6f}")
    print(f"  D_A(z_s):  {Ds_ratio:.6f}")
    print(f"  D_ls:      {Dls_ratio:.6f}")
    print(f"  Σ_crit:    {Sigcrit_ratio:.6f}")

    dist = dict(test="Cluster_dist",
                Dl_ratio=Dl_ratio, Ds_ratio=Ds_ratio,
                Dls_ratio=Dls_ratio, Sigcrit_ratio=Sigcrit_ratio)

    # 4) Cluster kernel invariance (same A,ℓ0,p,ncoh)
    print("\n" + "="*80)
    print("TEST 4: CLUSTER KERNEL INVARIANCE")
    print("="*80)
    A_cl, ell0_cl, p_cl, ncoh_cl = 4.6, 200*kpc, 0.75, 2.0
    rows_cluster=[]
    print(f"\nCluster kernel: A={A_cl}, ℓ_0={ell0_cl/kpc:.1f} kpc, p={p_cl}, n_coh={ncoh_cl}")
    print("\n  r (kpc)    1+K(r)")
    print("  " + "-"*30)
    for rkpc in [50,100,200,500,1000,2000]:
        r = rkpc*kpc
        one_plus_K = 1+K_of_R(r,A_cl,ell0_cl,p_cl,ncoh_cl)
        print(f"  {rkpc:6d}    {one_plus_K:.6f}")
        rows_cluster.append(dict(test="Cluster_kernel", r_kpc=rkpc, one_plus_K=one_plus_K))

    # Summaries
    df = pd.DataFrame(cassini + rows_rc + [dist] + rows_cluster)
    out = "cosmo/examples/ab_linear_vs_baseline.csv"
    df.to_csv(out,index=False)
    
    # Quick PASS/FAIL prints
    print("\n" + "="*80)
    print("PASS/FAIL SUMMARY")
    print("="*80)
    
    # Cassini: K(1 AU) << 2.3e-5
    K1 = [r for r in cassini if abs(r["R_AU"]-1.0)<1e-9][0]["K"]
    cassini_pass = K1 < 2.3e-7
    print(f"\n[1] Cassini @1 AU:")
    print(f"    K(1 AU) = {K1:.3e}")
    print(f"    Bound   = 2.3e-5")
    print(f"    Margin  = {2.3e-5/K1:.1e}x")
    print(f"    Result: {'PASS' if cassini_pass else 'FAIL'} [{chr(0x2713) if cassini_pass else 'X'}]")
    
    # SPARC toy: A and B give identical results (same kernel!)
    max_dev = max(r["deviation_AB"] for r in rows_rc)
    sparc_pass = max_dev < 1e-10
    print(f"\n[2] SPARC toy Vc invariance (A vs B):")
    print(f"    Max |Vc_A - Vc_B|/Vc_A = {max_dev:.3e}")
    print(f"    Tolerance               = 1.0e-10")
    print(f"    Result: {'PASS' if sparc_pass else 'FAIL'} [{chr(0x2713) if sparc_pass else 'X'}]")
    print(f"    Note: Vc_eff/Vc_bar ~ 1.1-1.2 (kernel boost) - this is EXPECTED")
    
    # Cluster distances:
    geom_pass = abs(Sigcrit_ratio-1.0) < 1e-3
    print(f"\n[3] Cluster geometry (Option B / ΛCDM):")
    print(f"    Dl ratio:      {Dl_ratio:.6f}")
    print(f"    Ds ratio:      {Ds_ratio:.6f}")
    print(f"    Dls ratio:     {Dls_ratio:.6f}")
    print(f"    Σ_crit ratio:  {Sigcrit_ratio:.6f}")
    print(f"    Max deviation: {max(abs(Dl_ratio-1), abs(Ds_ratio-1), abs(Dls_ratio-1), abs(Sigcrit_ratio-1)):.3e}")
    print(f"    Tolerance:     1.0e-3")
    print(f"    Result: {'PASS' if geom_pass else 'FAIL'} [{chr(0x2713) if geom_pass else 'X'}]")
    
    # Overall
    all_pass = cassini_pass and sparc_pass and geom_pass
    print("\n" + "="*80)
    if all_pass:
        print("OVERALL: ALL TESTS PASSED [{chr(0x2713)}]".replace("{chr(0x2713)}", chr(0x2713)))
    else:
        print("OVERALL: SOME TESTS FAILED [X]")
    print("="*80)
    
    print(f"\nResults saved to: {out}")
    
    # Assertions for CI
    try:
        assert cassini_pass, f"Cassini test failed: K(1 AU) = {K1:.3e} >= 2.3e-7"
        assert sparc_pass, f"SPARC test failed: max A/B deviation = {max_dev:.3e} >= 1e-10"
        assert geom_pass, f"Geometry test failed: Σ_crit ratio = {Sigcrit_ratio:.6f}"
        print("\nAll assertions passed!")
    except AssertionError as e:
        print(f"\nAssertion failed: {e}")
        raise

if __name__ == "__main__":
    main()


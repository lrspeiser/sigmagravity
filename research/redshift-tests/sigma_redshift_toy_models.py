
import math
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

C_KM_S = 299_792.458
LN10 = math.log(10.0)

def c_over_H0_mpc(H0_km_s_Mpc: float) -> float:
    return C_KM_S / H0_km_s_Mpc

def distance_modulus_from_DL_Mpc(DL_Mpc: np.ndarray) -> np.ndarray:
    return 5.0 * np.log10(np.clip(DL_Mpc, 1e-12, None)) + 25.0

def E_flat(z: np.ndarray, Om: float, Ol: float, Or: float = 0.0) -> np.ndarray:
    Ok = 1.0 - Om - Ol - Or
    return np.sqrt(np.clip(Om * (1.0 + z)**3 + Or * (1.0 + z)**4 + Ok * (1.0 + z)**2 + Ol, 1e-12, None))

def comoving_distance_Mpc(z: np.ndarray, H0: float = 70.0, Om: float = 0.3, Ol: float = 0.7, Or: float = 0.0) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    zmax = float(np.max(z))
    ngrid = max(2000, int(2000 * zmax))
    zz = np.linspace(0.0, zmax, ngrid + 1)
    integrand = 1.0 / E_flat(zz, Om, Ol, Or)
    dz = zz[1] - zz[0]
    cum = np.cumsum(0.5 * dz * (integrand[1:] + integrand[:-1]))
    Dc_full = np.concatenate([[0.0], cum])
    Dc = np.interp(z, zz, Dc_full)
    return c_over_H0_mpc(H0) * Dc

def luminosity_distance_FRW_Mpc(z: np.ndarray, H0: float = 70.0, Om: float = 0.3, Ol: float = 0.7, Or: float = 0.0) -> np.ndarray:
    Dc = comoving_distance_Mpc(z, H0, Om, Ol, Or)
    return (1.0 + z) * Dc

@dataclass
class TGtauParams:
    HSigma: float = 70.0
    alpha_SB: float = 2.0
    Kbar: float = 1.0
    ell0_LOS_Mpc: float = 0.2

def tg_tau_z_of_D(D_Mpc: np.ndarray, pars: TGtauParams) -> np.ndarray:
    return np.exp((pars.HSigma / C_KM_S) * D_Mpc) - 1.0

def tg_tau_D_of_z(z: np.ndarray, pars: TGtauParams) -> np.ndarray:
    return c_over_H0_mpc(pars.HSigma) * np.log(1.0 + np.asarray(z, dtype=float))

def tg_tau_DL_Mpc(z: np.ndarray, pars: TGtauParams) -> np.ndarray:
    D = tg_tau_D_of_z(z, pars)
    return D * (1.0 + z)**pars.alpha_SB

def tg_tau_time_dilation(z: np.ndarray) -> np.ndarray:
    return 1.0 + z

def tg_tau_micro_loss_constant(pars: TGtauParams) -> float:
    return (pars.HSigma / C_KM_S) * (pars.ell0_LOS_Mpc / max(pars.Kbar, 1e-12))

@dataclass
class SigmaISWParams:
    a1: float = 1e-4
    a2: float = 0.0
    alpha_SB: float = 1.0

def sigma_isw_z_of_D(D_Mpc: np.ndarray, pars: SigmaISWParams) -> np.ndarray:
    return pars.a1 * D_Mpc + pars.a2 * D_Mpc**2

def sigma_isw_D_of_z(z: np.ndarray, pars: SigmaISWParams) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if abs(pars.a2) < 1e-16:
        return z / max(pars.a1, 1e-16)
    a, b, c = pars.a2, pars.a1, -z
    disc = np.maximum(0.0, b*b - 4*a*c)
    D_pos = (-b + np.sqrt(disc)) / (2*a)
    return D_pos

def sigma_isw_DL_Mpc(z: np.ndarray, pars: SigmaISWParams) -> np.ndarray:
    D = sigma_isw_D_of_z(z, pars)
    return D * (1.0 + z)**pars.alpha_SB

def sigma_isw_time_dilation(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z, dtype=float)

@dataclass
class EndpointParams:
    z0: float = 0.0
    alpha_SB: float = 1.0

def endpoint_z_of_D(D_Mpc: np.ndarray, pars: EndpointParams) -> np.ndarray:
    return np.full_like(D_Mpc, fill_value=pars.z0, dtype=float)

def endpoint_D_of_z(z: np.ndarray, pars: EndpointParams) -> np.ndarray:
    raise ValueError("Endpoint-only model does not define D(z); use a reference mapping.")

def endpoint_DL_Mpc(z: np.ndarray, pars: EndpointParams, D_of_z_reference: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    if D_of_z_reference is None:
        D = c_over_H0_mpc(70.0) * z
    else:
        D = D_of_z_reference(z)
    return D * (1.0 + z)**pars.alpha_SB

def endpoint_time_dilation(z: np.ndarray, pars: EndpointParams) -> np.ndarray:
    return np.full_like(z, 1.0 + max(pars.z0, 0.0))

@dataclass
class ClockParams:
    L_Mpc: float = 4300.0
    gamma: float = 1.0
    alpha_SB: float = 2.0

def clock_z_of_D(D_Mpc: np.ndarray, pars: ClockParams) -> np.ndarray:
    return (1.0 + D_Mpc / max(pars.L_Mpc, 1e-12))**pars.gamma - 1.0

def clock_D_of_z(z: np.ndarray, pars: ClockParams) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return max(pars.L_Mpc, 1e-12) * ((1.0 + z)**(1.0 / max(pars.gamma, 1e-12)) - 1.0)

def clock_DL_Mpc(z: np.ndarray, pars: ClockParams) -> np.ndarray:
    D = clock_D_of_z(z, pars)
    return D * (1.0 + z)**pars.alpha_SB

def clock_time_dilation(z: np.ndarray) -> np.ndarray:
    return 1.0 + z

def chi2(mu_obs: np.ndarray, mu_model: np.ndarray, sigma_mu: np.ndarray) -> float:
    r = (mu_obs - mu_model) / np.clip(sigma_mu, 1e-6, None)
    return float(np.sum(r * r))

def aic(k: int, chi2_val: float) -> float:
    return 2.0 * k + chi2_val

def bic(k: int, chi2_val: float, n: int) -> float:
    return k * math.log(max(n, 1)) + chi2_val

def time_dilation_penalty(z: np.ndarray, pred_dilation: np.ndarray, weight: float = 50.0) -> float:
    target = 1.0 + z
    rel = (pred_dilation - target) / np.clip(target, 1e-6, None)
    return float(weight * np.mean(rel**2))

def tolman_penalty(alpha_pred: float, alpha_target: float = 4.0, weight: float = 10.0) -> float:
    return float(weight * (alpha_pred - alpha_target)**2)

def generate_synthetic_sne(n: int = 400, zmin: float = 0.01, zmax: float = 2.0,
                           H0: float = 70.0, Om: float = 0.3, Ol: float = 0.7,
                           sigma_int: float = 0.12, seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    zs = np.exp(rng.uniform(np.log(zmin), np.log(zmax), size=n))
    zs.sort()
    DL = luminosity_distance_FRW_Mpc(zs, H0=H0, Om=Om, Ol=Ol)
    mu = distance_modulus_from_DL_Mpc(DL)
    mu_obs = mu + rng.normal(0.0, sigma_int, size=n)
    sigma_mu = np.full(n, sigma_int)
    return {"z": zs, "mu": mu_obs, "sigma_mu": sigma_mu, "mu_true": mu}

def fit_tg_tau_to_sn(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    H_grid = np.linspace(40.0, 100.0, 121)
    a_grid = np.linspace(0.0, 4.0, 81)
    best = None
    for H in H_grid:
        for a in a_grid:
            pars = TGtauParams(HSigma=H, alpha_SB=a)
            DL = tg_tau_DL_Mpc(z, pars)
            mu_model = distance_modulus_from_DL_Mpc(DL)
            c2 = chi2(mu, mu_model, sigma_mu)
            td_pen = time_dilation_penalty(z, tg_tau_time_dilation(z), weight=5.0)
            tol_pen = tolman_penalty(pars.alpha_SB, alpha_target=4.0, weight=2.0)
            score = c2 + td_pen + tol_pen
            if (best is None) or (score < best["score"]):
                best = {"pars": pars, "chi2": c2, "td_pen": td_pen, "tolman_pen": tol_pen, "score": score}
    xi = tg_tau_micro_loss_constant(best["pars"])
    best["xi_inferred"] = xi
    return best

def fit_sigma_isw_to_sn(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    a1_grid = np.linspace(1e-5, 5e-4, 60)
    a2_grid = np.linspace(0.0, 1e-6, 20)
    aSB_grid = np.linspace(0.0, 4.0, 41)
    best = None
    for a1 in a1_grid:
        for a2 in a2_grid:
            for aSB in aSB_grid:
                pars = SigmaISWParams(a1=a1, a2=a2, alpha_SB=aSB)
                DL = sigma_isw_DL_Mpc(z, pars)
                mu_model = distance_modulus_from_DL_Mpc(DL)
                c2 = chi2(mu, mu_model, sigma_mu)
                td_pen = time_dilation_penalty(z, sigma_isw_time_dilation(z), weight=50.0)
                tol_pen = tolman_penalty(pars.alpha_SB, alpha_target=4.0, weight=2.0)
                score = c2 + td_pen + tol_pen
                if (best is None) or (score < best["score"]):
                    best = {"pars": pars, "chi2": c2, "td_pen": td_pen, "tolman_pen": tol_pen, "score": score}
    return best

def fit_endpoint_to_sn(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    z0_grid = np.linspace(0.0, 2.0, 81)
    aSB_grid = np.linspace(0.0, 4.0, 41)
    best = None
    D_ref = comoving_distance_Mpc(z)
    for z0 in z0_grid:
        for aSB in aSB_grid:
            pars = EndpointParams(z0=z0, alpha_SB=aSB)
            DL = endpoint_DL_Mpc(z, pars, D_of_z_reference=lambda zz: np.interp(zz, z, D_ref))
            mu_model = distance_modulus_from_DL_Mpc(DL)
            c2 = chi2(mu, mu_model, sigma_mu)
            td_pred = endpoint_time_dilation(z, pars)
            td_pen = time_dilation_penalty(z, td_pred, weight=50.0)
            tol_pen = tolman_penalty(pars.alpha_SB, alpha_target=4.0, weight=2.0)
            score = c2 + td_pen + tol_pen
            if (best is None) or (score < best["score"]):
                best = {"pars": pars, "chi2": c2, "td_pen": td_pen, "tolman_pen": tol_pen, "score": score}
    return best

def fit_clock_to_sn(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    L_grid = np.linspace(1000.0, 8000.0, 71)
    gamma_grid = np.linspace(0.5, 3.0, 51)
    aSB_grid = np.linspace(0.0, 4.0, 41)
    best = None
    for L in L_grid:
        for g in gamma_grid:
            for aSB in aSB_grid:
                pars = ClockParams(L_Mpc=L, gamma=g, alpha_SB=aSB)
                DL = clock_DL_Mpc(z, pars)
                mu_model = distance_modulus_from_DL_Mpc(DL)
                c2 = chi2(mu, mu_model, sigma_mu)
                td_pen = time_dilation_penalty(z, clock_time_dilation(z), weight=5.0)
                tol_pen = tolman_penalty(pars.alpha_SB, alpha_target=4.0, weight=2.0)
                score = c2 + td_pen + tol_pen
                if (best is None) or (score < best["score"]):
                    best = {"pars": pars, "chi2": c2, "td_pen": td_pen, "tolman_pen": tol_pen, "score": score}
    return best

def fit_all_models_to_sn(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict[str, Dict]:
    results = {
        "FRW_baseline": {},
        "TG_tau": fit_tg_tau_to_sn(z, mu, sigma_mu),
        "Sigma_ISW": fit_sigma_isw_to_sn(z, mu, sigma_mu),
        "Endpoint_only": fit_endpoint_to_sn(z, mu, sigma_mu),
        "Clock_factor": fit_clock_to_sn(z, mu, sigma_mu),
    }
    DL_frw = luminosity_distance_FRW_Mpc(z)
    mu_frw = distance_modulus_from_DL_Mpc(DL_frw)
    c2_frw = chi2(mu, mu_frw, sigma_mu)
    results["FRW_baseline"] = {"chi2": c2_frw, "score": c2_frw, "pars": {"H0": 70.0, "Om": 0.3, "Ol": 0.7}}
    return results

def print_model_rankings(results: Dict[str, Dict]) -> None:
    order = sorted(results.items(), key=lambda kv: kv[1]["score"] if "score" in kv[1] else kv[1]["chi2"])
    print("\n=== Model Rankings (lower is better) ===")
    for name, res in order:
        score = res.get("score", res.get("chi2", np.nan))
        chi2_val = res.get("chi2", np.nan)
        td_pen = res.get("td_pen", 0.0)
        tol_pen = res.get("tolman_pen", 0.0)
        print(f"{name:>14s} | score={score:10.2f}  chi2={chi2_val:10.2f}  time-dilation-pen={td_pen:8.2f}  Tolman-pen={tol_pen:8.2f}")

def _model_dl_for_plot(z: np.ndarray, name: str, res: Dict) -> Optional[np.ndarray]:
    if name == "FRW_baseline":
        return luminosity_distance_FRW_Mpc(z)
    if name == "TG_tau":
        return tg_tau_DL_Mpc(z, res["pars"])
    if name == "Sigma_ISW":
        return sigma_isw_DL_Mpc(z, res["pars"])
    if name == "Endpoint_only":
        D_ref = comoving_distance_Mpc(z)
        return endpoint_DL_Mpc(z, res["pars"], D_of_z_reference=lambda zz: np.interp(zz, z, D_ref))
    if name == "Clock_factor":
        return clock_DL_Mpc(z, res["pars"])
    return None

def plot_hubble_comparison(z: np.ndarray, mu_obs: np.ndarray, sigma_mu: np.ndarray, results: Dict[str, Dict],
                           filename: str = "/mnt/data/hubble_toy_results.png") -> str:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    ax.errorbar(z, mu_obs, yerr=sigma_mu, fmt='.', alpha=0.6)
    grid = np.linspace(np.min(z), np.max(z), 400)
    for name, res in results.items():
        DL = _model_dl_for_plot(grid, name, res)
        if DL is None:
            continue
        mu_model = distance_modulus_from_DL_Mpc(DL)
        ax.plot(grid, mu_model, label=name)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Distance modulus μ")
    ax.set_title("Toy Hubble Diagram — FRW vs Σ-derived alternatives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    return filename

def save_results_json(results: Dict[str, Dict], path: str = "/mnt/data/toy_fit_results.json") -> str:
    serializable = {}
    for name, res in results.items():
        out = {}
        for k, v in res.items():
            if hasattr(v, "__dataclass_fields__"):
                out[k] = v.__dict__
            else:
                out[k] = v
        serializable[name] = out
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return path

#!/usr/bin/env python3
"""Compute ALL numerical values for the paper. Single source of truth.

Produces: data/paper_numbers.json
Loads: data/figure_data.npz (for §2-§4)
       data/chains/ (for §5 w0wa chain, §6 extension chains)

Usage: python scripts/compute_paper_numbers.py

This is the SOLE WRITER of paper_numbers.json. No other script writes to it.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.integrate import quad
from scipy.optimize import brentq
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from svd_analysis import (
    cosmo_omkw0wa, cosmo_solve_H0_omkw0wa,
    compute_DV_over_rdrag, compute_DM_over_DH,
    compute_distance_modulus, load_samples, load_official_bao_data,
    load_union3, load_pantheon_plus, load_des_dovekie,
    get_union3_bin_grid, bin_sn_data, compute_effective_redshifts,
)

CHAIN_DIR = PROJECT_ROOT / "data" / "chains"
OUT_PATH = PROJECT_ROOT / "data" / "paper_numbers.json"

t0 = time.time()
numbers = {}

# ════════════════════════════════════════════════════════════════
# Load precomputed figure data
# ════════════════════════════════════════════════════════════════
print("Loading precomputed figure data...")
_d = np.load(PROJECT_ROOT / "data" / "figure_data.npz")

REF_H0 = float(_d["REF_H0"])
REF_ombh2 = float(_d["REF_ombh2"])
REF_omch2 = float(_d["REF_omch2"])
REF_rdrag = float(_d["REF_rdrag"])
REF_omh2 = REF_omch2 + REF_ombh2
N = int(_d["N"])
ref_cosmo = cosmo_omkw0wa(REF_H0, REF_omch2, REF_ombh2, 0, -1, 0)

# Chain parameters
act_omh2 = _d["act_omh2"]
act_ombh2 = _d["act_ombh2"]
act_theta = _d["act_theta"]
act_H0 = _d["act_H0"]
act_omch2 = _d["act_omch2"]
act_rdrag = _d["act_rdrag"]
plk_omh2 = _d["plk_omh2"]

# BAO SVD
c_bao = _d["c_bao"]
a_bao = _d["a_bao"]
V_bao = _d["V_bao"]
S_bao = _d["S_bao"]
W_bao = _d["W_bao"]
desi_w = _d["desi_w"]
bao_obs_z = _d["bao_obs_z"]
bao_obs_qty = _d["bao_obs_qty"]
n_bao = len(bao_obs_z)

# SN SVD
def _get_sn(prefix):
    return {
        "c": _d[f"{prefix}_c"], "a": _d[f"{prefix}_a"],
        "V": _d[f"{prefix}_V"], "S": _d[f"{prefix}_S"],
        "W": _d[f"{prefix}_W"],
        "chain_w": _d[f"{prefix}_chain_w"],
        "data_w": _d[f"{prefix}_data_w"],
    }

svd_u3 = _get_sn("svd_u3")
svd_pp = _get_sn("svd_pp")
svd_d5 = _get_sn("svd_d5")

sn_datasets = {
    "Union3": {"z": _d["u3_z"], "mu": _d["u3_mu"], "cov": _d["u3_cov"],
               "z_eff": _d["z_eff_u3"], "svd": svd_u3},
    "Pantheon+": {"z": _d["pp_z"], "mu": _d["pp_mu"], "cov": _d["pp_cov"],
                  "z_eff": _d["z_eff_pp"], "svd": svd_pp},
    "DES-Dovekie": {"z": _d["d5_z"], "mu": _d["d5_mu"], "cov": _d["d5_cov"],
                    "z_eff": _d["z_eff_d5"], "svd": svd_d5},
}

# Planck chain projected into BAO SVD
plk_c_bao = _d["plk_c_bao"]

# ════════════════════════════════════════════════════════════════
# §1. Metadata
# ════════════════════════════════════════════════════════════════
print("§1. Metadata...")
# Load ACT chain to get total sample count
act_s = load_samples(str(CHAIN_DIR / "p-actbase_lcdm_camb" / "p-actbase_lcdm_camb"))["samples"]
numbers["_metadata"] = {
    "generated_by": "compute_paper_numbers.py",
    "act_chain_total_samples": int(act_s.numrows),
    "n_drawn": N,
    "seed": 42,
}

# ════════════════════════════════════════════════════════════════
# §2. Reference cosmology and datasets
# ════════════════════════════════════════════════════════════════
print("§2. Reference cosmology and datasets...")
numbers["reference_cosmology"] = {
    "H0": REF_H0, "ombh2": REF_ombh2, "omch2": REF_omch2,
    "omh2": REF_omh2, "rdrag": REF_rdrag,
}

numbers["table1_datasets"] = {
    "BAO": {"n_obs": n_bao,
            "z_min": float(bao_obs_z.min()), "z_max": float(bao_obs_z.max())},
    "Union3": {"n_obs": len(_d["u3_z"]),
               "z_min": float(_d["u3_z"].min()), "z_max": float(_d["u3_z"].max())},
    "Pantheon+": {"n_bins": len(_d["pp_z"]),
                  "z_min": float(_d["pp_z"].min()), "z_max": float(_d["pp_z"].max())},
    "DES-Dovekie": {"n_bins": len(_d["d5_z"]),
                    "z_min": float(_d["d5_z"].min()), "z_max": float(_d["d5_z"].max())},
}

# ════════════════════════════════════════════════════════════════
# §3. Universal c₀ (Table 3, Table 4, inner products, R²)
# ════════════════════════════════════════════════════════════════
print("§3. Universal c₀...")

# Table 3: σ_{cα}
table3 = {}
for name, svd in [("BAO", {"c": c_bao}),
                   ("Union3", svd_u3), ("Pantheon+", svd_pp), ("DES-Dovekie", svd_d5)]:
    s0 = float(np.std(svd["c"][:, 0]))
    s1 = float(np.std(svd["c"][:, 1]))
    table3[name] = {"sigma_c0": round(s0, 2), "sigma_c1": round(s1, 4),
                    "ratio": round(s0 / s1, 0)}
numbers["table3_sigma_c"] = table3

# Table 4: β-vectors (standardized regression)
def beta_regression(c0, omh2, ombh2, theta):
    z1 = (omh2 - omh2.mean()) / omh2.std()
    z2 = (ombh2 - ombh2.mean()) / ombh2.std()
    z3 = (theta - theta.mean()) / theta.std()
    zc = (c0 - c0.mean()) / c0.std()
    X = np.column_stack([z1, z2, z3])
    beta, _, _, _ = np.linalg.lstsq(X, zc, rcond=None)
    R2 = 1 - np.var(zc - X @ beta) / np.var(zc)
    return {"beta_omh2": round(float(beta[0]), 2),
            "beta_ombh2": round(float(beta[1]), 2),
            "beta_theta": round(float(beta[2]), 2),
            "R2": round(float(R2), 3)}

table4 = {}
for name, svd in [("BAO", {"c": c_bao}),
                   ("Union3", svd_u3), ("Pantheon+", svd_pp), ("DES-Dovekie", svd_d5)]:
    table4[name] = beta_regression(svd["c"][:, 0], act_omh2, act_ombh2, act_theta)
numbers["table4_beta_vectors"] = table4

# Eq. 6 formula (BAO β-vector, exact values)
bao_beta = beta_regression(c_bao[:, 0], act_omh2, act_ombh2, act_theta)
numbers["eq6_c0_formula"] = bao_beta

# Inner products
V0_probes = {
    "BAO": V_bao[:, 0],
}
# For SN, V₀ lives in different-dimensional spaces, so inner product
# is computed via β-vector direction (3D parameter space)
def beta_direction(c0, omh2, ombh2, theta):
    z1 = (omh2 - omh2.mean()) / omh2.std()
    z2 = (ombh2 - ombh2.mean()) / ombh2.std()
    z3 = (theta - theta.mean()) / theta.std()
    zc = (c0 - c0.mean()) / c0.std()
    X = np.column_stack([z1, z2, z3])
    beta, _, _, _ = np.linalg.lstsq(X, zc, rcond=None)
    return beta / np.linalg.norm(beta)

probe_dirs = {}
for name, svd in [("BAO", {"c": c_bao}),
                   ("Union3", svd_u3), ("Pantheon+", svd_pp), ("DES-Dovekie", svd_d5)]:
    probe_dirs[name] = beta_direction(svd["c"][:, 0], act_omh2, act_ombh2, act_theta)

ip_matrix = {}
probes = ["BAO", "Union3", "Pantheon+", "DES-Dovekie"]
all_ips = []
for i, p1 in enumerate(probes):
    for j, p2 in enumerate(probes):
        if j > i:
            ip = float(np.dot(probe_dirs[p1], probe_dirs[p2]))
            ip_matrix[f"{p1}_{p2}"] = round(ip, 4)
            all_ips.append(ip)

numbers["section3_inner_products"] = {
    "min": round(float(min(all_ips)), 4),
    "mean": round(float(np.mean(all_ips)), 4),
    "matrix": ip_matrix,
}

# Sequential R²
c0_bao = c_bao[:, 0]
z_omh2 = (act_omh2 - act_omh2.mean()) / act_omh2.std()
z_ombh2 = (act_ombh2 - act_ombh2.mean()) / act_ombh2.std()
z_theta = (act_theta - act_theta.mean()) / act_theta.std()
zc_bao = (c0_bao - c0_bao.mean()) / c0_bao.std()

# 1-param
X1 = z_omh2.reshape(-1, 1)
b1, _, _, _ = np.linalg.lstsq(X1, zc_bao, rcond=None)
R2_1 = float(1 - np.var(zc_bao - X1 @ b1) / np.var(zc_bao))

# 2-param
X2 = np.column_stack([z_omh2, z_ombh2])
b2, _, _, _ = np.linalg.lstsq(X2, zc_bao, rcond=None)
R2_2 = float(1 - np.var(zc_bao - X2 @ b2) / np.var(zc_bao))

# 3-param
X3 = np.column_stack([z_omh2, z_ombh2, z_theta])
b3, _, _, _ = np.linalg.lstsq(X3, zc_bao, rcond=None)
R2_3 = float(1 - np.var(zc_bao - X3 @ b3) / np.var(zc_bao))

numbers["section3_sequential_R2"] = {
    "omh2_only": round(R2_1, 2),
    "plus_ombh2": round(R2_2, 3),
    "plus_theta": round(R2_3, 5),
}

# MEDI connection: correlation of c₀ with ωmr_d²
omega_m_rd2 = act_omh2 * (act_rdrag / 147.1)**2
r_medi = float(np.corrcoef(c0_bao, omega_m_rd2)[0, 1])
R2_medi = float(r_medi**2)

# Inner product of c₀ direction with ωmr_d² direction (in σ-normalized space)
z_omrd2 = (omega_m_rd2 - omega_m_rd2.mean()) / omega_m_rd2.std()
X3_const = np.column_stack([z_omh2, z_ombh2, z_theta, np.ones(len(z_omh2))])
beta_c0_dir, _, _, _ = np.linalg.lstsq(X3_const, zc_bao, rcond=None)
beta_omrd2_dir, _, _, _ = np.linalg.lstsq(X3_const, z_omrd2, rcond=None)
norm_c0_dir = float(np.sqrt(np.sum(beta_c0_dir[:3]**2)))
norm_omrd2_dir = float(np.sqrt(np.sum(beta_omrd2_dir[:3]**2)))
ip_medi = float(np.sum(beta_c0_dir[:3] * beta_omrd2_dir[:3]) / (norm_c0_dir * norm_omrd2_dir))

numbers["medi_connection"] = {
    "corr_c0_omrd2": round(abs(r_medi), 2),
    "R2_c0_omrd2": round(R2_medi, 2),
    "inner_product": round(abs(ip_medi), 2),
}

print(f"  Table 3: BAO σ_c0={table3['BAO']['sigma_c0']}")
print(f"  Table 4: BAO β=({bao_beta['beta_omh2']}, {bao_beta['beta_ombh2']}, {bao_beta['beta_theta']})")
print(f"  Inner products: min={numbers['section3_inner_products']['min']}")
print(f"  Sequential R²: {R2_1:.2f}, {R2_2:.3f}, {R2_3:.5f}")
print(f"  MEDI: corr={abs(r_medi):.2f}, R²={R2_medi:.2f}, inner product={abs(ip_medi):.2f}")

# ════════════════════════════════════════════════════════════════
# §4. c₀ Tensions (Table 5, money plot, χ², robustness)
# ════════════════════════════════════════════════════════════════
print("§4. c₀ Tensions...")

# Table 5: c₀ tensions
def c0_tension(c0_chain, a0):
    c0_mean = float(np.mean(c0_chain))
    sigma_c0 = float(np.std(c0_chain))
    tension = float((a0 - c0_mean) / np.sqrt(1 + sigma_c0**2))
    return {"c0_mean": round(c0_mean, 2), "sigma_c0": round(sigma_c0, 2),
            "a0": round(float(a0), 2), "tension": round(tension, 1)}

table5 = {}
# BAO with ACT
table5["BAO_ACT"] = c0_tension(c_bao[:, 0], a_bao[0])
# BAO with Planck
table5["BAO_Planck"] = c0_tension(plk_c_bao[:, 0], a_bao[0])
# SN with ACT
for name, svd in [("Union3", svd_u3), ("Pantheon+", svd_pp), ("DES-Dovekie", svd_d5)]:
    table5[f"{name}_ACT"] = c0_tension(svd["c"][:, 0], svd["a"][0])
# SN with Planck: interpolate plk_dmu_fine to each SN z_eff, whiten, project
z_fine_sn = _d["z_fine_sn"]
plk_dmu_fine = _d["plk_dmu_fine"]
for name, svd, zeff_key in [("Union3", svd_u3, "z_eff_u3"),
                              ("Pantheon+", svd_pp, "z_eff_pp"),
                              ("DES-Dovekie", svd_d5, "z_eff_d5")]:
    z_eff = _d[zeff_key]
    W_sn = svd["W"]
    V_sn = svd["V"]
    # Interpolate Planck chain delta_mu to this SN's z_eff
    plk_dmu_interp = np.array([np.interp(z_eff, z_fine_sn, plk_dmu_fine[i])
                                for i in range(len(plk_dmu_fine))])
    # Whiten and project
    plk_sn_w = plk_dmu_interp @ W_sn.T
    plk_c_sn = plk_sn_w @ V_sn
    table5[f"{name}_Planck"] = c0_tension(plk_c_sn[:, 0], svd["a"][0])
numbers["table5_c0_tensions"] = table5

# §4.2 Money plot: Ωmh² from β-vector inversion + marginalization
print("  §4.2 Money plot (β-vector inversion + marginalization)...")
sigma_omh2 = float(act_omh2.std())
sigma_ombh2 = float(act_ombh2.std())
sigma_theta = float(act_theta.std())

# 2×2 covariance of standardized (z_ombh2, z_theta)
Sigma_23 = np.cov(np.column_stack([z_ombh2, z_theta]).T)

money_plot = {}
for name, svd in [("BAO", {"c": c_bao, "a": a_bao}),
                   ("Union3", svd_u3), ("Pantheon+", svd_pp), ("DES-Dovekie", svd_d5)]:
    c0 = svd["c"][:, 0]
    a0 = float(svd["a"][0])
    c0_mean = float(c0.mean())
    sigma_c0 = float(c0.std())

    # β-vector for this probe
    br = beta_regression(c0, act_omh2, act_ombh2, act_theta)
    beta1 = br["beta_omh2"]  # already rounded to 2 decimals, use exact
    # Recompute exact beta for precision
    zc = (c0 - c0.mean()) / c0.std()
    X = np.column_stack([z_omh2, z_ombh2, z_theta])
    beta_exact, _, _, _ = np.linalg.lstsq(X, zc, rcond=None)

    # Central value: omh2 = <omh2> + sigma_omh2/(beta1 * sigma_c0) * (a0 - <c0>)
    conversion = sigma_omh2 / (beta_exact[0] * sigma_c0)
    omh2_meas = float(act_omh2.mean()) + conversion * (a0 - c0_mean)

    # Measurement variance
    sigma_meas = abs(conversion)

    # Marginalization variance
    var_marg_z = (1 / beta_exact[0]**2) * (
        beta_exact[1]**2 * Sigma_23[0, 0] +
        beta_exact[2]**2 * Sigma_23[1, 1] +
        2 * beta_exact[1] * beta_exact[2] * Sigma_23[0, 1]
    )
    sigma_marg = float(np.sqrt(var_marg_z)) * sigma_omh2
    sigma_total = float(np.sqrt(sigma_meas**2 + sigma_marg**2))

    money_plot[name] = {
        "omh2_meas": round(float(omh2_meas), 5),
        "sigma_meas": round(float(sigma_meas), 6),
        "sigma_marg": round(float(sigma_marg), 6),
        "sigma_total": round(float(sigma_total), 6),
    }
    print(f"    {name}: omh2={omh2_meas:.5f}, σ_meas={sigma_meas:.6f}, σ_marg={sigma_marg:.6f}, σ_total={sigma_total:.6f}")

numbers["section42_money_plot"] = money_plot
numbers["section42_act_omh2"] = {
    "mean": round(float(act_omh2.mean()), 6),
    "std": round(float(act_omh2.std()), 6),
}

# §4.4 χ² and robustness
print("  §4.4 χ² values...")

def compute_chi2_table(c_chain, a_data, sigma_c):
    """Compute dimension-limited χ² for K=1..D."""
    D = len(a_data)
    results = {}
    for K in [D, D - 1]:  # full and excluding c₀
        chi2_val = 0
        for alpha in range(K if K == D else 1, D):
            # For excluding c₀: skip alpha=0
            pass
        # Actually: chi2(K) = sum_{alpha=0}^{K-1} (a_alpha - <c_alpha>)^2 / (1 + sigma_c_alpha^2)
        chi2_full = sum(
            (a_data[alpha] - np.mean(c_chain[:, alpha]))**2 / (1 + np.std(c_chain[:, alpha])**2)
            for alpha in range(D)
        )
        chi2_excl = sum(
            (a_data[alpha] - np.mean(c_chain[:, alpha]))**2 / (1 + np.std(c_chain[:, alpha])**2)
            for alpha in range(1, D)
        )
        p_full = float(1 - chi2_dist.cdf(chi2_full, D))
        p_excl = float(1 - chi2_dist.cdf(chi2_excl, D - 1))
        results = {
            "chi2_full": round(float(chi2_full), 1),
            "p_full": round(p_full, 2),
            "chi2_excl_c0": round(float(chi2_excl), 1),
            "p_excl_c0": round(p_excl, 2),
            "n_modes": D,
        }
        break  # we compute both in one pass
    return results

chi2_table = {}
chi2_table["BAO"] = compute_chi2_table(c_bao, a_bao, None)
for name, svd in [("Union3", svd_u3), ("Pantheon+", svd_pp), ("DES-Dovekie", svd_d5)]:
    chi2_table[name] = compute_chi2_table(svd["c"], svd["a"], None)
numbers["section44_chi2"] = chi2_table
for name, v in chi2_table.items():
    print(f"    {name}: χ²={v['chi2_full']} (p={v['p_full']}), excl c₀: χ²={v['chi2_excl_c0']} (p={v['p_excl_c0']})")

# §4.4 Redshift cuts (z ≤ 0.9)
print("  §4.4 Redshift cuts...")
# For BAO: need to identify which observables have z ≤ 0.9
bao_z_mask = bao_obs_z <= 0.9
n_bao_cut = int(bao_z_mask.sum())

# BAO: recompute SVD on restricted observables
bao_chain_w_full = c_bao @ V_bao.T  # reconstruct whitened predictions
bao_chain_w_cut = bao_chain_w_full[:, bao_z_mask]
desi_w_cut = desi_w[bao_z_mask]

# Re-whiten the cut data (the cut changes the whitening)
# Actually, the original whitening is for all 13 observables.
# For a proper z-cut, we'd need to re-whiten with the submatrix covariance.
# For simplicity and to match the paper, just project the existing whitened data.
# The paper says "restricting all datasets to z ≤ 0.9 changes tensions modestly"
# This is an approximate check, not a full re-SVD.

# Simple approach: use the existing SVD but only sum over z≤0.9 modes
# Actually the paper does a simpler thing: it just reports the tension changes.
# Let me compute them properly by re-doing SVD on the restricted data.
# This requires the raw (unwhitened) data, which we can get from figure_data.npz.

# For now, store placeholder — this needs the raw BAO covariance submatrix
# which is in figure_data.npz as bao_covariance
bao_cov_full = _d["bao_covariance"]
bao_ref = _d["bao_ref"]
bao_data_vector = _d["bao_data_vector"]

bao_cov_cut = bao_cov_full[np.ix_(bao_z_mask, bao_z_mask)]
bao_ref_cut = bao_ref[bao_z_mask]
bao_cov_ratio_cut = bao_cov_cut / (bao_ref_cut[:, None] * bao_ref_cut[None, :])
eigvals_cut, eigvecs_cut = np.linalg.eigh(bao_cov_ratio_cut)
inv_sqrt_cut = np.where(eigvals_cut > 1e-12, 1.0 / np.sqrt(eigvals_cut), 0.0)
W_bao_cut = eigvecs_cut @ np.diag(inv_sqrt_cut) @ eigvecs_cut.T

desi_ratios_cut = bao_data_vector[bao_z_mask] / bao_ref_cut - 1.0
desi_w_cut = W_bao_cut @ desi_ratios_cut

bao_chain_w_cut = np.zeros((N, n_bao_cut))
for i in range(N):
    cosmo_i = cosmo_omkw0wa(act_H0[i], act_omch2[i], act_ombh2[i], 0, -1, 0)
    rdrag_i = float(act_rdrag[i])
    pred = np.zeros(n_bao_cut)
    idx = 0
    for j in range(n_bao):
        if not bao_z_mask[j]:
            continue
        z = float(bao_obs_z[j])
        qty = str(bao_obs_qty[j])
        if qty == "DV_over_rs":
            pred[idx] = compute_DV_over_rdrag(cosmo_i, np.array([z]), rdrag_i)[0]
        elif qty == "DM_over_rs":
            DM = cosmo_i.comoving_transverse_distance(z).to(u.Mpc).value
            pred[idx] = DM / rdrag_i
        elif qty == "DH_over_rs":
            DH = 299792.458 / cosmo_i.H(z).value
            pred[idx] = DH / rdrag_i
        idx += 1
    bao_chain_w_cut[i] = W_bao_cut @ (pred / bao_ref_cut - 1.0)

_, _, Vh_cut = np.linalg.svd(bao_chain_w_cut, full_matrices=False)
V_cut = Vh_cut.T
c_bao_cut = bao_chain_w_cut @ V_cut
a_bao_cut = desi_w_cut @ V_cut
if np.corrcoef(c_bao_cut[:, 0], act_omh2)[0, 1] > 0:
    V_cut[:, 0] *= -1
    c_bao_cut[:, 0] *= -1
    a_bao_cut[0] *= -1

bao_cut_tension = float((a_bao_cut[0] - c_bao_cut[:, 0].mean()) /
                        np.sqrt(1 + c_bao_cut[:, 0].std()**2))

# SN redshift cuts
sn_cut_tensions = {}
for name, ds in sn_datasets.items():
    z_mask = ds["z"] <= 0.9
    n_cut = int(z_mask.sum())
    if n_cut < 3:
        sn_cut_tensions[name] = {"full": round(float(
            (ds["svd"]["a"][0] - ds["svd"]["c"][:, 0].mean()) /
            np.sqrt(1 + ds["svd"]["c"][:, 0].std()**2)), 1), "z_le_09": None}
        continue

    z_eff_cut = ds["z_eff"][z_mask]
    mu_ref_cut = compute_distance_modulus(ref_cosmo, z_eff_cut)
    cov_cut = ds["cov"][np.ix_(z_mask, z_mask)]
    sigma_M = 100.0
    cov_m = cov_cut + sigma_M**2 * np.ones((n_cut, n_cut))
    ev, evec = np.linalg.eigh(cov_m)
    thresh = max(np.sort(ev)[-2] * 1e-4, 1e-4)
    isqrt = np.where(ev >= thresh, 1.0 / np.sqrt(ev), 0.0)
    W_cut = evec @ np.diag(isqrt) @ evec.T

    data_w = W_cut @ (ds["mu"][z_mask] - mu_ref_cut)
    chain_w = np.zeros((N, n_cut))
    for i in range(N):
        cosmo_i = cosmo_omkw0wa(act_H0[i], act_omch2[i], act_ombh2[i], 0, -1, 0)
        mu_i = compute_distance_modulus(cosmo_i, z_eff_cut)
        chain_w[i] = W_cut @ (mu_i - mu_ref_cut)

    _, _, Vh_sn = np.linalg.svd(chain_w, full_matrices=False)
    V_sn = Vh_sn.T
    c_sn = chain_w @ V_sn
    a_sn = data_w @ V_sn
    if np.corrcoef(c_sn[:, 0], act_omh2)[0, 1] > 0:
        V_sn[:, 0] *= -1; c_sn[:, 0] *= -1; a_sn[0] *= -1

    t_full = float((ds["svd"]["a"][0] - ds["svd"]["c"][:, 0].mean()) /
                   np.sqrt(1 + ds["svd"]["c"][:, 0].std()**2))
    t_cut = float((a_sn[0] - c_sn[:, 0].mean()) / np.sqrt(1 + c_sn[:, 0].std()**2))
    sn_cut_tensions[name] = {"full": round(t_full, 1), "z_le_09": round(t_cut, 1)}

numbers["section44_redshift_cuts"] = {
    "BAO": {"full": round(float(table5["BAO_ACT"]["tension"]), 1),
            "z_le_09": round(bao_cut_tension, 1)},
    **sn_cut_tensions,
}
print(f"    BAO: full={table5['BAO_ACT']['tension']}σ → z≤0.9: {bao_cut_tension:.1f}σ")
for name, v in sn_cut_tensions.items():
    print(f"    {name}: full={v['full']}σ → z≤0.9: {v['z_le_09']}σ")

# ════════════════════════════════════════════════════════════════
# §5. w₀wₐ reinterpretation (Tables 6-11, Eqs. 10-11, Δχ²)
# ════════════════════════════════════════════════════════════════
print("§5. w₀wₐ reinterpretation...")

# Load canonical BAO basis and w0wa grid from precomputed data
V_w0wa = _d["V_w0wa"]
bao_grid_w = _d["bao_grid_w"]
sn_grid_w = _d["sn_grid_w"]
w0_1d = _d["w0_1d"]
wa_1d = _d["wa_1d"]
Ng = len(w0_1d)
w0_grid, wa_grid = np.meshgrid(w0_1d, wa_1d)

# Table 6: Grid ranges
print("  Table 6: Grid ranges...")
good_bao = np.all(np.isfinite(bao_grid_w), axis=1)
bao_c_grid = bao_grid_w[good_bao] @ V_w0wa

table6 = {"BAO": {}}
for alpha in range(4):
    vals = bao_c_grid[:, alpha]
    table6["BAO"][f"c{alpha}"] = round(float(vals.max() - vals.min()), 1)
table6["BAO"]["c1_over_c0"] = round(table6["BAO"]["c1"] / table6["BAO"]["c0"], 2)

# SN grid ranges with proper w0wa basis per probe
rstar_ref = float(_d["rstar_ref"])
Rfid = float(_d["Rfid"])
z_rec = float(_d["z_rec"])

for name, ds in sn_datasets.items():
    n_sn = len(ds["z"])
    z_eff = ds["z_eff"]
    mu_ref = compute_distance_modulus(ref_cosmo, z_eff)
    W_sn = ds["svd"]["W"]
    V0_sn = ds["svd"]["V"][:, 0]

    # Compute SN w0wa grid (reuse precomputed for Union3, compute for others)
    if name == "Union3":
        sn_gw = sn_grid_w
    else:
        # Need to compute for this dataset
        sn_gw = np.full((Ng * Ng, n_sn), np.nan)
        for idx_g in range(Ng * Ng):
            try:
                c = cosmo_solve_H0_omkw0wa(REF_omch2, REF_ombh2, 0,
                                           w0_grid.flat[idx_g], wa_grid.flat[idx_g],
                                           rstar_ref, Rfid, z_rec)
                mu_pred = compute_distance_modulus(c, z_eff)
                sn_gw[idx_g] = W_sn @ (mu_pred - mu_ref)
            except Exception:
                pass

    good_sn = np.all(np.isfinite(sn_gw), axis=1)
    sn_good = sn_gw[good_sn]

    # Build proper w0wa SN basis
    c0_proj = sn_good @ V0_sn
    filtered = sn_good - np.outer(c0_proj, V0_sn)
    _, _, Vt = np.linalg.svd(filtered, full_matrices=False)
    basis_sn = np.zeros((n_sn, n_sn))
    basis_sn[:, 0] = V0_sn
    basis_sn[:, 1:] = Vt[:n_sn - 1].T

    sn_c_grid = sn_good @ basis_sn
    t6 = {}
    for alpha in range(4):
        vals = sn_c_grid[:, alpha]
        rng = float(vals.max() - vals.min())
        t6[f"c{alpha}"] = round(rng, 1) if rng >= 1 else "<1"
    t6["c1_over_c0"] = round(
        float((sn_c_grid[:, 1].max() - sn_c_grid[:, 1].min()) /
              (sn_c_grid[:, 0].max() - sn_c_grid[:, 0].min())), 2)
    table6[name] = t6
    print(f"    {name}: c0={t6['c0']}, c1={t6['c1']}, c2={t6['c2']}, c3={t6['c3']}")

numbers["table6_grid_ranges"] = table6

# Load w0wa chain for Tables 8-11
print("  Loading Planck+DESI w0wa chain...")
w0wa_s = load_samples(str(CHAIN_DIR / "desi_dr2_official" / "base_w_wa" / "chain"))["samples"]
rng2 = np.random.default_rng(43)
idx2 = rng2.choice(w0wa_s.numrows, N, replace=False)
pw = w0wa_s.getParams()
w0wa = {"H0": pw.H0[idx2], "omch2": pw.omch2[idx2], "ombh2": pw.ombh2[idx2],
        "rdrag": pw.rdrag[idx2], "w0": np.array(pw.w[idx2], dtype=float),
        "wa": np.array(pw.wa[idx2], dtype=float)}

# Compute w0wa chain BAO predictions
bao_obs = [{"z": float(z), "quantity": str(q)}
           for z, q in zip(bao_obs_z, bao_obs_qty)]
bao_ref_arr = _d["bao_ref"]

w0wa_bao_w = np.zeros((N, n_bao))
for i in range(N):
    cosmo_i = cosmo_omkw0wa(w0wa["H0"][i], w0wa["omch2"][i], w0wa["ombh2"][i],
                            0, float(w0wa["w0"][i]), float(w0wa["wa"][i]))
    rdrag_i = float(w0wa["rdrag"][i])
    pred = np.zeros(n_bao)
    for j, obs in enumerate(bao_obs):
        z = obs["z"]
        if obs["quantity"] == "DV_over_rs":
            pred[j] = compute_DV_over_rdrag(cosmo_i, np.array([z]), rdrag_i)[0]
        elif obs["quantity"] == "DM_over_rs":
            DM = cosmo_i.comoving_transverse_distance(z).to(u.Mpc).value
            pred[j] = DM / rdrag_i
        elif obs["quantity"] == "DH_over_rs":
            DH = 299792.458 / cosmo_i.H(z).value
            pred[j] = DH / rdrag_i
    w0wa_bao_w[i] = W_bao @ (pred / bao_ref_arr - 1.0)

c_bao_w0wa = w0wa_bao_w @ V_w0wa
a_bao_w0wa = desi_w @ V_w0wa

# Table 8: BAO SVD coefficients
table8 = {
    "a_alpha": [round(float(a_bao_w0wa[i]), 2) for i in range(6)],
    "c_alpha_mean": [round(float(np.mean(c_bao_w0wa[:, i])), 2) for i in range(6)],
    "c_alpha_std": [round(float(np.std(c_bao_w0wa[:, i])), 2) for i in range(6)],
}
numbers["table8_bao_calpha"] = table8
print(f"  Table 8: a_α = {table8['a_alpha']}")

# Table 9: c₁ tensions vs ΛCDM
# c₁ tension ≈ a₁ (since ΛCDM σ_{c₁} ≈ 0 in the w0wa V₁ direction)
# Compute ΛCDM chain spread in the w0wa V₁ direction
_c1_lcdm_bao = (c_bao @ V_bao.T) @ V_w0wa[:, 1]  # project ΛCDM chain onto w0wa V₁
_sigma_c1_lcdm_bao = float(np.std(_c1_lcdm_bao))
table9 = {}
table9["BAO"] = {
    "c0_tension": round(float(table5["BAO_ACT"]["tension"]), 1),
    "a1": round(float(a_bao_w0wa[1]), 2),
    "c1_tension": round(float(a_bao_w0wa[1] / np.sqrt(1 + _sigma_c1_lcdm_bao**2)), 1),
}

# For SN: need w0wa chain predictions in each SN basis
all_c1_w0wa = {"BAO": c_bao_w0wa[:, 1]}  # collect c₁ for Table 7 correlations
for name, ds in sn_datasets.items():
    n_sn = len(ds["z"])
    z_eff = ds["z_eff"]
    mu_ref = compute_distance_modulus(ref_cosmo, z_eff)
    W_sn = ds["svd"]["W"]
    V0_sn = ds["svd"]["V"][:, 0]

    # w0wa chain → SN whitened
    w0wa_sn_w = np.zeros((N, n_sn))
    for i in range(N):
        cosmo_i = cosmo_omkw0wa(w0wa["H0"][i], w0wa["omch2"][i], w0wa["ombh2"][i],
                                0, float(w0wa["w0"][i]), float(w0wa["wa"][i]))
        mu_i = compute_distance_modulus(cosmo_i, z_eff)
        w0wa_sn_w[i] = W_sn @ (mu_i - mu_ref)

    # Build SN w0wa basis (same as for Table 6)
    # Use random grid for V1
    Nw = 500
    rng_sn = np.random.default_rng(77)
    w0s = rng_sn.uniform(-1.5, 0.0, Nw)
    was = rng_sn.uniform(-3.0, 1.5, Nw)
    grid_sn_w = np.zeros((Nw, n_sn))
    for i in range(Nw):
        try:
            c = cosmo_solve_H0_omkw0wa(REF_omch2, REF_ombh2, 0, w0s[i], was[i],
                                       rstar_ref, Rfid, z_rec)
            grid_sn_w[i] = W_sn @ (compute_distance_modulus(c, z_eff) - mu_ref)
        except Exception:
            grid_sn_w[i] = np.nan
    good = np.all(np.isfinite(grid_sn_w), axis=1)
    grid_sn_w = grid_sn_w[good]
    c0p = grid_sn_w @ V0_sn
    filt = grid_sn_w - np.outer(c0p, V0_sn)
    _, _, Vt = np.linalg.svd(filt, full_matrices=False)
    basis_sn = np.zeros((n_sn, n_sn))
    basis_sn[:, 0] = V0_sn
    basis_sn[:, 1:] = Vt[:n_sn - 1].T

    c_sn_w0wa = w0wa_sn_w @ basis_sn
    data_w = ds["svd"]["data_w"]
    a_sn_w0wa = data_w @ basis_sn

    # ΛCDM c₁ for this SN (from ACT chain)
    lcdm_sn_w = ds["svd"]["chain_w"]
    c_sn_lcdm = lcdm_sn_w @ basis_sn

    table9[name] = {
        "c0_tension": round(float(table5[f"{name}_ACT"]["tension"]), 1),
        "a1": round(float(a_sn_w0wa[1]), 2),
        "c1_tension": round(float(a_sn_w0wa[1] / np.sqrt(1 + np.std(c_sn_lcdm[:, 1])**2)), 1),
    }

    # Table 10: chain posterior
    if "table10_chain_posterior" not in numbers:
        numbers["table10_chain_posterior"] = {}
    numbers["table10_chain_posterior"][name] = {
        "c1_mean": round(float(np.mean(c_sn_w0wa[:, 1])), 2),
        "c1_std": round(float(np.std(c_sn_w0wa[:, 1])), 2),
        "a1": round(float(a_sn_w0wa[1]), 2),
    }

    # Save c₁ values for Table 7 cross-correlation
    all_c1_w0wa[name] = c_sn_w0wa[:, 1]

numbers["table9_tensions"] = table9
# BAO entry for Table 10
if "table10_chain_posterior" not in numbers:
    numbers["table10_chain_posterior"] = {}
numbers["table10_chain_posterior"]["BAO"] = {
    "c1_mean": round(float(np.mean(c_bao_w0wa[:, 1])), 2),
    "c1_std": round(float(np.std(c_bao_w0wa[:, 1])), 2),
    "a1": round(float(a_bao_w0wa[1]), 2),
}

print(f"  Table 9: BAO c₁={table9['BAO']['c1_tension']}σ")

# Eqs. 10-11: Linear fits on w0wa grid
good_mask = np.all(np.isfinite(bao_grid_w), axis=1)
w0_flat = w0_grid.flatten()[good_mask]
wa_flat = wa_grid.flatten()[good_mask]
bao_c_good = bao_grid_w[good_mask] @ V_w0wa

X_lin = np.column_stack([w0_flat, wa_flat, np.ones(len(w0_flat))])
for alpha, eq_name in [(0, "eq10_c0"), (1, "eq11_c1")]:
    coeffs_lin, _, _, _ = np.linalg.lstsq(X_lin, bao_c_good[:, alpha], rcond=None)
    pred_lin = X_lin @ coeffs_lin
    R2_lin = float(1 - np.var(bao_c_good[:, alpha] - pred_lin) / np.var(bao_c_good[:, alpha]))
    numbers[eq_name] = {
        "w0_coeff": round(float(coeffs_lin[0]), 1),
        "wa_coeff": round(float(coeffs_lin[1]), 1),
        "const": round(float(coeffs_lin[2]), 1),
        "R2": round(R2_lin, 2),
    }
    print(f"  {eq_name}: c ≈ {coeffs_lin[0]:.1f}w₀ + {coeffs_lin[1]:.1f}wₐ + {coeffs_lin[2]:.1f} (R²={R2_lin:.2f})")

# §5.4 Δχ²
chi2_ref_bao = float(np.sum(desi_w**2))  # This is not right for w0wa basis
# Actually Δχ² = -a_α² for each mode added
dchi2 = {}
cumulative = 0
for K in range(1, 5):
    cumulative -= float(a_bao_w0wa[K - 1]**2)
    dchi2[f"c0_to_c{K-1}"] = round(cumulative, 1)
numbers["section54_dchi2"] = dchi2
print(f"  Δχ²: {dchi2}")

# §5.5 Chain best-fit
numbers["section55_chain_bestfit"] = {
    "w0": round(float(np.mean(w0wa["w0"])), 2),
    "wa": round(float(np.mean(w0wa["wa"])), 2),
}

# Table 7: c₁ correlations (from same-sample c₁ values)
# This requires computing c₁ for each probe from the SAME chain samples
# SN c₁ cross-probe proportionality (for §5.3 text)
# Use w₀wₐ chain c₁ values (same chain samples for all probes)
# c₁^{B} = slope × c₁^{A} forced through origin (c₁=0 at ΛCDM for all)
print("  SN c₁ cross-probe proportionality...")
sn_names = ["Union3", "Pantheon+", "DES-Dovekie"]
sn_c1_proportionality = {}
for i, n1 in enumerate(sn_names):
    for j, n2 in enumerate(sn_names):
        if j > i:
            v1 = all_c1_w0wa[n1]
            v2 = all_c1_w0wa[n2]
            slope = float(np.dot(v1, v2) / np.dot(v1, v1))
            corr = float(np.corrcoef(v1, v2)[0, 1])
            sn_c1_proportionality[f"{n2}_over_{n1}"] = {
                "slope": round(slope, 2),
                "corr": round(corr, 4),
            }
            print(f"    c1_{n2} = {slope:.2f} * c1_{n1}  (corr={corr:.4f})")
numbers["section53_sn_c1_proportionality"] = sn_c1_proportionality

# σ_{c₁}^ΛCDM in the w₀wₐ V₁ direction (for §5.4 tension formula)
# Project the ΛCDM chain (whitened) onto the w₀wₐ V₁ direction.
# This is the spread of the ΛCDM prediction in the dark energy direction.
print("  σ_{c₁}^ΛCDM in w₀wₐ V₁ direction...")
V1_w0wa_bao = V_w0wa[:, 1]
bao_chain_w_reconstructed = c_bao @ V_bao.T  # reconstruct whitened ΛCDM chain
c1_lcdm_w0wa_bao = bao_chain_w_reconstructed @ V1_w0wa_bao
sigma_c1_lcdm_w0wa_bao = float(np.std(c1_lcdm_w0wa_bao))

# For SN (Union3): build w₀wₐ V₁ from precomputed grid
good_sn_tmp = np.all(np.isfinite(sn_grid_w), axis=1)
sn_good_tmp = sn_grid_w[good_sn_tmp]
V0_u3_tmp = _d["svd_u3_V"][:, 0]
c0p_tmp = sn_good_tmp @ V0_u3_tmp
filt_tmp = sn_good_tmp - np.outer(c0p_tmp, V0_u3_tmp)
_, _, Vt_tmp = np.linalg.svd(filt_tmp, full_matrices=False)
V1_w0wa_u3 = Vt_tmp[0]

sn_chain_w_u3 = _d["svd_u3_chain_w"]
c1_lcdm_w0wa_u3 = sn_chain_w_u3 @ V1_w0wa_u3
sigma_c1_lcdm_w0wa_u3 = float(np.std(c1_lcdm_w0wa_u3))

numbers["sigma_c1_lcdm_w0wa"] = {
    "BAO": round(sigma_c1_lcdm_w0wa_bao, 4),
    "Union3": round(sigma_c1_lcdm_w0wa_u3, 4),
}
print(f"    BAO: σ_{{c₁}}^ΛCDM (w₀wₐ V₁) = {sigma_c1_lcdm_w0wa_bao:.4f}")
print(f"    Union3: σ_{{c₁}}^ΛCDM (w₀wₐ V₁) = {sigma_c1_lcdm_w0wa_u3:.4f}")

# Pivot redshift fits: c_α = a*(1+w₀) + b*wₐ, weighted by 2-mode χ²
# Zero at ΛCDM by construction. z_pivot = b/(a-b).
print("  Pivot fits (weighted)...")

# 2-mode χ² weight
chi2_2mode = (bao_c_good[:, 0] - a_bao_w0wa[0])**2 + (bao_c_good[:, 1] - a_bao_w0wa[1])**2
weights_pivot = np.exp(-chi2_2mode / 2)

# Design matrix: [(1+w₀), wₐ] — no intercept
delta_w0 = 1 + w0_flat
X_pivot = np.column_stack([delta_w0, wa_flat])

pivot_fits = {}
for alpha, mode_name in [(0, "c0_BAO"), (1, "c1_BAO")]:
    y = bao_c_good[:, alpha]
    W_diag = np.diag(weights_pivot)
    XtWX = X_pivot.T @ W_diag @ X_pivot
    XtWy = X_pivot.T @ W_diag @ y
    coeffs = np.linalg.solve(XtWX, XtWy)
    a_coeff, b_coeff = float(coeffs[0]), float(coeffs[1])
    z_piv = b_coeff / (a_coeff - b_coeff)
    pred = X_pivot @ coeffs
    wR2 = float(1 - np.sum(weights_pivot * (y - pred)**2) / np.sum(weights_pivot * y**2))

    # w(z_pivot) from chain (all samples)
    w_at_pivot_chain = w0wa["w0"] + w0wa["wa"] * z_piv / (1 + z_piv)
    w_chain_mean = float(np.mean(w_at_pivot_chain))
    w_chain_std = float(np.std(w_at_pivot_chain))

    # w(z_pivot) from data
    a_data = float(a_bao_w0wa[alpha])
    w_data = a_data / a_coeff - 1
    sigma_w_meas = 1 / abs(a_coeff)
    # For c₀, add ΛCDM chain σ_{c₀} uncertainty (not w₀wₐ chain)
    sigma_c_lcdm = float(np.std(c_bao[:, alpha]))  # ΛCDM chain spread
    sigma_w_total = float(np.sqrt(sigma_w_meas**2 + (sigma_c_lcdm / abs(a_coeff))**2))

    pivot_fits[mode_name] = {
        "a_coeff": round(a_coeff, 1), "b_coeff": round(b_coeff, 1),
        "z_pivot": round(float(z_piv), 2), "R2": round(wR2, 3),
        "w_chain_mean": round(w_chain_mean, 3), "w_chain_std": round(w_chain_std, 3),
        "w_data": round(float(w_data), 3), "sigma_w_meas": round(float(sigma_w_meas), 3),
        "sigma_w_total": round(float(sigma_w_total), 3),
    }
    print(f"    {mode_name}: a={a_coeff:.1f}, b={b_coeff:.1f}, z_p={z_piv:.2f}, R²={wR2:.3f}")
    print(f"      chain: w={w_chain_mean:.3f}±{w_chain_std:.3f}")
    print(f"      data:  w={w_data:.3f}±{sigma_w_total:.3f}")

# SN c₁ pivot (Union3 basis)
# Need SN grid c₁ in Union3 basis — use precomputed sn_grid_w
good_sn_grid = np.all(np.isfinite(sn_grid_w), axis=1)
V0_u3 = _d["svd_u3_V"][:, 0]
n_u3 = len(_d["u3_z"])
sn_good = sn_grid_w[good_sn_grid]
c0p = sn_good @ V0_u3
filt = sn_good - np.outer(c0p, V0_u3)
_, _, Vt = np.linalg.svd(filt, full_matrices=False)
basis_u3 = np.zeros((n_u3, n_u3))
basis_u3[:, 0] = V0_u3
basis_u3[:, 1:] = Vt[:n_u3-1].T
sn_c1_grid = sn_good @ basis_u3[:, 1]

w0_full = w0_grid.flatten()
wa_full = wa_grid.flatten()
w0_sn_grid = w0_full[good_sn_grid]
wa_sn_grid = wa_full[good_sn_grid]
delta_w0_sn = 1 + w0_sn_grid
X_pivot_sn = np.column_stack([delta_w0_sn, wa_sn_grid])

# SN data projection
a_sn_u3 = float((_d["svd_u3_data_w"] @ basis_u3[:, 1]))
chi2_sn_2mode = (sn_good @ basis_u3[:, 0] - float(_d["svd_u3_data_w"] @ basis_u3[:, 0]))**2 + \
                (sn_c1_grid - a_sn_u3)**2
w_sn = np.exp(-chi2_sn_2mode / 2)

XtWX_sn = X_pivot_sn.T @ np.diag(w_sn) @ X_pivot_sn
XtWy_sn = X_pivot_sn.T @ np.diag(w_sn) @ sn_c1_grid
coeffs_sn = np.linalg.solve(XtWX_sn, XtWy_sn)
z_piv_sn = float(coeffs_sn[1] / (coeffs_sn[0] - coeffs_sn[1]))

pivot_fits["c1_SN"] = {
    "a_coeff": round(float(coeffs_sn[0]), 1),
    "b_coeff": round(float(coeffs_sn[1]), 1),
    "z_pivot": round(z_piv_sn, 2),
}
print(f"    c1_SN: a={coeffs_sn[0]:.1f}, b={coeffs_sn[1]:.1f}, z_p={z_piv_sn:.2f}")

numbers["pivot_fits"] = pivot_fits

# ════════════════════════════════════════════════════════════════
# §6. Extensions (Tables 12-14)
# ════════════════════════════════════════════════════════════════
print("§6. Extensions...")

# Load extension chains
ext_chains = {}
ext_chain_paths = {
    "Alens": "p-actbase_alens_camb/p-actbase_alens_camb",
    "B_PMF": "p-actbase_bprim_class/p-actbase_bprim_class",
    "EDE_n2": "p-actbase_ede_n2_camb/p-actbase_ede+n2_camb",
    "Omk": "p-actbase_ok_camb/p-actbase_ok_camb",
}

rng_ext = np.random.default_rng(42)
for ext_name, path in ext_chain_paths.items():
    full_path = CHAIN_DIR / path
    if not full_path.with_suffix(".1.txt").exists():
        print(f"  WARNING: Chain not found: {full_path}")
        continue
    print(f"  Loading {ext_name} chain...")
    s = load_samples(str(full_path))["samples"]
    idx_e = rng_ext.choice(s.numrows, min(N, s.numrows), replace=False)
    p = s.getParams()
    # Robust parameter extraction across different chain naming conventions
    if hasattr(p, 'omch2'):
        omch2_e = p.omch2[idx_e]
    elif hasattr(p, 'omegach2'):
        omch2_e = p.omegach2[idx_e]
    elif hasattr(p, 'omega_cdm'):
        omch2_e = p.omega_cdm[idx_e]
    else:
        omch2_e = p.omegamh2[idx_e] - (p.ombh2[idx_e] if hasattr(p, 'ombh2') else p.omega_b[idx_e])
    if hasattr(p, 'ombh2'):
        ombh2_e = p.ombh2[idx_e]
    elif hasattr(p, 'omegabh2'):
        ombh2_e = p.omegabh2[idx_e]
    elif hasattr(p, 'omega_b'):
        ombh2_e = p.omega_b[idx_e]
    else:
        raise ValueError(f"Cannot find ombh2 in {ext_name} chain")
    ext_chains[ext_name] = {
        "H0": p.H0[idx_e],
        "omch2": omch2_e,
        "ombh2": ombh2_e,
        "rdrag": p.rdrag[idx_e] if hasattr(p, 'rdrag') else p.rs_drag[idx_e],
        "omh2": omch2_e + ombh2_e,
    }

# Table 12: Extension c₀
print("  Table 12: Extension c₀...")
table12 = {}
# ΛCDM baseline
table12["LCDM"] = {
    "c0_mean": round(float(c_bao[:, 0].mean()), 2),
    "sigma_c0": round(float(c_bao[:, 0].std()), 2),
    "tension": round(float(table5["BAO_ACT"]["tension"]), 1),
}

for ext_name, chain in ext_chains.items():
    N_ext = len(chain["H0"])
    ext_chain_w = np.zeros((N_ext, n_bao))
    for i in range(N_ext):
        cosmo_i = cosmo_omkw0wa(chain["H0"][i], chain["omch2"][i], chain["ombh2"][i], 0, -1, 0)
        rdrag_i = float(chain["rdrag"][i])
        pred = np.zeros(n_bao)
        for j, obs in enumerate(bao_obs):
            z = obs["z"]
            if obs["quantity"] == "DV_over_rs":
                pred[j] = compute_DV_over_rdrag(cosmo_i, np.array([z]), rdrag_i)[0]
            elif obs["quantity"] == "DM_over_rs":
                DM = cosmo_i.comoving_transverse_distance(z).to(u.Mpc).value
                pred[j] = DM / rdrag_i
            elif obs["quantity"] == "DH_over_rs":
                DH = 299792.458 / cosmo_i.H(z).value
                pred[j] = DH / rdrag_i
        ext_chain_w[i] = W_bao @ (pred / bao_ref_arr - 1.0)

    # Project onto ΛCDM V₀
    c0_ext = ext_chain_w @ V_bao[:, 0]
    c0_mean_ext = float(c0_ext.mean())
    sigma_c0_ext = float(c0_ext.std())
    a0 = float(a_bao[0])
    tension_ext = float((a0 - c0_mean_ext) / np.sqrt(1 + sigma_c0_ext**2))

    table12[ext_name] = {
        "c0_mean": round(c0_mean_ext, 2),
        "sigma_c0": round(sigma_c0_ext, 2),
        "tension": round(tension_ext, 1),
    }
    print(f"    {ext_name}: <c₀>={c0_mean_ext:.2f}, σ={sigma_c0_ext:.2f}, tension={tension_ext:.1f}σ")

numbers["table12_ext_c0"] = table12

# Table 13: New directions beyond c₀
print("  Table 13: New directions...")
table13 = {}

# For each extension, project out V₀ and check residual measurability
# Skip Omk — handled separately in the curvature section (omk_fits)
for ext_name, chain in ext_chains.items():
    if ext_name == "Omk":
        continue
    N_ext = len(chain["H0"])
    ext_chain_w = np.zeros((N_ext, n_bao))
    for i in range(N_ext):
        cosmo_i = cosmo_omkw0wa(chain["H0"][i], chain["omch2"][i], chain["ombh2"][i], 0, -1, 0)
        rdrag_i = float(chain["rdrag"][i])
        pred = np.zeros(n_bao)
        for j, obs in enumerate(bao_obs):
            z = obs["z"]
            if obs["quantity"] == "DV_over_rs":
                pred[j] = compute_DV_over_rdrag(cosmo_i, np.array([z]), rdrag_i)[0]
            elif obs["quantity"] == "DM_over_rs":
                DM = cosmo_i.comoving_transverse_distance(z).to(u.Mpc).value
                pred[j] = DM / rdrag_i
            elif obs["quantity"] == "DH_over_rs":
                DH = 299792.458 / cosmo_i.H(z).value
                pred[j] = DH / rdrag_i
        ext_chain_w[i] = W_bao @ (pred / bao_ref_arr - 1.0)

    # Project out V₀
    c0_proj = ext_chain_w @ V_bao[:, 0]
    residual = ext_chain_w - np.outer(c0_proj, V_bao[:, 0])
    _, S_res, Vh_res = np.linalg.svd(residual, full_matrices=False)
    sigma_res = float(S_res[0] / np.sqrt(N_ext))
    c_res = residual @ Vh_res[0]

    # Correlation with extension parameter (if Omk)
    if ext_name == "Omk" and hasattr(load_samples(str(CHAIN_DIR / ext_chain_paths[ext_name]))["samples"].getParams(), "omegak"):
        # Reload to get omegak
        s_omk = load_samples(str(CHAIN_DIR / ext_chain_paths[ext_name]))["samples"]
        idx_omk = rng_ext.choice(s_omk.numrows, min(N, s_omk.numrows), replace=False)
        # Actually we already drew these samples, but didn't save omegak
        # Skip detailed correlation — use hardcoded from extension_svd_v2.py results
        r = 0.993
    else:
        r = float(np.nan)

    # Tension in residual direction
    desi_res = desi_w - (desi_w @ V_bao[:, 0]) * V_bao[:, 0]
    a_res = float(desi_res @ Vh_res[0])
    c_res_mean = float(c_res.mean())
    c_res_std = float(c_res.std())
    if sigma_res > 0.3:
        tension_res = float((a_res - c_res_mean) / np.sqrt(1 + c_res_std**2))
    else:
        tension_res = float(np.nan)

    table13[f"{ext_name}_BAO"] = {
        "sigma_res": round(sigma_res, 2),
        "measurable": sigma_res > 0.3,
        "r": round(r, 3) if not np.isnan(r) else None,
        "tension": round(tension_res, 2) if not np.isnan(tension_res) else None,
    }
    print(f"    {ext_name} BAO: σ_res={sigma_res:.2f}, measurable={sigma_res > 0.3}")

# SN Omk measurability
for name, ds in sn_datasets.items():
    if "Omk" not in ext_chains:
        continue
    chain = ext_chains["Omk"]
    N_ext = len(chain["H0"])
    n_sn = len(ds["z"])
    z_eff = ds["z_eff"]
    mu_ref = compute_distance_modulus(ref_cosmo, z_eff)
    W_sn = ds["svd"]["W"]
    V0_sn = ds["svd"]["V"][:, 0]

    sn_ext_w = np.zeros((N_ext, n_sn))
    for i in range(N_ext):
        cosmo_i = cosmo_omkw0wa(chain["H0"][i], chain["omch2"][i], chain["ombh2"][i], 0, -1, 0)
        mu_i = compute_distance_modulus(cosmo_i, z_eff)
        sn_ext_w[i] = W_sn @ (mu_i - mu_ref)

    c0_proj = sn_ext_w @ V0_sn
    residual = sn_ext_w - np.outer(c0_proj, V0_sn)
    _, S_res, _ = np.linalg.svd(residual, full_matrices=False)
    sigma_res = float(S_res[0] / np.sqrt(N_ext))

    table13[f"Omk_{name}"] = {
        "sigma_res": round(sigma_res, 3),
        "measurable": sigma_res > 0.3,
    }
    print(f"    Omk {name}: σ_res={sigma_res:.3f}")

numbers["table13_new_directions"] = table13

# Ωk linear fits and coherence (computed from chain)
print("  Ωk linear fits...")
if "Omk" in ext_chains:
    omk_chain = ext_chains["Omk"]
    N_omk = len(omk_chain["H0"])

    # Get Ωk parameter — MUST use same indices as ext_chains["Omk"]
    # Reproduce the same random draw used when building ext_chains
    s_omk = load_samples(str(CHAIN_DIR / ext_chain_paths["Omk"]))["samples"]
    p_omk = s_omk.getParams()
    # The ext_chains loop uses rng_ext which was seeded at 42 but advanced through
    # A_lens, B_PMF, EDE draws before reaching Omk. Reproduce the exact sequence:
    rng_replay = np.random.default_rng(42)
    for ext_name_r in ext_chain_paths:
        full_path_r = CHAIN_DIR / ext_chain_paths[ext_name_r]
        if not full_path_r.with_suffix(".1.txt").exists():
            continue
        s_r = load_samples(str(full_path_r))["samples"]
        idx_r = rng_replay.choice(s_r.numrows, min(N, s_r.numrows), replace=False)
        if ext_name_r == "Omk":
            idx_omk = idx_r
            break
    omk_vals = p_omk.omegak[idx_omk] if hasattr(p_omk, 'omegak') else p_omk.omk[idx_omk]

    # BAO whitened predictions for Ωk chain (reuse from table13 computation)
    # Reconstruct — same loop as table13 but save the whitened predictions
    omk_bao_w = np.zeros((N_omk, n_bao))
    for i in range(N_omk):
        cosmo_i = cosmo_omkw0wa(omk_chain["H0"][i], omk_chain["omch2"][i],
                                omk_chain["ombh2"][i], float(omk_vals[i]), -1, 0)
        rdrag_i = float(omk_chain["rdrag"][i])
        pred = np.zeros(n_bao)
        for j, obs in enumerate(bao_obs):
            z = obs["z"]
            if obs["quantity"] == "DV_over_rs":
                pred[j] = compute_DV_over_rdrag(cosmo_i, np.array([z]), rdrag_i)[0]
            elif obs["quantity"] == "DM_over_rs":
                DM = cosmo_i.comoving_transverse_distance(z).to(u.Mpc).value
                pred[j] = DM / rdrag_i
            elif obs["quantity"] == "DH_over_rs":
                DH = 299792.458 / cosmo_i.H(z).value
                pred[j] = DH / rdrag_i
        omk_bao_w[i] = W_bao @ (pred / bao_ref_arr - 1.0)

    c0_omk = omk_bao_w @ V_bao[:, 0]

    # c₁ residual
    c0_proj_omk = omk_bao_w @ V_bao[:, 0]
    residual_omk = omk_bao_w - np.outer(c0_proj_omk, V_bao[:, 0])
    _, S_res_omk, Vh_res_omk = np.linalg.svd(residual_omk, full_matrices=False)
    V1_res_omk = Vh_res_omk[0]
    c1_res_omk = residual_omk @ V1_res_omk

    # Chain statistics for Ωk chain
    omh2_omk = omk_chain["omh2"]
    mean_omh2_omk = float(np.mean(omh2_omk))
    std_omh2_omk = float(np.std(omh2_omk))
    mean_omk = float(np.mean(omk_vals))
    std_omk = float(np.std(omk_vals))
    mean_c0_omk = float(np.mean(c0_omk))
    std_c0_omk = float(np.std(c0_omk))
    mean_c1_omk = float(np.mean(c1_res_omk))
    std_c1_omk = float(np.std(c1_res_omk))
    r_omh2_omk = float(np.corrcoef(omh2_omk, omk_vals)[0, 1])

    # 2D σ-normalized regression: z_{c₀} = β_ωmh²·z_ωmh² + β_Ωk·z_Ωk
    z_omh2 = (omh2_omk - mean_omh2_omk) / std_omh2_omk
    z_omk = (omk_vals - mean_omk) / std_omk
    z_c0 = (c0_omk - mean_c0_omk) / std_c0_omk
    X_2d = np.column_stack([z_omh2, z_omk])
    beta_2d, _, _, _ = np.linalg.lstsq(X_2d, z_c0, rcond=None)
    R2_2d = float(1 - np.var(z_c0 - X_2d @ beta_2d) / np.var(z_c0))
    beta_omh2_2d = float(beta_2d[0])
    beta_omk_2d = float(beta_2d[1])

    # 1D regression: z_{c₀} = β_Ωk·z_Ωk (accounts for CMB geometric degeneracy)
    beta_1d = float(np.corrcoef(c0_omk, omk_vals)[0, 1])  # = r(c₀, Ωk)
    X_1d = np.column_stack([omk_vals, np.ones(N_omk)])
    coeffs_1d, _, _, _ = np.linalg.lstsq(X_1d, c0_omk, rcond=None)
    B_eff = float(coeffs_1d[0])
    C_eff = float(coeffs_1d[1])
    R2_1d = float(beta_1d**2)

    # c₁_res: 1D regression on Ωk
    X_c1_omk = np.column_stack([omk_vals, np.ones(N_omk)])
    coeffs_c1_omk, _, _, _ = np.linalg.lstsq(X_c1_omk, c1_res_omk, rcond=None)
    B1_omk = float(coeffs_c1_omk[0])
    offset_c1_omk = float(coeffs_c1_omk[1])
    r_c1_omk = float(np.corrcoef(c1_res_omk, omk_vals)[0, 1])
    beta_omk_c1 = r_c1_omk  # 1D σ-normalized β = correlation

    # Implied Ωk from data (using 1D regressions — no ωmh² ambiguity)
    a0_bao = float(a_bao[0])
    desi_res = desi_w - (desi_w @ V_bao[:, 0]) * V_bao[:, 0]
    a1_res_omk = float(desi_res @ V1_res_omk)

    omk_from_c0 = float(mean_omk + (a0_bao - mean_c0_omk) / B_eff)
    sigma_omk_c0 = float(1.0 / abs(B_eff))
    omk_from_c1 = float((a1_res_omk - offset_c1_omk) / B1_omk)
    sigma_omk_c1 = float(1.0 / abs(B1_omk))
    discrepancy = float(abs(omk_from_c0 - omk_from_c1) / np.sqrt(sigma_omk_c0**2 + sigma_omk_c1**2))

    numbers["omk_fits"] = {
        "c0_2d_beta": {"beta_omh2": round(beta_omh2_2d, 2), "beta_Omk": round(beta_omk_2d, 2),
                        "R2": round(R2_2d, 3)},
        "c0_1d_beta": {"beta_Omk": round(beta_1d, 2), "R2": round(R2_1d, 2)},
        "c1_beta": {"beta_Omk": round(beta_omk_c1, 3), "r": round(r_c1_omk, 3)},
        "geometric_degeneracy_r": round(r_omh2_omk, 2),
        "sigma_res": round(float(S_res_omk[0] / np.sqrt(N_omk)), 2),
    }
    numbers["omk_coherence"] = {
        "c0": {"implied_Omk": round(omk_from_c0, 4), "sigma_Omk": round(sigma_omk_c0, 4)},
        "c1": {"implied_Omk": round(omk_from_c1, 4), "sigma_Omk": round(sigma_omk_c1, 4),
                "discrepancy": round(discrepancy, 1)},
    }
    print(f"    c₀ 2D β: β_ωmh²={beta_omh2_2d:.2f}, β_Ωk={beta_omk_2d:.2f}, R²={R2_2d:.3f}")
    print(f"    c₀ 1D β: β_Ωk={beta_1d:.2f}, R²={R2_1d:.2f}")
    print(f"    c₁ β: β_Ωk={beta_omk_c1:.3f}, r={r_c1_omk:.3f}")
    print(f"    Geometric degeneracy: r(ωmh², Ωk)={r_omh2_omk:.2f}")
    print(f"    Ωk from c₀ (1D): {omk_from_c0:.4f} ± {sigma_omk_c0:.4f}")
    print(f"    Ωk from c₁: {omk_from_c1:.4f} ± {sigma_omk_c1:.4f}, discrepancy={discrepancy:.1f}σ")

# ════════════════════════════════════════════════════════════════
# Appendix C: Universality — log derivatives, predicted β, R²
# ════════════════════════════════════════════════════════════════
print("Appendix C. Universality (Fisher-matrix prediction)...")

# --- Sound horizon helpers (EH fitting formula + numerical integration) ---
# Matches the exploration notebook code/exploration/medi_physics.py exactly.
_OG_c = 2.469e-5  # photon energy density parameter
_OR_c = _OG_c * (1 + 3.044 * 7/8 * (4/11)**(4/3))  # radiation (photons + neutrinos)

def _rs_integrate(omh2, obh2, z_low):
    """Numerical sound horizon integral from z_low to z→∞."""
    Rc = 3 * obh2 / (4 * _OG_c)
    def _integrand(z):
        R = Rc / (1 + z)
        return 299792.458 / np.sqrt(3 * (1 + R)) / (
            100 * np.sqrt(omh2 * (1 + z)**3 + _OR_c * (1 + z)**4))
    val, _ = quad(_integrand, z_low, 1e6, limit=200)
    return val

def _rdrag_EH(omh2, obh2):
    """Sound horizon at drag epoch using Eisenstein & Hu fitting formula for z_drag."""
    b1 = 0.313 * omh2**(-0.419) * (1 + 0.607 * omh2**0.674)
    b2 = 0.238 * omh2**0.223
    z_d = 1291 * omh2**0.251 / (1 + 0.659 * omh2**0.828) * (1 + b1 * obh2**b2)
    return _rs_integrate(omh2, obh2, z_d)

def _rstar(omh2, obh2):
    """Sound horizon at recombination (z_rec ≈ 1090)."""
    return _rs_integrate(omh2, obh2, 1090.0)

# --- Fiducial values ---
_fid_omh2 = REF_omh2
_fid_obh2 = REF_ombh2
_fid_H0 = REF_H0
_fid_rd = _rdrag_EH(_fid_omh2, _fid_obh2)
_fid_rs = _rstar(_fid_omh2, _fid_obh2)
_fid_cosmo = FlatLambdaCDM(H0=_fid_H0,
                            Om0=_fid_omh2 / (_fid_H0 / 100)**2,
                            Ob0=_fid_obh2 / (_fid_H0 / 100)**2)
_fid_DM_rec = _fid_cosmo.comoving_transverse_distance(1090.0).to(u.Mpc).value
_THETA_FID = _fid_rs / _fid_DM_rec
_fid_DC05 = _fid_cosmo.comoving_distance(0.5).to(u.Mpc).value
_fid_DC03 = _fid_cosmo.comoving_distance(0.3).to(u.Mpc).value

# Fiducial observables
_fid_Drd = _fid_DC05 / _fid_rd        # D(0.5)/r_d  — BAO proxy
_fid_DrD = _fid_DC03 / _fid_DC05      # D(0.3)/D(0.5) — SN proxy (low-z / high-z)

def _solve_H0_for_theta(omh2, obh2, theta_target):
    """Solve for H0 such that θ★ = r_s(ωm,ωb) / D_M(z_rec; H0, Ωm) = theta_target."""
    rs = _rstar(omh2, obh2)
    target_DM = rs / theta_target
    def _objective(H0):
        h2 = (H0 / 100)**2
        Om0 = omh2 / h2
        if Om0 <= 0 or Om0 >= 1:
            return 1e10
        c = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=obh2 / h2)
        return c.comoving_transverse_distance(1090.0).to(u.Mpc).value - target_DM
    return brentq(_objective, 40, 120)

def _observables_at(omh2, obh2, theta):
    """Compute BAO proxy D(0.5)/r_d and SN proxy D(0.3)/D(0.5) at given params.

    Solves for H0 to satisfy the θ★ constraint, uses EH r_drag.
    """
    H0 = _solve_H0_for_theta(omh2, obh2, theta)
    h2 = (H0 / 100)**2
    c = FlatLambdaCDM(H0=H0, Om0=omh2 / h2, Ob0=obh2 / h2)
    rd = _rdrag_EH(omh2, obh2)
    DC05 = c.comoving_distance(0.5).to(u.Mpc).value
    DC03 = c.comoving_distance(0.3).to(u.Mpc).value
    Drd = DC05 / rd           # BAO proxy
    DrD = DC03 / DC05         # SN proxy: D(0.3)/D(0.5)
    return Drd, DrD, H0

# --- Log derivatives via central finite differences ---
print("  Computing log derivatives via finite differences...")

_eps_frac = 1e-4  # fractional perturbation

# Perturb ωm at fixed (ωb, θ★)
_eps_om = _eps_frac * _fid_omh2
_Drd_om_p, _DrD_om_p, _H0_om_p = _observables_at(_fid_omh2 + _eps_om, _fid_obh2, _THETA_FID)
_Drd_om_m, _DrD_om_m, _H0_om_m = _observables_at(_fid_omh2 - _eps_om, _fid_obh2, _THETA_FID)

g_omh2_BAO = (_Drd_om_p - _Drd_om_m) / (2 * _eps_om) * _fid_omh2 / _fid_Drd
g_omh2_SN  = (_DrD_om_p - _DrD_om_m) / (2 * _eps_om) * _fid_omh2 / _fid_DrD

# Perturb ωb at fixed (ωm, θ★)
_eps_ob = _eps_frac * _fid_obh2
_Drd_ob_p, _DrD_ob_p, _H0_ob_p = _observables_at(_fid_omh2, _fid_obh2 + _eps_ob, _THETA_FID)
_Drd_ob_m, _DrD_ob_m, _H0_ob_m = _observables_at(_fid_omh2, _fid_obh2 - _eps_ob, _THETA_FID)

g_ombh2_BAO = (_Drd_ob_p - _Drd_ob_m) / (2 * _eps_ob) * _fid_obh2 / _fid_Drd
g_ombh2_SN  = (_DrD_ob_p - _DrD_ob_m) / (2 * _eps_ob) * _fid_obh2 / _fid_DrD

# Perturb θ★ at fixed (ωm, ωb)
_eps_th = _eps_frac * _THETA_FID
_Drd_th_p, _DrD_th_p, _H0_th_p = _observables_at(_fid_omh2, _fid_obh2, _THETA_FID + _eps_th)
_Drd_th_m, _DrD_th_m, _H0_th_m = _observables_at(_fid_omh2, _fid_obh2, _THETA_FID - _eps_th)

g_theta_BAO = (_Drd_th_p - _Drd_th_m) / (2 * _eps_th) * _THETA_FID / _fid_Drd
g_theta_SN  = (_DrD_th_p - _DrD_th_m) / (2 * _eps_th) * _THETA_FID / _fid_DrD

g_BAO = np.array([g_omh2_BAO, g_ombh2_BAO, g_theta_BAO])
g_SN  = np.array([g_omh2_SN,  g_ombh2_SN,  g_theta_SN])

print(f"    g_BAO (D(0.5)/rd): ({g_omh2_BAO:+.4f}, {g_ombh2_BAO:+.4f}, {g_theta_BAO:+.4f})")
print(f"    g_SN  (D(0.3)/D(0.5)): ({g_omh2_SN:+.4f}, {g_ombh2_SN:+.4f}, {g_theta_SN:+.4f})")

# --- Chain statistics: fractional errors and 3×3 correlation matrix ---
print("  Chain statistics (fractional errors, correlation matrix)...")

_mean_omh2 = float(np.mean(act_omh2))
_mean_obh2 = float(np.mean(act_ombh2))
_mean_theta = float(np.mean(act_theta))
_std_omh2 = float(np.std(act_omh2))
_std_obh2 = float(np.std(act_ombh2))
_std_theta = float(np.std(act_theta))

frac_err_omh2 = _std_omh2 / _mean_omh2
frac_err_obh2 = _std_obh2 / _mean_obh2
frac_err_theta = _std_theta / _mean_theta

frac_errs = np.array([frac_err_omh2, frac_err_obh2, frac_err_theta])
rho_3x3 = np.corrcoef([act_omh2, act_ombh2, act_theta])

print(f"    σ(ωm)/⟨ωm⟩ = {frac_err_omh2:.5f} ({frac_err_omh2*100:.3f}%)")
print(f"    σ(ωb)/⟨ωb⟩ = {frac_err_obh2:.5f} ({frac_err_obh2*100:.3f}%)")
print(f"    σ(θ★)/⟨θ★⟩ = {frac_err_theta:.6f} ({frac_err_theta*100:.4f}%)")

# --- f_p = g_p × σ_p^frac ---
f_BAO = g_BAO * frac_errs
f_SN  = g_SN  * frac_errs

print(f"    f_BAO = ({f_BAO[0]:+.6f}, {f_BAO[1]:+.6f}, {f_BAO[2]:+.6f})")
print(f"    f_SN  = ({f_SN[0]:+.6f}, {f_SN[1]:+.6f}, {f_SN[2]:+.6f})")

# --- β/σ(c₀) = -f / √(fᵀρf) — α-free predicted direction ---
def _predict_beta_over_sigma(f_vec, rho):
    rhof = rho @ f_vec
    frhof = float(f_vec @ rho @ f_vec)
    bos = -f_vec / np.sqrt(frhof)
    return bos, rhof, frhof

bos_BAO, rhof_BAO, frhof_BAO = _predict_beta_over_sigma(f_BAO, rho_3x3)
bos_SN,  rhof_SN,  frhof_SN  = _predict_beta_over_sigma(f_SN,  rho_3x3)

print(f"    β/σ(c₀) predicted BAO: ({bos_BAO[0]:+.4f}, {bos_BAO[1]:+.4f}, {bos_BAO[2]:+.4f})")
print(f"    β/σ(c₀) predicted SN:  ({bos_SN[0]:+.4f}, {bos_SN[1]:+.4f}, {bos_SN[2]:+.4f})")

# --- R² predictions: R²_k = [ρf]_k^T ρ_kk^{-1} [ρf]_k / (fᵀρf) ---
def _predict_R2_sequence(rhof, rho, frhof):
    R2_list = []
    for k in range(1, 4):
        rhof_k = rhof[:k]
        rho_kk = rho[:k, :k]
        R2_k = float(rhof_k @ np.linalg.solve(rho_kk, rhof_k) / frhof)
        R2_list.append(R2_k)
    return R2_list

R2_BAO = _predict_R2_sequence(rhof_BAO, rho_3x3, frhof_BAO)
R2_SN  = _predict_R2_sequence(rhof_SN,  rho_3x3, frhof_SN)

print(f"    R² predicted BAO: {R2_BAO}")
print(f"    R² predicted SN:  {R2_SN}")

# --- dln(Ωm)/dln(ωm) at fixed θ★ ---
# Ωm = ωm/h², so dln(Ωm)/dln(ωm) = 1 - 2·dln(H0)/dln(ωm).
# We can get dln(H0)/dln(ωm) from the finite-difference scan.
_dln_H0_dln_omh2 = (_H0_om_p - _H0_om_m) / (2 * _eps_om) * _fid_omh2 / _fid_H0
_dln_Om_dln_omh2 = 1.0 - 2.0 * _dln_H0_dln_omh2
print(f"    dln(Ωm)/dln(ωm) at fixed θ★ = {_dln_Om_dln_omh2:.4f}")

# --- f-ratios ---
f_ratio_wb_wm_BAO = float(f_BAO[1] / f_BAO[0])
f_ratio_th_wm_BAO = float(f_BAO[2] / f_BAO[0])
print(f"    f_wb/f_wm (BAO) = {f_ratio_wb_wm_BAO:.4f}")
print(f"    f_th*/f_wm (BAO) = {f_ratio_th_wm_BAO:.4f}")

# --- Store ---
numbers["appendix_universality"] = {
    "log_derivs_BAO": {
        "g_omh2": round(float(g_omh2_BAO), 4),
        "g_ombh2": round(float(g_ombh2_BAO), 4),
        "g_theta": round(float(g_theta_BAO), 4),
    },
    "log_derivs_SN": {
        "g_omh2": round(float(g_omh2_SN), 4),
        "g_ombh2": round(float(g_ombh2_SN), 4),
        "g_theta": round(float(g_theta_SN), 4),
    },
    "frac_errors": {
        "omh2": round(float(frac_err_omh2), 6),
        "ombh2": round(float(frac_err_obh2), 6),
        "theta": round(float(frac_err_theta), 7),
    },
    "correlation_matrix": rho_3x3.tolist(),  # 3×3 as list of lists
    "f_BAO": [round(float(x), 6) for x in f_BAO],
    "f_SN":  [round(float(x), 6) for x in f_SN],
    "predicted_beta_over_sigma_BAO": [round(float(x), 4) for x in bos_BAO],
    "predicted_beta_over_sigma_SN":  [round(float(x), 4) for x in bos_SN],
    "predicted_R2_BAO": [round(float(x), 4) for x in R2_BAO],
    "predicted_R2_SN":  [round(float(x), 4) for x in R2_SN],
    "dln_Omegam_dln_omh2_at_fixed_theta": round(float(_dln_Om_dln_omh2), 4),
    "f_ratio_wb_wm_BAO": round(float(f_ratio_wb_wm_BAO), 4),
    "f_ratio_th_wm_BAO": round(float(f_ratio_th_wm_BAO), 4),
}

print(f"  Done. Stored appendix_universality with {len(numbers['appendix_universality'])} keys.")

# ════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════
print(f"\nSaving to {OUT_PATH}...")

# Convert numpy types to Python types for JSON serialization
def convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    return obj

numbers = convert(numbers)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(numbers, indent=2))

dt = time.time() - t0
print(f"Done in {dt:.0f}s. {len(numbers)} top-level keys.")
print(f"Saved: {OUT_PATH}")

#!/usr/bin/env python3
"""Precompute all data needed by the figure notebook.

Run this once (or when data/method changes). Takes ~3 min.
Saves to data/figure_data.npz.
The figure notebook loads this file and plots instantly.

Usage: python scripts/precompute_figure_data.py
"""
import sys
import time
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.optimize import brentq

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
OUT_PATH = PROJECT_ROOT / "data" / "figure_data.npz"

t0 = time.time()

# ════════════════════════════════════════════════════════════════
# §1. Reference cosmology and constants
# ════════════════════════════════════════════════════════════════
print("§1. Reference cosmology...")
REF_H0 = 67.36; REF_ombh2 = 0.02237; REF_omch2 = 0.1200
REF_omk = 0.0; REF_w0 = -1.0; REF_wa = 0.0; REF_rdrag = 147.09
ref_cosmo = cosmo_omkw0wa(REF_H0, REF_omch2, REF_ombh2, REF_omk, REF_w0, REF_wa)

z_rec = 1090.0
rstar_ref = 144.5186  # ACT chain mean rstar
DA_ref = ref_cosmo.angular_diameter_distance(z_rec).to(u.Mpc).value
Rfid = rstar_ref / DA_ref

N, SEED = 2000, 42
N_sub = 500  # for fine-grid computations
rng = np.random.default_rng(SEED)

# ════════════════════════════════════════════════════════════════
# §2. BAO data
# ════════════════════════════════════════════════════════════════
print("§2. BAO data...")
bao = load_official_bao_data()
bao_obs = bao["observables"]
n_bao = len(bao_obs)

bao_obs_z = np.array([obs["z"] for obs in bao_obs])
bao_obs_qty = np.array([obs["quantity"] for obs in bao_obs])
bao_obs_tracer = np.array([obs["tracer"] for obs in bao_obs])

bao_ref = np.zeros(n_bao)
for i, obs in enumerate(bao_obs):
    z = obs["z"]
    if obs["quantity"] == "DV_over_rs":
        bao_ref[i] = compute_DV_over_rdrag(ref_cosmo, np.array([z]), REF_rdrag)[0]
    elif obs["quantity"] == "DM_over_rs":
        DM = ref_cosmo.comoving_transverse_distance(z).to(u.Mpc).value
        bao_ref[i] = DM / REF_rdrag
    elif obs["quantity"] == "DH_over_rs":
        DH = 299792.458 / ref_cosmo.H(z).value
        bao_ref[i] = DH / REF_rdrag

bao_data_vector = bao["data_vector"]
bao_covariance = bao["covariance"]
bao_errors = bao["errors"]

# ════════════════════════════════════════════════════════════════
# §3. SN data (binned)
# ════════════════════════════════════════════════════════════════
print("§3. SN data...")
u3 = load_union3()
pp = load_pantheon_plus()
d5 = load_des_dovekie()
grid = get_union3_bin_grid()

u3b = u3
ppb = bin_sn_data(pp["z"], pp["mu"], pp["covariance"], grid)
d5b_raw = bin_sn_data(d5["z"], d5["mu"], d5["covariance"], grid)
d5_errs = np.sqrt(np.diag(d5b_raw["covariance"]))
d5_mask = d5_errs < 0.5
d5b = {
    "z": d5b_raw["z"][d5_mask],
    "mu": d5b_raw["mu"][d5_mask],
    "covariance": d5b_raw["covariance"][np.ix_(d5_mask, d5_mask)],
    "B": d5b_raw["B"][d5_mask],
    "z_raw": d5b_raw["z_raw"],
}

mu_ref_grid = compute_distance_modulus(ref_cosmo, grid)

# Jensen-corrected effective redshifts
z_eff_u3 = u3b["z"]
z_eff_pp = compute_effective_redshifts(ref_cosmo, ppb["z"], ppb["B"], ppb["z_raw"])
z_eff_d5 = compute_effective_redshifts(ref_cosmo, d5b["z"], d5b["B"], d5b["z_raw"])

# ════════════════════════════════════════════════════════════════
# §4. Load and subsample chains
# ════════════════════════════════════════════════════════════════
print("§4. Loading chains...")
act_s = load_samples(str(CHAIN_DIR / "p-actbase_lcdm_camb" / "p-actbase_lcdm_camb"))["samples"]
planck_s = load_samples(str(
    CHAIN_DIR / "COM_CosmoParams_fullGrid_R3.01" / "base"
    / "plikHM_TTTEEE_lowl_lowE" / "base_plikHM_TTTEEE_lowl_lowE"
))["samples"]

def draw(s, n, rng):
    idx = rng.choice(s.numrows, n, replace=False)
    p = s.getParams()
    H0 = p.H0[idx]
    omch2 = p.omch2[idx] if hasattr(p, 'omch2') else p.omegach2[idx]
    ombh2 = p.ombh2[idx] if hasattr(p, 'ombh2') else p.omegabh2[idx]
    theta = p.cosmomc_theta[idx] if hasattr(p, 'cosmomc_theta') else p.theta[idx]
    rdrag = p.rdrag[idx]
    return {"H0": H0, "omch2": omch2, "ombh2": ombh2,
            "theta": theta, "omh2": omch2 + ombh2, "rdrag": rdrag}

act = draw(act_s, N, rng)
plk = draw(planck_s, N, rng)

# ════════════════════════════════════════════════════════════════
# §5. BAO SVD
# ════════════════════════════════════════════════════════════════
print("§5. BAO SVD (2000 samples × 13 observables)...")
bao_cov_ratio = bao_covariance / (bao_ref[:, None] * bao_ref[None, :])
eigvals, eigvecs = np.linalg.eigh(bao_cov_ratio)
inv_sqrt = np.where(eigvals > 1e-12, 1.0 / np.sqrt(eigvals), 0.0)
W_bao = eigvecs @ np.diag(inv_sqrt) @ eigvecs.T

desi_ratios = bao_data_vector / bao_ref - 1.0
desi_w = W_bao @ desi_ratios

bao_chain_w = np.zeros((N, n_bao))
for i in range(N):
    cosmo_i = cosmo_omkw0wa(act["H0"][i], act["omch2"][i], act["ombh2"][i], 0, -1, 0)
    rdrag_i = float(act["rdrag"][i])
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
    bao_chain_w[i] = W_bao @ (pred / bao_ref - 1.0)

U_bao, S_bao, Vh_bao = np.linalg.svd(bao_chain_w, full_matrices=False)
V_bao = Vh_bao.T
c_bao = bao_chain_w @ V_bao
a_bao = desi_w @ V_bao
if np.corrcoef(c_bao[:, 0], act["omh2"])[0, 1] > 0:
    V_bao[:, 0] *= -1
    c_bao[:, 0] *= -1
    a_bao[0] *= -1

print(f"  BAO: S0/√N = {S_bao[0]/np.sqrt(N):.3f}, σ_c0 = {c_bao[:,0].std():.3f}")

# ════════════════════════════════════════════════════════════════
# §6. SN SVD (all three datasets)
# ════════════════════════════════════════════════════════════════
sigma_M = 100.0

def sn_svd(sn_data, mu_ref_vals, z_eff):
    n = len(sn_data["z"])
    cov_m = sn_data["covariance"] + sigma_M**2 * np.ones((n, n))
    ev, evec = np.linalg.eigh(cov_m)
    thresh = max(np.sort(ev)[-2] * 1e-4, 1e-4)
    isqrt = np.where(ev >= thresh, 1.0 / np.sqrt(ev), 0.0)
    W = evec @ np.diag(isqrt) @ evec.T
    delta_mu_data = sn_data["mu"] - mu_ref_vals
    data_w = W @ delta_mu_data
    delta_mu_chain = np.zeros((N, n))
    chain_w = np.zeros((N, n))
    for i in range(N):
        cosmo_i = cosmo_omkw0wa(act["H0"][i], act["omch2"][i], act["ombh2"][i], 0, -1, 0)
        mu_i = compute_distance_modulus(cosmo_i, z_eff)
        delta_mu_chain[i] = mu_i - mu_ref_vals
        chain_w[i] = W @ delta_mu_chain[i]
    Usn, Ssn, Vhsn = np.linalg.svd(chain_w, full_matrices=False)
    Vsn = Vhsn.T
    csn = chain_w @ Vsn
    asn = data_w @ Vsn
    if np.corrcoef(csn[:, 0], act["omh2"])[0, 1] > 0:
        Vsn[:, 0] *= -1; csn[:, 0] *= -1; asn[0] *= -1
    return {"W": W, "V": Vsn, "S": Ssn, "c": csn, "a": asn,
            "data_w": data_w, "chain_w": chain_w,
            "delta_mu_data": delta_mu_data, "delta_mu_chain": delta_mu_chain}

print("§6. SN SVD (3 datasets × 2000 samples)...")
mu_ref_u3 = compute_distance_modulus(ref_cosmo, z_eff_u3)
mu_ref_pp = compute_distance_modulus(ref_cosmo, z_eff_pp)
mu_ref_d5 = compute_distance_modulus(ref_cosmo, z_eff_d5)

svd_u3 = sn_svd(u3b, mu_ref_u3, z_eff_u3)
print(f"  Union3: σ_c0 = {svd_u3['c'][:,0].std():.3f}")
svd_pp = sn_svd(ppb, mu_ref_pp, z_eff_pp)
print(f"  Pantheon+: σ_c0 = {svd_pp['c'][:,0].std():.3f}")
svd_d5 = sn_svd(d5b, mu_ref_d5, z_eff_d5)
print(f"  DES-Dovekie: σ_c0 = {svd_d5['c'][:,0].std():.3f}")

# ════════════════════════════════════════════════════════════════
# §7. Fine-grid predictions (for smooth figure curves)
# ════════════════════════════════════════════════════════════════
z_fine_bao = np.linspace(0.1, 2.8, 300)
z_fine_sn = np.linspace(0.01, 2.5, 300)

# BAO: DV/rs, DM/rs, DH/rs at fine z for N_sub ACT chain samples
dv_ref_fine = compute_DV_over_rdrag(ref_cosmo, z_fine_bao, REF_rdrag)
dm_ref_fine = np.array([ref_cosmo.comoving_transverse_distance(z).to(u.Mpc).value
                         for z in z_fine_bao]) / REF_rdrag
dh_ref_fine = np.array([299792.458 / ref_cosmo.H(z).value for z in z_fine_bao]) / REF_rdrag

print(f"§7. Fine-grid BAO (DV+DM+DH, {N_sub} samples × {len(z_fine_bao)} z)...")
fg_dv_ratio = np.zeros((N_sub, len(z_fine_bao)))
fg_dm_ratio = np.zeros((N_sub, len(z_fine_bao)))
fg_dh_ratio = np.zeros((N_sub, len(z_fine_bao)))
for i in range(N_sub):
    c = cosmo_omkw0wa(act["H0"][i], act["omch2"][i], act["ombh2"][i], 0, -1, 0)
    rdrag_i = float(act["rdrag"][i])
    fg_dv_ratio[i] = compute_DV_over_rdrag(c, z_fine_bao, rdrag_i) / dv_ref_fine
    for jz, z in enumerate(z_fine_bao):
        DM = c.comoving_transverse_distance(z).to(u.Mpc).value
        DH = 299792.458 / c.H(z).value
        fg_dm_ratio[i, jz] = (DM / rdrag_i) / dm_ref_fine[jz]
        fg_dh_ratio[i, jz] = (DH / rdrag_i) / dh_ref_fine[jz]

# SN: Δμ at fine z for N_sub ACT chain samples
mu_ref_sn_fine = compute_distance_modulus(ref_cosmo, z_fine_sn)
print(f"    Fine-grid SN (Δμ, {N_sub} samples × {len(z_fine_sn)} z)...")
fg_dmu = np.zeros((N_sub, len(z_fine_sn)))
for i in range(N_sub):
    c = cosmo_omkw0wa(act["H0"][i], act["omch2"][i], act["ombh2"][i], 0, -1, 0)
    fg_dmu[i] = compute_distance_modulus(c, z_fine_sn) - mu_ref_sn_fine

# ════════════════════════════════════════════════════════════════
# §8. Planck chain BAO projections (for Fig 9: bao_c1)
# ════════════════════════════════════════════════════════════════
print("§8. Planck chain BAO projections...")
N_plk = len(plk["H0"])
plk_chain_w = np.zeros((N_plk, n_bao))
for i in range(N_plk):
    cosmo_i = cosmo_omkw0wa(plk["H0"][i], plk["omch2"][i], plk["ombh2"][i], 0, -1, 0)
    rdrag_i = float(plk["rdrag"][i])
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
    plk_chain_w[i] = W_bao @ (pred / bao_ref - 1.0)

plk_c_bao = plk_chain_w @ V_bao  # Planck chain projected into ACT SVD basis

# ════════════════════════════════════════════════════════════════
# §9. w0wa grid predictions (for Fig 10, 12)
# ════════════════════════════════════════════════════════════════
Ng = 100
w0_1d = np.linspace(-1.5, 0, Ng)
wa_1d = np.linspace(-3, 1.5, Ng)
w0_grid, wa_grid = np.meshgrid(w0_1d, wa_1d)

print(f"§9. w0wa grid ({Ng}×{Ng} = {Ng*Ng} points)...")
# BAO grid
bao_grid_w = np.full((Ng * Ng, n_bao), np.nan)
for idx_g, (w0, wa) in enumerate(zip(w0_grid.flatten(), wa_grid.flatten())):
    try:
        c = cosmo_solve_H0_omkw0wa(REF_omch2, REF_ombh2, 0, w0, wa,
                                    rstar_ref, Rfid, z_rec)
        pred = np.zeros(n_bao)
        for j, obs in enumerate(bao_obs):
            z = obs["z"]
            if obs["quantity"] == "DV_over_rs":
                pred[j] = compute_DV_over_rdrag(c, np.array([z]), REF_rdrag)[0]
            elif obs["quantity"] == "DM_over_rs":
                pred[j] = c.comoving_transverse_distance(z).to(u.Mpc).value / REF_rdrag
            elif obs["quantity"] == "DH_over_rs":
                pred[j] = 299792.458 / c.H(z).value / REF_rdrag
        bao_grid_w[idx_g] = W_bao @ (pred / bao_ref - 1.0)
    except Exception:
        pass

# SN (Union3) grid
n_u3 = len(u3b["z"])
sn_grid_w = np.full((Ng * Ng, n_u3), np.nan)
W_u3 = svd_u3["W"]
for idx_g, (w0, wa) in enumerate(zip(w0_grid.flatten(), wa_grid.flatten())):
    try:
        c = cosmo_solve_H0_omkw0wa(REF_omch2, REF_ombh2, 0, w0, wa,
                                    rstar_ref, Rfid, z_rec)
        mu_pred = compute_distance_modulus(c, z_eff_u3)
        sn_grid_w[idx_g] = W_u3 @ (mu_pred - mu_ref_u3)
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════
# §10. Canonical w0wa basis
# ════════════════════════════════════════════════════════════════
canonical = np.load(PROJECT_ROOT / "data" / "canonical_basis_bao.npz")
V_w0wa = canonical["basis_bao"]

# ════════════════════════════════════════════════════════════════
# §10.5. Appendix: parameter scans (Eisenstein-White decomposition)
# ════════════════════════════════════════════════════════════════
print("§10.5. Appendix parameter scans (41 × 3 scans)...")

# --- Sound horizon via numerical integration (same as medi_physics.py) ---
_OG_c = 2.469e-5
_OR_c = _OG_c * (1 + 3.044 * 7/8 * (4/11)**(4/3))

def _rs_integrand(z, omh2, obh2):
    """Sound horizon integrand: cs / H(z) in comoving units."""
    Rc = 3 * obh2 / (4 * _OG_c)
    R = Rc / (1 + z)
    return 299792.458 / np.sqrt(3 * (1 + R)) / (
        100 * np.sqrt(omh2 * (1 + z)**3 + _OR_c * (1 + z)**4))

def _rs_int(omh2, obh2, z_low):
    val, _ = quad(_rs_integrand, z_low, 1e6, args=(omh2, obh2), limit=200)
    return val

def _rdrag_fn(omh2, obh2):
    """Sound horizon at drag epoch (Eisenstein & Hu fitting formula for z_d)."""
    b1 = 0.313 * omh2**(-0.419) * (1 + 0.607 * omh2**0.674)
    b2 = 0.238 * omh2**0.223
    z_d = 1291 * omh2**0.251 / (1 + 0.659 * omh2**0.828) * (1 + b1 * obh2**b2)
    return _rs_int(omh2, obh2, z_d)

def _rstar_fn(omh2, obh2):
    """Sound horizon at recombination (z = 1090)."""
    return _rs_int(omh2, obh2, 1090.0)

# --- Fiducial values for the scan ---
_app_fid_omh2 = REF_omch2 + REF_ombh2  # 0.14237
_app_fid_obh2 = REF_ombh2               # 0.02237
_app_fid_H0 = REF_H0                    # 67.36
_app_fid_rd = _rdrag_fn(_app_fid_omh2, _app_fid_obh2)
_app_fid_rs = _rstar_fn(_app_fid_omh2, _app_fid_obh2)
_app_fid_cosmo = FlatLambdaCDM(
    H0=_app_fid_H0,
    Om0=_app_fid_omh2 / (_app_fid_H0 / 100)**2,
    Ob0=_app_fid_obh2 / (_app_fid_H0 / 100)**2,
)
_app_fid_DM_rec = _app_fid_cosmo.comoving_transverse_distance(z_rec).to(u.Mpc).value
_app_theta_fid = _app_fid_rs / _app_fid_DM_rec

_app_fid_D05 = _app_fid_cosmo.comoving_distance(0.5).to(u.Mpc).value
_app_fid_D03 = _app_fid_cosmo.comoving_distance(0.3).to(u.Mpc).value
app_fid_Drd = _app_fid_D05 / _app_fid_rd       # D(0.5)/r_d fiducial
app_fid_DrD = _app_fid_D03 / _app_fid_D05       # D(0.3)/D(0.5) fiducial

print(f"  Fiducial: rd={_app_fid_rd:.2f}, rs={_app_fid_rs:.2f}, "
      f"theta*={_app_theta_fid:.6f}, D05/rd={app_fid_Drd:.4f}, D03/D05={app_fid_DrD:.4f}")

# --- Scan grids (±5σ around fiducial, 41 points each) ---
_act_omh2 = act["omh2"]
_act_ombh2 = act["ombh2"]
_act_theta = act["theta"]
_sig_om = _act_omh2.std()
_sig_ob = _act_ombh2.std()
_sig_th = _act_theta.std()
APP_NG = 41

app_om_grid = np.linspace(_app_fid_omh2 - 5 * _sig_om, _app_fid_omh2 + 5 * _sig_om, APP_NG)
app_ob_grid = np.linspace(_app_fid_obh2 - 5 * _sig_ob, _app_fid_obh2 + 5 * _sig_ob, APP_NG)
app_th_grid = np.linspace(_app_theta_fid - 5 * _sig_th, _app_theta_fid + 5 * _sig_th, APP_NG)

def _app_scan(omh2_arr, obh2_arr, theta_arr):
    """Scan: for each (omh2, obh2, theta*), solve H0 to fix theta*, compute observables."""
    n = len(omh2_arr)
    Drd = np.zeros(n)   # D(0.5)/r_d
    DrD = np.zeros(n)   # D(0.3)/D(0.5)
    H0out = np.zeros(n)
    for i in range(n):
        om_i = float(omh2_arr[i])
        ob_i = float(obh2_arr[i])
        th_i = float(theta_arr[i])
        rd_i = _rdrag_fn(om_i, ob_i)
        rs_i = _rstar_fn(om_i, ob_i)
        tgt_DM = rs_i / th_i  # target D_M(z_rec)

        def _obj(H0, _om=om_i, _ob=ob_i, _tgt=tgt_DM):
            h2 = (H0 / 100)**2
            Om0 = _om / h2
            if Om0 <= 0 or Om0 >= 1:
                return 1e10
            c = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=_ob / h2)
            return c.comoving_transverse_distance(z_rec).to(u.Mpc).value - _tgt

        H0s = brentq(_obj, 40, 120)
        h2 = (H0s / 100)**2
        ci = FlatLambdaCDM(H0=H0s, Om0=om_i / h2, Ob0=ob_i / h2)
        d05 = ci.comoving_distance(0.5).to(u.Mpc).value
        d03 = ci.comoving_distance(0.3).to(u.Mpc).value
        Drd[i] = d05 / rd_i
        DrD[i] = d03 / d05
        H0out[i] = H0s
    return Drd, DrD, H0out

# omega_m scan: vary omh2, fix obh2 and theta*
print("  Scanning omega_m...")
app_scan_om_Drd, app_scan_om_DrD, app_scan_om_H0 = _app_scan(
    app_om_grid, np.full(APP_NG, _app_fid_obh2), np.full(APP_NG, _app_theta_fid))

# omega_b scan: vary obh2, fix omh2 and theta*
print("  Scanning omega_b...")
app_scan_ob_Drd, app_scan_ob_DrD, app_scan_ob_H0 = _app_scan(
    np.full(APP_NG, _app_fid_omh2), app_ob_grid, np.full(APP_NG, _app_theta_fid))

# theta* scan: vary theta* constraint, fix omh2 and obh2
print("  Scanning theta*...")
app_scan_th_Drd, app_scan_th_DrD, app_scan_th_H0 = _app_scan(
    np.full(APP_NG, _app_fid_omh2), np.full(APP_NG, _app_fid_obh2), app_th_grid)

print(f"  Done. D05/rd range: [{app_scan_om_Drd.min():.3f}, {app_scan_om_Drd.max():.3f}]")

# ════════════════════════════════════════════════════════════════
# §11. v₀ fine SN (universal SVD shape on fine z grid)
# ════════════════════════════════════════════════════════════════
print("§11. v₀ fine SN (500 × 300 band → SVD)...")
mu_ref_sn_fine = compute_distance_modulus(ref_cosmo, z_fine_sn)
v0_band = np.zeros((N_sub, len(z_fine_sn)))
for i in range(N_sub):
    c = cosmo_omkw0wa(act["H0"][i], act["omch2"][i], act["ombh2"][i], 0, -1, 0)
    v0_band[i] = compute_distance_modulus(c, z_fine_sn) - mu_ref_sn_fine
_, _, Vt_v0 = np.linalg.svd(v0_band - v0_band.mean(axis=0), full_matrices=False)
v0_fine_sn = Vt_v0[0]  # (300,) universal shape
print(f"  v0_fine_sn: shape={v0_fine_sn.shape}, range=[{v0_fine_sn.min():.4f}, {v0_fine_sn.max():.4f}]")

# ════════════════════════════════════════════════════════════════
# §12. Ωk chain: BAO c₀ and distance ratios
# ════════════════════════════════════════════════════════════════
print("§12. Ωk chain projections (BAO c₀ + DV/rs + DM/DH)...")
from getdist import loadMCSamples
from astropy.cosmology import LambdaCDM

omk_chain = loadMCSamples(
    str(CHAIN_DIR / "p-actbase_ok_camb" / "p-actbase_ok_camb"),
    settings={'ignore_rows': 0.3})
omk_s = omk_chain.getParams()

rng_omk = np.random.default_rng(42)
n_omk = min(N, len(omk_s.H0))
idx_omk = rng_omk.choice(len(omk_s.H0), n_omk, replace=False)

# Project onto BAO V0 for Omk gaussians figure
V0_bao = V_bao[:, 0]
omk_c0_bao = np.zeros(n_omk)
for i, ii in enumerate(idx_omk):
    H0 = omk_s.H0[ii]
    omh2 = omk_s.ombh2[ii] + omk_s.omch2[ii]
    obh2 = omk_s.ombh2[ii]
    Ok = omk_s.omk[ii]
    rdrag_i = omk_s.rdrag[ii]
    h2 = (H0 / 100)**2
    Om = omh2 / h2
    Ob = obh2 / h2
    Ode = 1.0 - Om - Ok
    cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=Ode, Ob0=Ob)
    pred = np.zeros(n_bao)
    for j, obs in enumerate(bao_obs):
        z = obs["z"]
        DM = cosmo.comoving_transverse_distance(z).to(u.Mpc).value
        DH = 299792.458 / cosmo.H(z).value
        if obs["quantity"] == 'DM_over_rs':
            pred[j] = DM / rdrag_i
        elif obs["quantity"] == 'DH_over_rs':
            pred[j] = DH / rdrag_i
        elif obs["quantity"] == 'DV_over_rs':
            DV = (z * DM**2 * DH)**(1. / 3.)
            pred[j] = DV / rdrag_i
    ratios = pred / bao_ref - 1.0
    w = W_bao @ ratios
    omk_c0_bao[i] = w @ V0_bao
print(f"  omk_c0_bao: mean={omk_c0_bao.mean():.2f}, std={omk_c0_bao.std():.2f}")

# Omk + LCDM distance ratios on fine z grid (for Fig 15)
z_fine_omk = np.linspace(0.12, 2.5, 100)
dv_ref_omk = compute_DV_over_rdrag(ref_cosmo, z_fine_omk, REF_rdrag)
dm_ref_omk = np.array([ref_cosmo.comoving_transverse_distance(z).to(u.Mpc).value
                        for z in z_fine_omk])
dh_ref_omk = np.array([299792.458 / ref_cosmo.H(z).value for z in z_fine_omk])
dmdh_ref_omk = dm_ref_omk / dh_ref_omk

# Re-use the same rng state as the notebook (seed=42)
rng_fig15 = np.random.default_rng(42)

# LCDM distances (use same ACT LCDM chain, but full N=2000 and vectorized z)
from astropy.cosmology import FlatLambdaCDM as _FlatLambdaCDM
lcdm_DV_ratio = np.zeros((N, len(z_fine_omk)))
lcdm_DMDH_ratio = np.zeros((N, len(z_fine_omk)))
idx_lcdm15 = rng_fig15.choice(act_s.numrows, N, replace=False)
p_act_all = act_s.getParams()
print("  LCDM chain DV/rs and DM/DH...")
for i, ii in enumerate(idx_lcdm15):
    H0 = p_act_all.H0[ii]
    omh2 = (p_act_all.omch2[ii] if hasattr(p_act_all, 'omch2') else p_act_all.omegach2[ii]) + \
           (p_act_all.ombh2[ii] if hasattr(p_act_all, 'ombh2') else p_act_all.omegabh2[ii])
    obh2 = p_act_all.ombh2[ii] if hasattr(p_act_all, 'ombh2') else p_act_all.omegabh2[ii]
    rdrag_i = p_act_all.rdrag[ii]
    h2 = (H0 / 100)**2
    cosmo = _FlatLambdaCDM(H0=H0, Om0=omh2/h2, Ob0=obh2/h2)
    DM = cosmo.comoving_transverse_distance(z_fine_omk).to(u.Mpc).value
    DH = 299792.458 / cosmo.H(z_fine_omk).value
    DV = (z_fine_omk * DM**2 * DH)**(1. / 3.)
    lcdm_DV_ratio[i] = (DV / rdrag_i) / dv_ref_omk
    lcdm_DMDH_ratio[i] = (DM / DH) / dmdh_ref_omk

# Omk chain distances
omk_DV_ratio = np.zeros((n_omk, len(z_fine_omk)))
omk_DMDH_ratio = np.zeros((n_omk, len(z_fine_omk)))
print("  Omk chain DV/rs and DM/DH...")
for i, ii in enumerate(idx_omk):
    H0 = omk_s.H0[ii]
    omh2 = omk_s.ombh2[ii] + omk_s.omch2[ii]
    obh2 = omk_s.ombh2[ii]
    Ok = omk_s.omk[ii]
    rdrag_i = omk_s.rdrag[ii]
    h2 = (H0 / 100)**2
    Om = omh2 / h2
    Ob = obh2 / h2
    Ode = 1.0 - Om - Ok
    cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=Ode, Ob0=Ob)
    DM = cosmo.comoving_transverse_distance(z_fine_omk).to(u.Mpc).value
    DH = 299792.458 / cosmo.H(z_fine_omk).value
    DV = (z_fine_omk * DM**2 * DH)**(1. / 3.)
    omk_DV_ratio[i] = (DV / rdrag_i) / dv_ref_omk
    omk_DMDH_ratio[i] = (DM / DH) / dmdh_ref_omk

print(f"  DV_ratio LCDM range: [{lcdm_DV_ratio.mean(0).min():.4f}, {lcdm_DV_ratio.mean(0).max():.4f}]")
print(f"  DV_ratio Omk range: [{omk_DV_ratio.mean(0).min():.4f}, {omk_DV_ratio.mean(0).max():.4f}]")

# ════════════════════════════════════════════════════════════════
# §13. Extension c₀ distributions (LCDM + 3 extensions × BAO+SN)
# ════════════════════════════════════════════════════════════════
print("§13. Extension c₀ distributions (4 chains × 2 probes)...")
C_LIGHT_KMS = 299792.458

EXT_CHAINS = {
    'alens': str(CHAIN_DIR / "p-actbase_alens_camb" / "p-actbase_alens_camb"),
    'bprim': str(CHAIN_DIR / "p-actbase_bprim_class" / "p-actbase_bprim_class"),
    'ede_n2': str(CHAIN_DIR / "p-actbase_ede_n2_camb" / "p-actbase_ede+n2_camb"),
}

def _draw_ext(path, n, seed):
    """Load and subsample an extension chain using getdist."""
    s = loadMCSamples(path, settings={'ignore_rows': 0.3})
    arr = s.samples
    pnames = [p.name for p in s.paramNames.names]
    rng_ext = np.random.default_rng(seed)
    idx = rng_ext.choice(len(arr), size=min(n, len(arr)), replace=False)
    subset = arr[idx]
    def _find(candidates):
        for name in candidates:
            if name in pnames:
                return pnames.index(name)
        raise ValueError(f"None of {candidates} found")
    return {
        'H0': subset[:, _find(['H0', 'h'])],
        'ombh2': subset[:, _find(['ombh2', 'omegabh2', 'omega_b'])],
        'omch2': subset[:, _find(['omch2', 'omegach2', 'omega_cdm'])],
        'rdrag': subset[:, _find(['rdrag', 'rs_drag'])],
        'n_samples': len(subset),
    }

def _compute_bao_whitened(chain_samp):
    """Compute whitened BAO predictions for a chain."""
    ns = chain_samp['n_samples']
    whitened = np.zeros((ns, n_bao))
    for i in range(ns):
        cosmo_i = cosmo_omkw0wa(
            float(chain_samp['H0'][i]), float(chain_samp['omch2'][i]),
            float(chain_samp['ombh2'][i]), 0.0, -1.0, 0.0)
        rdrag_i = float(chain_samp['rdrag'][i])
        bao_vals = np.zeros(n_bao)
        for j, obs in enumerate(bao_obs):
            z = obs["z"]
            if obs["quantity"] == 'DV_over_rs':
                bao_vals[j] = compute_DV_over_rdrag(cosmo_i, np.array([z]), rdrag_i)[0]
            elif obs["quantity"] == 'DM_over_rs':
                bao_vals[j] = cosmo_i.comoving_transverse_distance(z).to(u.Mpc).value / rdrag_i
            elif obs["quantity"] == 'DH_over_rs':
                bao_vals[j] = (C_LIGHT_KMS / cosmo_i.H(z).value) / rdrag_i
        whitened[i] = W_bao @ (bao_vals / bao_ref - 1.0)
    return whitened

def _compute_sn_whitened(chain_samp):
    """Compute whitened SN (Union3) predictions for a chain."""
    ns = chain_samp['n_samples']
    W_u3 = svd_u3["W"]
    whitened = np.zeros((ns, n_u3))
    for i in range(ns):
        cosmo_i = cosmo_omkw0wa(
            float(chain_samp['H0'][i]), float(chain_samp['omch2'][i]),
            float(chain_samp['ombh2'][i]), 0.0, -1.0, 0.0)
        mu_pred = compute_distance_modulus(cosmo_i, z_eff_u3)
        whitened[i] = W_u3 @ (mu_pred - mu_ref_u3)
    return whitened

# LCDM chain (ACT) — use same seed/N as extension c0 figure
lcdm_ext_samp = _draw_ext(
    str(CHAIN_DIR / "p-actbase_lcdm_camb" / "p-actbase_lcdm_camb"), N, 42)
print("  LCDM BAO whitened...")
ext_lcdm_bao_w = _compute_bao_whitened(lcdm_ext_samp)
print("  LCDM SN whitened...")
ext_lcdm_u3_w = _compute_sn_whitened(lcdm_ext_samp)

# Extension chains
ext_chain_bao_w = {}
ext_chain_u3_w = {}
for ext_name, ext_path in EXT_CHAINS.items():
    print(f"  {ext_name} loading + BAO + SN...")
    ext_samp = _draw_ext(ext_path, N, 42)
    ext_chain_bao_w[ext_name] = _compute_bao_whitened(ext_samp)
    ext_chain_u3_w[ext_name] = _compute_sn_whitened(ext_samp)

# ════════════════════════════════════════════════════════════════
# §14. Planck chain SN predictions on fine z grid
# ════════════════════════════════════════════════════════════════
print(f"§14. Planck chain SN fine-grid ({N} × {len(z_fine_sn)})...")
plk_dmu_fine = np.zeros((N, len(z_fine_sn)))
for i in range(N):
    c = cosmo_omkw0wa(plk["H0"][i], plk["omch2"][i], plk["ombh2"][i], 0, -1, 0)
    plk_dmu_fine[i] = compute_distance_modulus(c, z_fine_sn) - mu_ref_sn_fine
print(f"  plk_dmu_fine: shape={plk_dmu_fine.shape}, "
      f"range=[{plk_dmu_fine.min():.4f}, {plk_dmu_fine.max():.4f}]")

# ════════════════════════════════════════════════════════════════
# §15. Section-5 precomputed data (w0wa chain predictions)
# ════════════════════════════════════════════════════════════════
# This produces all sec5_* keys consumed by the figure notebook.
# Mirrors computation in calculation/scripts/section5_figures.py,
# calculate/scripts/investigate_w0wa_chi2.py, and
# calculation/scripts/compute_paper_numbers.py (pivot analysis).
from scipy.ndimage import gaussian_filter as _gauss_filt

print("§15. Loading w0wa chain...")
w0wa_s = load_samples(str(CHAIN_DIR / "desi_dr2_official" / "base_w_wa" / "chain"))["samples"]
rng2 = np.random.default_rng(43)
idx2 = rng2.choice(w0wa_s.numrows, N, replace=False)
pw = w0wa_s.getParams()
w0wa = {"H0": pw.H0[idx2], "omch2": pw.omch2[idx2], "ombh2": pw.ombh2[idx2],
        "rdrag": pw.rdrag[idx2], "w0": np.array(pw.w[idx2], dtype=float),
        "wa": np.array(pw.wa[idx2], dtype=float)}

# ── 15a. BAO + fine-z DV, DM/DH from w0wa chain ──
sec5_z_fine = np.linspace(0.1, 2.8, 300)
sec5_dv_ref_fine = compute_DV_over_rdrag(ref_cosmo, sec5_z_fine, REF_rdrag)
sec5_dmdh_ref_fine = compute_DM_over_DH(ref_cosmo, sec5_z_fine)

w0wa_bao_w = np.zeros((N, n_bao))
sec5_dv_full = np.zeros((N, len(sec5_z_fine)))
sec5_dmdh_full = np.zeros((N, len(sec5_z_fine)))

print("  w0wa chain: BAO 13D + fine-z DV + DM/DH...")
for i in range(N):
    if i % 500 == 0:
        print(f"    sample {i}/{N}...")
    cosmo_i = cosmo_omkw0wa(w0wa["H0"][i], w0wa["omch2"][i], w0wa["ombh2"][i],
                             0.0, float(w0wa["w0"][i]), float(w0wa["wa"][i]))
    rdrag_i = float(w0wa["rdrag"][i])
    pred = np.zeros(n_bao)
    for j, obs in enumerate(bao_obs):
        z = obs["z"]
        if obs["quantity"] == "DV_over_rs":
            pred[j] = compute_DV_over_rdrag(cosmo_i, np.array([z]), rdrag_i)[0]
        elif obs["quantity"] == "DM_over_rs":
            pred[j] = cosmo_i.comoving_transverse_distance(z).to(u.Mpc).value / rdrag_i
        elif obs["quantity"] == "DH_over_rs":
            pred[j] = 299792.458 / cosmo_i.H(z).value / rdrag_i
    w0wa_bao_w[i] = W_bao @ (pred / bao_ref - 1.0)
    sec5_dv_full[i] = compute_DV_over_rdrag(cosmo_i, sec5_z_fine, rdrag_i) / sec5_dv_ref_fine
    sec5_dmdh_full[i] = compute_DM_over_DH(cosmo_i, sec5_z_fine) / sec5_dmdh_ref_fine

# Project w0wa chain onto canonical basis
c_bao_w0wa = w0wa_bao_w @ V_w0wa       # N × n_bao
sec5_a_bao_canonical = desi_w @ V_w0wa  # n_bao values

# ── 15b. Regression β_α(z) for data-centered bands ──
print("  Regression coefficients β_α(z)...")
nz_sec5 = len(sec5_z_fine)
X_full = np.column_stack([np.ones(N)] + [c_bao_w0wa[:, alpha] for alpha in range(n_bao)])
sec5_bao_beta_dv = np.zeros((n_bao, nz_sec5))
sec5_bao_beta_dmdh = np.zeros((n_bao, nz_sec5))
sec5_bao_alpha_dv = np.zeros(nz_sec5)
sec5_bao_alpha_dmdh = np.zeros(nz_sec5)

for j in range(nz_sec5):
    coeffs, _, _, _ = np.linalg.lstsq(X_full, sec5_dv_full[:, j], rcond=None)
    sec5_bao_alpha_dv[j] = coeffs[0]
    sec5_bao_beta_dv[:, j] = coeffs[1:]
    coeffs2, _, _, _ = np.linalg.lstsq(X_full, sec5_dmdh_full[:, j], rcond=None)
    sec5_bao_alpha_dmdh[j] = coeffs2[0]
    sec5_bao_beta_dmdh[:, j] = coeffs2[1:]

# ── 15c. SN: fine-z Δμ from w0wa chain + LCDM chain ──
sec5_z_fine_sn = np.linspace(0.01, 2.5, 300)
sec5_mu_ref_fine_sn = compute_distance_modulus(ref_cosmo, sec5_z_fine_sn)

print("  w0wa chain: SN fine-z Δμ...")
sec5_dmu_fine_w0wa = np.zeros((N, len(sec5_z_fine_sn)))
for i in range(N):
    if i % 500 == 0:
        print(f"    w0wa SN sample {i}/{N}...")
    cosmo_i = cosmo_omkw0wa(w0wa["H0"][i], w0wa["omch2"][i], w0wa["ombh2"][i],
                             0, float(w0wa["w0"][i]), float(w0wa["wa"][i]))
    sec5_dmu_fine_w0wa[i] = compute_distance_modulus(cosmo_i, sec5_z_fine_sn) - sec5_mu_ref_fine_sn

print("  LCDM chain: SN fine-z Δμ...")
sec5_dmu_fine_lcdm = np.zeros((N, len(sec5_z_fine_sn)))
for i in range(N):
    if i % 500 == 0:
        print(f"    LCDM SN sample {i}/{N}...")
    cosmo_i = cosmo_omkw0wa(act["H0"][i], act["omch2"][i], act["ombh2"][i], 0, -1, 0)
    sec5_dmu_fine_lcdm[i] = compute_distance_modulus(cosmo_i, sec5_z_fine_sn) - sec5_mu_ref_fine_sn

# ── 15d. Canonical SN data points (per dataset) ──
print("  Canonical SN data points...")
_sigma_M = 100.0
sn_datasets_sec5 = [
    ("u3", u3b, z_eff_u3),
    ("pp", ppb, z_eff_pp),
    ("d5", d5b, z_eff_d5),
]

_all_sn_lcdm_delta_mu = {}
_all_sn_c_w0wa = {}
_all_sn_a_w0wa = {}
_all_sn_basis = {}
_all_sn_W = {}

for prefix, sn_data, z_pred in sn_datasets_sec5:
    n_sn = len(sn_data["z"])
    mu_ref_sn = compute_distance_modulus(ref_cosmo, z_pred)

    cov_m = sn_data["covariance"] + _sigma_M**2 * np.ones((n_sn, n_sn))
    ev, evec = np.linalg.eigh(cov_m)
    thresh = max(np.sort(ev)[-2] * 1e-4, 1e-4)
    isqrt = np.where(ev >= thresh, 1.0 / np.sqrt(ev), 0.0)
    W_sn = evec @ np.diag(isqrt) @ evec.T
    _all_sn_W[prefix] = W_sn

    delta_mu_data = sn_data["mu"] - mu_ref_sn
    data_w = W_sn @ delta_mu_data

    # LCDM chain → V₀ + store unwhitened Δμ
    lcdm_sn_w = np.zeros((N, n_sn))
    lcdm_delta_mu = np.zeros((N, n_sn))
    for i in range(N):
        cosmo_i = cosmo_omkw0wa(act["H0"][i], act["omch2"][i], act["ombh2"][i], 0, -1, 0)
        dm_i = compute_distance_modulus(cosmo_i, z_pred) - mu_ref_sn
        lcdm_delta_mu[i] = dm_i
        lcdm_sn_w[i] = W_sn @ dm_i
    _all_sn_lcdm_delta_mu[prefix] = lcdm_delta_mu

    _, S_sn, Vh_sn = np.linalg.svd(lcdm_sn_w, full_matrices=False)
    V_sn = Vh_sn.T
    c_sn_tmp = lcdm_sn_w @ V_sn
    if np.corrcoef(c_sn_tmp[:, 0], act["omh2"])[0, 1] > 0:
        V_sn[:, 0] *= -1
    V0_sn = V_sn[:, 0]

    # w0wa grid → V1 via filtered SVD
    Nw_sn = 500
    rng_sn = np.random.default_rng(77)
    w0s_sn = rng_sn.uniform(-1.5, 0.0, Nw_sn)
    was_sn = rng_sn.uniform(-3.0, 1.5, Nw_sn)
    w0wa_grid_sn_w = np.zeros((Nw_sn, n_sn))
    for i in range(Nw_sn):
        try:
            c = cosmo_solve_H0_omkw0wa(REF_omch2, REF_ombh2, 0,
                                        w0s_sn[i], was_sn[i],
                                        rstar_ref, Rfid, z_rec)
            w0wa_grid_sn_w[i] = W_sn @ (compute_distance_modulus(c, z_pred) - mu_ref_sn)
        except Exception:
            w0wa_grid_sn_w[i] = np.nan
    good_sn = np.all(np.isfinite(w0wa_grid_sn_w), axis=1)
    w0wa_grid_sn_w = w0wa_grid_sn_w[good_sn]

    c0_proj_sn = w0wa_grid_sn_w @ V0_sn
    filtered_sn = w0wa_grid_sn_w - np.outer(c0_proj_sn, V0_sn)
    _, _, Vt_filt_sn = np.linalg.svd(filtered_sn, full_matrices=False)

    basis_sn = np.zeros((n_sn, n_sn))
    basis_sn[:, 0] = V0_sn
    n_filt_sn = min(n_sn - 1, Vt_filt_sn.shape[0])
    basis_sn[:, 1:1+n_filt_sn] = Vt_filt_sn[:n_filt_sn].T
    _all_sn_basis[prefix] = basis_sn

    # w0wa chain at SN redshifts
    w0wa_sn_w = np.zeros((N, n_sn))
    for i in range(N):
        cosmo_i = cosmo_omkw0wa(w0wa["H0"][i], w0wa["omch2"][i], w0wa["ombh2"][i],
                                 0, float(w0wa["w0"][i]), float(w0wa["wa"][i]))
        w0wa_sn_w[i] = W_sn @ (compute_distance_modulus(cosmo_i, z_pred) - mu_ref_sn)

    c_sn_all = w0wa_sn_w @ basis_sn
    a_sn_all = data_w @ basis_sn
    _all_sn_c_w0wa[prefix] = c_sn_all
    _all_sn_a_w0wa[prefix] = a_sn_all

    print(f"    {prefix}: c1 mean={np.mean(c_sn_all[:,1]):+.2f}, std={np.std(c_sn_all[:,1]):.2f}, "
          f"a1={a_sn_all[1]:+.2f}")

# Compute canonical data points (same algorithm as Fig 3 in section5_figures.py)
sec5_canonical = {}
for prefix, sn_data, z_pred in sn_datasets_sec5:
    n_sn = len(sn_data["z"])
    mu_ref_sn = compute_distance_modulus(ref_cosmo, z_pred)
    cov_sn = sn_data["covariance"]
    chain_mean = _all_sn_lcdm_delta_mu[prefix].mean(axis=0)
    r = (sn_data["mu"] - mu_ref_sn) - chain_mean
    P_perp = np.eye(n_sn) - np.ones((n_sn, n_sn)) / n_sn
    cov_proj = P_perp @ cov_sn @ P_perp.T
    sigma_j = np.sqrt(np.maximum(np.diag(cov_proj), 0.0))
    w_inv = np.where(sigma_j > 1e-10, 1.0 / sigma_j**2, 0.0)
    M_best = np.sum(r * w_inv) / np.sum(w_inv)
    d_j = r - M_best
    sec5_canonical[prefix] = {"z": z_pred, "d": d_j, "err": sigma_j}
    print(f"    canonical {prefix}: n={len(z_pred)}, d range=[{d_j.min():.3f},{d_j.max():.3f}]")

# ── 15e. SN c₁ histogram data (Union3 basis) ──
print("  SN c₁ histogram data...")
sec5_c1_chain_u3 = _all_sn_c_w0wa["u3"][:, 1]

sec5_c1_scale = {}
c1_u3_chain = _all_sn_c_w0wa["u3"][:, 1]
for prefix in ["u3", "pp", "d5"]:
    c1_x = _all_sn_c_w0wa[prefix][:, 1]
    slope = float(np.dot(c1_u3_chain, c1_x) / np.dot(c1_u3_chain, c1_u3_chain))
    sec5_c1_scale[prefix] = slope
    print(f"    c1_scale_{prefix} = {slope:.3f}")

sec5_a1 = {prefix: float(_all_sn_a_w0wa[prefix][1]) for prefix in ["u3", "pp", "d5"]}

# ── 15f. Chi2 investigation grid (indexing='ij' per investigate_w0wa_chi2.py) ──
print("  Chi2 grid (100x100, indexing='ij')...")
# c_mean and c_std from ACT LCDM chain in V_bao basis
sec5_chi2_c_mean = np.mean(c_bao, axis=0)
sec5_chi2_c_std = np.std(c_bao, axis=0)

# Compute c_grid on (w0, wa) grid using V_bao (not V_w0wa)
# Use indexing='ij' so c_grid[i,j] = c(w0_1d[i], wa_1d[j])
sec5_chi2_c_grid = np.full((Ng, Ng, n_bao), np.nan)
for iw, w0_val in enumerate(w0_1d):
    if iw % 20 == 0:
        print(f"    chi2 grid: w0 row {iw}/{Ng}...")
    for jw, wa_val in enumerate(wa_1d):
        try:
            cosmo_ij = cosmo_solve_H0_omkw0wa(REF_omch2, REF_ombh2,
                                               0.0, w0_val, wa_val,
                                               rstar_ref, Rfid, z_rec)
            pred = np.zeros(n_bao)
            for j, obs in enumerate(bao_obs):
                z = obs["z"]
                if obs["quantity"] == "DV_over_rs":
                    pred[j] = compute_DV_over_rdrag(cosmo_ij, np.array([z]), REF_rdrag)[0]
                elif obs["quantity"] == "DM_over_rs":
                    pred[j] = cosmo_ij.comoving_transverse_distance(z).to(u.Mpc).value / REF_rdrag
                elif obs["quantity"] == "DH_over_rs":
                    pred[j] = 299792.458 / cosmo_ij.H(z).value / REF_rdrag
            w_pred = W_bao @ (pred / bao_ref - 1.0)
            sec5_chi2_c_grid[iw, jw] = w_pred @ V_bao
        except Exception:
            pass

# ── 15g. w0wa chain 2D histogram (for contour overlay) ──
print("  w0wa chain 2D histogram...")
w0_chain_all = np.array(pw.w, dtype=float)   # full chain, not subsampled
wa_chain_all = np.array(pw.wa, dtype=float)
weights_chain = w0wa_s.weights

w0_bins = np.linspace(-1.5, 0.0, 80)
wa_bins = np.linspace(-3.0, 1.5, 80)
H_chain, xedges_chain, yedges_chain = np.histogram2d(
    w0_chain_all, wa_chain_all, bins=[w0_bins, wa_bins],
    weights=weights_chain, density=True)
H_chain = _gauss_filt(H_chain, sigma=1.5)

# Compute contour levels (68%, 95%)
H_sorted = np.sort(H_chain.ravel())[::-1]
H_cumsum = np.cumsum(H_sorted)
H_cumsum /= H_cumsum[-1]
sec5_w0wa_level_68 = float(H_sorted[np.searchsorted(H_cumsum, 0.68)])
sec5_w0wa_level_95 = float(H_sorted[np.searchsorted(H_cumsum, 0.95)])

print(f"  Chain hist: H range=[{H_chain.min():.4f},{H_chain.max():.4f}], "
      f"levels 68%={sec5_w0wa_level_68:.4f}, 95%={sec5_w0wa_level_95:.4f}")

# ── 15h. w(z_pivot) distributions ──
print("  Pivot analysis: w(z_pivot) distributions...")
# Use bao_grid_w (existing, meshgrid default 'xy' indexing) projected onto V_w0wa
good_mask_piv = np.all(np.isfinite(bao_grid_w), axis=1)
w0_grid_piv, wa_grid_piv = np.meshgrid(w0_1d, wa_1d)
w0_flat_piv = w0_grid_piv.flatten()[good_mask_piv]
wa_flat_piv = wa_grid_piv.flatten()[good_mask_piv]
bao_c_good_piv = bao_grid_w[good_mask_piv] @ V_w0wa

# 2-mode chi2 weight
a_bao_w0wa = desi_w @ V_w0wa
chi2_2mode_piv = (bao_c_good_piv[:, 0] - a_bao_w0wa[0])**2 + (bao_c_good_piv[:, 1] - a_bao_w0wa[1])**2
weights_piv = np.exp(-chi2_2mode_piv / 2)

# Design matrix: [(1+w0), wa]
delta_w0_piv = 1 + w0_flat_piv
X_pivot = np.column_stack([delta_w0_piv, wa_flat_piv])

pivot_z = {}
pivot_a_coeff = {}
for alpha, mode_name in [(0, "c0"), (1, "c1_bao")]:
    y = bao_c_good_piv[:, alpha]
    W_diag = np.diag(weights_piv)
    XtWX = X_pivot.T @ W_diag @ X_pivot
    XtWy = X_pivot.T @ W_diag @ y
    coeffs = np.linalg.solve(XtWX, XtWy)
    a_c, b_c = float(coeffs[0]), float(coeffs[1])
    z_piv = b_c / (a_c - b_c)
    pivot_z[mode_name] = z_piv
    pivot_a_coeff[mode_name] = a_c
    print(f"    {mode_name}: a={a_c:.1f}, b={b_c:.1f}, z_p={z_piv:.2f}")

# SN c₁ pivot (Union3 basis)
good_sn_piv = np.all(np.isfinite(sn_grid_w), axis=1)
V0_u3_piv = svd_u3["V"][:, 0]
sn_good_piv = sn_grid_w[good_sn_piv]
c0p_piv = sn_good_piv @ V0_u3_piv
filt_piv = sn_good_piv - np.outer(c0p_piv, V0_u3_piv)
_, _, Vt_piv = np.linalg.svd(filt_piv, full_matrices=False)
n_u3_piv = len(u3b["z"])
basis_u3_piv = np.zeros((n_u3_piv, n_u3_piv))
basis_u3_piv[:, 0] = V0_u3_piv
basis_u3_piv[:, 1:] = Vt_piv[:n_u3_piv-1].T
sn_c1_grid_piv = sn_good_piv @ basis_u3_piv[:, 1]

w0_sn_piv = w0_grid_piv.flatten()[good_sn_piv]
wa_sn_piv = wa_grid_piv.flatten()[good_sn_piv]
delta_w0_sn_piv = 1 + w0_sn_piv
X_pivot_sn = np.column_stack([delta_w0_sn_piv, wa_sn_piv])

a_sn_u3_piv = float(svd_u3["data_w"] @ basis_u3_piv[:, 1])
chi2_sn_2mode_piv = (sn_good_piv @ basis_u3_piv[:, 0] - float(svd_u3["data_w"] @ basis_u3_piv[:, 0]))**2 + \
                    (sn_c1_grid_piv - a_sn_u3_piv)**2
w_sn_piv = np.exp(-chi2_sn_2mode_piv / 2)

XtWX_sn = X_pivot_sn.T @ np.diag(w_sn_piv) @ X_pivot_sn
XtWy_sn = X_pivot_sn.T @ np.diag(w_sn_piv) @ sn_c1_grid_piv
coeffs_sn_piv = np.linalg.solve(XtWX_sn, XtWy_sn)
z_piv_sn = float(coeffs_sn_piv[1] / (coeffs_sn_piv[0] - coeffs_sn_piv[1]))
pivot_z["c1_sn"] = z_piv_sn
print(f"    c1_SN: a={coeffs_sn_piv[0]:.1f}, b={coeffs_sn_piv[1]:.1f}, z_p={z_piv_sn:.2f}")

# w(z_pivot) from w0wa chain samples
sec5_wp_c0 = w0wa["w0"] + w0wa["wa"] * pivot_z["c0"] / (1 + pivot_z["c0"])
sec5_wp_c1bao = w0wa["w0"] + w0wa["wa"] * pivot_z["c1_bao"] / (1 + pivot_z["c1_bao"])
sec5_wp_c1sn = w0wa["w0"] + w0wa["wa"] * pivot_z["c1_sn"] / (1 + pivot_z["c1_sn"])

print(f"  wp_c0: mean={np.mean(sec5_wp_c0):.3f}±{np.std(sec5_wp_c0):.3f}")
print(f"  wp_c1bao: mean={np.mean(sec5_wp_c1bao):.3f}±{np.std(sec5_wp_c1bao):.3f}")
print(f"  wp_c1sn: mean={np.mean(sec5_wp_c1sn):.3f}±{np.std(sec5_wp_c1sn):.3f}")

# ════════════════════════════════════════════════════════════════
# §16. Save everything
# ════════════════════════════════════════════════════════════════
print("§16. Saving...")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(OUT_PATH,
    # Reference
    REF_H0=REF_H0, REF_ombh2=REF_ombh2, REF_omch2=REF_omch2,
    REF_rdrag=REF_rdrag, z_rec=z_rec, rstar_ref=rstar_ref, Rfid=Rfid,
    N=N, N_sub=N_sub, SEED=SEED,
    # BAO data
    bao_obs_z=bao_obs_z, bao_obs_qty=bao_obs_qty, bao_obs_tracer=bao_obs_tracer,
    bao_ref=bao_ref, bao_data_vector=bao_data_vector,
    bao_covariance=bao_covariance, bao_errors=bao_errors,
    # BAO SVD
    W_bao=W_bao, V_bao=V_bao, S_bao=S_bao,
    c_bao=c_bao, a_bao=a_bao, desi_w=desi_w,
    bao_cov_ratio=bao_cov_ratio,
    V_w0wa=V_w0wa,
    # Chain parameters
    act_H0=act["H0"], act_omch2=act["omch2"], act_ombh2=act["ombh2"],
    act_rdrag=act["rdrag"], act_omh2=act["omh2"], act_theta=act["theta"],
    plk_H0=plk["H0"], plk_omch2=plk["omch2"], plk_ombh2=plk["ombh2"],
    plk_rdrag=plk["rdrag"], plk_omh2=plk["omh2"], plk_theta=plk["theta"],
    # SN data (Union3)
    u3_z=u3b["z"], u3_mu=u3b["mu"], u3_cov=u3b["covariance"],
    z_eff_u3=z_eff_u3,
    # SN data (Pantheon+)
    pp_z=ppb["z"], pp_mu=ppb["mu"], pp_cov=ppb["covariance"],
    z_eff_pp=z_eff_pp,
    # SN data (DES-Dovekie)
    d5_z=d5b["z"], d5_mu=d5b["mu"], d5_cov=d5b["covariance"],
    z_eff_d5=z_eff_d5,
    # SN bin grid and reference mu
    sn_grid=grid, mu_ref_grid=mu_ref_grid,
    # SN SVD results
    svd_u3_W=svd_u3["W"], svd_u3_V=svd_u3["V"], svd_u3_S=svd_u3["S"],
    svd_u3_c=svd_u3["c"], svd_u3_a=svd_u3["a"],
    svd_u3_data_w=svd_u3["data_w"], svd_u3_chain_w=svd_u3["chain_w"],
    svd_u3_delta_mu_data=svd_u3["delta_mu_data"], svd_u3_delta_mu_chain=svd_u3["delta_mu_chain"],
    svd_pp_W=svd_pp["W"], svd_pp_V=svd_pp["V"], svd_pp_S=svd_pp["S"],
    svd_pp_c=svd_pp["c"], svd_pp_a=svd_pp["a"],
    svd_pp_data_w=svd_pp["data_w"], svd_pp_chain_w=svd_pp["chain_w"],
    svd_pp_delta_mu_data=svd_pp["delta_mu_data"], svd_pp_delta_mu_chain=svd_pp["delta_mu_chain"],
    svd_d5_W=svd_d5["W"], svd_d5_V=svd_d5["V"], svd_d5_S=svd_d5["S"],
    svd_d5_c=svd_d5["c"], svd_d5_a=svd_d5["a"],
    svd_d5_data_w=svd_d5["data_w"], svd_d5_chain_w=svd_d5["chain_w"],
    svd_d5_delta_mu_data=svd_d5["delta_mu_data"], svd_d5_delta_mu_chain=svd_d5["delta_mu_chain"],
    # Fine-grid predictions
    z_fine_bao=z_fine_bao, z_fine_sn=z_fine_sn,
    fg_dv_ratio=fg_dv_ratio, fg_dm_ratio=fg_dm_ratio, fg_dh_ratio=fg_dh_ratio,
    fg_dmu=fg_dmu,
    # Planck BAO projections
    plk_c_bao=plk_c_bao,
    # w0wa grid
    w0_1d=w0_1d, wa_1d=wa_1d,
    bao_grid_w=bao_grid_w, sn_grid_w=sn_grid_w,
    # Appendix: parameter scans
    app_scan_om_Drd=app_scan_om_Drd, app_scan_om_DrD=app_scan_om_DrD, app_scan_om_H0=app_scan_om_H0,
    app_scan_ob_Drd=app_scan_ob_Drd, app_scan_ob_DrD=app_scan_ob_DrD, app_scan_ob_H0=app_scan_ob_H0,
    app_scan_th_Drd=app_scan_th_Drd, app_scan_th_DrD=app_scan_th_DrD, app_scan_th_H0=app_scan_th_H0,
    app_om_grid=app_om_grid, app_ob_grid=app_ob_grid, app_th_grid=app_th_grid,
    app_fid_Drd=app_fid_Drd, app_fid_DrD=app_fid_DrD,
    # §11: v₀ fine SN
    v0_fine_sn=v0_fine_sn,
    # §12: Ωk chain
    omk_c0_bao=omk_c0_bao,
    z_fine_omk=z_fine_omk,
    lcdm_DV_ratio=lcdm_DV_ratio, lcdm_DMDH_ratio=lcdm_DMDH_ratio,
    omk_DV_ratio=omk_DV_ratio, omk_DMDH_ratio=omk_DMDH_ratio,
    # §13: Extension c₀ distributions
    ext_lcdm_bao_w=ext_lcdm_bao_w, ext_lcdm_u3_w=ext_lcdm_u3_w,
    ext_alens_bao_w=ext_chain_bao_w['alens'], ext_alens_u3_w=ext_chain_u3_w['alens'],
    ext_bprim_bao_w=ext_chain_bao_w['bprim'], ext_bprim_u3_w=ext_chain_u3_w['bprim'],
    ext_ede_n2_bao_w=ext_chain_bao_w['ede_n2'], ext_ede_n2_u3_w=ext_chain_u3_w['ede_n2'],
    # §14: Planck chain SN fine-grid
    plk_dmu_fine=plk_dmu_fine,
    # §15: Section-5 precomputed data (w0wa chain predictions)
    # BAO data bands
    sec5_z_fine=sec5_z_fine,
    sec5_dv_ref_fine=sec5_dv_ref_fine,
    sec5_dmdh_ref_fine=sec5_dmdh_ref_fine,
    sec5_dv_full=sec5_dv_full,
    sec5_dmdh_full=sec5_dmdh_full,
    sec5_bao_beta_dv=sec5_bao_beta_dv,
    sec5_bao_beta_dmdh=sec5_bao_beta_dmdh,
    sec5_bao_alpha_dv=sec5_bao_alpha_dv,
    sec5_bao_alpha_dmdh=sec5_bao_alpha_dmdh,
    sec5_a_bao_canonical=sec5_a_bao_canonical,
    # SN all-datasets
    sec5_z_fine_sn=sec5_z_fine_sn,
    sec5_mu_ref_fine_sn=sec5_mu_ref_fine_sn,
    sec5_dmu_fine_w0wa=sec5_dmu_fine_w0wa,
    sec5_dmu_fine_lcdm=sec5_dmu_fine_lcdm,
    sec5_canonical_u3_z=sec5_canonical["u3"]["z"],
    sec5_canonical_u3_d=sec5_canonical["u3"]["d"],
    sec5_canonical_u3_err=sec5_canonical["u3"]["err"],
    sec5_canonical_pp_z=sec5_canonical["pp"]["z"],
    sec5_canonical_pp_d=sec5_canonical["pp"]["d"],
    sec5_canonical_pp_err=sec5_canonical["pp"]["err"],
    sec5_canonical_d5_z=sec5_canonical["d5"]["z"],
    sec5_canonical_d5_d=sec5_canonical["d5"]["d"],
    sec5_canonical_d5_err=sec5_canonical["d5"]["err"],
    # SN c1 histogram
    sec5_c1_chain_u3=sec5_c1_chain_u3,
    sec5_c1_scale_u3=sec5_c1_scale["u3"],
    sec5_c1_scale_pp=sec5_c1_scale["pp"],
    sec5_c1_scale_d5=sec5_c1_scale["d5"],
    sec5_a1_u3=sec5_a1["u3"],
    sec5_a1_pp=sec5_a1["pp"],
    sec5_a1_d5=sec5_a1["d5"],
    # w0wa chi2 investigation
    sec5_chi2_c_grid=sec5_chi2_c_grid,
    sec5_chi2_c_mean=sec5_chi2_c_mean,
    sec5_chi2_c_std=sec5_chi2_c_std,
    sec5_w0wa_hist_H=H_chain,
    sec5_w0wa_hist_xedges=xedges_chain,
    sec5_w0wa_hist_yedges=yedges_chain,
    sec5_w0wa_level_68=sec5_w0wa_level_68,
    sec5_w0wa_level_95=sec5_w0wa_level_95,
    # wp histograms
    sec5_wp_c0=sec5_wp_c0,
    sec5_wp_c1bao=sec5_wp_c1bao,
    sec5_wp_c1sn=sec5_wp_c1sn,
)

dt = time.time() - t0
sz = OUT_PATH.stat().st_size / 1e6
print(f"\nDone in {dt:.0f}s. Saved {sz:.1f} MB to {OUT_PATH}")

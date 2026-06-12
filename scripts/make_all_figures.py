import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def imports():
    import sys
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.stats import norm
    from astropy import units as u

    # Project paths
    _here = Path(__file__).resolve().parent if "__file__" in dir() else Path(".").resolve()
    PROJECT_ROOT = _here.parent
    PAPER_FIG_DIR = PROJECT_ROOT / "paper" / "figures"

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from svd_analysis import (
        cosmo_omkw0wa, cosmo_solve_H0_omkw0wa,
        compute_DV_over_rdrag, compute_DM_over_DH,
        compute_distance_modulus,
    )

    # ── Style ──
    SINGLE = 3.4  # aastex single column inches
    DOUBLE = 7.1  # aastex double column inches
    PCOL = {
        "BAO": "#1f77b4", "Union3": "#ff7f0e",
        "Pantheon+": "#2ca02c", "DES-Dovekie": "#d62728",
    }
    ECOL = {
        "LCDM": "#1f77b4", "Alens": "#ff7f0e", "B_PMF": "#2ca02c",
        "EDE_n2": "#9467bd", "Omk": "#d62728",
    }
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    return (
        np, plt, json, GridSpec, norm, u, Path,
        PROJECT_ROOT, PAPER_FIG_DIR,
        cosmo_omkw0wa, cosmo_solve_H0_omkw0wa,
        compute_DV_over_rdrag, compute_DM_over_DH,
        compute_distance_modulus,
        SINGLE, DOUBLE, PCOL, ECOL, matplotlib,
    )


@app.cell
def _(np, json, PROJECT_ROOT, cosmo_omkw0wa):
    """Load precomputed figure data (instant).

    Data produced by: python scripts/precompute_figure_data.py
    """
    print("Loading precomputed figure data...")
    _d = np.load(PROJECT_ROOT / "data" / "figure_data.npz")

    REF = {"H0": float(_d["REF_H0"]), "ombh2": float(_d["REF_ombh2"]),
           "omch2": float(_d["REF_omch2"]), "rdrag": float(_d["REF_rdrag"]),
           "omk": 0.0, "w0": -1.0, "wa": 0.0}
    ref_cosmo = cosmo_omkw0wa(REF["H0"], REF["omch2"], REF["ombh2"],
                              REF["omk"], REF["w0"], REF["wa"])
    N = int(_d["N"])

    # Reconstruct bao_obs list-of-dicts (needed by figure cells that iterate over it)
    bao_dv = _d["bao_data_vector"]
    bao_obs = [{"z": float(z), "quantity": str(q), "tracer": str(t), "value": float(bao_dv[i])}
               for i, (z, q, t) in enumerate(zip(_d["bao_obs_z"], _d["bao_obs_qty"], _d["bao_obs_tracer"]))]

    def _svd_dict(prefix):
        return {"W": _d[f"{prefix}_W"], "V": _d[f"{prefix}_V"],
                "S": _d[f"{prefix}_S"], "c": _d[f"{prefix}_c"],
                "a": _d[f"{prefix}_a"],
                "data_w": _d[f"{prefix}_data_w"],
                "chain_w": _d[f"{prefix}_chain_w"],
                "delta_mu_data": _d[f"{prefix}_delta_mu_data"],
                "delta_mu_chain": _d[f"{prefix}_delta_mu_chain"]}

    D = {
        "REF": REF, "ref_cosmo": ref_cosmo, "N": N,
        "rstar_ref": float(_d["rstar_ref"]), "Rfid": float(_d["Rfid"]),
        "z_rec": float(_d["z_rec"]),
        # BAO
        "bao": {"data_vector": _d["bao_data_vector"], "covariance": _d["bao_covariance"],
                "errors": _d["bao_errors"], "observables": bao_obs},
        "bao_ref": _d["bao_ref"], "bao_obs": bao_obs,
        "W_bao": _d["W_bao"], "V_bao": _d["V_bao"], "S_bao": _d["S_bao"],
        "bao_cov_ratio": _d["bao_cov_ratio"],
        "V_w0wa": _d["V_w0wa"],
        "c_bao": _d["c_bao"], "a_bao": _d["a_bao"], "desi_w": _d["desi_w"],
        # Chains
        "act": {"H0": _d["act_H0"], "omch2": _d["act_omch2"], "ombh2": _d["act_ombh2"],
                "rdrag": _d["act_rdrag"], "omh2": _d["act_omh2"], "theta": _d["act_theta"]},
        "plk": {"H0": _d["plk_H0"], "omch2": _d["plk_omch2"], "ombh2": _d["plk_ombh2"],
                "rdrag": _d["plk_rdrag"], "omh2": _d["plk_omh2"], "theta": _d["plk_theta"]},
        # SN data
        "u3": {"z": _d["u3_z"], "mu": _d["u3_mu"], "covariance": _d["u3_cov"]},
        "pp": {"z": _d["pp_z"], "mu": _d["pp_mu"], "covariance": _d["pp_cov"]},
        "d5": {"z": _d["d5_z"], "mu": _d["d5_mu"], "covariance": _d["d5_cov"]},
        "grid": _d["sn_grid"], "mu_ref": _d["mu_ref_grid"],
        "z_eff_u3": _d["z_eff_u3"], "z_eff_pp": _d["z_eff_pp"], "z_eff_d5": _d["z_eff_d5"],
        # SN SVD
        "svd_u3": _svd_dict("svd_u3"), "svd_pp": _svd_dict("svd_pp"), "svd_d5": _svd_dict("svd_d5"),
        # Fine-grid predictions (for smooth curves)
        "z_fine_bao": _d["z_fine_bao"], "z_fine_sn": _d["z_fine_sn"],
        "fg_dv_ratio": _d["fg_dv_ratio"], "fg_dm_ratio": _d["fg_dm_ratio"],
        "fg_dh_ratio": _d["fg_dh_ratio"], "fg_dmu": _d["fg_dmu"],
        # Planck BAO projections
        "plk_c_bao": _d["plk_c_bao"],
        # w0wa grid
        "w0_1d": _d["w0_1d"], "wa_1d": _d["wa_1d"],
        "bao_grid_w": _d["bao_grid_w"], "sn_grid_w": _d["sn_grid_w"],
        # Appendix scan data
        "app_om_grid": _d["app_om_grid"], "app_ob_grid": _d["app_ob_grid"],
        "app_th_grid": _d["app_th_grid"],
        "app_fid_Drd": float(_d["app_fid_Drd"]), "app_fid_DrD": float(_d["app_fid_DrD"]),
        "app_scan_om_Drd": _d["app_scan_om_Drd"], "app_scan_ob_Drd": _d["app_scan_ob_Drd"],
        "app_scan_th_Drd": _d["app_scan_th_Drd"],
        "app_scan_om_DrD": _d["app_scan_om_DrD"], "app_scan_ob_DrD": _d["app_scan_ob_DrD"],
        "app_scan_th_DrD": _d["app_scan_th_DrD"],
        # v0 fine SN (universal SVD shape on fine z grid)
        "v0_fine_sn": _d["v0_fine_sn"],
        # Omega_k chain
        "omk_c0_bao": _d["omk_c0_bao"],
        "z_fine_omk": _d["z_fine_omk"],
        "lcdm_DV_ratio": _d["lcdm_DV_ratio"], "lcdm_DMDH_ratio": _d["lcdm_DMDH_ratio"],
        "omk_DV_ratio": _d["omk_DV_ratio"], "omk_DMDH_ratio": _d["omk_DMDH_ratio"],
        # Extension c0 distributions (whitened chain predictions)
        "ext_lcdm_bao_w": _d["ext_lcdm_bao_w"], "ext_lcdm_u3_w": _d["ext_lcdm_u3_w"],
        "ext_alens_bao_w": _d["ext_alens_bao_w"], "ext_alens_u3_w": _d["ext_alens_u3_w"],
        "ext_bprim_bao_w": _d["ext_bprim_bao_w"], "ext_bprim_u3_w": _d["ext_bprim_u3_w"],
        "ext_ede_n2_bao_w": _d["ext_ede_n2_bao_w"], "ext_ede_n2_u3_w": _d["ext_ede_n2_u3_w"],
        # Planck chain SN fine-grid predictions
        "plk_dmu_fine": _d["plk_dmu_fine"],
    }

    # ── Load section 5 precomputed data (w0wa chain predictions) ──
    # These keys are added by precompute_figure_data.py for the 7 consolidated figures.
    _sec5_keys = [
        # BAO data bands
        "sec5_z_fine", "sec5_dv_ref_fine", "sec5_dmdh_ref_fine",
        "sec5_dv_full", "sec5_dmdh_full",
        "sec5_bao_beta_dv", "sec5_bao_beta_dmdh",
        "sec5_bao_alpha_dv", "sec5_bao_alpha_dmdh",
        "sec5_a_bao_canonical",
        # SN all-datasets
        "sec5_z_fine_sn", "sec5_mu_ref_fine_sn",
        "sec5_dmu_fine_w0wa", "sec5_dmu_fine_lcdm",
        "sec5_canonical_u3_z", "sec5_canonical_u3_d", "sec5_canonical_u3_err",
        "sec5_canonical_pp_z", "sec5_canonical_pp_d", "sec5_canonical_pp_err",
        "sec5_canonical_d5_z", "sec5_canonical_d5_d", "sec5_canonical_d5_err",
        # SN c1 histogram
        "sec5_c1_chain_u3", "sec5_c1_scale_u3", "sec5_c1_scale_pp", "sec5_c1_scale_d5",
        "sec5_a1_u3", "sec5_a1_pp", "sec5_a1_d5",
        # w0wa chi2 investigation
        "sec5_chi2_c_grid", "sec5_chi2_c_mean", "sec5_chi2_c_std",
        "sec5_w0wa_hist_H", "sec5_w0wa_hist_xedges", "sec5_w0wa_hist_yedges",
        "sec5_w0wa_level_68", "sec5_w0wa_level_95",
        # wp histograms
        "sec5_wp_c0", "sec5_wp_c1bao", "sec5_wp_c1sn",
    ]
    for key in _sec5_keys:
        if key in _d:
            D[key] = _d[key]

    # Load paper_numbers.json for pivot fits
    _pn_path = PROJECT_ROOT / "data" / "paper_numbers.json"
    if _pn_path.exists():
        with open(_pn_path) as f:
            D["paper_numbers"] = json.load(f)
    else:
        D["paper_numbers"] = {}

    print(f"Done. BAO sigma_c0={D['c_bao'][:,0].std():.3f}, "
          f"SN sigma_c0={D['svd_d5']['c'][:,0].std():.3f}")
    return D,


@app.cell
def _(np, plt, D, SINGLE, compute_DV_over_rdrag):
    """Fig 1: DV/rs DESI vs ACT LCDM."""
    def _make():
        fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
        z_fine = D["z_fine_bao"]
        ref = D["REF"]

        # -- Model band from precomputed ACT chain DV/rs ratios --
        ratios = D["fg_dv_ratio"]
        m, s = ratios.mean(0), ratios.std(0)
        ax.fill_between(z_fine, m - s, m + s, color="#1f77b4", alpha=0.25,
                         label="ACT $\\Lambda$CDM $1\\sigma$")
        ax.plot(z_fine, m, color="#1f77b4", lw=0.8)

        # -- Compute DV data points from BAO --
        bao_obs = D["bao_obs"]
        bao_errors = D["bao"]["errors"]
        unique_z = np.unique([obs["z"] for obs in bao_obs])

        z_dv, dv_vals, dv_errs = [], [], []
        for z in unique_z:
            obs_at_z = [o for o in bao_obs if abs(o["z"] - z) < 0.01]
            if len(obs_at_z) == 1 and obs_at_z[0]["quantity"] == "DV_over_rs":
                z_dv.append(z)
                dv_vals.append(obs_at_z[0]["value"])
                idx = bao_obs.index(obs_at_z[0])
                dv_errs.append(bao_errors[idx])
            else:
                dm_obs = [o for o in obs_at_z if o["quantity"] == "DM_over_rs"]
                dh_obs = [o for o in obs_at_z if o["quantity"] == "DH_over_rs"]
                if dm_obs and dh_obs:
                    dm = dm_obs[0]["value"]
                    dh = dh_obs[0]["value"]
                    dv = (z * dm**2 * dh)**(1.0/3.0)
                    z_dv.append(z)
                    dv_vals.append(dv)
                    idx_dm = bao_obs.index(dm_obs[0])
                    idx_dh = bao_obs.index(dh_obs[0])
                    rel_err_dm = bao_errors[idx_dm] / dm
                    rel_err_dh = bao_errors[idx_dh] / dh
                    rel_err_dv = np.sqrt((2*rel_err_dm/3)**2 + (rel_err_dh/3)**2)
                    dv_errs.append(dv * rel_err_dv)

        z_dv = np.array(z_dv)
        dv_vals = np.array(dv_vals)
        dv_errs = np.array(dv_errs)

        dv_ref_at_z = compute_DV_over_rdrag(D["ref_cosmo"], z_dv, ref["rdrag"])
        data_ratio = dv_vals / dv_ref_at_z
        data_ratio_err = dv_errs / dv_ref_at_z

        ax.errorbar(z_dv, data_ratio, yerr=data_ratio_err,
                    fmt="ko", ms=4, capsize=2, label="DESI DR2")

        ax.axhline(1, color="gray", ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 2.8)
        ax.set_xlabel("$z$")
        ax.set_ylabel("$(D_V/r_s)\\,/\\,(D_V/r_s)_{\\rm ref}$")
        ax.legend(fontsize=6, loc="lower left")
        ax.grid(True, alpha=0.3)

        ax.text(0.98, 0.02, "DESI DR2 (arXiv:2503.14738)",
                transform=ax.transAxes, fontsize=5, ha="right", va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.3))

        fig.tight_layout()
        return fig

    fig_dv_ratios = _make()
    return fig_dv_ratios,


@app.cell
def _(np, plt, D, SINGLE, compute_DM_over_DH):
    """Fig 2: DM/DH ratio plot."""
    def _make():
        fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
        z_fine = D["z_fine_bao"]

        ratios = D["fg_dm_ratio"] / D["fg_dh_ratio"]
        m, s = ratios.mean(0), ratios.std(0)
        ax.fill_between(z_fine, m - s, m + s, color="#1f77b4", alpha=0.25,
                         label="ACT $\\Lambda$CDM $1\\sigma$")
        ax.plot(z_fine, m, color="#1f77b4", lw=0.8)

        bao_obs = D["bao_obs"]
        bao_errors = D["bao"]["errors"]
        unique_z = np.unique([obs["z"] for obs in bao_obs])

        z_dmdh, dmdh_vals, dmdh_errs = [], [], []
        for z in unique_z:
            obs_at_z = [o for o in bao_obs if abs(o["z"] - z) < 0.01]
            dm_obs = [o for o in obs_at_z if o["quantity"] == "DM_over_rs"]
            dh_obs = [o for o in obs_at_z if o["quantity"] == "DH_over_rs"]
            if dm_obs and dh_obs:
                dm = dm_obs[0]["value"]
                dh = dh_obs[0]["value"]
                dmdh = dm / dh
                z_dmdh.append(z)
                dmdh_vals.append(dmdh)
                idx_dm = bao_obs.index(dm_obs[0])
                idx_dh = bao_obs.index(dh_obs[0])
                err_dm = bao_errors[idx_dm]
                err_dh = bao_errors[idx_dh]
                rel_err = np.sqrt((err_dm / dm)**2 + (err_dh / dh)**2)
                dmdh_errs.append(dmdh * rel_err)

        z_dmdh = np.array(z_dmdh)
        dmdh_vals = np.array(dmdh_vals)
        dmdh_errs = np.array(dmdh_errs)

        ref_dmdh_at_z = compute_DM_over_DH(D["ref_cosmo"], z_dmdh)
        data_ratio = dmdh_vals / ref_dmdh_at_z
        data_ratio_err = dmdh_errs / ref_dmdh_at_z

        ax.errorbar(z_dmdh, data_ratio, yerr=data_ratio_err,
                    fmt="ko", ms=4, capsize=2, label="DESI DR2")

        ax.axhline(1, color="gray", ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 2.8)
        ax.set_xlabel("$z$")
        ax.set_ylabel("$(D_M/D_H)\\,/\\,(D_M/D_H)_{\\rm ref}$")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        ax.text(0.98, 0.02, "DESI DR2 (arXiv:2503.14738)",
                transform=ax.transAxes, fontsize=5, ha="right", va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.3))

        fig.tight_layout()
        return fig

    fig_dmdh_ratios = _make()
    return fig_dmdh_ratios,


@app.cell
def _(np, plt, D, SINGLE, PCOL, compute_distance_modulus):
    """Fig 3: SN data overview -- three datasets on common log-z axis."""
    def _make():
        fig, ax = plt.subplots(figsize=(SINGLE, 2.5))

        dm_chain_u3 = D["svd_u3"]["delta_mu_chain"]
        chain_mean_u3 = dm_chain_u3.mean(axis=0)
        dm_residual = dm_chain_u3 - chain_mean_u3[None, :]
        band_std_u3 = dm_residual.std(axis=0)
        z_u3 = D["z_eff_u3"]
        z_all = np.unique(np.concatenate([z_u3, D["z_eff_pp"], D["z_eff_d5"]]))
        z_all.sort()
        band_std = np.interp(z_all, z_u3, band_std_u3)
        ax.fill_between(z_all, -band_std, band_std,
                         color="gray", alpha=0.2, label="ACT $\\Lambda$CDM $1\\sigma$")

        canonical_sn = {}
        for name, sn_key, svd_key, zeff_key, mk in [
            ("Union3",      "u3", "svd_u3", "z_eff_u3", "o"),
            ("Pantheon+",   "pp", "svd_pp", "z_eff_pp", "s"),
            ("DES-Dovekie", "d5", "svd_d5", "z_eff_d5", "^"),
        ]:
            sn = D[sn_key]
            z_eff = D[zeff_key]
            n = len(sn["z"])
            cov = sn["covariance"]

            mu_ref_at_zeff = compute_distance_modulus(D["ref_cosmo"], z_eff)
            chain_mean = D[svd_key]["delta_mu_chain"].mean(axis=0)
            r = (sn["mu"] - mu_ref_at_zeff) - chain_mean

            P_perp = np.eye(n) - np.ones((n, n)) / n
            cov_proj = P_perp @ cov @ P_perp.T
            sigma_j = np.sqrt(np.maximum(np.diag(cov_proj), 0.0))

            w_inv = np.where(sigma_j > 1e-10, 1.0 / sigma_j**2, 0.0)
            M_best = np.sum(r * w_inv) / np.sum(w_inv)
            d_j = r - M_best

            canonical_sn[name] = {"z": z_eff, "d": d_j, "err": sigma_j}

            ax.errorbar(z_eff, d_j, yerr=sigma_j, fmt=mk,
                         color=PCOL[name], ms=3, capsize=1.5, alpha=0.7, label=name)

        D["canonical_sn"] = canonical_sn

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("$z$")
        ax.set_ylabel("$\\Delta\\mu$ (mag)")
        ax.legend(fontsize=6, loc="upper left")
        ax.set_xscale("log")
        ax.set_ylim(-0.35, 0.35)
        fig.tight_layout()
        return fig

    fig_sn_datasets = _make()
    return fig_sn_datasets,


@app.cell
def _(np, plt, D, DOUBLE, PCOL, compute_DV_over_rdrag, compute_distance_modulus):
    """Fig 4: c0 family of curves in observed space (BAO DV/rs + SN dmu)."""
    def _make():
        ref = D["REF"]
        N_sub = D["fg_dv_ratio"].shape[0]
        c0_bao = D["c_bao"][:N_sub, 0]
        c0_var = c0_bao.var()
        c0_sn = D["svd_d5"]["c"][:N_sub, 0]
        c0_sn_var = c0_sn.var()

        z_fine = D["z_fine_bao"]
        ratios_fine = D["fg_dv_ratio"]

        alpha_bao = ratios_fine.mean(0)
        beta0_bao = np.array([np.cov(c0_bao, ratios_fine[:, j])[0, 1] / c0_var
                               for j in range(len(z_fine))])

        z_sn = D["z_fine_sn"]
        dmu_fine = D["fg_dmu"]
        alpha_sn = dmu_fine.mean(0)
        beta0_sn = np.array([np.cov(c0_sn, dmu_fine[:, j])[0, 1] / c0_sn_var
                              for j in range(len(z_sn))])

        offsets = [-5, -2.5, 0, 2.5, 5]
        cmap = plt.cm.coolwarm_r
        colors = [cmap(0.05), cmap(0.25), "0.3", cmap(0.75), cmap(0.95)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.8))

        for off, col in zip(offsets, colors):
            curve = alpha_bao + off * beta0_bao
            lw = 1.8 if off == 0 else 1.0
            ax1.plot(z_fine, curve, color=col, lw=lw,
                     label=f"$c_0 = {off:+.1f}$")

        bao_obs = D["bao_obs"]
        bao_errors = D["bao"]["errors"]
        unique_z = np.unique([obs["z"] for obs in bao_obs])
        z_dv, dv_vals, dv_errs = [], [], []
        for z in unique_z:
            obs_at_z = [o for o in bao_obs if abs(o["z"] - z) < 0.01]
            if len(obs_at_z) == 1 and obs_at_z[0]["quantity"] == "DV_over_rs":
                z_dv.append(z)
                dv_vals.append(obs_at_z[0]["value"])
                idx = bao_obs.index(obs_at_z[0])
                dv_errs.append(bao_errors[idx])
            else:
                dm_obs = [o for o in obs_at_z if o["quantity"] == "DM_over_rs"]
                dh_obs = [o for o in obs_at_z if o["quantity"] == "DH_over_rs"]
                if dm_obs and dh_obs:
                    dm, dh = dm_obs[0]["value"], dh_obs[0]["value"]
                    dv = (z * dm**2 * dh)**(1./3.)
                    z_dv.append(z)
                    dv_vals.append(dv)
                    idx_dm = bao_obs.index(dm_obs[0])
                    idx_dh = bao_obs.index(dh_obs[0])
                    rel_dm = bao_errors[idx_dm] / dm
                    rel_dh = bao_errors[idx_dh] / dh
                    dv_errs.append(dv * np.sqrt((2*rel_dm/3)**2 + (rel_dh/3)**2))
        z_dv = np.array(z_dv)
        dv_ref_z = compute_DV_over_rdrag(D["ref_cosmo"], z_dv, ref["rdrag"])
        ax1.errorbar(z_dv, np.array(dv_vals)/dv_ref_z,
                     yerr=np.array(dv_errs)/dv_ref_z,
                     fmt="ko", ms=4, capsize=2, zorder=10, label="DESI DR2")
        ax1.axhline(1, color="gray", ls="--", alpha=0.4)
        ax1.set_xscale("log")
        ax1.set_xlim(0.1, 2.8)
        ax1.set_xlabel("$z$")
        ax1.set_ylabel("$(D_V/r_s)\\,/\\,(D_V/r_s)_{\\rm ref}$")
        ax1.legend(fontsize=5, loc="lower left", ncol=2)
        ax1.grid(True, alpha=0.2)
        ax1.set_title("BAO", fontsize=9)

        sn_z_eff = D["z_eff_d5"]
        for off, col in zip(offsets, colors):
            curve = alpha_sn + off * beta0_sn
            curve_at_bins = np.interp(sn_z_eff, z_sn, curve)
            mean_at_bins = curve_at_bins.mean()
            curve_ms = curve - mean_at_bins
            lw = 1.8 if off == 0 else 1.0
            ax2.plot(z_sn, curve_ms, color=col, lw=lw,
                     label=f"$c_0 = {off:+.1f}$")

        sn = D["d5"]
        z_eff = D["z_eff_d5"]
        mu_r = compute_distance_modulus(D["ref_cosmo"], z_eff)
        dmu_data = sn["mu"] - mu_r
        n_sn = len(sn["z"])
        P = np.eye(n_sn) - np.ones((n_sn, n_sn)) / n_sn
        C_proj = P @ sn["covariance"] @ P.T
        sigma = np.sqrt(np.maximum(np.diag(C_proj), 0.0))
        data_ms = dmu_data - dmu_data.mean()
        sort_idx = np.argsort(sn["z"])
        ax2.errorbar(sn["z"][sort_idx], data_ms[sort_idx], yerr=sigma[sort_idx],
                     fmt="o", color=PCOL["DES-Dovekie"], ms=3, capsize=1.5, alpha=0.8,
                     label="DES-Dovekie", zorder=10)
        ax2.axhline(0, color="gray", ls="--", alpha=0.4)
        ax2.set_xscale("log")
        ax2.set_ylim(-0.15, 0.07)
        ax2.set_xlabel("$z$")
        ax2.set_ylabel(r"$\Delta\mu$ (mean-subtracted, mag)")
        ax2.legend(fontsize=5, loc="lower left", ncol=2)
        ax2.grid(True, alpha=0.2)
        ax2.set_title("SN (DES-Dovekie)", fontsize=9)

        fig.tight_layout()
        return fig

    fig_c0_families = _make()
    return fig_c0_families,


@app.cell
def _(np, plt, norm, D, SINGLE, PCOL):
    """Fig 5: Per-probe c0 Gaussians."""
    def _make():
        probes = [
            ("BAO",       D["c_bao"][:, 0],          D["a_bao"][0]),
            ("Union3",    D["svd_u3"]["c"][:, 0],     D["svd_u3"]["a"][0]),
            ("Pantheon+", D["svd_pp"]["c"][:, 0],     D["svd_pp"]["a"][0]),
            ("DES-Dovekie",    D["svd_d5"]["c"][:, 0],     D["svd_d5"]["a"][0]),
        ]

        fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
        x = np.linspace(-6, 8, 500)

        for name, c0_chain, a0 in probes:
            c0_mean = np.mean(c0_chain)
            c0_std  = np.std(c0_chain)
            combined_std = np.sqrt(1.0 + c0_std**2)
            tension = (a0 - c0_mean) / combined_std

            a0_shifted = a0 - c0_mean

            ax.fill_between(
                x, norm.pdf(x, 0, c0_std), alpha=0.15, color=PCOL[name],
            )
            ax.plot(
                x, norm.pdf(x, a0_shifted, 1.0),
                color=PCOL[name], ls="--", lw=1.5,
                label=f"{name} ({tension:+.1f}$\\sigma$)",
            )

        ax.axvline(0, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.set_xlabel("$c_0 - \\langle c_0 \\rangle_{\\rm chain}$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlim(-5, 8)
        fig.tight_layout()
        return fig

    fig_c0_gaussians = _make()
    return fig_c0_gaussians,


@app.cell
def _(np, plt, D, SINGLE, PCOL):
    """Fig 6: Omega_m h^2 money plot."""
    def _make():
        omegamh2_ref = D["REF"]["omch2"] + D["REF"]["ombh2"]
        omegamh2_chain = D["act"]["omh2"]

        probe_defs = [
            ("BAO",       D["c_bao"][:, 0],          D["a_bao"][0]),
            ("Union3",    D["svd_u3"]["c"][:, 0],     D["svd_u3"]["a"][0]),
            ("Pantheon+", D["svd_pp"]["c"][:, 0],     D["svd_pp"]["a"][0]),
            ("DES-Dovekie",    D["svd_d5"]["c"][:, 0],     D["svd_d5"]["a"][0]),
        ]

        z_omh2 = (omegamh2_chain - omegamh2_chain.mean()) / omegamh2_chain.std()
        z_ombh2 = (D["act"]["ombh2"] - D["act"]["ombh2"].mean()) / D["act"]["ombh2"].std()
        z_theta = (D["act"]["theta"] - D["act"]["theta"].mean()) / D["act"]["theta"].std()
        X_std = np.column_stack([z_omh2, z_ombh2, z_theta])
        Sigma_23 = np.cov(np.column_stack([z_ombh2, z_theta]).T)
        sigma_omh2 = float(omegamh2_chain.std())

        probe_results = []
        for name, c0_chain, a0 in probe_defs:
            c0_mean = float(np.mean(c0_chain))
            c0_std  = float(np.std(c0_chain))

            zc = (c0_chain - c0_mean) / c0_std
            beta, _, _, _ = np.linalg.lstsq(X_std, zc, rcond=None)

            conversion = sigma_omh2 / (beta[0] * c0_std)
            omegamh2_meas = float(omegamh2_chain.mean()) + conversion * (a0 - c0_mean)

            sigma_meas = abs(conversion)

            var_marg_z = (1 / beta[0]**2) * (
                beta[1]**2 * Sigma_23[0, 0] + beta[2]**2 * Sigma_23[1, 1] +
                2 * beta[1] * beta[2] * Sigma_23[0, 1])
            sigma_marg = float(np.sqrt(var_marg_z)) * sigma_omh2
            sigma_total = float(np.sqrt(sigma_meas**2 + sigma_marg**2))

            probe_results.append({
                "name": name, "omegamh2": omegamh2_meas,
                "sigma_meas": sigma_meas, "sigma_marg": sigma_marg,
                "sigma_total": sigma_total,
            })

        fig, ax = plt.subplots(figsize=(SINGLE, 2.4))

        n_probes = len(probe_results)
        y_probes = list(range(n_probes - 1, -1, -1))

        act_omh2_mean = np.mean(omegamh2_chain)
        act_omh2_std  = np.std(omegamh2_chain)
        ax.axvspan(act_omh2_mean - act_omh2_std, act_omh2_mean + act_omh2_std,
                   color="gray", alpha=0.15, zorder=0)
        ax.axvline(act_omh2_mean, color="gray", ls="--", lw=1.0, alpha=0.7,
                   label=f"ACT $\\Lambda$CDM")

        for i, p in enumerate(probe_results):
            y = y_probes[i]
            c = PCOL[p["name"]]
            ax.errorbar(p["omegamh2"], y, xerr=p["sigma_total"],
                        fmt="o", color=c, ms=6, capsize=4, lw=1.5)

        ax.set_yticks(y_probes)
        ax.set_yticklabels([p["name"] for p in probe_results],
                           fontsize=8)
        ax.set_xlabel("$\\Omega_m h^2$")
        all_omh2 = [p["omegamh2"] for p in probe_results]
        all_errs = [p["sigma_total"] for p in probe_results]
        max_dev = max(abs(v + e - act_omh2_mean) for v, e in zip(all_omh2, all_errs))
        max_dev = max(max_dev, max(abs(v - e - act_omh2_mean) for v, e in zip(all_omh2, all_errs)))
        margin = max_dev * 1.15
        ax.set_xlim(act_omh2_mean - margin, act_omh2_mean + margin)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.15, axis="x")
        fig.tight_layout()
        return fig

    fig_omegamh2_money = _make()
    return fig_omegamh2_money,


@app.cell
def _(np, plt, D, SINGLE, u, cosmo_omkw0wa, cosmo_solve_H0_omkw0wa):
    """Fig 7: H(z)/H_ref(z) for varying omega_m h^2 at fixed theta_star.

    This figure is cheap -- it only computes ~5 astropy cosmologies.
    No precomputed data needed.
    """
    def _make():
        REF = D["REF"]
        OMH2_REF = REF["omch2"] + REF["ombh2"]

        ref_cosmo = cosmo_omkw0wa(REF["H0"], REF["omch2"], REF["ombh2"], 0.0, -1.0, 0.0)

        # theta_star anchoring
        RSTAR = D["rstar_ref"]
        Z_REC = D["z_rec"]
        RFID = D["Rfid"]

        # Redshift grid: log-spaced from 0.01 to 1100
        z = np.geomspace(0.01, 1100, 500)
        H_ref = ref_cosmo.H(z).value

        # omega_m h^2 values to plot (match the Planck reference omega_m h^2 used
        # for the H_ref normalization, so the central curve is the reference; the
        # BAO-implied curve at -3 sigma reproduces the value quoted in the caption).
        ACT_OMH2 = 0.1424    # ACT chain mean / Planck reference omega_m h^2
        ACT_SIGMA = 0.0011    # ACT chain sigma

        omh2_values = [
            (ACT_OMH2 - 3.0 * ACT_SIGMA, "BAO-implied", "#d62728", "-", 2.0),
            (ACT_OMH2 - 2.0 * ACT_SIGMA, r"ACT $-2\sigma$", "#ff7f0e", "--", 1.2),
            (ACT_OMH2 - 1.0 * ACT_SIGMA, r"ACT $-1\sigma$", "#bcbd22", "--", 1.2),
            (ACT_OMH2,                    "ACT best-fit (ref)", "black", "-", 1.5),
            (ACT_OMH2 + 1.0 * ACT_SIGMA, r"ACT $+1\sigma$", "#17becf", "--", 1.2),
            (ACT_OMH2 + 2.0 * ACT_SIGMA, r"ACT $+2\sigma$", "#1f77b4", "--", 1.2),
        ]

        results = []
        for omh2, label, color, ls, lw in omh2_values:
            omch2 = omh2 - REF["ombh2"]

            if abs(omh2 - ACT_OMH2) < 1e-6:
                H_ratio = np.ones_like(z)
                H0_solved = ref_cosmo.H0.value
            else:
                cosmo = cosmo_solve_H0_omkw0wa(
                    omch2, REF["ombh2"], 0.0, -1.0, 0.0,
                    RSTAR, RFID, Z_REC,
                )
                H0_solved = cosmo.H0.value
                H_ratio = cosmo.H(z).value / H_ref

            results.append((z, H_ratio, label, color, ls, lw, omh2, H0_solved))

        fig, ax = plt.subplots(figsize=(7.1, 3.2))

        for z_arr, H_ratio, label, color, ls, lw, omh2, H0 in results:
            ax.plot(z_arr, H_ratio, color=color, ls=ls, lw=lw,
                    label=rf"{label}: $\omega_m = {omh2:.4f}$, $H_0 = {H0:.1f}$")

        ax.axvspan(0.3, 2.3, color="gold", alpha=0.07, label="DESI BAO range")
        ax.axvline(1090, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.text(1090, ax.get_ylim()[0] + 0.001, " $z_\\mathrm{rec}$", fontsize=8,
                color="gray", va="bottom")

        ax.axhline(1, color="gray", ls="--", lw=0.5, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Redshift $z$")
        ax.set_ylabel(r"$H(z)\,/\,H_\mathrm{ref}(z)$")
        ax.set_xlim(0.01, 1500)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        return fig

    fig_hz_omh2_variation = _make()
    return fig_hz_omh2_variation,


@app.cell
def _(np, plt, D, DOUBLE):
    """Fig 8: w0wa contours -- three-mode contours in (w0, wa) plane."""
    def _make():
        n_bao = len(D["bao_obs"])

        w0_1d = D["w0_1d"]
        wa_1d = D["wa_1d"]
        Ng = len(w0_1d)

        V0_bao = D["V_w0wa"][:, 0]
        V1_bao = D["V_w0wa"][:, 1]

        V0_u3 = D["svd_u3"]["V"][:, 0]

        sn_grid_w = D["sn_grid_w"]
        good_sn = np.all(np.isfinite(sn_grid_w), axis=1)
        sn_grid_good = sn_grid_w[good_sn]
        c0_proj_sn = sn_grid_good @ V0_u3
        filtered_sn = sn_grid_good - np.outer(c0_proj_sn, V0_u3)
        _, _, Vt_filt_sn = np.linalg.svd(filtered_sn, full_matrices=False)
        V1_u3 = Vt_filt_sn[0]

        bao_grid_w = D["bao_grid_w"]

        c0_bao_flat = bao_grid_w @ V0_bao
        c0_proj_bao = np.outer(c0_bao_flat, V0_bao)
        c1_bao_flat = (bao_grid_w - c0_proj_bao) @ V1_bao

        c0_proj_u3_flat = sn_grid_w @ V0_u3
        c1_u3_flat = (sn_grid_w - np.outer(c0_proj_u3_flat, V0_u3)) @ V1_u3

        c0_bao_map = c0_bao_flat.reshape(Ng, Ng).T
        c1_bao_map = c1_bao_flat.reshape(Ng, Ng).T
        c1_u3_map = c1_u3_flat.reshape(Ng, Ng).T

        c0_bao_map[~np.isfinite(c0_bao_map)] = np.nan
        c1_bao_map[~np.isfinite(c1_bao_map)] = np.nan
        c1_u3_map[~np.isfinite(c1_u3_map)] = np.nan

        step = Ng // 50 if Ng >= 50 else 1
        w0_vals = w0_1d[::step]
        wa_vals = wa_1d[::step]
        W0, WA = np.meshgrid(w0_vals, wa_vals, indexing="ij")
        c0_bao_map = c0_bao_map[::step, ::step]
        c1_bao_map = c1_bao_map[::step, ::step]
        c1_u3_map = c1_u3_map[::step, ::step]

        levels_c0 = list(range(-40, 41, 10))
        levels_c1b = list(range(-60, 61, 5))
        levels_c1u = list(range(-3, 4))

        fig, ax = plt.subplots(figsize=(DOUBLE * 0.55, DOUBLE * 0.55))

        mode_colors = {
            'c0': '#2ca02c',
            'c1_bao': '#d62728',
            'c1_u3': '#1f77b4',
        }

        # Deterministic label anchors for the c0 (green) contours: one label per
        # level where the contour crosses a rail line ~perpendicular to the (steep)
        # green contours, away from the upper-left legend (auto-placement otherwise
        # hid the green labels under the legend).
        from scipy.interpolate import RegularGridInterpolator

        def _rail_label_points(cmap, levels, slope, intercept,
                               w0_lo=-1.48, w0_hi=-0.42, wa_lo=-1.95, wa_hi=0.95):
            interp = RegularGridInterpolator((w0_vals, wa_vals), cmap,
                                             bounds_error=False, fill_value=np.nan)
            w0r = np.linspace(w0_lo, w0_hi, 600)
            war = slope * w0r + intercept
            m = (war >= wa_lo) & (war <= wa_hi)
            w0r, war = w0r[m], war[m]
            cr = interp(np.column_stack([w0r, war]))
            pts = []
            for L in levels:
                d = cr - L
                ok = np.isfinite(d)
                sc = np.where(ok[:-1] & ok[1:] & (np.sign(d[:-1]) != np.sign(d[1:])))[0]
                if len(sc):
                    i0 = sc[len(sc) // 2]
                    pts.append((float(w0r[i0]), float(war[i0])))
            return pts

        cs0 = ax.contour(W0, WA, c0_bao_map,
                          levels=levels_c0,
                          colors=[mode_colors['c0']], linestyles=['-'],
                          linewidths=1.5, alpha=0.8)
        _man_c0 = _rail_label_points(c0_bao_map, levels_c0, slope=2.42, intercept=1.69)
        ax.clabel(cs0, inline=True, fontsize=8, fmt='%.0f', manual=_man_c0)
        ax.plot([], [], color=mode_colors['c0'], ls='-', lw=2,
                label=r'$c_0$ (universal)')

        cs1b = ax.contour(W0, WA, c1_bao_map,
                           levels=levels_c1b,
                           colors=[mode_colors['c1_bao']], linestyles=['--'],
                           linewidths=1.5, alpha=0.8)
        ax.clabel(cs1b, inline=True, fontsize=8, fmt='%.0f')
        ax.plot([], [], color=mode_colors['c1_bao'], ls='--', lw=2,
                label=r'BAO $c_1$(w0wa)')

        cs1u = ax.contour(W0, WA, c1_u3_map,
                           levels=levels_c1u,
                           colors=[mode_colors['c1_u3']], linestyles=['-.'],
                           linewidths=1.5, alpha=0.8)
        ax.clabel(cs1u, inline=True, fontsize=8, fmt='%.0f')
        ax.plot([], [], color=mode_colors['c1_u3'], ls='-.', lw=2,
                label=r'SN $c_1$(w0wa, Union3)')

        w0_line = np.linspace(-1.5, -0.4, 200)
        z_refs = [0.0, 0.71, 0.46, 1.0]
        for zr in z_refs:
            if zr == 0:
                ax.axvline(-1, color='gray', ls='--', lw=0.8, alpha=0.4)
                ax.text(-0.98, 0.85, '$z{=}0$', fontsize=7, color='gray', alpha=0.6)
            else:
                fac = 1.0 - 1.0 / (1.0 + zr)
                wa_line = -(w0_line + 1.0) / fac
                ax.plot(w0_line, wa_line, '--', color='gray', lw=0.8, alpha=0.4)
                idx_label = min(len(w0_line) - 1,
                                np.searchsorted(w0_line, -0.5))
                wa_label = -(w0_line[idx_label] + 1.0) / fac
                if -2.0 <= wa_label <= 1.0:
                    ax.text(w0_line[idx_label] + 0.02, wa_label,
                            '$z\\!=\\!' + str(zr) + '$', fontsize=7, color='gray', alpha=0.6)

        ax.plot(-1, 0, 'k*', ms=12, zorder=10, label=r'$\Lambda$CDM')
        ax.set_xlabel('$w_0$')
        ax.set_ylabel('$w_a$')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlim(-1.5, -0.4)
        ax.set_ylim(-2, 1)
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        return fig

    fig_w0wa_contours = _make()
    return fig_w0wa_contours,


@app.cell
def _(np, plt, D, DOUBLE):
    """Fig 9+10: BAO data-centered convergence bands (DV and DM/DH).

    Uses precomputed w0wa chain regression coefficients from sec5_* keys.
    """
    def _make():
        # Precomputed from section5_figures.py via precompute_figure_data.py
        z_fine = D["sec5_z_fine"]
        dv_full = D["sec5_dv_full"]           # (N, nz) DV ratios
        dmdh_full = D["sec5_dmdh_full"]       # (N, nz) DM/DH ratios
        beta_dv = D["sec5_bao_beta_dv"]       # (n_bao, nz)
        beta_dmdh = D["sec5_bao_beta_dmdh"]   # (n_bao, nz)
        alpha_dv = D["sec5_bao_alpha_dv"]     # (nz,)
        alpha_dmdh = D["sec5_bao_alpha_dmdh"] # (nz,)
        a_bao_all = D["sec5_a_bao_canonical"] # (n_bao,)
        n_bao = len(a_bao_all)

        # Compute BAO data points for overlay
        bao_obs = D["bao_obs"]
        bao_errors = D["bao"]["errors"]
        ref_cosmo = D["ref_cosmo"]
        REF = D["REF"]
        unique_z = np.unique([obs["z"] for obs in bao_obs])

        from svd_analysis import compute_DV_over_rdrag, compute_DM_over_DH

        z_dv, dv_vals, dv_errs = [], [], []
        for zz in unique_z:
            obs_at_z = [o for o in bao_obs if abs(o["z"] - zz) < 0.01]
            if len(obs_at_z) == 1 and obs_at_z[0]["quantity"] == "DV_over_rs":
                z_dv.append(zz)
                dv_vals.append(obs_at_z[0]["value"])
                idx_o = bao_obs.index(obs_at_z[0])
                dv_errs.append(bao_errors[idx_o])
            else:
                dm_obs = [o for o in obs_at_z if o["quantity"] == "DM_over_rs"]
                dh_obs = [o for o in obs_at_z if o["quantity"] == "DH_over_rs"]
                if dm_obs and dh_obs:
                    dm, dh = dm_obs[0]["value"], dh_obs[0]["value"]
                    dv = (zz * dm**2 * dh)**(1.0/3.0)
                    z_dv.append(zz)
                    dv_vals.append(dv)
                    idx_dm = bao_obs.index(dm_obs[0])
                    idx_dh = bao_obs.index(dh_obs[0])
                    rel_dm = bao_errors[idx_dm] / dm
                    rel_dh = bao_errors[idx_dh] / dh
                    dv_errs.append(dv * np.sqrt((2*rel_dm/3)**2 + (rel_dh/3)**2))
        z_dv = np.array(z_dv)
        dv_data_ratio = np.array(dv_vals) / compute_DV_over_rdrag(ref_cosmo, z_dv, REF["rdrag"])
        dv_data_err = np.array(dv_errs) / compute_DV_over_rdrag(ref_cosmo, z_dv, REF["rdrag"])

        z_dmdh, dmdh_vals, dmdh_errs = [], [], []
        for zz in unique_z:
            obs_at_z = [o for o in bao_obs if abs(o["z"] - zz) < 0.01]
            dm_obs = [o for o in obs_at_z if o["quantity"] == "DM_over_rs"]
            dh_obs = [o for o in obs_at_z if o["quantity"] == "DH_over_rs"]
            if dm_obs and dh_obs:
                dm, dh = dm_obs[0]["value"], dh_obs[0]["value"]
                z_dmdh.append(zz)
                dmdh_vals.append(dm / dh)
                idx_dm = bao_obs.index(dm_obs[0])
                idx_dh = bao_obs.index(dh_obs[0])
                err_dm, err_dh = bao_errors[idx_dm], bao_errors[idx_dh]
                dmdh_errs.append(dm/dh * np.sqrt((err_dm/dm)**2 + (err_dh/dh)**2))
        z_dmdh = np.array(z_dmdh)
        dmdh_data_ratio = np.array(dmdh_vals) / compute_DM_over_DH(ref_cosmo, z_dmdh)
        dmdh_data_err = np.array(dmdh_errs) / compute_DM_over_DH(ref_cosmo, z_dmdh)

        K_list = [1, 2, 3, 4]
        band_colors = ["#a6cee3", "#6baed6", "#3182bd", "#08519c"]
        line_styles = [":", "--", "-.", "-"]
        band_alphas = [0.25, 0.25, 0.25, 0.30]

        figs = {}
        for obs_type, beta, alpha_reg, z_data, data_vals, data_errs, ylabel, fname in [
            ("DV", beta_dv, alpha_dv, z_dv, dv_data_ratio, dv_data_err,
             r"$(D_V/r_s)\,/\,(D_V/r_s)_{\rm ref}$", "sec5_bao_data_bands_dv"),
            ("DMDH", beta_dmdh, alpha_dmdh, z_dmdh, dmdh_data_ratio, dmdh_data_err,
             r"$(D_M/D_H)\,/\,(D_M/D_H)_{\rm ref}$", "sec5_bao_data_bands_dmdh"),
        ]:
            full_m = np.mean(dv_full if obs_type == "DV" else dmdh_full, axis=0)
            full_s = np.std(dv_full if obs_type == "DV" else dmdh_full, axis=0)

            fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6))

            ax.fill_between(z_fine, full_m - full_s, full_m + full_s,
                            color="gray", alpha=0.15, label="Full $w_0w_a$ chain")
            ax.plot(z_fine, full_m, color="gray", lw=0.8, alpha=0.6)

            for K, color, ls, ba in reversed(list(zip(K_list, band_colors, line_styles, band_alphas))):
                center = alpha_reg.copy()
                for alpha in range(K):
                    center = center + beta[alpha] * a_bao_all[alpha]
                width = np.sqrt(np.sum(beta[:K]**2, axis=0))

                mode_names = {1: "$c_0$", 2: "$c_0{+}c_1$", 3: "$c_0{+}c_1{+}c_2$",
                              4: "$c_0{+}c_1{+}c_2{+}c_3$"}
                cum_dchi2 = -sum(a_bao_all[alpha]**2 for alpha in range(K))
                label = f"{mode_names[K]}  ($\\Delta\\chi^2$={cum_dchi2:+.1f})"

                ax.fill_between(z_fine, center - width, center + width,
                                color=color, alpha=ba)
                ax.plot(z_fine, center, color=color, ls=ls, lw=1.5, label=label)

            ax.errorbar(z_data, data_vals, yerr=data_errs,
                        fmt="ko", ms=4, capsize=2, zorder=10, lw=1)

            ax.axhline(1, color="gray", ls="--", alpha=0.4, lw=0.8)
            ax.set_xscale("log")
            ax.set_xlim(0.1, 2.8)
            ax.set_xlabel("$z$")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            figs[fname] = fig

        return figs["sec5_bao_data_bands_dv"], figs["sec5_bao_data_bands_dmdh"]

    fig_sec5_bao_dv, fig_sec5_bao_dmdh = _make()
    return fig_sec5_bao_dv, fig_sec5_bao_dmdh,


@app.cell
def _(np, plt, D, DOUBLE):
    """Fig 11: w0wa chi2 investigation -- 4-panel figure.

    Uses precomputed c_grid from sec5_chi2_* keys and w0wa chain histogram
    from sec5_w0wa_hist_* keys.
    """
    def _make():
        from scipy.ndimage import gaussian_filter

        n_bao = len(D["bao_obs"])
        w0_arr = D["w0_1d"]
        wa_arr = D["wa_1d"]
        Ng = len(w0_arr)
        W0, WA = np.meshgrid(w0_arr, wa_arr, indexing='ij')

        c_grid = D["sec5_chi2_c_grid"]   # (Ng, Ng, n_bao)
        c_mean = D["sec5_chi2_c_mean"]   # (n_bao,)
        c_std = D["sec5_chi2_c_std"]     # (n_bao,)
        a_bao = D["a_bao"]

        # w0wa chain histogram
        H_chain = D["sec5_w0wa_hist_H"]
        xedges = D["sec5_w0wa_hist_xedges"]
        yedges = D["sec5_w0wa_hist_yedges"]
        level_68 = float(D["sec5_w0wa_level_68"])
        level_95 = float(D["sec5_w0wa_level_95"])
        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])

        # Chi2 surfaces
        def chi2_truncated(c_grid, a_bao, c_mean, c_std, n_modes):
            chi2 = np.zeros(c_grid.shape[:2])
            for alpha in range(n_modes):
                chi2 += (c_grid[:, :, alpha] + c_mean[alpha] - a_bao[alpha])**2 / (1.0 + c_std[alpha]**2)
            return chi2

        chi2_01 = chi2_truncated(c_grid, a_bao, c_mean, c_std, 2)
        chi2_0123 = chi2_truncated(c_grid, a_bao, c_mean, c_std, 4)

        # For 2-DOF Gaussian: 68% -> dchi2=2.30, 95% -> dchi2=6.18
        dchi2_1sig = 2.30
        dchi2_2sig = 6.18

        def add_lcdm(ax):
            ax.plot(-1.0, 0.0, marker='+', color='k', ms=12, mew=2, zorder=10)

        def format_ax(ax, title):
            ax.set_xlabel(r'$w_0$')
            ax.set_ylabel(r'$w_a$')
            ax.set_title(title)
            ax.set_xlim(-1.5, 0.0)
            ax.set_ylim(-3.0, 1.5)

        fig, axes = plt.subplots(2, 2, figsize=(10, 9))

        from matplotlib.lines import Line2D

        for ax, chi2_map, title in [
            (axes[0, 0], chi2_01, r'(A) $\chi^2(c_0, c_1)$ -- 2 modes'),
            (axes[0, 1], chi2_0123, r'(B) $\chi^2(c_0 ... c_3)$ -- 4 modes'),
        ]:
            dchi2 = chi2_map - np.nanmin(chi2_map)
            ax.contourf(W0, WA, dchi2, levels=[0, dchi2_1sig, dchi2_2sig, 20],
                        colors=['#3182bd', '#9ecae1', '#deebf7', 'white'], alpha=0.6)
            ax.contour(W0, WA, dchi2, levels=[dchi2_1sig, dchi2_2sig],
                        colors=['blue', 'blue'], linewidths=[2.0, 1.5],
                        linestyles=['-', '--'])
            ax.contour(x_centers, y_centers, H_chain.T,
                          levels=[level_95, level_68],
                          colors=['red', 'red'], linewidths=[1.5, 2.0],
                          linestyles=['--', '-'])
            add_lcdm(ax)
            format_ax(ax, title)
            ax.legend([Line2D([0],[0], color='blue', lw=2),
                       Line2D([0],[0], color='red', lw=2)],
                      [r'SVD $\Delta\chi^2$', 'Planck+DESI chain'],
                      fontsize=7, loc='upper right')

        # Panel C: c2 colormap
        ax = axes[1, 0]
        c2 = c_grid[:, :, 2]
        vmax_c2 = np.nanpercentile(np.abs(c2), 98)
        cf = ax.contourf(W0, WA, c2, levels=np.linspace(-vmax_c2, vmax_c2, 21),
                         cmap='RdBu_r', extend='both')
        fig.colorbar(cf, ax=ax, shrink=0.8, label=r'$c_2(w_0, w_a)$')
        ax.contour(x_centers, y_centers, H_chain.T, levels=[level_95, level_68],
                   colors='black', linewidths=[1.0, 1.5])
        add_lcdm(ax)
        ax.contour(W0, WA, c2, levels=[a_bao[2]], colors='lime', linewidths=2, linestyles='-')
        format_ax(ax, rf'(C) $c_2$ (data $a_2$={a_bao[2]:.2f})')

        # Panel D: c3 colormap
        ax = axes[1, 1]
        c3 = c_grid[:, :, 3]
        vmax_c3 = np.nanpercentile(np.abs(c3), 98)
        cf = ax.contourf(W0, WA, c3, levels=np.linspace(-vmax_c3, vmax_c3, 21),
                         cmap='RdBu_r', extend='both')
        fig.colorbar(cf, ax=ax, shrink=0.8, label=r'$c_3(w_0, w_a)$')
        ax.contour(x_centers, y_centers, H_chain.T, levels=[level_95, level_68],
                   colors='black', linewidths=[1.0, 1.5])
        add_lcdm(ax)
        ax.contour(W0, WA, c3, levels=[a_bao[3]], colors='lime', linewidths=2, linestyles='-')
        format_ax(ax, rf'(D) $c_3$ (data $a_3$={a_bao[3]:.2f})')

        fig.suptitle(r'BAO SVD $\chi^2$ vs Planck+DESI chain in $(w_0, w_a)$ plane',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    fig_w0wa_chi2_investigation = _make()
    return fig_w0wa_chi2_investigation,


@app.cell
def _(np, plt, norm, D, DOUBLE):
    """Fig 12: w(z_pivot) histograms -- 3 panels from precomputed w0wa chain data."""
    def _make():
        # Precomputed w(z_pivot) distributions for 3 modes
        # sec5_wp_c0: array of w(z_pivot) values for c0 mode
        # sec5_wp_c1bao: for BAO c1 mode
        # sec5_wp_c1sn: for SN c1 mode
        wp_arrays = {
            'c0':        D["sec5_wp_c0"],
            'c1_bao':    D["sec5_wp_c1bao"],
            'c1_union3': D["sec5_wp_c1sn"],
        }

        # Pivot fits from paper_numbers.json
        pf = D["paper_numbers"].get("pivot_fits", {})
        z_pivots = {
            'c0':        pf.get("c0_BAO", {}).get("z_pivot", 0.74),
            'c1_bao':    pf.get("c1_BAO", {}).get("z_pivot", 0.46),
            'c1_union3': pf.get("c1_SN", {}).get("z_pivot", -0.06),
        }

        mode_labels = {
            'c0':        r'$c_0$ (universal)',
            'c1_bao':    r'BAO $c_1$',
            'c1_union3': r'SN $c_1$ (Union3)',
        }
        mode_colors = {
            'c0':        '#2ca02c',
            'c1_bao':    '#d62728',
            'c1_union3': '#1f77b4',
        }

        mode_keys = ['c0', 'c1_bao', 'c1_union3']
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5))

        for i_mode, mk in enumerate(mode_keys):
            z_piv = z_pivots[mk]
            wp = wp_arrays[mk]

            wp_mean = np.mean(wp)
            wp_std = np.std(wp)
            tension = (wp_mean - (-1.0)) / wp_std

            ax = axes[i_mode]
            ax.hist(wp, bins=40, density=True, alpha=0.5,
                    color=mode_colors[mk],
                    edgecolor=mode_colors[mk], linewidth=0.5)
            x_pdf = np.linspace(wp_mean - 4 * wp_std,
                                wp_mean + 4 * wp_std, 200)
            ax.plot(x_pdf, norm.pdf(x_pdf, wp_mean, wp_std),
                    color=mode_colors[mk], linewidth=1.5)
            ax.axvline(-1, color='gray', linestyle=':', linewidth=1.2,
                       label='$w=-1$')
            ax.set_xlabel(f'$w(z={z_piv:.2f})$')
            if i_mode == 0:
                ax.set_ylabel('PDF')
            ax.set_title(
                f'{mode_labels[mk]}\n'
                f'$w = {wp_mean:.3f} \\pm {wp_std:.3f}$'
                f' ({tension:+.1f}$\\sigma$)',
                fontsize=8)
            ax.legend(fontsize=6)

        fig.tight_layout()
        return fig

    fig_wp_histograms = _make()
    return fig_wp_histograms,


@app.cell
def _(np, plt, D):
    """Fig 13: SN all-3-datasets panel with w0wa chain prediction band."""
    def _make():
        z_fine_sn = D["sec5_z_fine_sn"]
        dmu_fine_w0wa = D["sec5_dmu_fine_w0wa"]   # (N, nz)
        dmu_fine_lcdm = D["sec5_dmu_fine_lcdm"]   # (N, nz)

        lcdm_fine_mean = np.mean(dmu_fine_lcdm, axis=0)
        w0wa_fine_mean = np.mean(dmu_fine_w0wa, axis=0)
        w0wa_fine_std = np.std(dmu_fine_w0wa, axis=0)
        de_signal = w0wa_fine_mean - lcdm_fine_mean

        sn_colors = {"Union3": "#ff7f0e", "Pantheon+": "#2ca02c", "DES-Dovekie": "#d62728"}
        sn_markers = {"Union3": "o", "Pantheon+": "s", "DES-Dovekie": "^"}

        canonical_sn = {}
        for sn_name, prefix in [("Union3", "u3"), ("Pantheon+", "pp"), ("DES-Dovekie", "d5")]:
            canonical_sn[sn_name] = {
                "z": D[f"sec5_canonical_{prefix}_z"],
                "d": D[f"sec5_canonical_{prefix}_d"],
                "err": D[f"sec5_canonical_{prefix}_err"],
            }

        # Shift w0wa band to best match DES-Dovekie
        des = canonical_sn["DES-Dovekie"]
        de_at_des = np.interp(des["z"], z_fine_sn, de_signal)
        w_inv_des = np.where(des["err"] > 1e-10, 1.0 / des["err"]**2, 0.0)
        M_band = np.sum((des["d"] - de_at_des) * w_inv_des) / np.sum(w_inv_des)

        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))

        band_center = de_signal + M_band
        ax.fill_between(z_fine_sn, band_center - w0wa_fine_std,
                        band_center + w0wa_fine_std,
                        color="gray", alpha=0.25, label=r"$w_0w_a$ chain $\pm 1\sigma$")
        ax.plot(z_fine_sn, band_center, color="gray", lw=1.2)

        for sn_name in ["Union3", "Pantheon+", "DES-Dovekie"]:
            cs = canonical_sn[sn_name]
            sort_idx = np.argsort(cs["z"])
            ax.errorbar(cs["z"][sort_idx], cs["d"][sort_idx], yerr=cs["err"][sort_idx],
                        fmt=sn_markers[sn_name], color=sn_colors[sn_name],
                        ms=3, lw=0.8, capsize=1.2, alpha=0.8, zorder=5,
                        label=sn_name)

        ax.axhline(0, color="gray", ls="--", alpha=0.4, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("$z$")
        ax.set_ylabel(r"$\Delta\mu$ (mag)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return fig

    fig_sec5_sn_all = _make()
    return fig_sec5_sn_all,


@app.cell
def _(np, plt, D):
    """Fig 14: SN c1 histogram with data measurements (Union3 basis)."""
    def _make():
        c1_chain = D["sec5_c1_chain_u3"]  # w0wa chain c1 in Union3 basis

        c1_scale = {
            "Union3": float(D["sec5_c1_scale_u3"]),
            "Pantheon+": float(D["sec5_c1_scale_pp"]),
            "DES-Dovekie": float(D["sec5_c1_scale_d5"]),
        }
        a1_native = {
            "Union3": float(D["sec5_a1_u3"]),
            "Pantheon+": float(D["sec5_a1_pp"]),
            "DES-Dovekie": float(D["sec5_a1_d5"]),
        }

        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5))

        ax.hist(c1_chain, bins=50, density=True, alpha=0.4, color="gray",
                label=f"$w_0w_a$ chain ($\\langle c_1 \\rangle$={np.mean(c1_chain):+.1f})")

        zz = np.linspace(-8, 8, 300)
        data_colors = {"Union3": "#ff7f0e", "Pantheon+": "#2ca02c", "DES-Dovekie": "#d62728"}
        for sn_name in ["Union3", "Pantheon+", "DES-Dovekie"]:
            a1_u3 = a1_native[sn_name] / c1_scale[sn_name]
            gauss = np.exp(-0.5*(zz - a1_u3)**2) / np.sqrt(2*np.pi)
            ax.plot(zz, gauss, color=data_colors[sn_name], lw=1.5, ls="--", alpha=0.7)
            scale_note = "" if sn_name == "Union3" else f" (/{c1_scale[sn_name]:.2f})"
            ax.axvline(a1_u3, color=data_colors[sn_name], lw=2,
                       label=f"{sn_name}: $a_1$={a1_u3:+.1f}{scale_note}")

        ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5, label=r"$\Lambda$CDM")
        ax.set_xlabel("$c_1$ (Union3 basis)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(-8, 8)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return fig

    fig_sec5_sn_c1 = _make()
    return fig_sec5_sn_c1,


@app.cell
def _(np, plt, norm, D, DOUBLE, ECOL):
    """Fig 15: Extension c0 distributions (BAO left, Union3 right)."""
    def _make():
        CHAINS = {
            'alens': {'label': r'$A_{\rm lens}$', 'color': ECOL["Alens"]},
            'bprim': {'label': r'$B_{\rm PMF}$', 'color': ECOL["B_PMF"]},
            'ede_n2': {'label': r'EDE $n{=}2$', 'color': ECOL["EDE_n2"]},
        }

        whitened_data_bao = D["desi_w"]
        whitened_data_u3 = D["svd_u3"]["data_w"]

        lcdm_bao_w = D["ext_lcdm_bao_w"]
        U_b, S_b, Vt_b = np.linalg.svd(lcdm_bao_w, full_matrices=False)
        V0_bao = Vt_b[0]
        c0_lcdm_bao = U_b[:, 0] * S_b[0]
        a0_bao = whitened_data_bao @ V0_bao

        lcdm_u3_w = D["ext_lcdm_u3_w"]
        U_u, S_u, Vt_u = np.linalg.svd(lcdm_u3_w, full_matrices=False)
        V0_u3 = Vt_u[0]
        c0_lcdm_u3 = U_u[:, 0] * S_u[0]
        a0_u3 = whitened_data_u3 @ V0_u3

        probes = {
            'BAO':   {'V0': V0_bao, 'c0_lcdm': c0_lcdm_bao, 'a0': a0_bao},
            'Union3': {'V0': V0_u3, 'c0_lcdm': c0_lcdm_u3, 'a0': a0_u3},
        }

        ext_c0 = {}
        for ext_name in CHAINS:
            ext_c0[ext_name] = {
                'BAO': D[f"ext_{ext_name}_bao_w"] @ V0_bao,
                'Union3': D[f"ext_{ext_name}_u3_w"] @ V0_u3,
            }

        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.8))

        for ax_idx, probe_name in enumerate(['BAO', 'Union3']):
            ax = axes[ax_idx]
            cd = probes[probe_name]
            c0_lcdm = cd['c0_lcdm']
            a0 = cd['a0']
            lcdm_mean = np.mean(c0_lcdm)
            lcdm_std = np.std(c0_lcdm)

            all_means = [lcdm_mean, a0]
            all_stds = [lcdm_std, 1.0]
            for ext_name in CHAINS:
                c0_ext = ext_c0[ext_name][probe_name]
                all_means.append(np.mean(c0_ext))
                all_stds.append(np.std(c0_ext))

            x_lo = min(all_means) - 3.5 * max(all_stds)
            x_hi = max(all_means) + 3.5 * max(all_stds)
            x = np.linspace(x_lo, x_hi, 400)

            ax.plot(x, norm.pdf(x, lcdm_mean, lcdm_std), '-', color=ECOL["LCDM"],
                    lw=1.8,
                    label=f'ACT LCDM ($\\mu$={lcdm_mean:.2f}, $\\sigma$={lcdm_std:.2f})')
            ax.fill_between(x, norm.pdf(x, lcdm_mean, lcdm_std),
                            color=ECOL["LCDM"], alpha=0.10)

            for ext_name, chain_info in CHAINS.items():
                c0_ext = ext_c0[ext_name][probe_name]
                ext_mean = np.mean(c0_ext)
                ext_std = np.std(c0_ext)
                ax.plot(x, norm.pdf(x, ext_mean, ext_std), '-',
                        color=chain_info['color'], lw=1.3,
                        label=f'{chain_info["label"]} ($\\mu$={ext_mean:.2f}, $\\sigma$={ext_std:.2f})')
                ax.fill_between(x, norm.pdf(x, ext_mean, ext_std),
                                color=chain_info['color'], alpha=0.06)

            tension = (a0 - lcdm_mean) / np.sqrt(1 + lcdm_std**2)
            ax.plot(x, norm.pdf(x, a0, 1.0), 'r-', lw=1.8,
                    label=f'DESI data ($a_0$={a0:.2f}, {tension:+.1f}$\\sigma$)')
            ax.fill_between(x, norm.pdf(x, a0, 1.0), color='r', alpha=0.10)

            ax.set_xlabel('$c_0$')
            ax.set_ylabel('PDF')
            ax.set_title(f'{probe_name}: $c_0$ under extensions', fontsize=9)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    fig_ext_c0_dists = _make()
    return fig_ext_c0_dists,


@app.cell
def _(np, plt, norm, D, SINGLE):
    """Fig 16: Omega_k 3-Gaussian plot."""
    def _make():
        c0_lcdm = D["c_bao"][:, 0]
        a0_data = D["a_bao"][0]
        c0_omk = D["omk_c0_bao"]

        mu_lcdm, sig_lcdm = np.mean(c0_lcdm), np.std(c0_lcdm)
        mu_omk, sig_omk = np.mean(c0_omk), np.std(c0_omk)

        tension_lcdm = (a0_data - mu_lcdm) / np.sqrt(1 + sig_lcdm**2)
        tension_omk = (a0_data - mu_omk) / np.sqrt(1 + sig_omk**2)

        fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
        x = np.linspace(-55, 12, 500)

        ax.plot(x, norm.pdf(x, mu_lcdm, sig_lcdm), color='red', lw=1.8,
                label=f'ACT $\\Lambda$CDM ($\\mu$={mu_lcdm:.1f}, $\\sigma$={sig_lcdm:.1f})')
        ax.fill_between(x, norm.pdf(x, mu_lcdm, sig_lcdm), alpha=0.25, color='red')

        ax.plot(x, norm.pdf(x, mu_omk, sig_omk), color='steelblue', lw=1.8,
                label=f'ACT $\\Omega_k$ ($\\mu$={mu_omk:.1f}, $\\sigma$={sig_omk:.1f})')
        ax.fill_between(x, norm.pdf(x, mu_omk, sig_omk), alpha=0.25, color='steelblue')

        ax.plot(x, norm.pdf(x, a0_data, 1.0), color='black', lw=1.8, ls='--',
                label=f'DESI BAO ($a_0$={a0_data:.1f}, $\\sigma$=1)')
        ax.fill_between(x, norm.pdf(x, a0_data, 1.0), alpha=0.12, color='black')

        ax.set_xlabel('$c_0$')
        ax.set_ylabel('PDF')
        ax.set_xlim(-55, 12)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6, loc='upper left')

        ax.text(0.97, 0.95,
                f'$\\Lambda$CDM tension: {tension_lcdm:+.1f}$\\sigma$\n'
                f'$\\Omega_k$ tension: {tension_omk:+.1f}$\\sigma$',
                transform=ax.transAxes, fontsize=7, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.tight_layout()
        return fig

    fig_omk_gaussians = _make()
    return fig_omk_gaussians,


@app.cell
def _(np, plt, D, DOUBLE):
    """Fig 17: Appendix -- observable derivatives vs delta_p / sigma_p."""
    def _make():
        fid_Drd = float(D["app_fid_Drd"])
        fid_DrD = float(D["app_fid_DrD"])
        om_grid = D["app_om_grid"]
        ob_grid = D["app_ob_grid"]
        th_grid = D["app_th_grid"]
        act = D["act"]
        sig_om = act["omh2"].std()
        sig_ob = act["ombh2"].std()
        sig_th = act["theta"].std()
        fid_om = om_grid[len(om_grid)//2]
        fid_ob = ob_grid[len(ob_grid)//2]
        fid_th = th_grid[len(th_grid)//2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.8))

        styles = [
            (r"$\omega_m$", "om", "C0", "-"),
            (r"$\omega_b$", "ob", "C1", "--"),
            (r"$\theta_\star$", "th", "C2", "-."),
        ]
        grids = {"om": om_grid, "ob": ob_grid, "th": th_grid}
        sigs = {"om": sig_om, "ob": sig_ob, "th": sig_th}
        fids_p = {"om": fid_om, "ob": fid_ob, "th": fid_th}

        for label, pkey, color, ls in styles:
            xsig = (grids[pkey] - fids_p[pkey]) / sigs[pkey]
            vals_Drd = D[f"app_scan_{pkey}_Drd"]
            ax1.plot(xsig, vals_Drd / fid_Drd, color=color, ls=ls, lw=1.5, label=label)
            vals_DrD = D[f"app_scan_{pkey}_DrD"]
            ax2.plot(xsig, vals_DrD / fid_DrD, color=color, ls=ls, lw=1.5, label=label)

        for ax, title in [(ax1, r"$D_M(0.5)/r_d$"), (ax2, r"$D_M(0.3)/D_M(0.5)$")]:
            ax.axhline(1, color="gray", lw=0.4, ls="--")
            ax.axvline(0, color="gray", lw=0.4, ls="--")
            ax.set_xlabel(r"$\delta p \,/\, \sigma_p$", fontsize=9)
            ax.set_ylabel("ratio to fiducial", fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(-5, 5)
            ax.legend(fontsize=7, loc="best")

        fig.tight_layout()
        return fig

    fig_app_derivatives = _make()
    return fig_app_derivatives,


@app.cell
def _(np, plt, D, SINGLE):
    """Fig 18: Appendix -- predicted vs chain beta/sigma(c0)."""
    def _make():
        act = D["act"]
        c0 = D["c_bao"][:, 0]

        def norm_fn(x):
            return (x - x.mean()) / x.std()
        Xf = np.column_stack([norm_fn(act["omh2"]), norm_fn(act["ombh2"]), norm_fn(act["theta"]),
                               np.ones(len(c0))])
        bf, _, _, _ = np.linalg.lstsq(Xf, c0, rcond=None)
        chain_bos = bf[:3] / c0.std()

        om_grid = D["app_om_grid"]
        ob_grid = D["app_ob_grid"]
        th_grid = D["app_th_grid"]
        mid = len(om_grid) // 2
        fid_Drd = float(D["app_fid_Drd"])
        fid_DrD = float(D["app_fid_DrD"])

        rho = np.corrcoef([act["omh2"], act["ombh2"], act["theta"]])
        frac = np.array([act["omh2"].std()/act["omh2"].mean(),
                          act["ombh2"].std()/act["ombh2"].mean(),
                          act["theta"].std()/act["theta"].mean()])

        g_bao = np.zeros(3)
        g_sn = np.zeros(3)
        for idx, (grid, pkey) in enumerate([(om_grid, "om"), (ob_grid, "ob"), (th_grid, "th")]):
            dp = grid[mid+1] - grid[mid-1]
            g_bao[idx] = (D[f"app_scan_{pkey}_Drd"][mid+1] - D[f"app_scan_{pkey}_Drd"][mid-1]) / dp * grid[mid] / fid_Drd
            g_sn[idx] = (D[f"app_scan_{pkey}_DrD"][mid+1] - D[f"app_scan_{pkey}_DrD"][mid-1]) / dp * grid[mid] / fid_DrD

        f_bao = g_bao * frac
        f_sn = g_sn * frac
        bos_bao = -f_bao / np.sqrt(f_bao @ rho @ f_bao)
        bos_sn = -f_sn / np.sqrt(f_sn @ rho @ f_sn)

        fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
        xx = np.arange(3)
        w = 0.22
        ax.bar(xx - w, chain_bos, w, label="Chain", color="#1f77b4", alpha=0.85)
        ax.bar(xx, bos_bao, w, label=r"$D_M(0.5)/r_d$", color="#ff7f0e", alpha=0.85)
        ax.bar(xx + w, bos_sn, w, label=r"$D_M(0.3)/D_M(0.5)$", color="#2ca02c", alpha=0.85)

        ax.set_xticks(xx)
        ax.set_xticklabels([r"$\omega_m$", r"$\omega_b$", r"$\theta_\star$"])
        ax.set_ylabel(r"$\beta_p / \sigma(c_0)$", fontsize=9)
        ax.axhline(0, color="k", lw=0.4)
        ax.legend(fontsize=7)
        fig.tight_layout()
        return fig

    fig_app_beta_prediction = _make()
    return fig_app_beta_prediction,


@app.cell
def _(
    PAPER_FIG_DIR,
    fig_dv_ratios, fig_dmdh_ratios, fig_sn_datasets,
    fig_c0_families, fig_c0_gaussians, fig_omegamh2_money,
    fig_hz_omh2_variation,
    fig_w0wa_contours,
    fig_sec5_bao_dv, fig_sec5_bao_dmdh,
    fig_w0wa_chi2_investigation,
    fig_wp_histograms,
    fig_sec5_sn_all, fig_sec5_sn_c1,
    fig_ext_c0_dists, fig_omk_gaussians,
    fig_app_derivatives, fig_app_beta_prediction,
):
    """Save all 18 figures as PDF."""
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    _figs = {
        # Section 2: Data and method
        "fig_dv_ratios": fig_dv_ratios,
        "fig_dmdh_ratios": fig_dmdh_ratios,
        "fig_sn_datasets": fig_sn_datasets,
        # Section 3: Universal c0
        "fig_c0_families": fig_c0_families,
        # Section 4: c0 tensions
        "fig_c0_gaussians": fig_c0_gaussians,
        "fig_omegamh2_money": fig_omegamh2_money,
        "fig_hz_omh2_variation": fig_hz_omh2_variation,
        # Section 5: w0wa reinterpretation
        "fig_w0wa_contours": fig_w0wa_contours,
        "sec5_bao_data_bands_dv": fig_sec5_bao_dv,
        "sec5_bao_data_bands_dmdh": fig_sec5_bao_dmdh,
        "fig_w0wa_chi2_investigation": fig_w0wa_chi2_investigation,
        "fig_wp_histograms": fig_wp_histograms,
        "sec5_sn_all_datasets": fig_sec5_sn_all,
        "sec5_sn_c1_histogram": fig_sec5_sn_c1,
        # Section 6: Extensions and curvature
        "fig_ext_c0_dists": fig_ext_c0_dists,
        "fig_omk_gaussians": fig_omk_gaussians,
        # Appendix A
        "fig_app_derivatives": fig_app_derivatives,
        # fig_app_beta_prediction: dropped from paper (not referenced)
    }

    for _name, _f in _figs.items():
        _f.savefig(PAPER_FIG_DIR / f"{_name}.pdf", format="pdf")
        print(f"  Saved {_name}.pdf")

    print(f"\nTotal: {len(_figs)} figures saved to {PAPER_FIG_DIR}")
    return


if __name__ == "__main__":
    app.run()

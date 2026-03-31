"""
MCMC chain loading and sample processing functions.

This module provides functions to load MCMC chains and compute
distance samples from cosmological parameters.
"""

from __future__ import annotations

from typing import TypedDict, Any

import numpy as np
from numpy.typing import NDArray
from getdist import loadMCSamples
from getdist.mcsamples import MCSamples

from .cosmology import (
    CosmoDict,
    cosmo_omkw0wa,
    compute_DV_over_rdrag,
    compute_DM_over_DH,
    compute_distance_modulus,
)


class SamplesDict(TypedDict, total=False):
    """Type definition for samples dictionary."""

    samples: MCSamples
    param_names: list[str]
    summary: dict[str, dict[str, float]]
    redshifts: NDArray[np.floating]
    DV_over_rdrag_samples: NDArray[np.floating]
    DM_over_DH_samples: NDArray[np.floating]
    mu_samples: NDArray[np.floating]
    distance_samples_parameters: NDArray[np.floating]
    DV_over_rdrag: NDArray[np.floating]
    std_DV_over_rdrag: NDArray[np.floating]
    DM_over_DH: NDArray[np.floating]
    std_DM_over_DH: NDArray[np.floating]
    mu: NDArray[np.floating]
    std_mu: NDArray[np.floating]
    DV_over_rdrag_interp: NDArray[np.floating]
    DM_over_DH_interp: NDArray[np.floating]


class ChainSummary(TypedDict):
    """Type definition for parameter summary statistics."""

    mean: float
    std: float
    frac_err_pct: float


def load_samples(file: str) -> SamplesDict:
    """
    Load MCMC samples, add a derived omega_m h^2 parameter (omh2),
    and compute summary statistics for all valid parameters.

    Parameters
    ----------
    file : str
        Path to the chain file (without extension)

    Returns
    -------
    SamplesDict
        Dictionary with keys:
        - 'samples': the getdist MCSamples object (with 'omh2' added)
        - 'param_names': list of valid parameter names (including 'omh2')
        - 'summary': dict mapping each parameter to its mean, std, and % error
    """
    samples_obj = loadMCSamples(file)

    # Add derived omh2 = ombh2 + omch2 if both exist
    names = [p.name for p in samples_obj.paramNames.names]
    ib = next(
        (names.index(n) for n in ["ombh2", "omegabh2", "omega_b"] if n in names),
        None,
    )
    ic = next(
        (names.index(n) for n in ["omch2", "omegach2", "omega_cdm"] if n in names),
        None,
    )
    if ib is not None and ic is not None:
        chain = samples_obj.samples
        omh2 = chain[:, ib] + chain[:, ic]
        samples_obj.addDerived(omh2, "omh2", label="\\omega_m h^2")

    # Gather valid parameter names
    params = samples_obj.getParams()
    valid_param_names = [
        name
        for name in dir(params)
        if not name.startswith("_")
        and hasattr(getattr(params, name), "mean")
        and callable(getattr(params, name).mean)
    ]

    # Compute summary statistics
    summary: dict[str, dict[str, float]] = {}
    for p in valid_param_names:
        obj = getattr(params, p)
        m = obj.mean()
        s = obj.std()
        fe = (s / m * 100) if m != 0 else float("nan")
        summary[p] = {"mean": m, "std": s, "frac_err %": fe}

    return {
        "samples": samples_obj,
        "param_names": valid_param_names,
        "summary": summary,
    }


def _find_param_index(
    param_names: list[str],
    possible_names: list[str],
    required: bool = True,
) -> int | None:
    """
    Find the index of a parameter from a list of possible names.

    Parameters
    ----------
    param_names : list[str]
        List of parameter names in the chain
    possible_names : list[str]
        List of possible names to search for
    required : bool
        If True, raise error if not found; if False, return None

    Returns
    -------
    int | None
        Index of the parameter, or None if not found and not required
    """
    for name in possible_names:
        if name in param_names:
            return param_names.index(name)
    if required:
        raise ValueError(
            f"Required parameter not found. Looked for: {possible_names}. "
            f"Available: {param_names[:15]}..."
        )
    return None


def compute_distance_samples(
    samples_dic: SamplesDict,
    cosmo_dic_ref: CosmoDict,
    n_random: int = 500,
    seed: int = 0,
) -> SamplesDict:
    """
    Compute distance ratios for random samples from MCMC chains.

    This function draws random samples from the MCMC chain, computes
    cosmological distances for each sample, and returns the results
    as ratios relative to a reference cosmology.

    Parameters
    ----------
    samples_dic : SamplesDict
        Dictionary containing 'samples' key with MCSamples object
    cosmo_dic_ref : CosmoDict
        Reference cosmology dictionary with precomputed distances
    n_random : int, optional
        Number of random samples to draw (default: 500)
    seed : int, optional
        Random seed for reproducibility (default: 0)

    Returns
    -------
    SamplesDict
        New dictionary with original samples plus computed distances:
        - 'redshifts': array of redshift values
        - 'DV_over_rdrag_samples': (n_random, n_z) array of DV/rd ratios
        - 'DM_over_DH_samples': (n_random, n_z) array of DM/DH ratios
        - 'mu_samples': (n_random, n_z) array of distance modulus differences
        - 'distance_samples_parameters': (n_random, 13) array of sample parameters
        - 'DV_over_rdrag': mean DV/rd ratio
        - 'std_DV_over_rdrag': std of DV/rd ratio
        - 'DM_over_DH': mean DM/DH ratio
        - 'std_DM_over_DH': std of DM/DH ratio
        - 'mu': mean distance modulus difference
        - 'std_mu': std of distance modulus difference
    """
    samples_array = samples_dic["samples"].samples
    param_names = [p.name for p in samples_dic["samples"].paramNames.names]
    redshifts = cosmo_dic_ref["redshifts"]

    # Find parameter indices
    h_index = _find_param_index(param_names, ["H0", "h"])
    ombh2_index = _find_param_index(param_names, ["ombh2", "omegabh2", "omega_b"])
    omch2_index = _find_param_index(param_names, ["omch2", "omegach2", "omega_cdm"])
    rdrag_index = _find_param_index(param_names, ["rdrag", "rs_drag"])
    nnu_index = _find_param_index(param_names, ["nnu", "Neff", "n_nu"], required=False)
    w0_index = _find_param_index(param_names, ["w"], required=False)
    wa_index = _find_param_index(param_names, ["wa"], required=False)
    tau_index = _find_param_index(param_names, ["tau", "tau_reio"])
    thetastar_index = _find_param_index(param_names, ["thetastar", "theta_s_100"])
    rstar_index = _find_param_index(param_names, ["rstar"], required=False)

    # Initialize output arrays
    DV_over_rdrag_samples: list[NDArray] = []
    DM_over_DH_samples: list[NDArray] = []
    mu_samples: list[NDArray] = []
    distance_samples_parameters: list[list[float]] = []

    # Set random seed and draw sample indices
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(samples_array), size=n_random, replace=False)

    for i in indices:
        s = samples_array[i]
        H0 = float(s[h_index])
        ombh2 = float(s[ombh2_index])
        omch2 = float(s[omch2_index])
        rdrag = float(s[rdrag_index])

        rstar = float(s[rstar_index]) if rstar_index is not None else 0.0
        tau = float(s[tau_index]) if tau_index is not None else 0.0
        thetastar = float(s[thetastar_index]) if thetastar_index is not None else 0.0
        Neff = float(s[nnu_index]) if nnu_index is not None else 3.04
        omk = 0.0
        w0 = float(s[w0_index]) if w0_index is not None else -1.0
        wa = float(s[wa_index]) if wa_index is not None else 0.0

        # Create cosmology and compute distances
        cosmo = cosmo_omkw0wa(H0, omch2, ombh2, omk, w0, wa, Neff=Neff)

        dv_rd = compute_DV_over_rdrag(cosmo, redshifts, rdrag)
        dm_dh = compute_DM_over_DH(cosmo, redshifts)
        mu = compute_distance_modulus(cosmo, redshifts)

        DV_over_rdrag_samples.append(dv_rd)
        DM_over_DH_samples.append(dm_dh)
        mu_samples.append(mu)

        # Compute DV/rd ratio at z=0.5 for parameter tracking
        dv_rd_ratio = dv_rd / cosmo_dic_ref["DV_over_rdrag"]
        DVrd05 = float(np.interp(0.5, redshifts, dv_rd_ratio))
        distance_samples_parameters.append([
            DVrd05, H0, omch2, ombh2, omk, w0, wa, rdrag,
            rstar, Neff, tau, thetastar, rdrag,
        ])

    # Convert to arrays
    DV_over_rdrag_arr = np.array(DV_over_rdrag_samples)
    DM_over_DH_arr = np.array(DM_over_DH_samples)
    mu_arr = np.array(mu_samples)
    params_arr = np.array(distance_samples_parameters)

    # Compute ratios relative to reference
    DV_ratio = DV_over_rdrag_arr / cosmo_dic_ref["DV_over_rdrag"]
    DM_ratio = DM_over_DH_arr / cosmo_dic_ref["DM_over_DH"]
    mu_diff = mu_arr - cosmo_dic_ref["mu"]

    # Return new dictionary with all results
    return {
        **samples_dic,
        "redshifts": redshifts,
        "DV_over_rdrag_samples": DV_ratio,
        "DM_over_DH_samples": DM_ratio,
        "mu_samples": mu_diff,
        "distance_samples_parameters": params_arr,
        "DV_over_rdrag": np.mean(DV_ratio, axis=0),
        "std_DV_over_rdrag": np.std(DV_ratio, axis=0),
        "DM_over_DH": np.mean(DM_ratio, axis=0),
        "std_DM_over_DH": np.std(DM_ratio, axis=0),
        "mu": np.mean(mu_diff, axis=0),
        "std_mu": np.std(mu_diff, axis=0),
    }


# Legacy compatibility alias (deprecated)
def distances_samples(
    samples_dic: dict[str, Any],
    cosmo_dic_ref: dict[str, Any],
    n_random: int = 500,
    seed: int = 0,
) -> None:
    """Deprecated: Use compute_distance_samples instead (returns new dict)."""
    # For backward compatibility, modify in place
    # Note: uses old np.random.seed for exact reproducibility with notebook
    samples_array = samples_dic["samples"].samples
    param_names = [p.name for p in samples_dic["samples"].paramNames.names]
    redshifts = cosmo_dic_ref["redshifts"]

    h_index = _find_param_index(param_names, ["H0", "h"])
    ombh2_index = _find_param_index(param_names, ["ombh2", "omegabh2", "omega_b"])
    omch2_index = _find_param_index(param_names, ["omch2", "omegach2", "omega_cdm"])
    rdrag_index = _find_param_index(param_names, ["rdrag", "rs_drag"])
    nnu_index = _find_param_index(param_names, ["nnu", "Neff", "n_nu"], required=False)
    w0_index = _find_param_index(param_names, ["w"], required=False)
    wa_index = _find_param_index(param_names, ["wa"], required=False)
    tau_index = _find_param_index(param_names, ["tau", "tau_reio"])
    thetastar_index = _find_param_index(param_names, ["thetastar", "theta_s_100"])
    rstar_index = _find_param_index(param_names, ["rstar"], required=False)

    DV_over_rdrag_samples = []
    DM_over_DH_samples = []
    mu_samples = []
    distance_samples_parameters = []

    np.random.seed(seed)
    indices = np.random.choice(len(samples_array), size=n_random, replace=False)

    for i in indices:
        s = samples_array[i]
        H0 = s[h_index]
        ombh2 = s[ombh2_index]
        omch2 = s[omch2_index]
        rdrag = s[rdrag_index]
        rstar = s[rstar_index] if rstar_index is not None else 0
        tau = s[tau_index] if tau_index is not None else 0
        thetastar = s[thetastar_index] if thetastar_index is not None else 0
        Neff = s[nnu_index] if nnu_index is not None else 3.04
        omk = 0
        w0 = s[w0_index] if w0_index is not None else -1
        wa = s[wa_index] if wa_index is not None else 0

        cosmo = cosmo_omkw0wa(H0, omch2, ombh2, omk, w0, wa, Neff=Neff)
        dv_rd = compute_DV_over_rdrag(cosmo, redshifts, rdrag)
        dm_dh = compute_DM_over_DH(cosmo, redshifts)
        mu = compute_distance_modulus(cosmo, redshifts)

        DV_over_rdrag_samples.append(dv_rd)
        DM_over_DH_samples.append(dm_dh)
        mu_samples.append(mu)

        dv_rd_ratio = dv_rd / cosmo_dic_ref["DV_over_rdrag"]
        DVrd05 = np.interp(0.5, redshifts, dv_rd_ratio)
        distance_samples_parameters.append([
            DVrd05, H0, omch2, ombh2, omk, w0, wa, rdrag,
            rstar, Neff, tau, thetastar, rdrag,
        ])

    DV_over_rdrag_arr = np.array(DV_over_rdrag_samples)
    DM_over_DH_arr = np.array(DM_over_DH_samples)
    mu_arr = np.array(mu_samples)

    samples_dic["redshifts"] = redshifts
    samples_dic["DV_over_rdrag_samples"] = DV_over_rdrag_arr / cosmo_dic_ref["DV_over_rdrag"]
    samples_dic["DM_over_DH_samples"] = DM_over_DH_arr / cosmo_dic_ref["DM_over_DH"]
    samples_dic["mu_samples"] = mu_arr - cosmo_dic_ref["mu"]
    samples_dic["distance_samples_parameters"] = np.array(distance_samples_parameters)
    samples_dic["DV_over_rdrag"] = np.mean(samples_dic["DV_over_rdrag_samples"], axis=0)
    samples_dic["std_DV_over_rdrag"] = np.std(samples_dic["DV_over_rdrag_samples"], axis=0)
    samples_dic["DM_over_DH"] = np.mean(samples_dic["DM_over_DH_samples"], axis=0)
    samples_dic["std_DM_over_DH"] = np.std(samples_dic["DM_over_DH_samples"], axis=0)
    samples_dic["mu"] = np.mean(samples_dic["mu_samples"], axis=0)
    samples_dic["std_mu"] = np.std(samples_dic["mu_samples"], axis=0)

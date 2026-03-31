"""
Utility functions for interpolation and cosmology solving.

This module provides helper functions for interpolating distance
measurements to BAO redshifts and solving for H0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from astropy.cosmology import w0waCDM

from .cosmology import CosmoDict, cosmo_omkw0wa
from .data import BAOData


def interpolate_to_redshifts(
    z_orig: NDArray[np.floating],
    values: NDArray[np.floating],
    z_target: list[float] | NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Interpolate values from original redshifts to target redshifts.

    Parameters
    ----------
    z_orig : NDArray
        Original redshift grid
    values : NDArray
        Values at original redshifts (1D or 2D with samples on axis 0)
    z_target : array-like
        Target redshifts for interpolation

    Returns
    -------
    NDArray
        Interpolated values at target redshifts
    """
    if values.ndim == 1:
        interp_func = interp1d(z_orig, values, kind="linear", fill_value="extrapolate")
        return interp_func(z_target)
    else:
        interp_func = interp1d(
            z_orig, values, kind="linear", axis=1, fill_value="extrapolate"
        )
        return interp_func(z_target)


def interpolate_bao_samples(
    samples: dict[str, Any],
    bao_data: BAOData,
) -> dict[str, Any]:
    """
    Interpolate distance samples to BAO effective redshifts.

    Parameters
    ----------
    samples : dict
        Dictionary containing:
        - 'redshifts': original redshift grid
        - 'DV_over_rdrag_samples': (n_samples, n_z) array
        - 'DM_over_DH_samples': (n_samples, n_z) array
    bao_data : BAOData
        Dictionary containing 'zeff' key with BAO redshifts

    Returns
    -------
    dict
        New dictionary with original values plus interpolated samples:
        - 'DV_over_rdrag_interp': interpolated DV/rd samples at BAO redshifts
        - 'DM_over_DH_interp': interpolated DM/DH samples at BAO redshifts
    """
    z_orig = samples["redshifts"]
    z_target = bao_data["zeff"]

    return {
        **samples,
        "DV_over_rdrag_interp": interpolate_to_redshifts(
            z_orig, samples["DV_over_rdrag_samples"], z_target
        ),
        "DM_over_DH_interp": interpolate_to_redshifts(
            z_orig, samples["DM_over_DH_samples"], z_target
        ),
    }


def interpolate_cosmo_to_bao(
    cosmo_dict: CosmoDict,
    bao_data: BAOData,
) -> CosmoDict:
    """
    Interpolate cosmology distance ratios to BAO effective redshifts.

    Parameters
    ----------
    cosmo_dict : CosmoDict
        Dictionary containing:
        - 'redshifts': original redshift grid
        - 'DV_over_rdrag_ratio': DV/rd ratio array
        - 'DM_over_DH_ratio': DM/DH ratio array
    bao_data : BAOData
        Dictionary containing 'zeff' key with BAO redshifts

    Returns
    -------
    CosmoDict
        New dictionary with original values plus interpolated ratios:
        - 'DV_over_rdrag_interp': interpolated DV/rd ratio at BAO redshifts
        - 'DM_over_DH_interp': interpolated DM/DH ratio at BAO redshifts
    """
    z_orig = cosmo_dict["redshifts"]
    z_target = bao_data["zeff"]

    return {
        **cosmo_dict,
        "DV_over_rdrag_interp": interpolate_to_redshifts(
            z_orig, cosmo_dict["DV_over_rdrag_ratio"], z_target
        ),
        "DM_over_DH_interp": interpolate_to_redshifts(
            z_orig, cosmo_dict["DM_over_DH_ratio"], z_target
        ),
    }


def cosmo_solve_H0_omkw0wa(
    omch2: float,
    ombh2: float,
    omk: float,
    w0: float,
    wa: float,
    rstar: float,
    Rfid: float,
    z_rec: float,
) -> w0waCDM:
    """
    Solve for H0 such that the angular diameter distance at z_rec satisfies
    D_A(z_rec) = rstar / Rfid.

    This uses Brent's method to find H0 that matches the acoustic scale
    constraint from CMB observations.

    Parameters
    ----------
    omch2 : float
        Physical cold dark matter density
    ombh2 : float
        Physical baryon density
    omk : float
        Curvature density parameter
    w0 : float
        Dark energy equation of state at z=0
    wa : float
        Dark energy equation of state evolution
    rstar : float
        Sound horizon at recombination in Mpc
    Rfid : float
        Fiducial angular scale (rstar / D_A at z_rec)
    z_rec : float
        Redshift of recombination

    Returns
    -------
    w0waCDM
        Cosmology object with the solved H0 value
    """
    def objective(H0: float) -> float:
        cosmo = cosmo_omkw0wa(H0, omch2, ombh2, omk, w0, wa)
        DA = cosmo.angular_diameter_distance(z_rec).value
        return DA - (rstar / Rfid)

    H0_solution = brentq(objective, 30, 1000)
    return cosmo_omkw0wa(H0_solution, omch2, ombh2, omk, w0, wa)

"""
BAO and supernova data constants.

This module contains the DESI BAO measurements and supernova data
used in the SVD analysis.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

# Default redshift grid for distance calculations
DEFAULT_REDSHIFTS: NDArray[np.floating] = np.logspace(-2, 1, 200)

# DESI BAO data from the DESI paper
ZEFF: list[float] = [0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330, 0.922, 0.955]
DV_RD: list[float] = [7.942, 12.720, 16.050, 19.721, 24.252, 26.055, 31.267, 19.656, 20.008]
DV_RD_ERR: list[float] = [0.075, 0.099, 0.110, 0.091, 0.174, 0.398, 0.256, 0.105, 0.183]
DM_DH: list[float | None] = [None, 0.622, 0.892, 1.223, 1.948, 2.386, 4.518, 1.232, 1.220]
DM_DH_ERR: list[float | None] = [None, 0.017, 0.021, 0.019, 0.045, 0.136, 0.097, 0.021, 0.033]
R_VM_H: list[float | None] = [None, 0.050, -0.018, 0.056, 0.202, 0.044, 0.574, 0.106, 0.420]
TRACERS: list[str] = ["BGS", "LRG1", "LRG2", "LRG3+ELG1", "ELG2", "QSO", "Lya", "LRG3", "ELG1"]

# Digitized BAO data from https://arxiv.org/pdf/2503.24343
Z_BAO_DIGITIZED: NDArray[np.floating] = np.asarray([
    0.2795468926242813,
    0.4916916558914964,
    0.6987812594569315,
    0.9311466615312664,
    1.317436518198575,
    1.4781011857272552,
    2.343530771135382,
])
Y_BAO_DIGITIZED: NDArray[np.floating] = np.asarray([
    -0.013879506276443233,
    -0.008858351690618552,
    -0.025086891150040213,
    -0.011522780516801289,
    -0.00830029021993776,
    0.003521102136438338,
    -0.0027126822616175376,
])
YERR_BAO_DIGITIZED: NDArray[np.floating] = np.asarray([
    0.00881989,
    0.00775552,
    0.00691213,
    0.00473723,
    0.00658764,
    0.01601594,
    0.00834435,
])

# Supernova data
Z_SN: NDArray[np.floating] = np.asarray([
    0.0645894569128918,
    0.21711809466252782,
    0.35937972145152874,
    0.5050064555119084,
    0.7093165398456172,
    1.0868266351674314,
])
Y_SN: NDArray[np.floating] = np.asarray([
    0.04172966191655676,
    -0.0020153118666966924,
    0.007348730980795831,
    -0.005351111397046998,
    -0.03980602824331715,
    0.024857657541737705,
])
YERR_SN: NDArray[np.floating] = np.asarray([
    0.01787242,
    0.01182488,
    0.01310516,
    0.01114614,
    0.01737382,
    0.02713353,
])


class BAOData(TypedDict, total=False):
    """Type definition for BAO data dictionary."""

    z: NDArray[np.floating]
    DV_over_rdrag: NDArray[np.floating]
    eDV_over_rdrag: NDArray[np.floating]
    zeff: list[float]
    Dv_rd: list[float]
    Dv_rd_err: list[float]
    DM_DH: list[float | None]
    DM_DH_err: list[float | None]
    r_VM_H: list[float | None]
    tracers: list[str]
    Dv_rd_ratio: NDArray[np.floating]
    Dv_rd_err_ratio: NDArray[np.floating]
    DM_DH_ratio: NDArray[np.floating]
    DM_DH_err_ratio: NDArray[np.floating]


class SNData(TypedDict):
    """Type definition for supernova data dictionary."""

    z: NDArray[np.floating]
    mu: NDArray[np.floating]
    emu: NDArray[np.floating]


def get_bao_data() -> BAOData:
    """
    Get BAO data dictionary with all measurements.

    Returns
    -------
    BAOData
        Dictionary containing BAO measurements with keys:
        - 'z': digitized redshifts
        - 'DV_over_rdrag': digitized DV/rd values
        - 'eDV_over_rdrag': digitized DV/rd errors
        - 'zeff': effective redshifts from DESI paper
        - 'Dv_rd': DV/rd values
        - 'Dv_rd_err': DV/rd errors
        - 'DM_DH': DM/DH values
        - 'DM_DH_err': DM/DH errors
        - 'r_VM_H': correlation coefficients
        - 'tracers': tracer names
    """
    return {
        "z": Z_BAO_DIGITIZED,
        "DV_over_rdrag": Y_BAO_DIGITIZED,
        "eDV_over_rdrag": YERR_BAO_DIGITIZED,
        "zeff": ZEFF,
        "Dv_rd": DV_RD,
        "Dv_rd_err": DV_RD_ERR,
        "DM_DH": DM_DH,
        "DM_DH_err": DM_DH_ERR,
        "r_VM_H": R_VM_H,
        "tracers": TRACERS,
    }


def get_sn_data() -> SNData:
    """
    Get supernova data dictionary.

    Returns
    -------
    SNData
        Dictionary containing supernova measurements with keys:
        - 'z': redshifts
        - 'mu': distance modulus values
        - 'emu': distance modulus errors
    """
    return {
        "z": Z_SN,
        "mu": Y_SN,
        "emu": YERR_SN,
    }

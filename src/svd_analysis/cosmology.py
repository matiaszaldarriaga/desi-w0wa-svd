"""
Cosmology functions for distance calculations.

This module provides functions to create cosmology objects and compute
various distance measures used in BAO analysis.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from astropy.cosmology import w0waCDM
import astropy.units as u

# Speed of light in km/s
C_LIGHT_KMS: float = 299792.458


class CosmoDict(TypedDict, total=False):
    """Type definition for cosmology dictionary."""

    cosmo: w0waCDM
    redshifts: NDArray[np.floating]
    rdrag: float
    rstar: float
    H0: float
    ombh2: float
    omch2: float
    omk: float
    w0: float
    wa: float
    Rfid: float
    z_rec: float
    DV_over_rdrag: NDArray[np.floating]
    DM_over_DH: NDArray[np.floating]
    mu: NDArray[np.floating]
    DV_over_rdrag_ratio: NDArray[np.floating]
    DM_over_DH_ratio: NDArray[np.floating]
    mu_diff: NDArray[np.floating]


def cosmo_omkw0wa(
    H0: float,
    omch2: float,
    ombh2: float,
    omk: float,
    w0: float,
    wa: float,
    Neff: float = 3.046,
) -> w0waCDM:
    """
    Create a w0waCDM cosmology with given parameters.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc
    omch2 : float
        Physical cold dark matter density (Omega_c * h^2)
    ombh2 : float
        Physical baryon density (Omega_b * h^2)
    omk : float
        Curvature density parameter (0 for flat)
    w0 : float
        Dark energy equation of state at z=0
    wa : float
        Dark energy equation of state evolution parameter
    Neff : float, optional
        Effective number of neutrino species (default: 3.046)

    Returns
    -------
    w0waCDM
        Cosmology object with the specified parameters
    """
    h = H0 / 100.0
    Oc = omch2 / h**2
    Ob = ombh2 / h**2
    Om = Oc + Ob
    Ode = 1 - Om - omk
    return w0waCDM(
        H0=H0,
        Om0=Om,
        Ode0=Ode,
        Ob0=Ob,
        w0=w0,
        wa=wa,
        Tcmb0=2.7255 * u.K,
        Neff=Neff,
    )


def compute_DV_over_rdrag(
    cosmo: w0waCDM,
    redshifts: NDArray[np.floating],
    rdrag: float,
) -> NDArray[np.floating]:
    """
    Calculate the volume-averaged distance divided by the sound horizon at drag epoch.

    Parameters
    ----------
    cosmo : w0waCDM
        Astropy cosmology object
    redshifts : NDArray
        Array of redshift values
    rdrag : float
        Sound horizon at drag epoch in Mpc

    Returns
    -------
    NDArray
        DV/rdrag values at each redshift
    """
    DM = cosmo.comoving_transverse_distance(redshifts).value  # Mpc
    DH = C_LIGHT_KMS * redshifts / cosmo.H(redshifts).value  # Mpc
    DV = (DM**2 * DH) ** (1 / 3)  # Mpc
    return DV / rdrag


def compute_DM_over_DH(
    cosmo: w0waCDM,
    redshifts: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Calculate the ratio of comoving angular diameter distance to the Hubble distance.

    Parameters
    ----------
    cosmo : w0waCDM
        Astropy cosmology object
    redshifts : NDArray
        Array of redshift values

    Returns
    -------
    NDArray
        DM/DH values at each redshift
    """
    DM = cosmo.comoving_transverse_distance(redshifts).value  # Mpc
    DH = C_LIGHT_KMS / cosmo.H(redshifts).value  # Mpc
    return DM / DH


def compute_distance_modulus(
    cosmo: w0waCDM,
    redshifts: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Calculate the distance modulus mu = 5 * log10(dL / 10 pc).

    Parameters
    ----------
    cosmo : w0waCDM
        Astropy cosmology object
    redshifts : NDArray
        Array of redshift values

    Returns
    -------
    NDArray
        Distance modulus values at each redshift
    """
    dL = cosmo.luminosity_distance(redshifts).to("pc").value
    return 5 * np.log10(dL / 10.0)


def compute_distances(
    cosmo: w0waCDM,
    redshifts: NDArray[np.floating],
    rdrag: float,
) -> CosmoDict:
    """
    Compute all distance measures for a cosmology.

    Parameters
    ----------
    cosmo : w0waCDM
        Astropy cosmology object
    redshifts : NDArray
        Array of redshift values
    rdrag : float
        Sound horizon at drag epoch in Mpc

    Returns
    -------
    CosmoDict
        Dictionary containing:
        - cosmo: the cosmology object
        - redshifts: the redshift array
        - rdrag: the sound horizon
        - DV_over_rdrag: volume-averaged distance / rdrag
        - DM_over_DH: comoving distance / Hubble distance
        - mu: distance modulus
    """
    return {
        "cosmo": cosmo,
        "redshifts": redshifts,
        "rdrag": rdrag,
        "DV_over_rdrag": compute_DV_over_rdrag(cosmo, redshifts, rdrag),
        "DM_over_DH": compute_DM_over_DH(cosmo, redshifts),
        "mu": compute_distance_modulus(cosmo, redshifts),
    }


def compute_distance_ratios(
    cosmo_dic: CosmoDict,
    cosmo_dic_ref: CosmoDict,
) -> CosmoDict:
    """
    Calculate ratios of distances between two cosmologies.

    Parameters
    ----------
    cosmo_dic : CosmoDict
        Target cosmology dictionary with computed distances
    cosmo_dic_ref : CosmoDict
        Reference cosmology dictionary with computed distances

    Returns
    -------
    CosmoDict
        New dictionary with original values plus ratio keys:
        - DV_over_rdrag_ratio
        - DM_over_DH_ratio
        - mu_diff
    """
    return {
        **cosmo_dic,
        "DV_over_rdrag_ratio": cosmo_dic["DV_over_rdrag"] / cosmo_dic_ref["DV_over_rdrag"],
        "DM_over_DH_ratio": cosmo_dic["DM_over_DH"] / cosmo_dic_ref["DM_over_DH"],
        "mu_diff": cosmo_dic["mu"] - cosmo_dic_ref["mu"],
    }


# Legacy compatibility aliases (deprecated, will be removed)
def DV_over_rdrag(cosmo_dic: dict) -> None:
    """Deprecated: Use compute_DV_over_rdrag instead."""
    cosmo_dic["DV_over_rdrag"] = compute_DV_over_rdrag(
        cosmo_dic["cosmo"], cosmo_dic["redshifts"], cosmo_dic["rdrag"]
    )


def DM_over_DH(cosmo_dic: dict) -> None:
    """Deprecated: Use compute_DM_over_DH instead."""
    cosmo_dic["DM_over_DH"] = compute_DM_over_DH(
        cosmo_dic["cosmo"], cosmo_dic["redshifts"]
    )


def distance_modulus(cosmo_dic: dict) -> None:
    """Deprecated: Use compute_distance_modulus instead."""
    cosmo_dic["mu"] = compute_distance_modulus(
        cosmo_dic["cosmo"], cosmo_dic["redshifts"]
    )


def DV_over_rdrag_ratio(cosmo_dic: dict, cosmo_dic_ref: dict) -> None:
    """Deprecated: Use compute_distance_ratios instead."""
    cosmo_dic["redshifts"] = cosmo_dic_ref["redshifts"]
    cosmo_dic["DV_over_rdrag_ratio"] = (
        cosmo_dic["DV_over_rdrag"] / cosmo_dic_ref["DV_over_rdrag"]
    )
    cosmo_dic["DM_over_DH_ratio"] = (
        cosmo_dic["DM_over_DH"] / cosmo_dic_ref["DM_over_DH"]
    )


def distance_modulus_diff(cosmo_dic: dict, cosmo_dic_ref: dict) -> None:
    """Deprecated: Use compute_distance_ratios instead."""
    cosmo_dic["redshifts"] = cosmo_dic_ref["redshifts"]
    cosmo_dic["mu"] = compute_distance_modulus(
        cosmo_dic["cosmo"], cosmo_dic["redshifts"]
    )
    cosmo_dic["mu_diff"] = cosmo_dic["mu"] - cosmo_dic_ref["mu"]

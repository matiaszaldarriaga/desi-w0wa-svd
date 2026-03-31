"""
Plotting functions for BAO distance analysis.

This module provides visualization functions for BAO distance
measurements and cosmological model comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .cosmology import CosmoDict
from .data import BAOData


def BAO_dist_plot_v2(
    samples_dic: dict[str, Any],
    bao_data: BAOData,
    title: str = "",
    cosmo_list: list[CosmoDict] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Create a 2-panel plot showing DV/rd ratio and DM/DH ratio vs redshift.

    Parameters
    ----------
    samples_dic : dict
        Dictionary containing:
        - 'redshifts': redshift array
        - 'DV_over_rdrag': mean DV/rd ratio
        - 'std_DV_over_rdrag': std of DV/rd ratio
        - 'DM_over_DH': mean DM/DH ratio
        - 'std_DM_over_DH': std of DM/DH ratio
    bao_data : BAOData
        Dictionary containing:
        - 'zeff': effective redshifts
        - 'Dv_rd_ratio': DV/rd ratio data
        - 'Dv_rd_err_ratio': DV/rd ratio errors
        - 'DM_DH_ratio': DM/DH ratio data
        - 'DM_DH_err_ratio': DM/DH ratio errors
    title : str, optional
        Plot title
    cosmo_list : list[CosmoDict], optional
        List of cosmology dictionaries to overlay. Each must have:
        - 'redshifts': redshift array
        - 'DV_over_rdrag_ratio': DV/rd ratio
        - 'DM_over_DH_ratio': DM/DH ratio
        - 'label': legend label
    save_path : str | Path, optional
        If provided, save figure to this path

    Returns
    -------
    Figure
        The created matplotlib figure object
    """
    redshifts: NDArray = samples_dic["redshifts"]
    mean_DV_rdrag: NDArray = samples_dic["DV_over_rdrag"]
    std_DV_rdrag: NDArray = samples_dic["std_DV_over_rdrag"]
    mean_DM_DH: NDArray = samples_dic["DM_over_DH"]
    std_DM_DH: NDArray = samples_dic["std_DM_over_DH"]

    # BAO data for DV/rd (first 7 tracers)
    z_bao_dv = bao_data["zeff"][0:7]
    y_bao_dv = bao_data["Dv_rd_ratio"][0:7]
    yerr_bao_dv = bao_data["Dv_rd_err_ratio"][0:7]

    fig = plt.figure(figsize=(8, 4))

    # Left panel: DV/rd
    plt.subplot(121)
    plt.plot(
        redshifts,
        mean_DV_rdrag,
        label=r"$D_V/r_d$",
        color="red",
        linestyle="--",
    )
    plt.fill_between(
        redshifts,
        mean_DV_rdrag - std_DV_rdrag,
        mean_DV_rdrag + std_DV_rdrag,
        color="red",
        alpha=0.4,
        label=r"1$\sigma$ range",
    )

    plt.axhline(1.0, color="k", linestyle="-", linewidth=0.8)

    plt.errorbar(
        z_bao_dv,
        np.array(y_bao_dv),
        yerr=yerr_bao_dv,
        fmt="o",
        color="black",
        label="DESI DR2",
    )

    if cosmo_list is not None:
        for cosmo in cosmo_list:
            plt.plot(
                cosmo["redshifts"],
                cosmo["DV_over_rdrag_ratio"],
                linestyle="--",
                label=cosmo.get("label", ""),
            )

    plt.xlabel("Redshift z")
    plt.ylabel(r"$(D_V/r_d)/ D_V/r_d|_{ref}$")
    plt.xlim(0.1, 2.8)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xscale("log")

    # Right panel: DM/DH
    plt.subplot(122)

    plt.plot(
        redshifts,
        mean_DM_DH,
        label=r"$D_M/D_H$",
        color="red",
        linestyle="--",
    )
    plt.fill_between(
        redshifts,
        mean_DM_DH - std_DM_DH,
        mean_DM_DH + std_DM_DH,
        color="red",
        alpha=0.4,
        label=r"1$\sigma$ range",
    )

    plt.axhline(1.0, color="k", linestyle="-", linewidth=0.8)

    # BAO data for DM/DH (tracers 1-6, excluding BGS)
    z_bao_dm = bao_data["zeff"][1:7]
    y_bao_dm = bao_data["DM_DH_ratio"][1:7]
    yerr_bao_dm = bao_data["DM_DH_err_ratio"][1:7]

    plt.errorbar(
        z_bao_dm,
        np.array(y_bao_dm),
        yerr=yerr_bao_dm,
        fmt="o",
        color="black",
        label="DESI DR2",
    )

    if cosmo_list is not None:
        for cosmo in cosmo_list:
            plt.plot(
                cosmo["redshifts"],
                cosmo["DM_over_DH_ratio"],
                linestyle="--",
                label=cosmo.get("label", ""),
            )

    plt.xlabel("Redshift z")
    plt.ylabel(r"$(D_M/D_H)/ D_M/D_H|_{ref}$")
    plt.xlim(0.1, 2.8)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xscale("log")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

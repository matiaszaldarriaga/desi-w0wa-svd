"""
SVD Analysis Package for DESI BAO w0wa Investigation

This package provides tools for analyzing DESI BAO data using SVD decomposition
to investigate dark energy equation of state parameters (w0, wa).

Modules:
    cosmology: Cosmology model creation and distance calculations
    chains: MCMC chain loading and sample processing
    data: BAO and supernova data constants
    plotting: Visualization functions
    utils: Interpolation and helper functions
"""

from .cosmology import (
    CosmoDict,
    cosmo_omkw0wa,
    compute_DV_over_rdrag,
    compute_DM_over_DH,
    compute_distance_modulus,
    compute_distances,
    compute_distance_ratios,
    # Legacy aliases (deprecated)
    DV_over_rdrag,
    DM_over_DH,
    distance_modulus,
    DV_over_rdrag_ratio,
    distance_modulus_diff,
)

from .data import (
    BAOData,
    SNData,
    get_bao_data,
    get_sn_data,
    DEFAULT_REDSHIFTS,
)

from .official_data import (
    OfficialBAOData,
    BinnedSNData,
    load_official_bao_data,
    load_pantheon_plus,
    load_des_y5,
    load_des_dovekie,
    load_union3,
    get_union3_bin_grid,
    bin_sn_data,
    compute_effective_redshifts,
    print_bao_summary,
)

from .utils import (
    interpolate_to_redshifts,
    interpolate_bao_samples,
    interpolate_cosmo_to_bao,
    cosmo_solve_H0_omkw0wa,
)

from .plotting import (
    BAO_dist_plot_v2,
)

# Chain functions require getdist - import conditionally
try:
    from .chains import (
        SamplesDict,
        ChainSummary,
        load_samples,
        compute_distance_samples,
        # Legacy alias (deprecated)
        distances_samples,
    )
    _HAS_GETDIST = True
except ImportError:
    _HAS_GETDIST = False
    SamplesDict = None
    ChainSummary = None
    load_samples = None
    compute_distance_samples = None
    distances_samples = None

__version__ = "0.1.0"
__all__ = [
    # Type definitions
    "CosmoDict",
    "BAOData",
    "SNData",
    "SamplesDict",
    "ChainSummary",
    # Cosmology (new API)
    "cosmo_omkw0wa",
    "compute_DV_over_rdrag",
    "compute_DM_over_DH",
    "compute_distance_modulus",
    "compute_distances",
    "compute_distance_ratios",
    # Cosmology (legacy, deprecated)
    "DV_over_rdrag",
    "DM_over_DH",
    "distance_modulus",
    "DV_over_rdrag_ratio",
    "distance_modulus_diff",
    # Chains (new API)
    "load_samples",
    "compute_distance_samples",
    # Chains (legacy, deprecated)
    "distances_samples",
    # Data (legacy)
    "get_bao_data",
    "get_sn_data",
    "DEFAULT_REDSHIFTS",
    # Official data (paper-grade)
    "OfficialBAOData",
    "BinnedSNData",
    "load_official_bao_data",
    "load_pantheon_plus",
    "load_des_y5",
    "load_des_dovekie",
    "load_union3",
    "get_union3_bin_grid",
    "bin_sn_data",
    "compute_effective_redshifts",
    "print_bao_summary",
    # Utils
    "interpolate_to_redshifts",
    "interpolate_bao_samples",
    "interpolate_cosmo_to_bao",
    "cosmo_solve_H0_omkw0wa",
    # Plotting
    "BAO_dist_plot_v2",
]

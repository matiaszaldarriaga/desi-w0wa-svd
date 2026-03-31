"""
Pytest configuration and fixtures for svd_analysis tests.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

# Get paths relative to this file
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent  # release/ is one level up from tests/
REFERENCE_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = PROJECT_ROOT / "data" / "chains"


@pytest.fixture
def reference_chi2():
    """Load reference chi-squared values from JSON."""
    with open(REFERENCE_DATA_DIR / "chi2_values.json") as f:
        return json.load(f)


@pytest.fixture
def planck_reference_params():
    """
    Planck 2018 ΛCDM reference cosmology parameters.

    These are approximate values; actual values should be loaded from chains.
    """
    return {
        "H0": 67.36,
        "ombh2": 0.02237,
        "omch2": 0.1200,
        "rdrag": 147.09,
        "rstar": 144.43,
        "omk": 0,
        "w0": -1,
        "wa": 0,
        "z_rec": 1090,
    }


@pytest.fixture
def default_redshifts():
    """Default redshift grid for distance calculations."""
    return np.logspace(-2, 1, 200)


@pytest.fixture
def bao_zeff():
    """DESI BAO effective redshifts."""
    return [0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330, 0.922, 0.955]


@pytest.fixture
def chain_paths():
    """
    Paths to MCMC chain files.

    Returns a dictionary with chain identifiers and their paths.
    Returns None for chains that don't exist.
    """
    paths = {
        "act_lcdm": DATA_DIR / "p-actbase_lcdm_camb" / "p-actbase_lcdm_camb",
        "planck18": DATA_DIR / "COM_CosmoParams_fullGrid_R3.01" / "base" / "plikHM_TTTEEE_lowl_lowE" / "base_plikHM_TTTEEE_lowl_lowE",
        "w0wa": DATA_DIR / "desi_dr2_official" / "base_w_wa" / "chain",
        "alens": DATA_DIR / "p-actbase_alens_camb" / "p-actbase_alens_camb",
        "bprim": DATA_DIR / "p-actbase_bprim_class" / "p-actbase_bprim_class",
    }

    # Convert to strings and check existence
    result = {}
    for key, path in paths.items():
        # Check if any chain file exists (e.g., *.1.txt)
        chain_file = Path(str(path) + ".1.txt")
        if chain_file.exists():
            result[key] = str(path)
        else:
            result[key] = None

    return result


@pytest.fixture
def has_chain_data(chain_paths):
    """Check if any chain data is available."""
    return any(p is not None for p in chain_paths.values())


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_chains: mark test as requiring MCMC chain data"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require chain data if chains are not available."""
    # Check if chain data exists
    chain_file = DATA_DIR / "p-actbase_lcdm_camb" / "p-actbase_lcdm_camb.1.txt"
    has_chains = chain_file.exists()

    if not has_chains:
        skip_chains = pytest.mark.skip(reason="MCMC chain data not available")
        for item in items:
            if "requires_chains" in item.keywords:
                item.add_marker(skip_chains)

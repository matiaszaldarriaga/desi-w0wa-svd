"""
Tests for SN data binning to common redshift grid.

Tests cover:
- Identity property (Union3 binned to its own grid)
- Positive-definite binned covariance for all datasets
- Correct output shape (22 bins)
- Empty bin handling
- Consistency with cross_dataset_analysis.py visualization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from svd_analysis.official_data import (
    bin_sn_data,
    get_union3_bin_grid,
    load_union3,
    load_pantheon_plus,
    load_des_y5,
)


@pytest.fixture(scope="module")
def union3_data():
    try:
        return load_union3()
    except FileNotFoundError:
        pytest.skip("Union3 data not available")


@pytest.fixture(scope="module")
def pantheon_data():
    try:
        return load_pantheon_plus()
    except FileNotFoundError:
        pytest.skip("Pantheon+ data not available")


@pytest.fixture(scope="module")
def des_y5_data():
    try:
        return load_des_y5()
    except FileNotFoundError:
        pytest.skip("DES Y5 data not available")


@pytest.fixture(scope="module")
def union3_grid(union3_data):
    return get_union3_bin_grid()


class TestUnion3Identity:
    """Binning Union3 to its own grid should approximately recover the same data."""

    def test_mu_unchanged(self, union3_data, union3_grid):
        """Binned mu should be very close to original mu."""
        binned = bin_sn_data(
            union3_data['z'], union3_data['mu'],
            union3_data['covariance'], union3_grid
        )
        # Each Union3 bin has exactly 1 SN, so binned mu = original mu
        np.testing.assert_allclose(binned['mu'], union3_data['mu'], atol=1e-10)

    def test_z_unchanged(self, union3_data, union3_grid):
        """Binned z should match original z."""
        binned = bin_sn_data(
            union3_data['z'], union3_data['mu'],
            union3_data['covariance'], union3_grid
        )
        np.testing.assert_allclose(binned['z'], union3_data['z'], atol=1e-10)

    def test_covariance_close(self, union3_data, union3_grid):
        """Binned covariance should be very close to original."""
        binned = bin_sn_data(
            union3_data['z'], union3_data['mu'],
            union3_data['covariance'], union3_grid
        )
        np.testing.assert_allclose(
            binned['covariance'], union3_data['covariance'], atol=1e-10
        )

    def test_one_per_bin(self, union3_data, union3_grid):
        """Each bin should contain exactly 1 SN for Union3."""
        binned = bin_sn_data(
            union3_data['z'], union3_data['mu'],
            union3_data['covariance'], union3_grid
        )
        np.testing.assert_array_equal(binned['n_per_bin'], np.ones(22, dtype=int))


class TestBinnedCovariancePositiveDefinite:
    """Binned covariance should be positive semi-definite for all datasets."""

    def test_union3_pd(self, union3_data, union3_grid):
        binned = bin_sn_data(
            union3_data['z'], union3_data['mu'],
            union3_data['covariance'], union3_grid
        )
        eigvals = np.linalg.eigvalsh(binned['covariance'])
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalues: {eigvals[eigvals < -1e-10]}"

    def test_pantheon_pd(self, pantheon_data, union3_grid):
        binned = bin_sn_data(
            pantheon_data['z'], pantheon_data['mu'],
            pantheon_data['covariance'], union3_grid
        )
        eigvals = np.linalg.eigvalsh(binned['covariance'])
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalues: {eigvals[eigvals < -1e-10]}"

    def test_des_y5_pd(self, des_y5_data, union3_grid):
        """DES Y5 binned covariance should be PD despite raw cov having 805 negative eigenvalues."""
        binned = bin_sn_data(
            des_y5_data['z'], des_y5_data['mu'],
            des_y5_data['covariance'], union3_grid
        )
        eigvals = np.linalg.eigvalsh(binned['covariance'])
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalues: {eigvals[eigvals < -1e-10]}"


class TestBinnedShape:
    """All binned datasets should produce (22, 22) covariance."""

    def test_union3_shape(self, union3_data, union3_grid):
        binned = bin_sn_data(
            union3_data['z'], union3_data['mu'],
            union3_data['covariance'], union3_grid
        )
        assert binned['covariance'].shape == (22, 22)
        assert len(binned['mu']) == 22
        assert len(binned['z']) == 22

    def test_pantheon_shape(self, pantheon_data, union3_grid):
        binned = bin_sn_data(
            pantheon_data['z'], pantheon_data['mu'],
            pantheon_data['covariance'], union3_grid
        )
        assert binned['covariance'].shape == (22, 22)
        assert len(binned['mu']) == 22

    def test_des_y5_shape(self, des_y5_data, union3_grid):
        binned = bin_sn_data(
            des_y5_data['z'], des_y5_data['mu'],
            des_y5_data['covariance'], union3_grid
        )
        assert binned['covariance'].shape == (22, 22)
        assert len(binned['mu']) == 22


class TestEmptyBins:
    """DES Y5 should have empty bins at high z (z > 1.12)."""

    def test_des_y5_high_z_empty(self, des_y5_data, union3_grid):
        binned = bin_sn_data(
            des_y5_data['z'], des_y5_data['mu'],
            des_y5_data['covariance'], union3_grid
        )
        # DES Y5 max z is ~1.12; Union3 goes to ~1.5
        # High-z bins with no data should have large variance
        high_z_mask = union3_grid > des_y5_data['z'].max() + 0.05
        for j in np.where(high_z_mask)[0]:
            assert binned['n_per_bin'][j] == 0
            assert binned['covariance'][j, j] >= 1e5, (
                f"Empty bin {j} (z={union3_grid[j]:.2f}) should have large variance"
            )

    def test_empty_bins_dont_affect_others(self, des_y5_data, union3_grid):
        """Empty bins shouldn't have correlations with populated bins."""
        binned = bin_sn_data(
            des_y5_data['z'], des_y5_data['mu'],
            des_y5_data['covariance'], union3_grid
        )
        empty = binned['n_per_bin'] == 0
        populated = ~empty
        if np.any(empty):
            # Off-diagonal blocks between empty and populated should be zero
            cross_cov = binned['covariance'][np.ix_(empty, populated)]
            np.testing.assert_allclose(cross_cov, 0.0, atol=1e-10)


class TestBinnedMeanConsistency:
    """Binned mu values should be consistent with simple inverse-variance weighted means."""

    def test_pantheon_binned_means(self, pantheon_data, union3_grid):
        """Check a few bins manually against direct computation."""
        binned = bin_sn_data(
            pantheon_data['z'], pantheon_data['mu'],
            pantheon_data['covariance'], union3_grid
        )
        z = pantheon_data['z']
        mu = pantheon_data['mu']
        var = np.diag(pantheon_data['covariance'])

        # Build bin edges the same way
        n_bins = len(union3_grid)
        bin_edges = np.zeros(n_bins + 1)
        bin_edges[0] = max(0, union3_grid[0] - (union3_grid[1] - union3_grid[0]) / 2)
        bin_edges[-1] = union3_grid[-1] + (union3_grid[-1] - union3_grid[-2]) / 2
        for i in range(1, n_bins):
            bin_edges[i] = (union3_grid[i - 1] + union3_grid[i]) / 2

        # Check a few populated bins
        for j in [0, 5, 10, 15]:
            mask = (z >= bin_edges[j]) & (z < bin_edges[j + 1])
            if np.sum(mask) > 0:
                w = 1.0 / var[mask]
                expected_mu = np.average(mu[mask], weights=w)
                np.testing.assert_allclose(
                    binned['mu'][j], expected_mu, atol=1e-10,
                    err_msg=f"Bin {j} mu mismatch"
                )


class TestCustomGrid:
    """Test binning to a custom (non-Union3) grid."""

    def test_coarse_grid(self, union3_data):
        """Binning to 5 coarse bins should still work."""
        coarse_bins = np.array([0.1, 0.3, 0.5, 0.8, 1.2])
        binned = bin_sn_data(
            union3_data['z'], union3_data['mu'],
            union3_data['covariance'], coarse_bins
        )
        assert binned['covariance'].shape == (5, 5)
        assert len(binned['mu']) == 5
        # At least some bins should be populated
        assert np.sum(binned['n_per_bin'] > 0) >= 3

"""
Unit tests for chain loading and sample processing functions.
"""

import numpy as np
import pytest

from svd_analysis.chains import load_samples, distances_samples
from svd_analysis.cosmology import cosmo_omkw0wa, DV_over_rdrag, DM_over_DH, distance_modulus


@pytest.mark.requires_chains
class TestLoadSamples:
    """Tests for load_samples function."""

    def test_load_act_lcdm(self, chain_paths):
        """Test loading ACT LCDM chains."""
        if chain_paths["act_lcdm"] is None:
            pytest.skip("ACT LCDM chains not available")

        samples = load_samples(chain_paths["act_lcdm"])

        assert "samples" in samples
        assert "param_names" in samples
        assert "summary" in samples

    def test_act_lcdm_sample_count(self, chain_paths):
        """Test ACT LCDM chain has expected number of samples."""
        if chain_paths["act_lcdm"] is None:
            pytest.skip("ACT LCDM chains not available")

        samples = load_samples(chain_paths["act_lcdm"])

        # Expected: 755,302 samples according to CODE_EXPLORATION.md
        assert samples["samples"].numrows == 755302

    def test_derived_omh2_added(self, chain_paths):
        """Test that derived omh2 parameter is added."""
        if chain_paths["act_lcdm"] is None:
            pytest.skip("ACT LCDM chains not available")

        samples = load_samples(chain_paths["act_lcdm"])

        assert "omh2" in samples["param_names"]
        assert "omh2" in samples["summary"]

    def test_summary_statistics_computed(self, chain_paths):
        """Test that summary statistics are computed for parameters."""
        if chain_paths["act_lcdm"] is None:
            pytest.skip("ACT LCDM chains not available")

        samples = load_samples(chain_paths["act_lcdm"])

        for param in samples["param_names"]:
            assert "mean" in samples["summary"][param]
            assert "std" in samples["summary"][param]
            assert "frac_err %" in samples["summary"][param]

    def test_load_planck18(self, chain_paths):
        """Test loading Planck 2018 chains."""
        if chain_paths["planck18"] is None:
            pytest.skip("Planck 2018 chains not available")

        samples = load_samples(chain_paths["planck18"])

        assert "samples" in samples
        assert samples["samples"].numrows > 0

    def test_load_w0wa(self, chain_paths):
        """Test loading w0wa chains."""
        if chain_paths["w0wa"] is None:
            pytest.skip("w0wa chains not available")

        samples = load_samples(chain_paths["w0wa"])

        assert "samples" in samples
        # w0wa chains should have ~40-50 parameters
        param_count = len([p.name for p in samples["samples"].paramNames.names])
        assert param_count == pytest.approx(45, abs=10)


@pytest.mark.requires_chains
@pytest.mark.slow
class TestDistancesSamples:
    """Tests for distances_samples function."""

    @pytest.fixture
    def reference_cosmology(self, chain_paths, default_redshifts):
        """Create reference cosmology from Planck parameters."""
        if chain_paths["planck18"] is None:
            pytest.skip("Planck 2018 chains not available")

        samples = load_samples(chain_paths["planck18"])

        H0 = samples["summary"]["H0"]["mean"]
        ombh2 = samples["summary"]["omegabh2"]["mean"]
        omch2 = samples["summary"]["omegach2"]["mean"]
        rdrag = samples["summary"]["rdrag"]["mean"]

        cosmo = cosmo_omkw0wa(H0, omch2, ombh2, 0, -1, 0)

        cosmo_dic = {
            "cosmo": cosmo,
            "redshifts": default_redshifts,
            "rdrag": rdrag,
        }
        DV_over_rdrag(cosmo_dic)
        DM_over_DH(cosmo_dic)
        distance_modulus(cosmo_dic)

        return cosmo_dic

    def test_distances_samples_computes(self, chain_paths, reference_cosmology):
        """Test that distances_samples computes distance samples."""
        if chain_paths["planck18"] is None:
            pytest.skip("Planck 2018 chains not available")

        samples = load_samples(chain_paths["planck18"])
        distances_samples(samples, reference_cosmology, n_random=100)

        assert "DV_over_rdrag_samples" in samples
        assert "DM_over_DH_samples" in samples
        assert "mu_samples" in samples

    def test_distances_samples_shape(self, chain_paths, reference_cosmology):
        """Test that distance sample arrays have correct shape."""
        if chain_paths["planck18"] is None:
            pytest.skip("Planck 2018 chains not available")

        samples = load_samples(chain_paths["planck18"])
        n_random = 100
        distances_samples(samples, reference_cosmology, n_random=n_random)

        n_z = len(reference_cosmology["redshifts"])
        assert samples["DV_over_rdrag_samples"].shape == (n_random, n_z)
        assert samples["DM_over_DH_samples"].shape == (n_random, n_z)

    def test_distances_samples_statistics(self, chain_paths, reference_cosmology):
        """Test that mean and std are computed."""
        if chain_paths["planck18"] is None:
            pytest.skip("Planck 2018 chains not available")

        samples = load_samples(chain_paths["planck18"])
        distances_samples(samples, reference_cosmology, n_random=100)

        assert "DV_over_rdrag" in samples
        assert "std_DV_over_rdrag" in samples
        assert "DM_over_DH" in samples
        assert "std_DM_over_DH" in samples

    def test_distances_samples_reproducibility(self, chain_paths, reference_cosmology):
        """Test that results are reproducible with same seed."""
        if chain_paths["planck18"] is None:
            pytest.skip("Planck 2018 chains not available")

        samples1 = load_samples(chain_paths["planck18"])
        samples2 = load_samples(chain_paths["planck18"])

        distances_samples(samples1, reference_cosmology, n_random=50, seed=42)
        distances_samples(samples2, reference_cosmology, n_random=50, seed=42)

        np.testing.assert_array_equal(
            samples1["DV_over_rdrag_samples"],
            samples2["DV_over_rdrag_samples"],
        )

    def test_lcdm_ratios_near_one(self, chain_paths, reference_cosmology):
        """Test that ΛCDM sample ratios are near 1 (self-consistent)."""
        if chain_paths["planck18"] is None:
            pytest.skip("Planck 2018 chains not available")

        samples = load_samples(chain_paths["planck18"])
        distances_samples(samples, reference_cosmology, n_random=100)

        # Mean should be close to 1 for self-consistent chains
        assert samples["DV_over_rdrag"].mean() == pytest.approx(1.0, abs=0.05)
        assert samples["DM_over_DH"].mean() == pytest.approx(1.0, abs=0.05)


class TestParameterIndexing:
    """Tests for parameter index finding logic."""

    def test_find_h0_or_h(self):
        """Test that H0 or h parameter is found."""
        # This tests the internal logic via distances_samples
        # We test indirectly by ensuring no error is raised
        pass  # Covered by integration tests

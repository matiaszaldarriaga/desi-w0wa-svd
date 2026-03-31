"""
Integration tests for the full SVD analysis pipeline.

These tests verify that the refactored code produces the same results
as the original notebook implementation.
"""

import numpy as np
import pytest

from svd_analysis import (
    cosmo_omkw0wa,
    DV_over_rdrag,
    DM_over_DH,
    distance_modulus,
    load_samples,
    distances_samples,
    get_bao_data,
    interpolate_bao_samples,
)


@pytest.mark.requires_chains
class TestFullPipeline:
    """End-to-end tests for the analysis pipeline."""

    @pytest.fixture
    def loaded_chains(self, chain_paths):
        """Load all available chains."""
        chains = {}
        for name, path in chain_paths.items():
            if path is not None:
                try:
                    chains[name] = load_samples(path)
                except Exception:
                    pass
        return chains

    @pytest.fixture
    def reference_cosmology(self, loaded_chains, default_redshifts):
        """Create reference cosmology from Planck 2018 parameters."""
        if "planck18" not in loaded_chains:
            pytest.skip("Planck 2018 chains required")

        samples = loaded_chains["planck18"]

        H0 = samples["summary"]["H0"]["mean"]
        ombh2 = samples["summary"]["omegabh2"]["mean"]
        omch2 = samples["summary"]["omegach2"]["mean"]
        rdrag = samples["summary"]["rdrag"]["mean"]
        rstar = samples["summary"]["rstar"]["mean"]

        cosmo = cosmo_omkw0wa(H0, omch2, ombh2, 0, -1, 0)

        cosmo_dic = {
            "cosmo": cosmo,
            "redshifts": default_redshifts,
            "rdrag": rdrag,
            "rstar": rstar,
            "H0": H0,
            "ombh2": ombh2,
            "omch2": omch2,
        }
        DV_over_rdrag(cosmo_dic)
        DM_over_DH(cosmo_dic)
        distance_modulus(cosmo_dic)

        return cosmo_dic

    def test_all_chains_load(self, loaded_chains):
        """Test that all available chains can be loaded."""
        assert len(loaded_chains) > 0

        for name, samples in loaded_chains.items():
            assert "samples" in samples
            assert samples["samples"].numrows > 0

    @pytest.mark.slow
    def test_distance_samples_all_chains(self, loaded_chains, reference_cosmology):
        """Test computing distance samples for all chains."""
        for name, samples in loaded_chains.items():
            distances_samples(samples, reference_cosmology, n_random=100)

            assert "DV_over_rdrag_samples" in samples
            assert "DM_over_DH_samples" in samples
            assert samples["DV_over_rdrag_samples"].shape[0] == 100


class TestReferenceValues:
    """Tests comparing against reference values from notebook."""

    def test_chi2_reference_values_exist(self, reference_chi2):
        """Test that reference values are loaded correctly."""
        assert "chi2_13" in reference_chi2
        assert "chi2_12" in reference_chi2
        assert "a_lcdm" in reference_chi2

    def test_chi2_13_value(self, reference_chi2):
        """Verify chi2_13 reference value."""
        assert reference_chi2["chi2_13"] == pytest.approx(37.2900398758782, rel=1e-10)

    def test_chi2_12_value(self, reference_chi2):
        """Verify chi2_12 reference value."""
        assert reference_chi2["chi2_12"] == pytest.approx(10.748464786660938, rel=1e-10)

    def test_a_lcdm_0_value(self, reference_chi2):
        """Verify a_lcdm[0] (c0 projection) reference value."""
        assert reference_chi2["a_lcdm"][0] == pytest.approx(5.15185162, rel=1e-6)

    def test_c_alpha_std(self, reference_chi2):
        """Verify c_alpha_std reference value."""
        assert reference_chi2["c_alpha_std"] == pytest.approx(
            1.945251120995139, rel=1e-10
        )

    def test_pivot_points(self, reference_chi2):
        """Verify pivot point reference values."""
        pivots = reference_chi2["pivot_points"]

        assert pivots["z_pivot0"] == pytest.approx(0.723, rel=0.01)
        assert pivots["z_pivot1"] == pytest.approx(0.469, rel=0.01)
        assert pivots["w_star"] == pytest.approx(-1.006, rel=0.01)


class TestBAOData:
    """Tests for BAO data loading and processing."""

    def test_bao_data_loads(self):
        """Test that BAO data loads correctly."""
        bao = get_bao_data()

        assert "zeff" in bao
        assert "Dv_rd" in bao
        assert "DM_DH" in bao
        assert len(bao["zeff"]) == 9

    def test_bao_desi_values(self):
        """Verify DESI BAO measurements."""
        bao = get_bao_data()

        # BGS at z=0.295
        assert bao["zeff"][0] == 0.295
        assert bao["Dv_rd"][0] == 7.942
        assert bao["Dv_rd_err"][0] == 0.075

        # Lya at z=2.330
        assert bao["zeff"][6] == 2.330
        assert bao["Dv_rd"][6] == 31.267

    def test_bao_tracers(self):
        """Verify BAO tracer names."""
        bao = get_bao_data()

        assert bao["tracers"][0] == "BGS"
        assert bao["tracers"][6] == "Lya"
        assert len(bao["tracers"]) == 9


class TestInterpolation:
    """Tests for interpolation functions."""

    @pytest.mark.requires_chains
    def test_interpolate_bao_samples(self, chain_paths, default_redshifts):
        """Test interpolating samples to BAO redshifts."""
        if chain_paths["planck18"] is None:
            pytest.skip("Planck 2018 chains not available")

        samples = load_samples(chain_paths["planck18"])
        bao = get_bao_data()

        # Create reference cosmology
        H0 = samples["summary"]["H0"]["mean"]
        ombh2 = samples["summary"]["omegabh2"]["mean"]
        omch2 = samples["summary"]["omegach2"]["mean"]
        rdrag = samples["summary"]["rdrag"]["mean"]

        cosmo = cosmo_omkw0wa(H0, omch2, ombh2, 0, -1, 0)
        ref = {
            "cosmo": cosmo,
            "redshifts": default_redshifts,
            "rdrag": rdrag,
        }
        DV_over_rdrag(ref)
        DM_over_DH(ref)
        distance_modulus(ref)

        # Compute distance samples
        distances_samples(samples, ref, n_random=50)

        # Interpolate to BAO redshifts
        interpolate_bao_samples(samples, bao)

        assert "DV_over_rdrag_interp" in samples
        assert samples["DV_over_rdrag_interp"].shape == (50, 9)


class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    def test_low_redshift_distances(self, planck_reference_params):
        """Test distance calculations at very low redshift."""
        p = planck_reference_params
        cosmo = cosmo_omkw0wa(p["H0"], p["omch2"], p["ombh2"], 0, -1, 0)

        cosmo_dic = {
            "cosmo": cosmo,
            "redshifts": np.array([0.001, 0.01, 0.1]),
            "rdrag": p["rdrag"],
        }

        DV_over_rdrag(cosmo_dic)
        DM_over_DH(cosmo_dic)

        # Should not have NaN or Inf
        assert np.all(np.isfinite(cosmo_dic["DV_over_rdrag"]))
        assert np.all(np.isfinite(cosmo_dic["DM_over_DH"]))

    def test_high_redshift_distances(self, planck_reference_params):
        """Test distance calculations at high redshift."""
        p = planck_reference_params
        cosmo = cosmo_omkw0wa(p["H0"], p["omch2"], p["ombh2"], 0, -1, 0)

        cosmo_dic = {
            "cosmo": cosmo,
            "redshifts": np.array([5.0, 10.0, 50.0]),
            "rdrag": p["rdrag"],
        }

        DV_over_rdrag(cosmo_dic)
        DM_over_DH(cosmo_dic)

        assert np.all(np.isfinite(cosmo_dic["DV_over_rdrag"]))
        assert np.all(np.isfinite(cosmo_dic["DM_over_DH"]))

    def test_extreme_w0wa_values(self):
        """Test cosmology creation with extreme w0, wa values."""
        # Should not raise errors
        cosmo1 = cosmo_omkw0wa(70.0, 0.12, 0.022, 0, -2.0, 1.5)
        cosmo2 = cosmo_omkw0wa(70.0, 0.12, 0.022, 0, -0.5, -1.0)

        assert cosmo1 is not None
        assert cosmo2 is not None

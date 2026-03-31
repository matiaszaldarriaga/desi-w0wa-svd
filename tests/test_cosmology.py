"""
Unit tests for cosmology functions.
"""

import numpy as np
import pytest

from svd_analysis.cosmology import (
    cosmo_omkw0wa,
    DV_over_rdrag,
    DM_over_DH,
    distance_modulus,
    DV_over_rdrag_ratio,
    distance_modulus_diff,
)


class TestCosmoCreation:
    """Tests for cosmo_omkw0wa function."""

    def test_flat_lcdm_creation(self, planck_reference_params):
        """Test creating a flat ΛCDM cosmology."""
        p = planck_reference_params
        cosmo = cosmo_omkw0wa(p["H0"], p["omch2"], p["ombh2"], 0, -1, 0)

        # Check basic properties
        assert cosmo.H0.value == pytest.approx(p["H0"], rel=1e-6)
        assert cosmo.w0 == -1
        assert cosmo.wa == 0

    def test_om0_calculation(self, planck_reference_params):
        """Test that Om0 is correctly computed from omch2 and ombh2."""
        p = planck_reference_params
        cosmo = cosmo_omkw0wa(p["H0"], p["omch2"], p["ombh2"], 0, -1, 0)

        h = p["H0"] / 100.0
        expected_Om0 = (p["omch2"] + p["ombh2"]) / h**2

        assert cosmo.Om0 == pytest.approx(expected_Om0, rel=1e-6)

    def test_w0wa_creation(self):
        """Test creating a w0waCDM cosmology with non-trivial w0, wa."""
        cosmo = cosmo_omkw0wa(70.0, 0.12, 0.022, 0, -0.9, 0.3)

        assert cosmo.w0 == -0.9
        assert cosmo.wa == 0.3

    def test_curved_cosmology(self):
        """Test creating a curved cosmology."""
        cosmo = cosmo_omkw0wa(70.0, 0.12, 0.022, 0.01, -1, 0)

        # Om0 + Ode0 should equal 1 - omk
        assert cosmo.Om0 + cosmo.Ode0 == pytest.approx(0.99, rel=1e-6)


class TestDistanceFunctions:
    """Tests for distance calculation functions."""

    @pytest.fixture
    def reference_cosmo_dic(self, planck_reference_params, default_redshifts):
        """Create a reference cosmology dictionary."""
        p = planck_reference_params
        cosmo = cosmo_omkw0wa(p["H0"], p["omch2"], p["ombh2"], 0, -1, 0)

        return {
            "cosmo": cosmo,
            "redshifts": default_redshifts,
            "rdrag": p["rdrag"],
        }

    def test_DV_over_rdrag_computes(self, reference_cosmo_dic):
        """Test that DV_over_rdrag computes and stores values."""
        DV_over_rdrag(reference_cosmo_dic)

        assert "DV_over_rdrag" in reference_cosmo_dic
        assert len(reference_cosmo_dic["DV_over_rdrag"]) == len(
            reference_cosmo_dic["redshifts"]
        )

    def test_DV_over_rdrag_values(self, reference_cosmo_dic):
        """Test DV/rd values at specific redshifts."""
        DV_over_rdrag(reference_cosmo_dic)

        z = reference_cosmo_dic["redshifts"]
        dv_rd = reference_cosmo_dic["DV_over_rdrag"]

        # Interpolate to DESI redshifts and check approximate values
        # DV/rd should be ~7.94 at z=0.295, ~12.72 at z=0.51, ~31.27 at z=2.33
        dv_rd_0295 = np.interp(0.295, z, dv_rd)
        dv_rd_051 = np.interp(0.51, z, dv_rd)
        dv_rd_233 = np.interp(2.33, z, dv_rd)

        assert dv_rd_0295 == pytest.approx(7.94, rel=0.02)  # 2% tolerance
        assert dv_rd_051 == pytest.approx(12.72, rel=0.02)
        assert dv_rd_233 == pytest.approx(31.27, rel=0.02)

    def test_DM_over_DH_computes(self, reference_cosmo_dic):
        """Test that DM_over_DH computes and stores values."""
        DM_over_DH(reference_cosmo_dic)

        assert "DM_over_DH" in reference_cosmo_dic
        assert len(reference_cosmo_dic["DM_over_DH"]) == len(
            reference_cosmo_dic["redshifts"]
        )

    def test_DM_over_DH_monotonic(self, reference_cosmo_dic):
        """Test that DM/DH is monotonically increasing with redshift."""
        DM_over_DH(reference_cosmo_dic)

        dm_dh = reference_cosmo_dic["DM_over_DH"]
        assert np.all(np.diff(dm_dh) > 0)

    def test_distance_modulus_computes(self, reference_cosmo_dic):
        """Test that distance_modulus computes and stores values."""
        distance_modulus(reference_cosmo_dic)

        assert "mu" in reference_cosmo_dic
        assert len(reference_cosmo_dic["mu"]) == len(reference_cosmo_dic["redshifts"])

    def test_distance_modulus_increases(self, reference_cosmo_dic):
        """Test that distance modulus increases with redshift."""
        distance_modulus(reference_cosmo_dic)

        mu = reference_cosmo_dic["mu"]
        assert np.all(np.diff(mu) > 0)


class TestRatioFunctions:
    """Tests for ratio calculation functions."""

    @pytest.fixture
    def two_cosmologies(self, planck_reference_params, default_redshifts):
        """Create two cosmology dictionaries for ratio tests."""
        p = planck_reference_params

        # Reference: ΛCDM
        cosmo_ref = cosmo_omkw0wa(p["H0"], p["omch2"], p["ombh2"], 0, -1, 0)
        ref_dic = {
            "cosmo": cosmo_ref,
            "redshifts": default_redshifts,
            "rdrag": p["rdrag"],
        }
        DV_over_rdrag(ref_dic)
        DM_over_DH(ref_dic)
        distance_modulus(ref_dic)

        # Target: w0waCDM
        cosmo_target = cosmo_omkw0wa(p["H0"], p["omch2"], p["ombh2"], 0, -0.9, 0.2)
        target_dic = {
            "cosmo": cosmo_target,
            "redshifts": default_redshifts,
            "rdrag": p["rdrag"],
        }
        DV_over_rdrag(target_dic)
        DM_over_DH(target_dic)

        return ref_dic, target_dic

    def test_DV_ratio_computes(self, two_cosmologies):
        """Test that DV_over_rdrag_ratio computes ratios."""
        ref_dic, target_dic = two_cosmologies
        DV_over_rdrag_ratio(target_dic, ref_dic)

        assert "DV_over_rdrag_ratio" in target_dic
        assert "DM_over_DH_ratio" in target_dic

    def test_self_ratio_is_one(self, two_cosmologies):
        """Test that ratio of cosmology with itself is 1."""
        ref_dic, _ = two_cosmologies

        # Create a copy to compute ratio with itself
        self_dic = dict(ref_dic)
        DV_over_rdrag_ratio(self_dic, ref_dic)

        assert np.allclose(self_dic["DV_over_rdrag_ratio"], 1.0)
        assert np.allclose(self_dic["DM_over_DH_ratio"], 1.0)

    def test_distance_modulus_diff_computes(self, two_cosmologies):
        """Test that distance_modulus_diff computes differences."""
        ref_dic, target_dic = two_cosmologies
        distance_modulus_diff(target_dic, ref_dic)

        assert "mu_diff" in target_dic


class TestAstropyConsistency:
    """Tests verifying consistency with direct astropy calculations."""

    def test_dm_dh_against_astropy(self, planck_reference_params):
        """Verify DM/DH calculation matches direct astropy computation."""
        p = planck_reference_params
        cosmo = cosmo_omkw0wa(p["H0"], p["omch2"], p["ombh2"], 0, -1, 0)

        z = 0.5
        cosmo_dic = {
            "cosmo": cosmo,
            "redshifts": np.array([z]),
            "rdrag": p["rdrag"],
        }
        DM_over_DH(cosmo_dic)

        # Direct calculation
        c = 299792.458
        DM_direct = cosmo.comoving_transverse_distance(z).value
        DH_direct = c / cosmo.H(z).value
        expected = DM_direct / DH_direct

        assert cosmo_dic["DM_over_DH"][0] == pytest.approx(expected, rel=1e-10)

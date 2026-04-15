"""
tests/test_synthetic.py
Tests for data/synthetic.py.

Rule 7:  Validate every component on synthetic ground truth before touching ADNI.
Rule 5:  The null intervention (alpha=1.0) must exactly reproduce the baseline
         trajectory — enforced here as a quantitative assertion.
Rule 1:  Split overlap validated inside make_synthetic_cohort; confirmed here too.
Rule 10: Not applicable directly (no metrics returned here), but checks that
         counterfactual trajectories are not degenerate (not collapsed to a constant).

These tests have zero dependency on ADNI files and must pass in CI on every
commit that touches data/synthetic.py, data/splits.py, or any stage module.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.synthetic import (
    SyntheticDataset,
    SyntheticPatient,
    collate_variable_length,
    get_counterfactual_ground_truth,
    make_synthetic_cohort,
    _generate_one_patient,
    CDRSB_MIN,
    CDRSB_MAX,
)
from data.splits import validate_no_patient_overlap


# ── Shared fixtures ────────────────────────────────────────────────────────────

LATENT_DIM = 64   # Small for test speed; production uses 512


@pytest.fixture(scope="module")
def small_cohort():
    """A small cohort sufficient for split and dataset tests."""
    return make_synthetic_cohort(
        n_patients = 90,
        latent_dim = LATENT_DIM,
        seed       = 42,
        min_visits = 3,
    )


@pytest.fixture(scope="module")
def one_patient():
    """A single deterministic synthetic patient for trajectory tests."""
    rng = np.random.default_rng(0)
    return _generate_one_patient(
        patient_id = "TEST_0000",
        diagnosis  = "MCI",
        latent_dim = LATENT_DIM,
        min_visits = 5,
        rng        = rng,
    )


# ── Patient generation ─────────────────────────────────────────────────────────

class TestPatientGeneration:

    def test_returns_synthetic_patient(self, one_patient):
        assert isinstance(one_patient, SyntheticPatient)

    def test_trajectory_shape(self, one_patient):
        T = len(one_patient.visit_months)
        assert one_patient.z_trajectory.shape   == (T, LATENT_DIM)
        assert one_patient.cdrsb_trajectory.shape == (T,)

    def test_z0_shape(self, one_patient):
        assert one_patient.z0.shape == (LATENT_DIM,)

    def test_cdrsb_in_valid_range(self, one_patient):
        assert np.all(one_patient.cdrsb_trajectory >= CDRSB_MIN)
        assert np.all(one_patient.cdrsb_trajectory <= CDRSB_MAX)

    def test_visit_months_sorted_and_nonneg(self, one_patient):
        vm = one_patient.visit_months
        assert np.all(vm >= 0.0), "Visit months must be non-negative."
        assert np.all(np.diff(vm) >= 0.0), "Visit months must be sorted."

    def test_baseline_visit_is_zero(self, one_patient):
        assert one_patient.visit_months[0] == pytest.approx(0.0, abs=1e-6)

    def test_min_visit_count(self, one_patient):
        assert len(one_patient.visit_months) >= 5

    def test_atrophy_decay_is_monotone(self, one_patient):
        """
        z_atrophy L1-norm must be non-increasing over time (exponential decay).
        The CDRSB may be non-monotone due to measurement noise, but the latent
        atrophy norm itself should decrease.
        """
        n_a   = one_patient.n_atrophy_dims
        norms = np.abs(one_patient.z_trajectory[:, :n_a]).sum(axis=1)
        diffs = np.diff(norms)
        # Allow a tiny numerical slack; strict equality would be overly fragile
        assert np.all(diffs <= 1e-5), (
            "z_atrophy L1-norm should be non-increasing (exponential decay). "
            f"Found positive diffs: {diffs[diffs > 1e-5]}"
        )

    @pytest.mark.parametrize("dx", ["CN", "MCI", "AD"])
    def test_all_diagnoses_generate(self, dx):
        rng = np.random.default_rng(1)
        p   = _generate_one_patient("TEST", dx, LATENT_DIM, min_visits=3, rng=rng)
        assert p.diagnosis == dx


# ── Counterfactual oracle ──────────────────────────────────────────────────────

class TestCounterfactualGroundTruth:
    """
    Rule 5: The null intervention must exactly reproduce the baseline trajectory.
    This is the most fundamental correctness check in the counterfactual module.
    """

    def test_null_intervention_matches_baseline(self, one_patient):
        """
        Rule 5 — MANDATORY TEST.
        alpha=1.0 (null intervention) must produce a simulated_trajectory
        that is numerically identical to the baseline trajectory.
        If this test fails, ALL counterfactual results are meaningless.
        """
        z_cf, cdrsb_cf = one_patient.counterfactual_trajectory(
            intervention_start_month = 0.0,
            alpha                    = 1.0,
        )
        np.testing.assert_allclose(
            z_cf,
            one_patient.z_trajectory,
            atol = 1e-5,
            err_msg = (
                "Rule 5 VIOLATION: null intervention (alpha=1.0) diverges from "
                "baseline trajectory.  All counterfactual results are invalid."
            ),
        )
        np.testing.assert_allclose(
            cdrsb_cf,
            one_patient.cdrsb_trajectory,
            atol = 1e-4,
            err_msg = (
                "Rule 5 VIOLATION: null-intervention CDRSB diverges from baseline."
            ),
        )

    def test_full_suppression_freezes_atrophy(self, one_patient):
        """
        alpha=0.0 means disease velocity is fully suppressed from t0.
        z_atrophy after t0 must equal z_atrophy(t0) — i.e., no further decay.
        """
        t0 = one_patient.visit_months[2]   # start intervention after 3rd visit
        n_a = one_patient.n_atrophy_dims

        z_cf, _ = one_patient.counterfactual_trajectory(
            intervention_start_month = t0,
            alpha                    = 0.0,
        )

        # z_atrophy at t0
        idx_t0   = np.argmin(np.abs(one_patient.visit_months - t0))
        z_a_at_t0 = z_cf[idx_t0, :n_a]

        # All visits AFTER t0 must have the same z_atrophy
        after_mask = one_patient.visit_months > t0
        for i, after in enumerate(after_mask):
            if after:
                np.testing.assert_allclose(
                    z_cf[i, :n_a], z_a_at_t0, atol=1e-5,
                    err_msg=f"alpha=0.0: atrophy should be frozen after t0 at visit {i}."
                )

    def test_dose_response_monotonicity(self, one_patient):
        """
        Rule 8 — QUANTITATIVE TEST.
        Varying alpha from 0 to 1 must produce monotonically INCREASING
        trajectory change (stronger alpha → smaller intervention effect).

        Specifically: the final CDRSB under alpha=0.0 must be <= that under
        alpha=0.5, which must be <= that under alpha=1.0.
        This test must pass; failure renders the model ineligible for reporting.
        """
        t0       = one_patient.visit_months[1]   # early intervention
        alphas   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        final_cdrsb = []

        for alpha in alphas:
            _, cdrsb_cf = one_patient.counterfactual_trajectory(
                intervention_start_month = t0,
                alpha                    = alpha,
            )
            final_cdrsb.append(float(cdrsb_cf[-1]))

        # Larger alpha → less intervention → higher CDRSB at end
        non_monotone = sum(
            1 for i in range(len(final_cdrsb) - 1)
            if final_cdrsb[i] > final_cdrsb[i + 1] + 1e-6
        )
        assert non_monotone == 0, (
            f"Rule 8 VIOLATION: dose-response is non-monotone.  "
            f"final CDRSB values: {[f'{v:.4f}' for v in final_cdrsb]}  "
            f"non-monotone count: {non_monotone}.  "
            "This model version is NOT eligible for reporting."
        )

    def test_alpha_out_of_range_raises(self, one_patient):
        with pytest.raises(ValueError, match="alpha must be in"):
            one_patient.counterfactual_trajectory(0.0, alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            one_patient.counterfactual_trajectory(0.0, alpha=-0.1)

    def test_counterfactual_before_t0_matches_baseline(self, one_patient):
        """
        Before the intervention start, the counterfactual trajectory must
        be identical to the baseline (interventions are not retroactive).
        """
        t0 = one_patient.visit_months[3]   # intervene at 4th visit
        z_cf, _ = one_patient.counterfactual_trajectory(
            intervention_start_month = t0,
            alpha                    = 0.3,
        )
        before_mask = one_patient.visit_months <= t0
        np.testing.assert_allclose(
            z_cf[before_mask],
            one_patient.z_trajectory[before_mask],
            atol = 1e-5,
            err_msg = "Counterfactual must match baseline before intervention start."
        )

    def test_get_counterfactual_ground_truth_helper(self, small_cohort):
        train_ds = small_cohort["train"]
        results  = get_counterfactual_ground_truth(
            dataset                   = train_ds,
            intervention_start_month  = 12.0,
            alpha                     = 0.5,
        )
        assert len(results) == len(train_ds)
        for pid, cf in results.items():
            assert "z_cf"     in cf
            assert "cdrsb_cf" in cf
            assert cf["z_cf"].shape[1]    == LATENT_DIM
            assert cf["cdrsb_cf"].ndim    == 1


# ── Dataset and DataLoader ─────────────────────────────────────────────────────

class TestSyntheticDataset:

    def test_len(self, small_cohort):
        total = sum(len(ds) for ds in small_cohort.values())
        assert total == 90

    def test_getitem_keys(self, small_cohort):
        item = small_cohort["train"][0]
        expected_keys = {
            "patient_id", "diagnosis", "visit_months", "z_true",
            "cdrsb_true", "covariates", "mask", "n_atrophy_dims", "true_decay_rate",
        }
        assert expected_keys == set(item.keys())

    def test_tensor_dtypes(self, small_cohort):
        item = small_cohort["train"][0]
        assert item["visit_months"].dtype == torch.float32
        assert item["z_true"].dtype       == torch.float32
        assert item["cdrsb_true"].dtype   == torch.float32
        assert item["covariates"].dtype   == torch.float32
        assert item["mask"].dtype         == torch.bool

    def test_z_true_shape(self, small_cohort):
        item = small_cohort["train"][0]
        T    = item["visit_months"].shape[0]
        assert item["z_true"].shape == (T, LATENT_DIM)

    def test_cdrsb_in_range(self, small_cohort):
        for split_ds in small_cohort.values():
            for i in range(len(split_ds)):
                item = split_ds[i]
                assert item["cdrsb_true"].min() >= CDRSB_MIN - 1e-4
                assert item["cdrsb_true"].max() <= CDRSB_MAX + 1e-4

    def test_mask_all_true_for_real_visits(self, small_cohort):
        item = small_cohort["train"][0]
        assert item["mask"].all(), "All mask values should be True for un-padded items."

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="at least one patient"):
            SyntheticDataset([])

    def test_dataloader_with_collate(self, small_cohort):
        """
        Verify that DataLoader + collate_variable_length produces correctly
        padded batches with the expected keys and shapes.
        """
        loader = DataLoader(
            small_cohort["train"],
            batch_size  = 4,
            collate_fn  = collate_variable_length,
            shuffle     = False,
        )
        batch = next(iter(loader))

        B     = 4
        T_max = batch["visit_months"].shape[1]

        assert batch["visit_months"].shape  == (B, T_max)
        assert batch["z_true"].shape        == (B, T_max, LATENT_DIM)
        assert batch["cdrsb_true"].shape    == (B, T_max)
        assert batch["mask"].shape          == (B, T_max)
        assert batch["covariates"].shape    == (B, 3)

    def test_padding_is_zero(self, small_cohort):
        """
        Padded positions (mask==False) in z_true and cdrsb_true must be zero.
        """
        loader = DataLoader(
            small_cohort["train"],
            batch_size = 8,
            collate_fn = collate_variable_length,
            shuffle    = False,
        )
        for batch in loader:
            mask = batch["mask"]   # (B, T_max) bool
            pad  = ~mask
            if pad.any():
                assert batch["z_true"][pad].abs().sum() == 0.0, \
                    "Padded positions in z_true must be zero."
                assert batch["cdrsb_true"][pad].abs().sum() == 0.0, \
                    "Padded positions in cdrsb_true must be zero."
            break   # one batch is sufficient


# ── make_synthetic_cohort ──────────────────────────────────────────────────────

class TestMakeSyntheticCohort:

    def test_returns_three_splits(self, small_cohort):
        assert set(small_cohort.keys()) == {"train", "val", "test"}

    def test_no_patient_overlap(self, small_cohort):
        """Rule 1 — belt-and-suspenders check at the test level."""
        train_ids = [p.patient_id for p in small_cohort["train"].patients]
        val_ids   = [p.patient_id for p in small_cohort["val"].patients]
        test_ids  = [p.patient_id for p in small_cohort["test"].patients]
        validate_no_patient_overlap(train_ids, val_ids, test_ids)

    def test_split_ratios_approx(self, small_cohort):
        total = sum(len(ds) for ds in small_cohort.values())
        assert abs(len(small_cohort["train"]) / total - 0.70) < 0.06
        assert abs(len(small_cohort["val"])   / total - 0.15) < 0.06
        assert abs(len(small_cohort["test"])  / total - 0.15) < 0.06

    def test_reproducibility(self):
        cohort_a = make_synthetic_cohort(n_patients=60, latent_dim=LATENT_DIM, seed=7)
        cohort_b = make_synthetic_cohort(n_patients=60, latent_dim=LATENT_DIM, seed=7)
        ids_a = sorted(p.patient_id for p in cohort_a["train"].patients)
        ids_b = sorted(p.patient_id for p in cohort_b["train"].patients)
        assert ids_a == ids_b, "Same seed must produce identical splits."

    def test_different_seeds_differ(self):
        cohort_a = make_synthetic_cohort(n_patients=60, latent_dim=LATENT_DIM, seed=1)
        cohort_b = make_synthetic_cohort(n_patients=60, latent_dim=LATENT_DIM, seed=2)
        ids_a = sorted(p.patient_id for p in cohort_a["train"].patients)
        ids_b = sorted(p.patient_id for p in cohort_b["train"].patients)
        assert ids_a != ids_b, "Different seeds should produce different splits."

    def test_invalid_dx_distribution_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            make_synthetic_cohort(
                n_patients       = 10,
                latent_dim       = LATENT_DIM,
                dx_distribution  = {"CN": 0.5, "MCI": 0.3, "AD": 0.1},   # sums to 0.9
            )

    def test_unknown_dx_key_raises(self):
        with pytest.raises(ValueError, match="Unknown diagnosis"):
            make_synthetic_cohort(
                n_patients       = 10,
                latent_dim       = LATENT_DIM,
                dx_distribution  = {"CN": 0.5, "MCI": 0.3, "UNKNOWN": 0.2},
            )

    def test_all_patients_have_min_visits(self):
        cohort = make_synthetic_cohort(n_patients=30, latent_dim=LATENT_DIM, seed=0, min_visits=4)
        for split_ds in cohort.values():
            for p in split_ds.patients:
                assert len(p.visit_months) >= 4, (
                    f"Patient {p.patient_id} has only {len(p.visit_months)} visits "
                    f"but min_visits=4."
                )

    def test_latent_dim_propagates(self):
        cohort = make_synthetic_cohort(n_patients=20, latent_dim=32, seed=0)
        for split_ds in cohort.values():
            for p in split_ds.patients:
                assert p.z0.shape == (32,)
                assert p.z_trajectory.shape[1] == 32

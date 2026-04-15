"""
tests/test_splits.py
Unit tests for data/splits.py.

These tests use only synthetic data and have no dependency on ADNI files,
so they can (and should) run in CI on every commit.  Rule 1 is the single
most critical correctness invariant in the codebase — its test suite must
be airtight.
"""
import pytest
import pandas as pd
import numpy as np

from data.splits import (
    validate_no_patient_overlap,
    generate_patient_splits,
    save_splits,
    load_splits,
    MIN_VISITS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_visit_df(
    n_patients: int = 100,
    visits_per_patient: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a minimal synthetic visit DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        rid = f"P{i:04d}"
        dx  = rng.choice(["CN", "MCI", "AD"])
        base_cdrsb = {"CN": 0.0, "MCI": 1.5, "AD": 5.0}[dx]
        for v in range(visits_per_patient):
            rows.append({
                "RID":          rid,
                "DX_bl":        dx,
                "visit_month":  v * 12,
                "CDRSB":        base_cdrsb + v * rng.uniform(0, 1),
            })
    return pd.DataFrame(rows)


# ── validate_no_patient_overlap ───────────────────────────────────────────────

class TestValidateNoPatientOverlap:

    def test_clean_splits_pass(self):
        validate_no_patient_overlap(["A", "B"], ["C", "D"], ["E", "F"])

    def test_train_val_overlap_raises(self):
        with pytest.raises(AssertionError, match="DATA LEAK"):
            validate_no_patient_overlap(["A", "B"], ["B", "C"], ["D"])

    def test_train_test_overlap_raises(self):
        with pytest.raises(AssertionError, match="DATA LEAK"):
            validate_no_patient_overlap(["A", "B"], ["C"], ["B", "D"])

    def test_val_test_overlap_raises(self):
        with pytest.raises(AssertionError, match="DATA LEAK"):
            validate_no_patient_overlap(["A"], ["B", "C"], ["C", "D"])

    def test_empty_splits_pass(self):
        # Edge case: empty splits are technically leak-free
        validate_no_patient_overlap([], [], [])


# ── generate_patient_splits ───────────────────────────────────────────────────

class TestGeneratePatientSplits:

    def test_no_overlap(self):
        df = make_visit_df(n_patients=120)
        splits = generate_patient_splits(df, seed=42)
        # validate_no_patient_overlap is called inside; this just confirms no exception
        all_ids = splits["train"] + splits["val"] + splits["test"]
        assert len(all_ids) == len(set(all_ids))

    def test_split_sizes(self):
        df = make_visit_df(n_patients=200)
        splits = generate_patient_splits(df, train_ratio=0.70, val_ratio=0.15, seed=42)
        n_total = sum(len(v) for v in splits.values())
        assert abs(len(splits["train"]) / n_total - 0.70) < 0.05
        assert abs(len(splits["val"])   / n_total - 0.15) < 0.05
        assert abs(len(splits["test"])  / n_total - 0.15) < 0.05

    def test_min_visit_filter(self):
        # Inject patients with only 1 visit — they must be excluded
        df = make_visit_df(n_patients=100, visits_per_patient=5)
        sparse_rows = [{"RID": "SPARSE_001", "DX_bl": "CN",
                        "visit_month": 0, "CDRSB": 0.0}]
        df = pd.concat([df, pd.DataFrame(sparse_rows)], ignore_index=True)
        splits = generate_patient_splits(df, seed=42)
        all_ids = splits["train"] + splits["val"] + splits["test"]
        assert "SPARSE_001" not in all_ids

    def test_reproducibility(self):
        df = make_visit_df(n_patients=100)
        splits_a = generate_patient_splits(df, seed=99)
        splits_b = generate_patient_splits(df, seed=99)
        assert splits_a == splits_b

    def test_different_seeds_differ(self):
        df = make_visit_df(n_patients=100)
        splits_a = generate_patient_splits(df, seed=1)
        splits_b = generate_patient_splits(df, seed=2)
        assert splits_a["train"] != splits_b["train"]

    def test_invalid_ratios_raise(self):
        df = make_visit_df()
        with pytest.raises(ValueError):
            generate_patient_splits(df, train_ratio=0.90, val_ratio=0.20)


# ── save / load round-trip ────────────────────────────────────────────────────

class TestSplitPersistence:

    def test_round_trip(self, tmp_path):
        df = make_visit_df(n_patients=60)
        splits = generate_patient_splits(df, seed=42)
        save_splits(splits, tmp_path)
        loaded = load_splits(tmp_path / "patient_splits.csv")
        assert splits == loaded

    def test_load_detects_tampered_csv(self, tmp_path):
        """If someone hand-edits the CSV to duplicate a patient, load_splits catches it."""
        df = make_visit_df(n_patients=60)
        splits = generate_patient_splits(df, seed=42)
        save_splits(splits, tmp_path)

        # Tamper: duplicate a train patient into val
        csv_path = tmp_path / "patient_splits.csv"
        split_df = pd.read_csv(csv_path)
        first_train = split_df[split_df["split"] == "train"].iloc[0:1].copy()
        first_train["split"] = "val"
        tampered = pd.concat([split_df, first_train], ignore_index=True)
        tampered.to_csv(csv_path, index=False)

        with pytest.raises(AssertionError, match="DATA LEAK"):
            load_splits(csv_path)

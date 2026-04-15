"""
data/splits.py
Rule 1: Split by Patient, Always — validate_no_patient_overlap() must be
        called as the FIRST operation in every training script, before any
        data is loaded into memory.

A visit-level split is a silent data leak. It does not crash. It produces
optimistic metrics that look like results.
"""
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# Per design doc Section 4.3 — patients with fewer visits are excluded
MIN_VISITS: int = 3


# ── Core safety assertion ──────────────────────────────────────────────────────

def validate_no_patient_overlap(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> None:
    """
    Hard assertion: no patient ID appears in more than one split.

    This function must be the very first call in every training script,
    invoked before any tensors, datasets, or data loaders are constructed.
    It is intentionally impossible to skip — if the splits overlap, execution
    stops here with a descriptive error.

    Args:
        train_ids: List of patient IDs assigned to the training split.
        val_ids:   List of patient IDs assigned to the validation split.
        test_ids:  List of patient IDs assigned to the test split.

    Raises:
        AssertionError: immediately on any detected overlap.
    """
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

    train_val  = train_set & val_set
    train_test = train_set & test_set
    val_test   = val_set   & test_set

    assert not train_val, (
        f"DATA LEAK — {len(train_val)} patient(s) in both train and val: "
        f"{sorted(train_val)[:10]} ..."
    )
    assert not train_test, (
        f"DATA LEAK — {len(train_test)} patient(s) in both train and test: "
        f"{sorted(train_test)[:10]} ..."
    )
    assert not val_test, (
        f"DATA LEAK — {len(val_test)} patient(s) in both val and test: "
        f"{sorted(val_test)[:10]} ..."
    )

    total = len(train_set) + len(val_set) + len(test_set)
    logger.info(
        "[splits] Overlap validation PASSED — "
        "train=%d  val=%d  test=%d  total=%d",
        len(train_set), len(val_set), len(test_set), total,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _progression_label(group: pd.DataFrame, visit_month_col: str) -> str:
    """
    Assign a coarse progression label from a patient's CDRSB trajectory.
    Used only for stratification during split generation.
    Returns 'fast', 'slow', 'stable', or 'unknown'.
    """
    if "CDRSB" not in group.columns or group["CDRSB"].isna().all():
        return "unknown"
    series = group.sort_values(visit_month_col)["CDRSB"].dropna()
    if len(series) < 2:
        return "unknown"
    delta = float(series.iloc[-1]) - float(series.iloc[0])
    if delta >= 3.0:
        return "fast"
    if delta > 0.5:
        return "slow"
    return "stable"


# ── Split generation ───────────────────────────────────────────────────────────

def generate_patient_splits(
    visit_df: pd.DataFrame,
    patient_id_col: str = "RID",
    diagnosis_col: str = "DX_bl",
    visit_month_col: str = "visit_month",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Generate stratified, patient-level train / val / test splits.

    Stratification is over (baseline_diagnosis × progression_rate) so that
    CN/MCI/AD and fast/slow/stable progressors are proportionally represented
    in every split — per design doc Section 4.3.

    Args:
        visit_df:        One row per visit.
        patient_id_col:  Patient identifier column (default 'RID').
        diagnosis_col:   Baseline diagnosis column, values in {CN, MCI, AD}.
        visit_month_col: Numeric visit time in months (converted from VISCODE2).
        train_ratio:     Fraction of patients for training (default 0.70).
        val_ratio:       Fraction for validation (default 0.15).
                         test_ratio = 1 - train_ratio - val_ratio.
        seed:            Must match config["reproducibility"]["seed"] (Rule 2).

    Returns:
        {"train": [...], "val": [...], "test": [...]} — sorted lists of patient IDs.

    Raises:
        AssertionError: via validate_no_patient_overlap if any overlap exists.
        ValueError:     if ratio arguments are invalid.
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must both be in (0, 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    # ── Filter to patients with enough visits ──────────────────────────────────
    visit_counts = visit_df.groupby(patient_id_col).size()
    eligible = visit_counts[visit_counts >= MIN_VISITS].index
    n_excluded = len(visit_counts) - len(eligible)
    if n_excluded > 0:
        logger.warning(
            "[splits] Excluded %d patient(s) with < %d visits.",
            n_excluded, MIN_VISITS,
        )
    visit_df = visit_df[visit_df[patient_id_col].isin(eligible)].copy()

    # ── Build one-row-per-patient summary for stratification ───────────────────
    def summarise(g: pd.DataFrame) -> pd.Series:
        baseline_row = g.sort_values(visit_month_col).iloc[0]
        return pd.Series({
            "baseline_dx":   str(baseline_row.get(diagnosis_col, "unknown")),
            "progression":   _progression_label(g, visit_month_col),
            "n_visits":      len(g),
        })

    patient_df = (
        visit_df.groupby(patient_id_col)
        .apply(summarise)
        .reset_index()
    )
    patient_df["stratum"] = (
        patient_df["baseline_dx"] + "_" + patient_df["progression"]
    )

    ids    = patient_df[patient_id_col].values
    strata = patient_df["stratum"].values

    # ── Two-step stratified split ──────────────────────────────────────────────
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)

    spl1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(spl1.split(ids, strata))

    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    spl2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adj, random_state=seed)
    rel_train_idx, rel_val_idx = next(
        spl2.split(ids[trainval_idx], strata[trainval_idx])
    )
    train_idx = trainval_idx[rel_train_idx]
    val_idx   = trainval_idx[rel_val_idx]

    train_ids = sorted(ids[train_idx].tolist())
    val_ids   = sorted(ids[val_idx].tolist())
    test_ids  = sorted(ids[test_idx].tolist())

    # Rule 1 — hard assertion before returning
    validate_no_patient_overlap(train_ids, val_ids, test_ids)

    n_total = len(ids)
    logger.info(
        "[splits] Done — train=%d (%.0f%%)  val=%d (%.0f%%)  test=%d (%.0f%%)",
        len(train_ids), 100 * len(train_ids) / n_total,
        len(val_ids),   100 * len(val_ids)   / n_total,
        len(test_ids),  100 * len(test_ids)  / n_total,
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


# ── Persistence ────────────────────────────────────────────────────────────────

def save_splits(splits: Dict[str, List[str]], output_dir: Path) -> None:
    """
    Write split assignments to CSV so the exact split is fully reproducible
    regardless of future code changes (Rule 2).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"patient_id": pid, "split": split_name}
        for split_name, ids in splits.items()
        for pid in ids
    ]
    out_file = output_dir / "patient_splits.csv"
    pd.DataFrame(rows).sort_values(["split", "patient_id"]).to_csv(out_file, index=False)
    logger.info("[splits] Split assignments saved → %s", out_file)


def load_splits(split_csv: Path) -> Dict[str, List[str]]:
    """
    Load a previously saved split CSV and re-validate overlap (Rule 1).
    This re-check catches the case where the CSV was edited by hand.
    """
    df = pd.read_csv(split_csv)
    splits = {
        name: sorted(grp["patient_id"].tolist())
        for name, grp in df.groupby("split")
    }
    validate_no_patient_overlap(
        splits.get("train", []),
        splits.get("val",   []),
        splits.get("test",  []),
    )
    return splits

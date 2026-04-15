"""
data/synthetic.py
Rule 7: Synthetic Data First, Real Data Second.

This module is the project's ground-truth oracle.  Every component of the
three-stage pipeline must be verified to work here before it ever touches
ADNI.  The module does three things:

  1. SyntheticPatient  — generates a single patient's latent trajectory and
                         CDRSB time-series using a mechanistic ODE whose
                         true parameters are known and stored.

  2. SyntheticDataset  — wraps a cohort of SyntheticPatient objects into a
                         torch.utils.data.Dataset that yields batches
                         compatible with all three pipeline stages.

  3. make_synthetic_cohort — the single public entry-point used by training
                             scripts and tests.  Returns a dict of
                             {split: SyntheticDataset} with guaranteed
                             patient-level splits (Rule 1).

Mechanistic model (the "ground truth" ODE)
──────────────────────────────────────────
Each patient's latent state z ∈ ℝ^d evolves as:

    dz/dt = -λ · z(t)      (exponential atrophy decay per design doc §5.2A)

where λ is a scalar decay rate drawn per patient.  The first few dimensions
of z are "atrophy dimensions" (disease-relevant, z_atrophy) and the rest are
identity dimensions (z_identity, frozen over time).

CDRSB is a nonlinear readout of z_atrophy:

    CDRSB(t) = clip( c₀ + k · ||z_atrophy(t)||₁ , 0, 18 )

where c₀ and k are per-patient constants.

Counterfactual ground truth
───────────────────────────
For any patient we can compute the exact counterfactual trajectory under
an alpha-scaled velocity intervention (design doc §3.3, Option 1):

    dz_atrophy/dt_cf = alpha · (-λ · z(t))

i.e., alpha=1.0  →  no intervention  (null trajectory, Rule 5)
      alpha=0.0  →  disease frozen    (maximum intervention)
      0 < alpha < 1  →  partially slowed progression

Because we solved the ODE analytically, the counterfactual is exact:

    z_atrophy_cf(t; t0, alpha) = z_atrophy(t0) · exp(-alpha · λ · (t - t0))

This enables MSE-based counterfactual quality evaluation (design doc §8.4)
without any approximation.

Irregular visit schedules
─────────────────────────
ADNI visit intervals are non-uniform.  Synthetic patients replicate this by
drawing visit times from a realistic irregular schedule (Rule 7: validate
irregularity handling before touching ADNI).

VISCODE2 time map (mirrors data/adni_utils.py when written)
────────────────────────────────────────────────────────────
Numeric times in months.  Kept in sync with the ADNI preprocessing module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.splits import generate_patient_splits, validate_no_patient_overlap

logger = logging.getLogger(__name__)


# ── Module-level constants ─────────────────────────────────────────────────────

# CDRSB range per design doc §4.1
CDRSB_MIN: float = 0.0
CDRSB_MAX: float = 18.0

# Fraction of latent dimensions that are "disease-relevant" (z_atrophy).
# The remainder are identity/noise dimensions (z_identity).
# Must be kept consistent with the autoencoder's disentanglement target.
ATROPHY_DIM_FRACTION: float = 0.25   # e.g., 128 of 512 dims

# Realistic visit schedule options (months from baseline).
# Mirrors ADNI VISCODE2 time-points per design doc §4.2.
# Some patients skip visits — modelled by sampling a subset.
_CANDIDATE_VISIT_MONTHS: List[int] = [0, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96]

# Baseline CDRSB intercepts by diagnosis group (matches ADNI distributions).
_CDRSB_INTERCEPT: Dict[str, float] = {"CN": 0.2, "MCI": 1.8, "AD": 5.5}

# Per-diagnosis decay rate ranges (month⁻¹).
# AD decays fastest; CN is near-stable.
_LAMBDA_RANGE: Dict[str, Tuple[float, float]] = {
    "CN":  (0.001, 0.008),
    "MCI": (0.005, 0.020),
    "AD":  (0.015, 0.040),
}

# APOE ε4 genotype accelerates decay; modelled as a multiplicative factor.
_APOE4_MULTIPLIER: float = 1.35   # carriers decay ~35% faster on average


# ── Per-patient ground-truth record ───────────────────────────────────────────

@dataclass
class SyntheticPatient:
    """
    A single patient whose complete trajectory is analytically known.

    All attributes are the TRUE parameters — never inferred.  This is what
    separates synthetic validation from real-data evaluation: we can measure
    how well the model recovers these exact values.

    Attributes
    ----------
    patient_id:         Unique string ID, e.g. "SYN_0001".
    diagnosis:          Baseline diagnosis: "CN", "MCI", or "AD".
    apoe4_carrier:      Whether the patient carries the APOE ε4 allele.
    age_at_baseline:    Age in years at the first visit.
    sex:                "M" or "F".
    decay_rate:         True λ (month⁻¹) governing atrophy velocity.
    cdrsb_intercept:    c₀ in the CDRSB readout function.
    cdrsb_slope:        k in the CDRSB readout function.
    z0:                 Initial latent state z(t=0), shape (latent_dim,).
    visit_months:       Observed visit times in months, shape (T,).
    z_trajectory:       True latent trajectory at each visit, shape (T, latent_dim).
    cdrsb_trajectory:   True CDRSB at each visit, shape (T,).
    n_atrophy_dims:     Number of atrophy dimensions (first n dims of z).
    """
    patient_id:         str
    diagnosis:          str
    apoe4_carrier:      bool
    age_at_baseline:    float
    sex:                str
    decay_rate:         float
    cdrsb_intercept:    float
    cdrsb_slope:        float
    z0:                 np.ndarray
    visit_months:       np.ndarray           # shape (T,)
    z_trajectory:       np.ndarray           # shape (T, latent_dim)
    cdrsb_trajectory:   np.ndarray           # shape (T,)
    n_atrophy_dims:     int

    # ── Counterfactual oracle ──────────────────────────────────────────────────

    def counterfactual_trajectory(
        self,
        intervention_start_month: float,
        alpha: float,
        query_months: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the EXACT counterfactual latent trajectory and CDRSB under an
        alpha-scaled velocity intervention applied from t0 onward.

        Rule 9 (causal language discipline): this is a simulated_trajectory
        under a hypothetical intervention, NOT a proved causal effect.

        Args:
            intervention_start_month: t0 — intervention begins at this time.
            alpha:  Velocity scaling factor in [0, 1].
                    alpha=1.0  →  null intervention (must match baseline).
                    alpha=0.0  →  disease velocity fully suppressed.
            query_months: Times at which to evaluate the counterfactual.
                          Defaults to self.visit_months.

        Returns:
            (z_cf, cdrsb_cf) where:
                z_cf      shape (T, latent_dim)  — counterfactual latent states
                cdrsb_cf  shape (T,)             — counterfactual CDRSB scores

        Raises:
            ValueError: if alpha is outside [0, 1].
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"alpha must be in [0, 1], got {alpha:.4f}. "
                "alpha=1.0 is the null intervention (Rule 5); "
                "alpha=0.0 fully suppresses atrophy velocity."
            )

        if query_months is None:
            query_months = self.visit_months

        lam = self.decay_rate
        n_a = self.n_atrophy_dims

        # ── Find z at t0 by integrating the baseline ODE up to t0 ─────────────
        z0_atrophy = self.z0[:n_a]
        z0_identity = self.z0[n_a:]

        z_at_t0_atrophy = z0_atrophy * np.exp(-lam * intervention_start_month)

        # ── Counterfactual: apply alpha-scaled decay from t0 onward ───────────
        z_cf = np.empty((len(query_months), len(self.z0)), dtype=np.float32)

        for i, t in enumerate(query_months):
            if t <= intervention_start_month:
                # Before intervention: identical to baseline
                z_cf[i, :n_a] = z0_atrophy * np.exp(-lam * t)
            else:
                # After intervention: alpha-scaled decay from t0
                dt = t - intervention_start_month
                z_cf[i, :n_a] = z_at_t0_atrophy * np.exp(-alpha * lam * dt)
            # Identity dimensions never change
            z_cf[i, n_a:] = z0_identity

        cdrsb_cf = self._cdrsb_from_z(z_cf)

        return z_cf, cdrsb_cf

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _cdrsb_from_z(self, z: np.ndarray) -> np.ndarray:
        """
        Nonlinear readout: CDRSB = clip(c0 + k * ||z_atrophy||_1, 0, 18).
        This exact function is used for ground-truth labels; the model must
        learn to approximate it from data.
        """
        atrophy_norm = np.abs(z[:, : self.n_atrophy_dims]).sum(axis=1)
        raw = self.cdrsb_intercept + self.cdrsb_slope * atrophy_norm
        return np.clip(raw, CDRSB_MIN, CDRSB_MAX).astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    """
    torch.utils.data.Dataset wrapping a list of SyntheticPatient objects.

    Each __getitem__ returns a dict with keys matching what all three pipeline
    stages expect.  Variable-length visit sequences are zero-padded within a
    batch by the collate function below.

    Returned dict keys
    ──────────────────
    "patient_id"      str
    "diagnosis"       str in {"CN", "MCI", "AD"}
    "visit_months"    FloatTensor (T,)
    "z_true"          FloatTensor (T, latent_dim)   — true latent trajectory
    "cdrsb_true"      FloatTensor (T,)              — true CDRSB scores
    "covariates"      FloatTensor (n_covariates,)   — [age, sex, apoe4]
                      age: normalised (raw - 70) / 10
                      sex: 0.0=F, 1.0=M
                      apoe4: 0.0=non-carrier, 1.0=carrier
    "mask"            BoolTensor (T,)               — True for real visits
    "n_atrophy_dims"  int — number of atrophy dimensions in z
    "true_decay_rate" float — ground-truth λ (for evaluation only)
    """

    def __init__(self, patients: List[SyntheticPatient]) -> None:
        if not patients:
            raise ValueError("SyntheticDataset requires at least one patient.")
        self.patients = patients
        self.latent_dim = patients[0].z0.shape[0]

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict:
        p = self.patients[idx]

        covariates = torch.tensor(
            [
                (p.age_at_baseline - 70.0) / 10.0,   # normalised age
                1.0 if p.sex == "M" else 0.0,
                1.0 if p.apoe4_carrier else 0.0,
            ],
            dtype=torch.float32,
        )

        return {
            "patient_id":      p.patient_id,
            "diagnosis":       p.diagnosis,
            "visit_months":    torch.tensor(p.visit_months,    dtype=torch.float32),
            "z_true":          torch.tensor(p.z_trajectory,   dtype=torch.float32),
            "cdrsb_true":      torch.tensor(p.cdrsb_trajectory, dtype=torch.float32),
            "covariates":      covariates,
            "mask":            torch.ones(len(p.visit_months), dtype=torch.bool),
            "n_atrophy_dims":  p.n_atrophy_dims,
            "true_decay_rate": p.decay_rate,
        }


# ── Collate function ───────────────────────────────────────────────────────────

def collate_variable_length(batch: List[Dict]) -> Dict:
    """
    Collate a list of per-patient dicts into a padded batch.

    Variable-length sequences are right-padded to the maximum T in the batch.
    The "mask" tensor marks which positions are real vs. padding.

    This function must be passed as collate_fn to DataLoader when using
    SyntheticDataset (or any real-data dataset with variable visit counts).

    Returns
    ───────
    Dict with keys matching SyntheticDataset.__getitem__, plus:
      "visit_months"  FloatTensor (B, T_max)
      "z_true"        FloatTensor (B, T_max, latent_dim)
      "cdrsb_true"    FloatTensor (B, T_max)
      "mask"          BoolTensor  (B, T_max)
    """
    T_max = max(item["visit_months"].shape[0] for item in batch)
    B = len(batch)
    latent_dim = batch[0]["z_true"].shape[1]

    visit_months = torch.zeros(B, T_max, dtype=torch.float32)
    z_true       = torch.zeros(B, T_max, latent_dim, dtype=torch.float32)
    cdrsb_true   = torch.zeros(B, T_max, dtype=torch.float32)
    mask         = torch.zeros(B, T_max, dtype=torch.bool)
    covariates   = torch.stack([item["covariates"] for item in batch])

    for i, item in enumerate(batch):
        T = item["visit_months"].shape[0]
        visit_months[i, :T] = item["visit_months"]
        z_true[i,       :T] = item["z_true"]
        cdrsb_true[i,   :T] = item["cdrsb_true"]
        mask[i,         :T] = item["mask"]

    return {
        "patient_ids":     [item["patient_id"]    for item in batch],
        "diagnoses":       [item["diagnosis"]     for item in batch],
        "visit_months":    visit_months,
        "z_true":          z_true,
        "cdrsb_true":      cdrsb_true,
        "covariates":      covariates,
        "mask":            mask,
        "n_atrophy_dims":  batch[0]["n_atrophy_dims"],
        "true_decay_rates": torch.tensor(
            [item["true_decay_rate"] for item in batch], dtype=torch.float32
        ),
    }


# ── Patient factory ────────────────────────────────────────────────────────────

def _generate_one_patient(
    patient_id: str,
    diagnosis: str,
    latent_dim: int,
    min_visits: int,
    rng: np.random.Generator,
    visit_month_noise_std: float = 1.5,
) -> SyntheticPatient:
    """
    Draw all per-patient parameters from their priors and integrate the
    ground-truth ODE to produce the complete visit-level trajectory.

    Args:
        patient_id:            e.g. "SYN_0001"
        diagnosis:             "CN", "MCI", or "AD"
        latent_dim:            Dimension of z (must match stage1.latent_dim in config)
        min_visits:            Minimum number of visits to retain (mirrors MIN_VISITS)
        rng:                   Seeded numpy Generator (Rule 2)
        visit_month_noise_std: Jitter added to canonical visit months to mimic
                               ADNI scheduling irregularity (months).

    Returns:
        SyntheticPatient with fully populated trajectory fields.
    """
    # ── Demographics ──────────────────────────────────────────────────────────
    age         = float(rng.uniform(55.0, 90.0))
    sex         = rng.choice(["M", "F"])
    apoe4       = bool(rng.random() < 0.35)   # ~35% prevalence in ADNI

    # ── Disease parameters ────────────────────────────────────────────────────
    lo, hi      = _LAMBDA_RANGE[diagnosis]
    decay_rate  = float(rng.uniform(lo, hi))
    if apoe4:
        decay_rate *= _APOE4_MULTIPLIER

    cdrsb_c0    = _CDRSB_INTERCEPT[diagnosis] + float(rng.normal(0, 0.3))
    cdrsb_c0    = max(0.0, cdrsb_c0)
    # Slope k maps L1-norm of z_atrophy → CDRSB units.
    # Calibrated so that a freshly initialised z (all ones) gives sensible scores.
    cdrsb_k     = float(rng.uniform(0.05, 0.20))

    # ── Initial latent state ──────────────────────────────────────────────────
    n_atrophy   = max(1, int(latent_dim * ATROPHY_DIM_FRACTION))
    n_identity  = latent_dim - n_atrophy

    # z_atrophy ~ Uniform(0.5, 1.5) per dimension (positive, so decay is meaningful)
    z0_atrophy  = rng.uniform(0.5, 1.5, size=n_atrophy).astype(np.float32)
    # z_identity ~ Normal(0, 1) — patient anatomy, does not evolve
    z0_identity = rng.standard_normal(n_identity).astype(np.float32)
    z0          = np.concatenate([z0_atrophy, z0_identity])

    # ── Visit schedule — irregular, mirroring ADNI ────────────────────────────
    # Start from the full candidate list; randomly drop some interior visits
    # to create gaps (e.g., missed 12-month visit).
    candidates  = list(_CANDIDATE_VISIT_MONTHS)
    # Always keep baseline (month 0) and last available
    interior    = candidates[1:-1]
    keep_mask   = rng.random(len(interior)) > 0.35   # ~35% dropout per visit
    kept        = [candidates[0]] + [m for m, k in zip(interior, keep_mask) if k] + [candidates[-1]]
    # Remove duplicates introduced if interior was empty, preserve order
    kept        = sorted(set(kept))
    # Ensure minimum visit count
    if len(kept) < min_visits:
        missing = [m for m in candidates if m not in kept]
        rng.shuffle(missing)
        kept = sorted(set(kept + missing[: min_visits - len(kept)]))

    # Add per-visit timing jitter (ADNI patients don't come in on exact month boundaries)
    jitter      = rng.normal(0, visit_month_noise_std, size=len(kept))
    visit_months = np.clip(
        np.array(kept, dtype=np.float32) + jitter.astype(np.float32),
        a_min=0.0,
        a_max=None,
    )
    # Keep sorted and ensure baseline is exactly 0
    visit_months[0] = 0.0
    visit_months    = np.sort(visit_months)

    # ── Integrate ground-truth ODE analytically ───────────────────────────────
    # dz_atrophy/dt = -λ · z_atrophy   →   z_atrophy(t) = z0_atrophy · exp(-λt)
    # z_identity is constant.
    T             = len(visit_months)
    z_trajectory  = np.empty((T, latent_dim), dtype=np.float32)

    for i, t in enumerate(visit_months):
        z_a             = z0_atrophy * np.exp(-decay_rate * t)
        z_trajectory[i, :n_atrophy] = z_a
        z_trajectory[i, n_atrophy:] = z0_identity

    # ── CDRSB trajectory ──────────────────────────────────────────────────────
    # Temporarily build a patient stub to reuse _cdrsb_from_z
    atrophy_norms  = np.abs(z_trajectory[:, :n_atrophy]).sum(axis=1)
    raw_cdrsb      = cdrsb_c0 + cdrsb_k * atrophy_norms
    cdrsb_traj     = np.clip(raw_cdrsb, CDRSB_MIN, CDRSB_MAX).astype(np.float32)

    # Add small measurement noise to CDRSB (clinical assessments aren't perfect)
    cdrsb_noise    = rng.normal(0, 0.15, size=T).astype(np.float32)
    cdrsb_traj     = np.clip(cdrsb_traj + cdrsb_noise, CDRSB_MIN, CDRSB_MAX)

    return SyntheticPatient(
        patient_id       = patient_id,
        diagnosis        = diagnosis,
        apoe4_carrier    = apoe4,
        age_at_baseline  = age,
        sex              = sex,
        decay_rate       = decay_rate,
        cdrsb_intercept  = cdrsb_c0,
        cdrsb_slope      = cdrsb_k,
        z0               = z0,
        visit_months     = visit_months,
        z_trajectory     = z_trajectory,
        cdrsb_trajectory = cdrsb_traj,
        n_atrophy_dims   = n_atrophy,
    )


# ── Public entry-point ─────────────────────────────────────────────────────────

def make_synthetic_cohort(
    n_patients: int = 300,
    latent_dim: int = 512,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    min_visits: int = 3,
    dx_distribution: Optional[Dict[str, float]] = None,
) -> Dict[str, SyntheticDataset]:
    """
    Generate a synthetic cohort and return stratified patient-level splits.

    This is the single entry-point for training scripts and tests.  It
    mirrors the interface of the real ADNI loader so that Stage 1–3 training
    scripts work identically on synthetic and real data.

    Args:
        n_patients:       Total number of synthetic patients to generate.
        latent_dim:       Dimension of z — must match stage1.latent_dim in config.
        seed:             Reproducibility seed (Rule 2).  Passed to both the
                          patient generator and generate_patient_splits().
        train_ratio:      Fraction of patients for training (default 0.70).
        val_ratio:        Fraction for validation (default 0.15).
        min_visits:       Minimum visits per patient (mirrors data.min_visits in config).
        dx_distribution:  Fraction of patients per diagnosis group.
                          Defaults to {"CN": 0.45, "MCI": 0.40, "AD": 0.15},
                          matching approximate ADNI proportions.

    Returns:
        {"train": SyntheticDataset, "val": SyntheticDataset, "test": SyntheticDataset}

    Raises:
        AssertionError:  via validate_no_patient_overlap if any split leaks.
        ValueError:      on invalid ratio arguments or dx_distribution.
    """
    if dx_distribution is None:
        dx_distribution = {"CN": 0.45, "MCI": 0.40, "AD": 0.15}

    _validate_dx_distribution(dx_distribution)

    rng = np.random.default_rng(seed)

    # ── Generate patients ─────────────────────────────────────────────────────
    diagnoses = _sample_diagnoses(n_patients, dx_distribution, rng)
    patients  = []
    for i, dx in enumerate(diagnoses):
        pid = f"SYN_{i:05d}"
        p   = _generate_one_patient(
            patient_id  = pid,
            diagnosis   = dx,
            latent_dim  = latent_dim,
            min_visits  = min_visits,
            rng         = rng,
        )
        patients.append(p)

    logger.info(
        "[synthetic] Generated %d patients  (CN=%d  MCI=%d  AD=%d)  "
        "latent_dim=%d  seed=%d",
        len(patients),
        sum(1 for p in patients if p.diagnosis == "CN"),
        sum(1 for p in patients if p.diagnosis == "MCI"),
        sum(1 for p in patients if p.diagnosis == "AD"),
        latent_dim,
        seed,
    )

    # ── Build a visit-level DataFrame for generate_patient_splits ─────────────
    # (mirrors the structure of the real ADNI visit DataFrame)
    import pandas as pd  # imported here to keep the import visible near use

    rows = []
    for p in patients:
        for t, cdrsb in zip(p.visit_months, p.cdrsb_trajectory):
            rows.append({
                "RID":         p.patient_id,
                "DX_bl":       p.diagnosis,
                "visit_month": float(t),
                "CDRSB":       float(cdrsb),
            })
    visit_df = pd.DataFrame(rows)

    # ── Patient-level stratified splits (Rule 1) ──────────────────────────────
    split_ids = generate_patient_splits(
        visit_df,
        patient_id_col  = "RID",
        diagnosis_col   = "DX_bl",
        visit_month_col = "visit_month",
        train_ratio     = train_ratio,
        val_ratio       = val_ratio,
        seed            = seed,
    )

    # Rule 1 — explicit re-check even though generate_patient_splits calls it
    # internally.  Belt and suspenders for a correctness-critical invariant.
    validate_no_patient_overlap(
        split_ids["train"], split_ids["val"], split_ids["test"]
    )

    # ── Assemble per-split datasets ───────────────────────────────────────────
    patient_map: Dict[str, SyntheticPatient] = {p.patient_id: p for p in patients}
    datasets    = {}
    for split_name, ids in split_ids.items():
        split_patients = [patient_map[pid] for pid in ids]
        datasets[split_name] = SyntheticDataset(split_patients)
        logger.info(
            "[synthetic] %s split: %d patients", split_name, len(split_patients)
        )

    return datasets


# ── Utilities ──────────────────────────────────────────────────────────────────

def _validate_dx_distribution(dx_dist: Dict[str, float]) -> None:
    """Validate that dx_distribution sums to 1.0 and has valid keys."""
    valid_keys = {"CN", "MCI", "AD"}
    unknown    = set(dx_dist.keys()) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown diagnosis keys in dx_distribution: {unknown}. "
            f"Valid keys are {valid_keys}."
        )
    total = sum(dx_dist.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"dx_distribution values must sum to 1.0, got {total:.6f}."
        )


def _sample_diagnoses(
    n: int,
    dx_distribution: Dict[str, float],
    rng: np.random.Generator,
) -> List[str]:
    """
    Sample n diagnosis labels according to dx_distribution.
    Uses np.random.Generator for reproducibility (Rule 2).
    """
    labels = list(dx_distribution.keys())
    probs  = np.array([dx_distribution[k] for k in labels], dtype=float)
    probs /= probs.sum()   # normalise to guard against floating-point drift
    indices = rng.choice(len(labels), size=n, p=probs)
    return [labels[i] for i in indices]


def get_counterfactual_ground_truth(
    dataset: SyntheticDataset,
    intervention_start_month: float,
    alpha: float,
) -> Dict[str, np.ndarray]:
    """
    Compute exact counterfactual trajectories for every patient in a dataset.

    This is the oracle used in synthetic validation experiments (design doc
    §5.2A, §8.4).  The returned dict maps patient_id → (z_cf, cdrsb_cf).

    Rule 9 (language discipline): returned arrays are named
    "simulated_trajectory" in calling code, not "causal_effect".

    Args:
        dataset:                   SyntheticDataset to evaluate.
        intervention_start_month:  t0 — when the intervention begins.
        alpha:                     Velocity scaling factor in [0, 1].

    Returns:
        {
          patient_id: {
            "z_cf":      np.ndarray shape (T, latent_dim),
            "cdrsb_cf":  np.ndarray shape (T,),
          }
        }
    """
    results = {}
    for patient in dataset.patients:
        z_cf, cdrsb_cf = patient.counterfactual_trajectory(
            intervention_start_month = intervention_start_month,
            alpha                    = alpha,
        )
        results[patient.patient_id] = {
            "z_cf":     z_cf,
            "cdrsb_cf": cdrsb_cf,
        }
    return results

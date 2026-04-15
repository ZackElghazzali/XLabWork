"""
utils/seed.py
Rule 2: Reproducibility is Mandatory.
Sets a global seed across Python, NumPy, PyTorch (CPU + CUDA) before any
computation. Call this as the second line of every training script, right
after validate_no_patient_overlap().
"""
import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Set deterministic seed across all relevant libraries.

    Args:
        seed: Integer seed. Must match the value in the YAML config
              (config["reproducibility"]["seed"]).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # covers multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # sacrifices speed for reproducibility

    logger.info("[seed] Global seed set to %d", seed)

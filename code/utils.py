"""Reproducibility utilities for DeepVCF experiments."""

import random

import numpy as np
import torch


def set_seeds(seed: int):
    """Seed Python, NumPy, and PyTorch for best-effort reproducibility.

    Python's hash seed must be set through ``PYTHONHASHSEED`` before starting
    the interpreter; changing that environment variable here would not affect
    the current process. The cuDNN flags below constrain cuDNN algorithm
    selection but do not guarantee that every PyTorch operation is
    deterministic.

    Args:
        seed (int): Random seed in the shared Python/NumPy/PyTorch range
            ``[0, 2**32)``.

    Raises:
        ValueError: If ``seed`` is not a non-boolean integer in the supported
            range.
    """
    if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or not 0 <= int(seed) < 2**32):
        raise ValueError('seed must be an integer in [0, 2**32).')
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

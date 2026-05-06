from __future__ import annotations

import torch


def is_gpu_available() -> bool:
    """Return True if any GPU backend (CUDA, ROCm/HIP, or MPS) is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def get_device(preferred: str = "") -> torch.device:
    """
    Resolve the best available torch device.

    Args:
        preferred: User-preferred device string (e.g. "cuda", "cuda:0",
                   "mps", "cpu"). If empty or unavailable, auto-detect.

    Returns:
        A ``torch.device`` ready for model/tensor placement.
    """
    if preferred:
        try:
            dev = torch.device(preferred)
            # Quick sanity check: if the user asked for cuda but no GPUs exist,
            # fall back rather than crashing later.
            if dev.type == "cuda" and not torch.cuda.is_available():
                pass  # fall through to auto-detect
            elif dev.type == "mps" and not torch.backends.mps.is_available():
                pass  # fall through to auto-detect
            else:
                return dev
        except RuntimeError:
            pass  # invalid device string, fall through

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed_all(seed: int) -> None:
    """Set RNG seed for CPU and all available GPU backends."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # MPS does not have a separate global seed API; torch.manual_seed covers it.

from __future__ import annotations
import os
import logging
import torch
from transformers.integrations.bitsandbytes import (
    _validate_bnb_multi_backend_availability,
)
import importlib.metadata

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _is_bnb_installed() -> bool:
    """Return True if the bitsandbytes package is installed."""
    try:
        importlib.metadata.version("bitsandbytes")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
    except Exception:
        return False


def detect_device_and_quant(user_opt_out: bool) -> tuple[dict | str, torch.dtype, bool]:
    """Detect device and desired quantisation settings.

    Returns
    -------
    device_map : dict | str
        Mapping passed to Transformers ``from_pretrained``.
    torch_dtype : torch.dtype
        Precision to load weights with.
    use_4bit : bool
        Whether to attempt 4-bit quantisation via bitsandbytes.
    """
    if user_opt_out:
        return "cpu", torch.float32, False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for 4-bit weights")

    if not _is_bnb_installed():
        raise RuntimeError("bitsandbytes is required for 4-bit weights")

    _validate_bnb_multi_backend_availability(raise_exception=True)
    return "auto", torch.bfloat16, True

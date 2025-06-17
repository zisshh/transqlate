from __future__ import annotations
import os
import torch
from transformers.integrations.bitsandbytes import (
    is_bitsandbytes_available,
    _validate_bnb_multi_backend_availability,
)


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

    if torch.cuda.is_available():
        if is_bitsandbytes_available():
            try:
                _validate_bnb_multi_backend_availability()
                return "auto", torch.bfloat16, True
            except Exception:
                pass
        return "auto", torch.float16, False

    if torch.backends.mps.is_available():
        return {"": "mps"}, torch.float16, False

    return "cpu", torch.float32, False

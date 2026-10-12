"""Deterministic pure-PyTorch bilinear resize for the UFM RGB-D trainer.

This module provides :func:`deterministic_bilinear_interpolate`, a drop-in for
``torch.nn.functional.interpolate(..., mode="bilinear")`` limited to **4D
floating tensors** (``NCHW``).  It is written so that the backward pass stays on
the deterministic CUDA path used by strict reproducibility runs:

* the interpolation is **separable** -- a horizontal two-point weighted sum,
  then a vertical two-point weighted sum;
* both gathers are plain ``Tensor.index_select`` calls (never
  ``F.interpolate`` / ``grid_sample`` / ``gather``-based helpers);
* under ``torch.use_deterministic_algorithms(True)`` the ``index_select``
  backward runs through the deterministic ``index_add`` path, so the whole
  resize has no atomics-dependent nondeterminism.

The forward math reproduces PyTorch's ``upsample_bilinear2d`` sampling:

* ``align_corners=True``  -> ``scale = (in - 1) / (out - 1)``,
  ``src = scale * dst``;
* ``align_corners=False`` -> half-pixel rule.  When a ``scale_factor`` is
  supplied the kernel scale is ``1 / scale_factor`` (unless
  ``recompute_scale_factor=True``), otherwise ``in / out``; then
  ``src = scale * (dst + 0.5) - 0.5`` clamped to ``>= 0``;
* the two taps are ``floor(src)`` and ``min(floor(src) + 1, in - 1)`` with
  weight ``frac = src - floor(src)``.

Nothing here embeds private data, host paths, dataset identifiers, or secrets.
The helper's arithmetic is a pure function; the *hook-up* lives in
:func:`deterministic_forward_context`, which patches ``F.interpolate`` only for
the duration of a single forward scope (and restores it in ``finally``).
"""

from __future__ import annotations

import contextlib
import math
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = [
    "DeterministicInterpolateError",
    "DeterministicForwardError",
    "deterministic_bilinear_interpolate",
    "deterministic_forward_context",
]


class DeterministicInterpolateError(ValueError):
    """Raised when the requested resize is outside the supported contract."""


def _as_pair(value, name: str) -> Tuple[object, object]:
    """Normalise an int / 1-tuple / 2-tuple into a ``(h, w)`` pair."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise DeterministicInterpolateError(
                f"{name} must be an int or a length-2 sequence for 4D input, "
                f"got length {len(value)}"
            )
        return value[0], value[1]
    return value, value


def _require_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeterministicInterpolateError(
            f"{name} entries must be int, got {type(value).__name__}"
        )
    if value <= 0:
        raise DeterministicInterpolateError(f"{name} entries must be > 0, got {value}")
    return value


def _require_scale(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeterministicInterpolateError(
            f"{name} entries must be a number, got {type(value).__name__}"
        )
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise DeterministicInterpolateError(
            f"{name} entries must be a finite positive number, got {value}"
        )
    return value


def _source_index(scale: float, dst: int, align_corners: bool) -> float:
    if align_corners:
        return scale * dst
    src = scale * (dst + 0.5) - 0.5
    return src if src >= 0.0 else 0.0


def _axis_plan(in_size: int, out_size: int, align_corners: bool, scale) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(lower_idx, upper_idx, upper_weight)`` for one spatial axis.

    ``lower_idx`` / ``upper_idx`` are the two taps and ``upper_weight`` is the
    weight applied to the upper tap (the lower tap gets ``1 - upper_weight``).
    """
    if align_corners:
        if out_size > 1:
            kernel_scale = (in_size - 1) / (out_size - 1)
        else:
            kernel_scale = 0.0
    elif scale is not None:
        kernel_scale = 1.0 / scale
    else:
        kernel_scale = in_size / out_size

    lower = []
    upper = []
    frac = []
    for dst in range(out_size):
        src = _source_index(kernel_scale, dst, align_corners)
        lo = int(math.floor(src))
        if lo > in_size - 1:
            lo = in_size - 1
        if lo < 0:
            lo = 0
        hi = lo + 1
        if hi > in_size - 1:
            hi = in_size - 1
        lower.append(lo)
        upper.append(hi)
        frac.append(src - lo)
    return (
        torch.tensor(lower),
        torch.tensor(upper),
        torch.tensor(frac, dtype=torch.float64),
    )


def deterministic_bilinear_interpolate(
    input: torch.Tensor,
    size=None,
    scale_factor=None,
    align_corners: bool = False,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> torch.Tensor:
    """Bilinear resize of a 4D floating tensor with a deterministic backward.

    Mirrors ``F.interpolate(input, size, scale_factor, mode="bilinear",
    align_corners=..., recompute_scale_factor=...)`` for the supported subset.
    """
    if antialias:
        raise DeterministicInterpolateError(
            "antialias=True is not supported by deterministic_bilinear_interpolate"
        )
    if recompute_scale_factor not in (None, False, True):
        raise DeterministicInterpolateError(
            f"recompute_scale_factor must be None/bool, got {recompute_scale_factor!r}"
        )
    if not isinstance(align_corners, bool):
        raise DeterministicInterpolateError(
            f"align_corners must be bool, got {type(align_corners).__name__}"
        )
    if not isinstance(input, torch.Tensor):
        raise DeterministicInterpolateError(
            f"input must be a torch.Tensor, got {type(input).__name__}"
        )
    if input.dim() != 4:
        raise DeterministicInterpolateError(
            f"input must be 4D (N, C, H, W), got {input.dim()}D"
        )
    if not input.is_floating_point():
        raise DeterministicInterpolateError(
            f"input must be a floating tensor, got dtype {input.dtype}"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise DeterministicInterpolateError(
            f"only float32 / float64 are supported, got {input.dtype}"
        )
    if size is not None and scale_factor is not None:
        raise DeterministicInterpolateError(
            "specify only one of size or scale_factor, not both"
        )
    if size is None and scale_factor is None:
        raise DeterministicInterpolateError(
            "one of size or scale_factor must be specified"
        )

    n, c, in_h, in_w = input.shape

    if size is not None:
        pair = _as_pair(size, "size")
        out_h = _require_int(pair[0], "size")
        out_w = _require_int(pair[1], "size")
        scale_h = scale_w = None
    else:
        pair = _as_pair(scale_factor, "scale_factor")
        scale_h = _require_scale(pair[0], "scale_factor")
        scale_w = _require_scale(pair[1], "scale_factor")
        out_h = int(math.floor(in_h * scale_h))
        out_w = int(math.floor(in_w * scale_w))
        if out_h <= 0 or out_w <= 0:
            raise DeterministicInterpolateError(
                f"scale_factor yields non-positive output size ({out_h}, {out_w})"
            )
        if recompute_scale_factor:
            scale_h = scale_w = None

    if (out_h, out_w) == (in_h, in_w):
        return input

    device = input.device
    weight_dtype = input.dtype

    lo_h, hi_h, w_h = _axis_plan(in_h, out_h, align_corners, scale_h)
    lo_w, hi_w, w_w = _axis_plan(in_w, out_w, align_corners, scale_w)
    lo_h = lo_h.to(device)
    hi_h = hi_h.to(device)
    w_h = w_h.to(device=device, dtype=weight_dtype).view(out_h, 1)
    lo_w = lo_w.to(device)
    hi_w = hi_w.to(device)
    w_w = w_w.to(device=device, dtype=weight_dtype).view(out_w)

    lower = input.index_select(3, lo_w)
    upper = input.index_select(3, hi_w)
    horizontal = lower * (1.0 - w_w) + upper * w_w

    lower = horizontal.index_select(2, lo_h)
    upper = horizontal.index_select(2, hi_h)
    result = lower * (1.0 - w_h) + upper * w_h
    return result


# --------------------------------------------------------------------------- #
# bounded forward hook-up (Step 19 attempt-10)
# --------------------------------------------------------------------------- #
class DeterministicForwardError(ValueError):
    """Raised when a forward-hook determinism setting is outside its contract."""


#: captured once so the wrapper can always delegate non-bilinear modes to the
#: *genuine* torch implementation (never to itself / an outer wrapper).
_ORIGINAL_INTERPOLATE = F.interpolate

#: accepted ``determinism.sdpa_backend`` / ``determinism.bilinear`` values.
_ALLOWED_SDPA_BACKENDS: Tuple[str, ...] = ("default", "math")
_ALLOWED_BILINEAR_MODES: Tuple[str, ...] = ("default", "deterministic")

#: SDPA enable-flag names on ``torch.backends.cuda`` (query + setter suffixes).
_SDPA_FLAG_NAMES: Tuple[str, ...] = ("math", "flash", "mem_efficient", "cudnn")


def _snapshot_sdpa_flags() -> dict:
    """Capture the four SDPA backend enable flags (works on CPU and CUDA)."""
    cuda = torch.backends.cuda
    return {name: bool(getattr(cuda, f"{name}_sdp_enabled")()) for name in _SDPA_FLAG_NAMES}


def _restore_sdpa_flags(flags: Mapping) -> None:
    """Restore all four SDPA backend enable flags verbatim."""
    cuda = torch.backends.cuda
    for name in _SDPA_FLAG_NAMES:
        getattr(cuda, f"enable_{name}_sdp")(bool(flags[name]))


def _deterministic_interpolate(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
    antialias=False,
):
    """Scope-only ``F.interpolate`` wrapper: bilinear/4D -> deterministic helper.

    Any other mode (or non-4D input) is forwarded to the genuine
    ``torch.nn.functional.interpolate`` captured at import time, so the wrapper
    never recurses and never changes unrelated resize behaviour.
    """
    if mode == "bilinear" and isinstance(input, torch.Tensor) and input.dim() == 4:
        return deterministic_bilinear_interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            align_corners=False if align_corners is None else align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )
    return _ORIGINAL_INTERPOLATE(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


@contextlib.contextmanager
def _patched_interpolate():
    """Replace ``F.interpolate`` for the scope and restore it in ``finally``."""
    previous = F.interpolate
    F.interpolate = _deterministic_interpolate
    try:
        yield
    finally:
        F.interpolate = previous


def _determinism_group(cfg) -> Mapping:
    """Return the ``determinism`` mapping from a dict-like / OmegaConf cfg."""
    getter = getattr(cfg, "get", None)
    det = getter("determinism") if callable(getter) else None
    if det is None:
        return {}
    if isinstance(det, Mapping):
        return det
    raise DeterministicForwardError(
        f"determinism must be a mapping, got {type(det).__name__}"
    )


def _forward_setting(det: Mapping, key: str, allowed: Tuple[str, ...]) -> str:
    value = det.get(key, "default")
    if value not in allowed:
        raise DeterministicForwardError(
            f"determinism.{key} must be one of {allowed}, got {value!r}"
        )
    return value


@contextlib.contextmanager
def deterministic_forward_context(cfg):
    """Bounded, single-scope determinism hook for one model forward pass.

    Reads ``cfg.determinism.sdpa_backend`` / ``cfg.determinism.bilinear`` (a
    missing group / key means the no-op ``"default"``):

    * ``sdpa_backend == "math"`` -> all scaled-dot-product-attention backends are
      restricted to the math kernel via
      :func:`torch.nn.attention.sdpa_kernel` ``(SDPBackend.MATH)``;
    * ``bilinear == "deterministic"`` -> ``F.interpolate`` is replaced *only
      inside this scope* by :func:`_deterministic_interpolate` (mode
      ``"bilinear"`` / 4D -> :func:`deterministic_bilinear_interpolate`; every
      other call is unchanged).

    On exit -- including on exception -- ``F.interpolate`` and *all* SDPA enable
    flags are restored in ``finally``.  The context nests: an inner scope
    restores exactly the flags/function that were active when it was entered.
    This never mutates the model source or the public ``predict`` path; it only
    affects calls made while the scope is active.
    """
    det = _determinism_group(cfg)
    sdpa = _forward_setting(det, "sdpa_backend", _ALLOWED_SDPA_BACKENDS)
    bilinear = _forward_setting(det, "bilinear", _ALLOWED_BILINEAR_MODES)

    if sdpa == "default" and bilinear == "default":
        yield
        return

    with contextlib.ExitStack() as stack:
        if sdpa == "math":
            # belt-and-suspenders: snapshot the flags ourselves and restore them
            # in finally even if the (beta) sdpa_kernel internals change.
            flags = _snapshot_sdpa_flags()
            stack.callback(_restore_sdpa_flags, flags)
            stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        if bilinear == "deterministic":
            stack.enter_context(_patched_interpolate())
        yield

"""Generic closest-to-camera (minimum) valid-depth dilation.

Backend-agnostic utility implementing a 3x3 closest-to-camera minimum
valid-depth dilation. It is deliberately dataset- and aligner-independent:
no dataset-specific paths, calibration or private identifiers are referenced.

Semantics
---------
For every output pixel the value is the **minimum** (closest-to-camera) valid
depth found in its 3x3 neighbourhood (``radius = 1``). Regular value-max
dilation is explicitly forbidden.

Invalid inputs (never used as a source, never propagated):

* integer ``uint16`` (millimetres): value ``0`` and the sentinel ``65535``.
* floating-point (metric, e.g. metres): value ``0``, any non-finite value
  (``NaN``/``+inf``/``-inf``) and any negative value.

The ``65535`` sentinel is a **uint16 millimetre** rule only. It is deliberately
*not* applied to floating-point metric inputs, where ``65535.0`` is a normal,
valid measurement. This keeps the sentinel unambiguous across units.

If every neighbour is invalid the output pixel is ``0`` and its provenance is
``none``.

Borders use invalid padding: out-of-image neighbours are skipped (treated as
invalid), so support is clipped at the image edges.

Mask policy and validation
--------------------------
``mask`` is treated **independently** of the depth values. When a mask is
provided, a pixel is a valid source only when the depth value is valid *and*
``mask`` is true. A false mask therefore **rejects/invalidates** an otherwise
numeric-valid depth: that value is ignored as a source and never dilated into
neighbours. The mask is never modified, and it does not by itself create
values.

The mask must match the depth input in **shape**, **backend** and **device**.
A mismatch fails fast instead of silently broadcasting:

* shape mismatch -> ``ValueError`` (e.g. a ``(1, 3)`` or ``(3,)`` mask for a
  ``3x3`` depth is rejected, not broadcast).
* backend mismatch -> ``TypeError``. The NumPy backend accepts array-like
  masks (``numpy.ndarray``, Python lists/tuples), which are converted with
  ``numpy.asarray``; a ``torch.Tensor`` mask with NumPy depth is rejected. The
  torch backend requires an actual ``torch.Tensor`` mask, so a Python list or
  ``numpy.ndarray`` mask with torch depth is rejected.
* device mismatch -> ``ValueError`` (e.g. a CPU mask with a CUDA depth).

A mask may be boolean or any bool-castable array-like; it is cast to boolean
internally and never written to.

Supported inputs
----------------
``depth`` must be 2D and either ``uint16`` or a floating-point dtype
(``float16``/``float32``/``float64``/``bfloat16`` for torch). Unsupported
dtypes (signed/unsigned integers other than ``uint16``, ``bool``, complex, ...)
are rejected with ``TypeError``.

Supported backends
------------------
NumPy arrays and ``torch.Tensor`` (CPU or CUDA). Output dtype and device follow
the input. ``torch`` is not a required eager dependency: it is detected from
``sys.modules`` only and imported lazily when a real ``torch.Tensor`` is used.

Immutability
------------
The input depth and mask arrays are never written to. The returned ``depth``,
``valid_mask``, ``added_support``, ``changed_measured`` and ``provenance`` are
newly allocated arrays.

Existing measured pixels may still change: a measured pixel whose neighbour is
closer (smaller) becomes smaller. Those pixels are reported through
``changed_measured`` and ``provenance == modified``; newly supported pixels are
reported through ``added_support`` and ``provenance == filled``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Provenance codes for every output pixel.
PROVENANCE_NONE = 0        # no valid source reached this pixel (output 0)
PROVENANCE_UNCHANGED = 1   # measured pixel, value unchanged
PROVENANCE_FILLED = 2      # previously invalid pixel, now supported
PROVENANCE_MODIFIED = 3    # measured pixel, value changed (became smaller)

# The 3x3 closest/min-valid dilation radius is fixed. KERNEL = 2*radius+1 = 3.
DEFAULT_RADIUS = 1
UINT16_SENTINEL = 65535


@dataclass
class MinValidDepthResult:
    """Result of :func:`min_valid_depth_dilate`.

    Attributes
    ----------
    depth:
        Dilation output (separate array; input is untouched). Same dtype/device
        as the input. Invalid output pixels are ``0``.
    valid_mask:
        Boolean mask of output pixels that received a valid depth.
    added_support:
        Boolean mask of pixels that were invalid on input but are valid on
        output (``provenance == PROVENANCE_FILLED``).
    changed_measured:
        Boolean mask of measured pixels whose value changed
        (``provenance == PROVENANCE_MODIFIED``).
    provenance:
        ``uint8`` map with values ``0=none, 1=unchanged, 2=filled, 3=modified``.
    """

    depth: Any
    valid_mask: Any
    added_support: Any
    changed_measured: Any
    provenance: Any


def _torch_module():
    """Return the already-imported ``torch`` module, or ``None``.

    ``torch`` is not a required eager dependency. A value can only be a real
    ``torch.Tensor`` when ``torch`` is already imported, so inspecting
    ``sys.modules`` is both sufficient and safe. Backend detection never relies
    on a ``type.__module__`` string, which monkey-patching could spoof.
    """
    module = sys.modules.get("torch")
    if module is not None and hasattr(module, "Tensor"):
        return module
    return None


def _is_torch_tensor(value: Any) -> bool:
    torch = _torch_module()
    return torch is not None and isinstance(value, torch.Tensor)


def _validate_depth_dtype_numpy(depth: np.ndarray) -> None:
    if depth.dtype == np.uint16 or np.issubdtype(depth.dtype, np.floating):
        return
    raise TypeError(
        "depth dtype must be uint16 or floating-point, got "
        f"{depth.dtype!r}"
    )


def _validate_depth_dtype_torch(depth: Any, torch) -> None:
    allowed = (torch.uint16, torch.float16, torch.float32, torch.float64, torch.bfloat16)
    if depth.dtype in allowed:
        return
    raise TypeError(
        "depth dtype must be uint16 or floating-point, got "
        f"{depth.dtype}"
    )


def _validate_mask_numpy(mask: Any, depth: np.ndarray):
    """Validate/convert a NumPy-backend mask, failing fast on mismatch."""
    if mask is None:
        return None
    if _is_torch_tensor(mask):
        raise TypeError(
            "mask backend (torch.Tensor) does not match depth backend "
            "(NumPy array)"
        )
    mask_array = np.asarray(mask)
    if mask_array.shape != depth.shape:
        raise ValueError(
            f"mask shape {mask_array.shape} does not match depth shape "
            f"{depth.shape}"
        )
    return mask_array.astype(bool)


def _validate_mask_torch(mask: Any, depth: Any, torch):
    """Validate/convert a torch-backend mask, failing fast on mismatch."""
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            "mask backend does not match depth backend (torch.Tensor); "
            f"got {type(mask).__name__}"
        )
    if tuple(mask.shape) != tuple(depth.shape):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match depth shape "
            f"{tuple(depth.shape)}"
        )
    if mask.device != depth.device:
        raise ValueError(
            f"mask device {mask.device} does not match depth device "
            f"{depth.device}"
        )
    return mask.to(torch.bool)


def _min_filter_numpy(value: np.ndarray, radius: int) -> np.ndarray:
    h, w = value.shape
    padded = np.pad(value, radius, mode="constant", constant_values=np.inf)
    acc = np.full(value.shape, np.inf, dtype=value.dtype)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            acc = np.minimum(acc, padded[dy:dy + h, dx:dx + w])
    return acc


def _min_valid_depth_numpy(depth, mask, sentinel, radius):
    is_uint16 = depth.dtype == np.uint16
    if is_uint16:
        invalid = (depth == 0) | (depth == sentinel)
    else:
        d64 = depth.astype(np.float64)
        invalid = (depth == 0) | ~np.isfinite(d64) | (d64 < 0)

    measured = ~invalid
    if mask is not None:
        measured = measured & mask.astype(bool)

    value = np.where(measured, depth.astype(np.float64), np.inf)
    acc = _min_filter_numpy(value, radius)
    valid_out = np.isfinite(acc)
    out = np.where(valid_out, acc, 0.0).astype(depth.dtype)

    provenance = np.zeros(depth.shape, dtype=np.uint8)
    provenance[valid_out] = PROVENANCE_FILLED
    provenance[valid_out & measured] = PROVENANCE_UNCHANGED
    provenance[measured & valid_out & (out != depth)] = PROVENANCE_MODIFIED

    return MinValidDepthResult(
        depth=out,
        valid_mask=valid_out,
        added_support=valid_out & ~measured,
        changed_measured=measured & valid_out & (out != depth),
        provenance=provenance,
    )


def _min_valid_depth_torch(depth, mask, sentinel, radius):
    import torch
    import torch.nn.functional as F

    is_uint16 = depth.dtype == torch.uint16
    if is_uint16:
        invalid = (depth == 0) | (depth == sentinel)
    else:
        d64 = depth.to(torch.float64)
        invalid = (depth == 0) | (~torch.isfinite(d64)) | (d64 < 0)

    measured = ~invalid
    if mask is not None:
        measured = measured & mask.to(torch.bool)

    value = torch.where(measured, depth.to(torch.float64), torch.full_like(
        depth.to(torch.float64), float("inf")))
    h, w = value.shape
    padded = F.pad(value[None, None], (radius, radius, radius, radius), value=float("inf"))[0, 0]
    acc = torch.full_like(value, float("inf"))
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            acc = torch.minimum(acc, padded[dy:dy + h, dx:dx + w])

    valid_out = torch.isfinite(acc)
    out = torch.where(valid_out, acc, torch.zeros_like(acc)).to(depth.dtype)

    provenance = torch.zeros(depth.shape, dtype=torch.uint8, device=depth.device)
    provenance[valid_out] = PROVENANCE_FILLED
    provenance[valid_out & measured] = PROVENANCE_UNCHANGED
    provenance[measured & valid_out & (out != depth)] = PROVENANCE_MODIFIED

    return MinValidDepthResult(
        depth=out,
        valid_mask=valid_out,
        added_support=valid_out & ~measured,
        changed_measured=measured & valid_out & (out != depth),
        provenance=provenance,
    )


def min_valid_depth_dilate(
    depth,
    mask: Optional[Any] = None,
    *,
    sentinel: int = UINT16_SENTINEL,
    radius: int = DEFAULT_RADIUS,
) -> MinValidDepthResult:
    """3x3 closest-to-camera (minimum) valid-depth dilation.

    Parameters
    ----------
    depth:
        Depth image, ``uint16`` (millimetres) or floating-point (metric, e.g.
        metres), 2D. NumPy array or ``torch.Tensor``. Other dtypes raise
        ``TypeError``.
    mask:
        Optional mask matching ``depth`` in shape, backend and device. NumPy
        depth accepts array-like masks (``numpy.ndarray``, Python lists/tuples),
        converted with ``numpy.asarray``; torch depth requires a
        ``torch.Tensor``. Treated independently of depth values: false rejects
        an otherwise valid depth pixel. Invalid masks fail fast with
        ``ValueError`` (shape/device) or ``TypeError`` (backend) instead of
        silently broadcasting.
    sentinel:
        Invalid ``uint16`` sentinel value (default ``65535``). Not applied to
        floating-point inputs.
    radius:
        Neighbourhood radius; fixed default ``1`` (3x3 kernel). Must be a
        non-negative integer.

    Returns
    -------
    MinValidDepthResult
    """
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError(f"radius must be a non-negative integer, got {radius!r}")

    torch = _torch_module()
    if torch is not None and isinstance(depth, torch.Tensor):
        if depth.dim() != 2:
            raise ValueError(f"depth must be 2D, got shape {tuple(depth.shape)}")
        _validate_depth_dtype_torch(depth, torch)
        mask = _validate_mask_torch(mask, depth, torch)
        return _min_valid_depth_torch(depth, mask, sentinel, radius)

    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got shape {depth.shape}")
    _validate_depth_dtype_numpy(depth)
    mask = _validate_mask_numpy(mask, depth)
    return _min_valid_depth_numpy(depth, mask, sentinel, radius)


__all__ = [
    "PROVENANCE_NONE",
    "PROVENANCE_UNCHANGED",
    "PROVENANCE_FILLED",
    "PROVENANCE_MODIFIED",
    "DEFAULT_RADIUS",
    "UINT16_SENTINEL",
    "MinValidDepthResult",
    "min_valid_depth_dilate",
]

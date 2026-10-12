"""Checkpoint identity, safe loading and RGB->RGB-D migration helpers.

This module is a *generic*, dependency-light utility for pinning the identity of
a released checkpoint (bytes + SHA256 + revision), loading its state safely, and
migrating a released RGB state onto an RGB-D model that adds a new
zero-initialised module.

Motivation (workdoc procedure 16)
---------------------------------
* ``huggingface_hub.PyTorchModelHubMixin.from_pretrained`` silently calls
  ``load_state_dict(..., strict=False)`` and therefore *ignores* missing and
  unexpected keys.  That is unacceptable for a scientific migration: a truncated
  or wrong checkpoint would load without any error.  The helpers here always
  compute the full missing / unexpected / shape-mismatch diff and fail fast on
  any *genuine* discrepancy.
* The released UFM-Base safetensors deduplicates shared parameters: the
  ``input_process`` sequential re-uses the ``scratch.layer*_rn`` convolution
  objects, so only the canonical name is stored and the alias name shows up as a
  "missing" key.  Because the two names share one storage, loading the canonical
  name fills both.  ``classify_missing`` recognises exactly this benign case
  (a missing key whose storage is shared with a loaded key) and rejects any
  missing key that is *not* an alias -- so the mixin's silent behaviour can never
  hide a truly absent weight.
* The RGB-D model adds exactly one new module (``depth_fusion``: a small CNN
  adapter + a zero-initialised scalar gate).  Migrating the released RGB state
  must report *exactly* those new keys as the only genuine missing keys, and must
  preserve the zero-initialised gate so the fused model is numerically identical
  to RGB at initialisation.
* Checkpoints are loaded with ``weights_only=True`` by default so a hostile or
  corrupt pickle can never execute arbitrary code.

Nothing in this module embeds private data, calibration, credentials, or
absolute host paths; callers pass file paths and expected digests explicitly.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

__all__ = [
    "CheckpointError",
    "CheckpointIdentityError",
    "CheckpointFormatError",
    "CheckpointStateError",
    "CheckpointNonFiniteError",
    "CheckpointMigrationError",
    "sha256_file",
    "verify_file_identity",
    "load_checkpoint",
    "load_safetensors_state",
    "extract_state_dict",
    "state_dict_diff",
    "alias_groups",
    "classify_missing",
    "check_state_finite",
    "strict_load_state_dict",
    "load_released_state",
    "expected_new_keys",
    "migrate_rgb_to_rgbd",
]


# --------------------------------------------------------------------------- #
# typed fail-fast exceptions
# --------------------------------------------------------------------------- #
class CheckpointError(Exception):
    """Base class for checkpoint contract violations."""


class CheckpointIdentityError(CheckpointError):
    """File bytes / SHA256 do not match the pinned identity."""


class CheckpointFormatError(CheckpointError):
    """The checkpoint file is unreadable, truncated or of an unknown format."""


class CheckpointStateError(CheckpointError):
    """State dict keys/shapes do not match the target model."""


class CheckpointNonFiniteError(CheckpointError):
    """A state tensor contains NaN/inf (or the wrong dtype)."""


class CheckpointMigrationError(CheckpointError):
    """The RGB->RGB-D migration is not exactly the expected new-module delta."""


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase hex SHA256 of a file, streamed (no full read)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_identity(
    path: str,
    *,
    expected_sha256: Optional[str] = None,
    expected_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify a file's size and/or SHA256 against a pinned identity.

    Returns ``{"path", "bytes", "sha256"}`` on success; raises
    :class:`CheckpointIdentityError` on any mismatch (including a missing file
    or a truncated file whose byte count differs).
    """
    if not os.path.isfile(path):
        raise CheckpointIdentityError(f"checkpoint not found: {path}")
    size = os.path.getsize(path)
    if expected_bytes is not None and size != int(expected_bytes):
        raise CheckpointIdentityError(
            f"byte-size mismatch for {path}: expected {int(expected_bytes)}, got {size} "
            f"(truncated or wrong file)"
        )
    digest = sha256_file(path)
    if expected_sha256 is not None and digest.lower() != str(expected_sha256).lower():
        raise CheckpointIdentityError(
            f"SHA256 mismatch for {path}: expected {expected_sha256}, got {digest}"
        )
    return {"path": path, "bytes": size, "sha256": digest}


# --------------------------------------------------------------------------- #
# safe loading
# --------------------------------------------------------------------------- #
def load_safetensors_state(path: str) -> Dict[str, torch.Tensor]:
    """Load a ``.safetensors`` file into a ``{name: tensor}`` dict."""
    try:
        from safetensors.torch import load_file  # local import: optional dep

        return dict(load_file(path))
    except Exception as exc:  # noqa: BLE001 - normalize every loader failure
        raise CheckpointFormatError(f"failed to load safetensors {path}: {exc}") from exc


def _looks_like_state_dict(obj: Mapping) -> bool:
    if not obj:
        return False
    return all(isinstance(v, torch.Tensor) for v in obj.values())


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """Extract a ``{name: tensor}`` state dict from a loaded checkpoint object.

    Handles the released safetensors mapping, a training checkpoint with a
    ``"state_dict"`` entry, and the ModelHub ``"model"`` entry.  A bare
    tensor-only mapping is returned as-is.  Anything else is rejected.
    """
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            inner = obj.get(key)
            if isinstance(inner, Mapping) and _looks_like_state_dict(inner):
                return {str(k): v for k, v in inner.items()}
        if _looks_like_state_dict(obj):
            return {str(k): v for k, v in obj.items()}
    raise CheckpointFormatError(
        "checkpoint does not contain a recognisable tensor state dict "
        "(expected a safetensors mapping, or a dict with 'state_dict'/'model')"
    )


def load_checkpoint(path: str, *, weights_only: bool = True, map_location: str = "cpu") -> Any:
    """Safely load a checkpoint file.

    ``.safetensors`` is read with the safetensors loader; ``.pt``/``.pth``/``.bin``
    with ``torch.load(..., weights_only=True)``.  Loading is wrapped so a
    truncated or otherwise unreadable file becomes a
    :class:`CheckpointFormatError` rather than an opaque traceback.
    """
    if not os.path.isfile(path):
        raise CheckpointFormatError(f"checkpoint not found: {path}")
    if path.endswith(".safetensors"):
        return load_safetensors_state(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointFormatError(f"failed to load checkpoint {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# state validation
# --------------------------------------------------------------------------- #
def state_dict_diff(
    model_state: Mapping[str, torch.Tensor],
    loaded_state: Mapping[str, torch.Tensor],
) -> Tuple[List[str], List[str], List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]]:
    """Return ``(missing, unexpected, shape_mismatch)`` between two states.

    ``missing``   : keys required by the model but absent from ``loaded_state``.
    ``unexpected``: keys present in ``loaded_state`` but unknown to the model.
    ``shape_mismatch``: ``(key, model_shape, loaded_shape)`` for differing shapes.
    """
    model_keys = set(model_state)
    loaded_keys = set(loaded_state)
    missing = sorted(model_keys - loaded_keys)
    unexpected = sorted(loaded_keys - model_keys)
    mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for key in sorted(model_keys & loaded_keys):
        ms = tuple(model_state[key].shape)
        ls = tuple(loaded_state[key].shape)
        if ms != ls:
            mismatch.append((key, ms, ls))
    return missing, unexpected, mismatch


def _storage_key(t: torch.Tensor) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...], Any, Any]:
    """A hashable identity for a tensor's storage + view geometry.

    Two names denote the *same* weight only when they share storage **and**
    agree on view geometry, device and dtype; a reinterpreted view (``.view``
    to another dtype) or a copy on another device is not the same weight.
    """
    return (
        t.data_ptr(),
        t.storage_offset(),
        tuple(t.shape),
        tuple(t.stride()),
        t.device,
        t.dtype,
    )


def alias_groups(model_state: Mapping[str, torch.Tensor]) -> Dict[Any, List[str]]:
    """Group model state keys that share one storage (aliased parameters)."""
    groups: Dict[Any, List[str]] = defaultdict(list)
    for key, value in model_state.items():
        groups[_storage_key(value)].append(key)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def classify_missing(
    model: torch.nn.Module,
    loaded_state: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Split the model's missing keys into benign aliases and genuine misses.

    A missing key is *benign* when it shares storage with a key that **is**
    present in ``loaded_state`` (the released checkpoint deduplicates shared
    parameters).  Any other missing key is *genuine* and must be treated as an
    incomplete checkpoint.

    Returns ``(benign, genuine)`` where ``benign`` maps ``missing_key ->
    [canonical_keys_present_in_checkpoint]`` and ``genuine`` is a sorted list.
    """
    model_state = model.state_dict()
    loaded_keys = set(loaded_state)
    missing, _, _ = state_dict_diff(model_state, loaded_state)
    groups = alias_groups(model_state)
    benign: Dict[str, List[str]] = {}
    genuine: List[str] = []
    for key in missing:
        alias = [k for k in groups.get(_storage_key(model_state[key]), []) if k in loaded_keys]
        if alias:
            benign[key] = alias
        else:
            genuine.append(key)
    return benign, sorted(genuine)


def check_state_finite(
    state: Mapping[str, torch.Tensor],
    *,
    require_fp32: bool = True,
    name: str = "state_dict",
) -> None:
    """Fail fast if any tensor is non-finite or (optionally) not float32.

    Integer/bool buffers are allowed when ``require_fp32`` is set (only
    floating-point tensors are dtype-checked), but *every* tensor is checked for
    finiteness.
    """
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise CheckpointStateError(f"{name}[{key!r}] is not a tensor")
        if require_fp32 and value.is_floating_point() and value.dtype != torch.float32:
            raise CheckpointNonFiniteError(
                f"{name}[{key!r}] must be float32, got {value.dtype}"
            )
        if not bool(torch.isfinite(value).all()):
            raise CheckpointNonFiniteError(f"{name}[{key!r}] contains NaN/inf")


def _check_state_dtype(
    model_state: Mapping[str, torch.Tensor],
    loaded_state: Mapping[str, torch.Tensor],
) -> None:
    """Reject a loaded tensor whose dtype differs from the model tensor's.

    Every key here is already known to be present with a matching shape.  A
    loaded tensor must have exactly the *same* dtype as the destination
    parameter/buffer: silently promoting ``int``/``bool``/``float64`` into a
    float32 weight, or casting between integer buffer widths, would hide a
    corrupt or wrong checkpoint.  Genuine integer/bool buffers with a matching
    dtype are preserved.  This contract is independent of ``require_fp32`` (an
    fp16 model with matching fp16 state is allowed; fp16 state for an fp32
    model is not).
    """
    for key in sorted(set(model_state) & set(loaded_state)):
        model_dtype = model_state[key].dtype
        loaded_dtype = loaded_state[key].dtype
        if loaded_dtype != model_dtype:
            raise CheckpointStateError(
                f"dtype mismatch for {key!r}: model expects {model_dtype}, "
                f"loaded state has {loaded_dtype}"
            )


def _check_alias_value_consistency(
    model_state: Mapping[str, torch.Tensor],
    loaded_state: Mapping[str, torch.Tensor],
) -> None:
    """Reject contradictory values supplied for one aliased model storage.

    Storage aliasing licenses the *same weight under different names*, never
    *different values under those names*.  When a loaded state provides several
    names of one storage group (e.g. both the canonical and the alias key) their
    loaded values must be equal; otherwise the sequential ``copy_`` would
    silently let the last one win.  Reject before any mutation.
    """
    for keys in alias_groups(model_state).values():
        present = [k for k in keys if k in loaded_state]
        if len(present) < 2:
            continue
        reference = loaded_state[present[0]]
        for other in present[1:]:
            if not torch.equal(reference, loaded_state[other]):
                raise CheckpointStateError(
                    "contradictory values for aliased state keys "
                    f"{present}: {present[0]!r} and {other!r} share one model "
                    "storage but carry different loaded values"
                )


def strict_load_state_dict(
    model: torch.nn.Module,
    loaded_state: Mapping[str, torch.Tensor],
    *,
    require_fp32: bool = True,
    check_finite: bool = True,
    allow_aliases: bool = False,
) -> Dict[str, Any]:
    """Load a state dict requiring exact key/shape agreement.

    Unlike the ModelHub mixin, any *genuine* missing key, unexpected key or shape
    mismatch raises :class:`CheckpointStateError`; non-finite (or non-fp32)
    tensors raise :class:`CheckpointNonFiniteError`.  ``copy_`` is used so a
    wrong-shaped or extra tensor can never be silently broadcast.

    With ``allow_aliases=True`` a missing key is tolerated only when it is a
    storage alias of a key that *is* present (see :func:`classify_missing`);
    otherwise it is rejected.  In addition, every loaded tensor must match the
    destination model tensor's **dtype**, and several names of one aliased
    storage must carry **equal** loaded values; both are checked before any
    ``copy_`` so a rejected load never mutates the model.  Returns a small
    report.
    """
    model_state = model.state_dict()
    missing, unexpected, mismatch = state_dict_diff(model_state, loaded_state)
    benign: Dict[str, List[str]] = {}
    genuine = missing
    if allow_aliases:
        benign, genuine = classify_missing(model, loaded_state)
    if unexpected or mismatch or genuine:
        raise CheckpointStateError(
            f"state dict does not match model: missing={genuine} "
            f"unexpected={unexpected} shape_mismatch={mismatch} "
            f"(benign_aliases={sorted(benign)})"
        )
    _check_state_dtype(model_state, loaded_state)
    _check_alias_value_consistency(model_state, loaded_state)
    if check_finite:
        check_state_finite(loaded_state, require_fp32=require_fp32)
    with torch.no_grad():
        for key, value in loaded_state.items():
            model_state[key].copy_(value)
    return {
        "loaded": len(loaded_state),
        "benign_aliases": benign,
        "unexpected": unexpected,
        "shape_mismatch": mismatch,
    }


def load_released_state(
    model: torch.nn.Module,
    loaded_state: Mapping[str, torch.Tensor],
    *,
    require_fp32: bool = True,
) -> Dict[str, Any]:
    """Alias-aware strict load of a released (deduplicated) checkpoint.

    Enforces: zero unexpected keys, zero shape mismatches, and every missing key
    must be a storage alias of a loaded key.  This is the safe replacement for
    the ModelHub mixin's silent ``strict=False`` behaviour.
    """
    return strict_load_state_dict(
        model, loaded_state, require_fp32=require_fp32, check_finite=True, allow_aliases=True
    )


# --------------------------------------------------------------------------- #
# RGB -> RGB-D migration
# --------------------------------------------------------------------------- #
def expected_new_keys(model: torch.nn.Module, module_name: str) -> List[str]:
    """Sorted names of the parameters/buffers under ``module_name`` in ``model``."""
    keys = [n for n, _ in model.named_parameters()]
    keys += [n for n, _ in model.named_buffers()]
    prefix = module_name.rstrip(".")
    return sorted(k for k in keys if k == prefix or k.startswith(prefix + "."))


def _new_module_gate(model: torch.nn.Module, module_name: str) -> Optional[torch.Tensor]:
    """Return the ``gate`` parameter of the named module, if it has one."""
    module: Any = model
    for part in module_name.rstrip(".").split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    gate = getattr(module, "gate", None)
    if isinstance(gate, torch.Tensor):
        return gate.detach()
    return None


def migrate_rgb_to_rgbd(
    model: torch.nn.Module,
    rgb_state: Mapping[str, torch.Tensor],
    *,
    new_module_prefix: str = "depth_fusion",
    require_fp32: bool = True,
) -> Dict[str, Any]:
    """Migrate a released RGB state onto an RGB-D ``model`` in place.

    Contract enforced here (workdoc procedure 16):

    * after alias resolution, the only keys missing from ``rgb_state`` are
      *exactly* the parameters and buffers of the new ``new_module_prefix``
      module (the five ``depth_fusion.*`` keys);
    * there are **no** unexpected keys and **no** shape mismatches;
    * every loaded (legacy) tensor is finite and (optionally) float32;
    * the new module's gate is exactly zero-initialised before and after the
      load (so the fused model is numerically identical to RGB at init).

    ``rgb_state`` may be either the raw deduplicated released state or the
    materialised RGB model state; benign storage aliases are resolved in both
    cases.  Returns a report dict.  Raises :class:`CheckpointMigrationError` on
    violation.
    """
    model_state = model.state_dict()
    _, unexpected, mismatch = state_dict_diff(model_state, rgb_state)
    if unexpected or mismatch:
        raise CheckpointMigrationError(
            f"RGB->RGB-D migration has unexpected={unexpected} shape_mismatch={mismatch}; "
            f"only the new '{new_module_prefix}' module may differ"
        )
    benign, genuine_missing = classify_missing(model, rgb_state)
    expected_missing = expected_new_keys(model, new_module_prefix)
    if sorted(genuine_missing) != sorted(expected_missing):
        raise CheckpointMigrationError(
            f"genuine missing keys are not exactly the new module: "
            f"missing={genuine_missing} expected={expected_missing}"
        )

    _check_state_dtype(model_state, rgb_state)
    _check_alias_value_consistency(model_state, rgb_state)
    check_state_finite(rgb_state, require_fp32=require_fp32)

    gate_before = _new_module_gate(model, new_module_prefix)

    with torch.no_grad():
        for key, value in rgb_state.items():
            model_state[key].copy_(value)

    # legacy values must be preserved exactly (compare against the loaded state)
    for key, value in rgb_state.items():
        if not torch.equal(model_state[key], value):
            raise CheckpointMigrationError(f"legacy tensor {key!r} not preserved by migration")

    gate_after = _new_module_gate(model, new_module_prefix)
    if gate_after is not None:
        if gate_before is not None and not torch.equal(
            gate_before.to(torch.float32), gate_after.to(torch.float32)
        ):
            raise CheckpointMigrationError("new module gate changed during migration")
        if not bool(torch.all(gate_after == 0)):
            raise CheckpointMigrationError(
                f"new module gate must stay zero-initialised, got {gate_after}"
            )

    return {
        "genuine_missing": genuine_missing,
        "benign_aliases": benign,
        "unexpected": unexpected,
        "shape_mismatch": mismatch,
        "new_module_keys": expected_missing,
        "gate_value": None if gate_after is None else float(gate_after.reshape(-1)[0]),
        "legacy_keys_loaded": len(rgb_state),
    }

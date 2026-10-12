"""Strict, atomic training checkpoints + resume identity for the RGB-D trainer (17C).

This module owns everything that must survive a stop/resume:

* a **semantic resume identity** (public source SHA list, released-weight SHA,
  dataset-index SHA, resume-config SHA, UniCeption pin) that is compared
  *before* any model / optimizer / sampler state is copied;
* **RNG capture** for the CPU / CUDA / Python / numpy generators in a form that
  ``torch.load(..., weights_only=True)`` can read safely (numpy arrays are
  stored as primitive lists, never pickled objects);
* the **stateful permutation + epoch sampler** whose consumed cursor advances
  *only* when the trainer commits a completed optimizer update;
* **atomic last-good writes** (``tmp`` + ``os.replace``) and a distinction
  between the training ``y`` checkpoint (resume-only) and the best averaged ``x``
  checkpoint (evaluation-only, resuming from it is rejected).

Policy on output locations
--------------------------
Two resume attempts of the *same experiment* may legitimately use different
Git-external output roots / resolved-config paths.  The resume identity therefore
hashes only the **semantic** config (model / dataset / loss / optimizer /
experiment / precision / freeze / seed) with the output-location keys removed;
the config *file* SHA is recorded separately but is not part of the equality test.

Nothing here embeds private data, host paths, or dataset identifiers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "CheckpointContractError",
    "ResumeIdentityError",
    "ResumeKindError",
    "capture_rng_state",
    "restore_rng_state",
    "semantic_config_sha",
    "build_identity",
    "assert_resume_identity",
    "StatefulPermutationSampler",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "trainable_mask",
]

_FORMAT = "ufm_rgbd_training_checkpoint"
_FORMAT_VERSION = 1

#: config keys whose *location* may change between resume attempts of one run.
_OUTPUT_LOCATION_KEYS: Tuple[Tuple[str, ...], ...] = (
    ("resolved_output_dir",),
    ("external_config",),
    ("checkpoint", "dir"),
)


class CheckpointContractError(ValueError):
    """A training checkpoint is missing, malformed or violates the contract."""


class ResumeIdentityError(CheckpointContractError):
    """The saved checkpoint belongs to a different experiment identity."""


class ResumeKindError(CheckpointContractError):
    """A resume was attempted from a non-resumable (best-x / eval-only) checkpoint."""


# --------------------------------------------------------------------------- #
# RNG
# --------------------------------------------------------------------------- #
def _numpy_state_to_serializable() -> Dict[str, Any]:
    st = np.random.get_state()  # legacy RandomState tuple
    return {
        "kind": str(st[0]),
        "keys": [int(x) for x in st[1]],
        "pos": int(st[2]),
        "has_gauss": int(st[3]),
        "cached_gaussian": float(st[4]),
    }


def _numpy_state_from_serializable(d: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(d["kind"]),
        np.array([int(x) for x in d["keys"]], dtype=np.uint32),
        int(d["pos"]),
        int(d["has_gauss"]),
        float(d["cached_gaussian"]),
    )


def capture_rng_state() -> Dict[str, Any]:
    """Capture CPU/CUDA/Python/numpy RNG state in a ``weights_only``-safe form."""
    state: Dict[str, Any] = {
        "cpu": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": _numpy_state_to_serializable(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = [torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    if "cpu" in state and state["cpu"] is not None:
        torch.set_rng_state(state["cpu"])
    if "python" in state and state["python"] is not None:
        random.setstate(tuple(state["python"]) if isinstance(state["python"], list) else state["python"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(_numpy_state_from_serializable(state["numpy"]))
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        for d, s in enumerate(cuda):
            if d < torch.cuda.device_count():
                torch.cuda.set_rng_state(s, d)


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def _without_output_locations(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(cfg))
    out.pop("hydra", None)
    out.pop("defaults", None)
    for path in _OUTPUT_LOCATION_KEYS:
        node: Any = out
        for part in path[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return out


def semantic_config_sha(cfg: Mapping[str, Any]) -> str:
    """SHA256 of the semantic config (output-location keys removed)."""
    payload = json.dumps(_without_output_locations(cfg), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build_identity(
    *,
    source_sha_list: Sequence[Sequence[str]],
    weight_sha: str,
    data_sha: str,
    resume_config_sha: str,
    uni_ception_pin: Optional[str] = None,
    public_modules: Sequence[str] = (),
    dino_python_tree_sha256: Optional[str] = None,
    asset_config_sha: Optional[str] = None,
    asset_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the semantic resume identity.

    ``source_sha_list`` is a deterministic list of ``(relative_path, sha256)``
    pairs for the public modules that define the run.  ``uni_ception_pin`` is the
    real UniCeption git commit; ``dino_python_tree_sha256`` is kept as a
    *separate* field (it is the local DINO hub tree aggregate, never reused as the
    UniCeption pin).  ``asset_config_sha`` / ``asset_sha`` record the actually
    used released-weight config and weights.
    """
    payload: Dict[str, Any] = {
        "source_sha_list": sorted([list(map(str, p)) for p in source_sha_list]),
        "weight_sha": str(weight_sha),
        "data_sha": str(data_sha),
        "resume_config_sha": str(resume_config_sha),
        "uni_ception_pin": None if uni_ception_pin is None else str(uni_ception_pin),
        "dino_python_tree_sha256": (
            None if dino_python_tree_sha256 is None else str(dino_python_tree_sha256)
        ),
        "asset_config_sha": None if asset_config_sha is None else str(asset_config_sha),
        "asset_sha": None if asset_sha is None else str(asset_sha),
        "public_modules": sorted(map(str, public_modules)),
    }
    payload["sha"] = _sha256_json(payload)
    return payload


def assert_resume_identity(expected: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    """Raise :class:`ResumeIdentityError` if the two identities disagree.

    Called *before* any state copy, so a mismatch can never partially mutate the
    live model / optimizer.
    """
    if not isinstance(saved, Mapping):
        raise ResumeIdentityError("checkpoint has no resume identity")
    if saved.get("sha") != expected.get("sha"):
        differing = sorted(
            k for k in set(expected) | set(saved) if expected.get(k) != saved.get(k) and k != "sha"
        )
        raise ResumeIdentityError(
            "resume identity mismatch (source / data / config / weight / UniCeption pin); "
            f"differing fields: {differing}; refusing to copy any state"
        )


# --------------------------------------------------------------------------- #
# stateful permutation / epoch sampler
# --------------------------------------------------------------------------- #
class StatefulPermutationSampler:
    """A permutation sampler whose consumed cursor advances only on ``commit``.

    ``commit`` must be called by the trainer **after** a completed optimizer
    update (accumulation boundary).  Iterator creation uses a dedicated
    ``torch.Generator`` so it never consumes the global RNG.
    """

    def __init__(self, n: int, *, seed: int = 0, generator: Optional[torch.Generator] = None):
        if int(n) <= 0:
            raise CheckpointContractError(f"sampler size must be > 0, got {n}")
        self.n = int(n)
        self.seed = int(seed)
        self.generator = generator if generator is not None else torch.Generator().manual_seed(self.seed)
        self.epoch = 0
        self.perm: List[int] = self._new_perm()
        self.cursor = 0

    def _new_perm(self) -> List[int]:
        return torch.randperm(self.n, generator=self.generator).tolist()

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for i in range(self.cursor, self.n):
            yield self.perm[i]

    def remaining(self) -> int:
        return self.n - self.cursor

    def advance_epoch(self) -> None:
        self.epoch += 1
        self.perm = self._new_perm()
        self.cursor = 0

    def commit(self) -> None:
        """Advance the consumed cursor by exactly one (post-update)."""
        self.cursor += 1
        if self.cursor >= self.n:
            self.advance_epoch()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "seed": self.seed,
            "epoch": self.epoch,
            "perm": list(self.perm),
            "cursor": self.cursor,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("n", -1)) != self.n:
            raise CheckpointContractError(
                f"sampler size mismatch: checkpoint {state.get('n')} vs current {self.n}"
            )
        self.epoch = int(state["epoch"])
        self.perm = [int(x) for x in state["perm"]]
        self.cursor = int(state["cursor"])
        self.generator.set_state(state["generator_state"])


# --------------------------------------------------------------------------- #
# trainable mask
# --------------------------------------------------------------------------- #
def trainable_mask(model: torch.nn.Module) -> Dict[str, bool]:
    return {n: bool(p.requires_grad) for n, p in model.named_parameters()}


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def save_training_checkpoint(
    path: Any,
    *,
    kind: str,
    model: torch.nn.Module,
    optimizer: Any,
    identity: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    resume_config_sha: str,
    global_step: int,
    epoch: int,
    sampler_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
    best_monitor: Mapping[str, Any],
    trainable: Mapping[str, bool],
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Atomically write a training checkpoint (last-good tmp + ``os.replace``)."""
    if kind not in ("last_y", "best_averaged_x"):
        raise CheckpointContractError(f"unknown checkpoint kind {kind!r}")
    payload = {
        "format": _FORMAT,
        "format_version": _FORMAT_VERSION,
        "kind": kind,
        "identity": copy.deepcopy(dict(identity)),
        "resolved_config": copy.deepcopy(dict(resolved_config)),
        "resume_config_sha": str(resume_config_sha),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "sampler": copy.deepcopy(dict(sampler_state)),
        "rng": copy.deepcopy(dict(rng_state)),
        "best_monitor": copy.deepcopy(dict(best_monitor)),
        "trainable_mask": {str(k): bool(v) for k, v in trainable.items()},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "extra": copy.deepcopy(dict(extra or {})),
    }
    target = Path(str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return str(target)


def load_training_checkpoint(
    path: Any,
    *,
    expected_identity: Optional[Mapping[str, Any]] = None,
    require_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Load + validate a training checkpoint with ``weights_only=True``.

    If ``require_kind`` is ``"last_y"`` (resume), a best-x / eval-only checkpoint
    is rejected before any state is returned.
    """
    target = Path(str(path))
    if not target.is_file():
        raise CheckpointContractError(f"checkpoint not found: {target}")
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointContractError(f"failed to load checkpoint {target}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("format") != _FORMAT:
        raise CheckpointContractError("not a UFM RGB-D training checkpoint")
    kind = payload.get("kind")
    if require_kind is not None and kind != require_kind:
        raise ResumeKindError(
            f"resume requires a {require_kind!r} checkpoint but the file is {kind!r}; "
            "the best averaged x is evaluation-only and must never be resumed from"
        )
    if expected_identity is not None:
        assert_resume_identity(expected_identity, payload.get("identity", {}))
    return dict(payload)

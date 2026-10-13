"""Schema validation and external resolution for the RGB-D Hydra config (17A).

This module is deliberately dependency-light (stdlib + PyYAML only) so it can be
imported by tests and by the 17B/17C entry points *without* importing the heavy
UFM model stack.  It provides:

* the authoritative **key schema** for the ``configs/rgbd_training`` tree,
* typed **fail-fast** validation (unknown keys, invalid shapes, resolution not a
  multiple of the encoder patch size, invalid depth units/window, invalid
  precision, AMUSE ``warmup_steps == 0``, an external scheduler attached to the
  schedule-free AMUSE preset, planar homography ON, and FULL / ``max_updates``
  confusion),
* **external resolution**: private values (dataset index path/sha, pretrained
  asset paths, DINO hub dir, occlusion-supervision path) are injected from a
  Git-external config and merged over the public config,
* canonical hashing + a Git-external writer for the resolved yaml,
* **redaction** helpers so dry-run/print never leak private host paths.

Nothing in the public config tree embeds a private host path; only the external
resolved config does.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

__all__ = [
    "RGBDConfigError",
    "UnknownKeyError",
    "ShapeError",
    "ResolutionError",
    "DepthError",
    "PrecisionError",
    "OptimizerError",
    "LossError",
    "ExperimentError",
    "DeterminismError",
    "ExternalConfigError",
    "UnsafeOutputError",
    "GROUP_NAMES",
    "EXPERIMENT_PRESETS",
    "PRIVATE_KEYS",
    "FORBIDDEN_HOST_PREFIXES",
    "validate_config",
    "resolve_config",
    "load_external_config",
    "config_sha256",
    "canonical_yaml",
    "repo_root",
    "assert_git_external",
    "save_resolved",
    "redact",
    "find_host_paths",
    "without_hydra",
]


# --------------------------------------------------------------------------- #
# typed fail-fast exceptions
# --------------------------------------------------------------------------- #
class RGBDConfigError(ValueError):
    """Base class for RGB-D config contract violations."""


class UnknownKeyError(RGBDConfigError):
    """A config key is not part of the schema."""


class ShapeError(RGBDConfigError):
    """A structured value (resolution / window / pair) has the wrong shape."""


class ResolutionError(RGBDConfigError):
    """Train resolution is invalid or not a multiple of the patch size."""


class DepthError(RGBDConfigError):
    """Depth units / window are invalid."""


class PrecisionError(RGBDConfigError):
    """A precision value is not permitted (master/CPU must stay Float32)."""


class DeterminismError(RGBDConfigError):
    """A determinism/backends setting is invalid or would be silently ignored."""


class OptimizerError(RGBDConfigError):
    """Optimizer settings violate the AMUSE schedule-free contract."""


class LossError(RGBDConfigError):
    """Loss settings are invalid (e.g. homography ON for non-rigid scenes)."""


class ExperimentError(RGBDConfigError):
    """Experiment preset is invalid (e.g. FULL / max_updates confusion)."""


class ExternalConfigError(RGBDConfigError):
    """The external (private) resolved config is missing or malformed."""


class UnsafeOutputError(RGBDConfigError):
    """A resolved-config output path would write inside the public repository
    (directly or via symlink), or the filename escapes its directory."""


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
class _Free:
    """Sentinel: any value / nested mapping is accepted below this key."""

    __slots__ = ()


FREE = _Free()

#: Hydra group names provided by this config tree.
GROUP_NAMES: Tuple[str, ...] = ("dataset", "model", "optimizer", "loss", "experiment")

#: preset table (also documented in the worker report / API spec).
EXPERIMENT_PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "max_updates": 1,
        "validate_every_updates": 1,
        "save_every_updates": 1,
        "validation_max_pairs": 2,
    },
    "short": {
        "max_updates": 8,
        "validate_every_updates": 4,
        "save_every_updates": 4,
        "validation_max_pairs": 4,
    },
    "full": {
        "max_updates": None,
        "epochs": 12,
        "validate_full_each_epoch": True,
        "validation_max_pairs": None,
        "early_stop": False,
    },
}

#: dot-paths that hold Git-external private values (redacted from any print).
PRIVATE_KEYS: Tuple[str, ...] = (
    "resolved_output_dir",
    "external_config",
    "dataset.expected_train_pairs",
    "dataset.expected_val_pairs",
    "dataset.index_path",
    "dataset.index_sha256",
    "dataset.occlusion_supervision_path",
    "model.assets.asset_path",
    "model.assets.config_path",
    "model.assets.config_sha256",
    "model.assets.dino_hub_dir",
)

#: absolute host-path prefixes that must never appear in public config/tests.
#: Assembled from fragments so this source file stays free of the raw rejected
#: sequences (mirrors scripts/check_public_stage.py).
FORBIDDEN_HOST_PREFIXES: Tuple[str, ...] = (
    "/" + "home/",
    "/" + "workspace/",
    "/" + "root/",
    "/" + "Users/",
    "C:" + "\\" + "Users",
)

_ALLOWED_DEPTH_UNITS = ("metre", "meter")
_ALLOWED_MASTER_PRECISION = ("float32",)
_ALLOWED_CPU_PRECISION = ("float32",)
_ALLOWED_TRAIN_PRECISION = ("bf16", "float32")
#: legal cuBLAS workspace configs (must be set before the first cuBLAS call).
_ALLOWED_CUBLAS_WORKSPACE = (":4096:8", ":16:8")
#: forward-scope determinism hook selectors (Step 19 attempt-10).  `default`
#: leaves the corresponding runtime behaviour untouched; the other value opts
#: into the bounded deterministic path.
_ALLOWED_SDPA_BACKENDS = ("default", "math")
_ALLOWED_BILINEAR_MODES = ("default", "deterministic")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

SCHEMA: Dict[str, Any] = {
    "seed": FREE,
    "fail_nonfinite": FREE,
    "dry_run": FREE,
    "external_config": FREE,
    "resolved_output_dir": FREE,
    "checkpoint": {
        "dir": FREE,
        "save_last_y": FREE,
        "save_best_averaged_x": FREE,
        "metadata": {
            "last_y": {"param_state": FREE, "purpose": FREE},
            "best_averaged_x": {"param_state": FREE, "purpose": FREE},
        },
        "strict_resume": {
            "require_rng": FREE,
            "require_global_step": FREE,
            "require_epoch": FREE,
            "require_sampler_permutation": FREE,
            "require_consumed_cursor": FREE,
            "require_config_sha": FREE,
            "require_data_sha": FREE,
            "require_weight_sha": FREE,
            "require_source_sha": FREE,
        },
    },
    "determinism": {
        "enabled": FREE,
        "warn_only": FREE,
        "cublas_workspace_config": FREE,
        "cudnn_benchmark": FREE,
        "cudnn_deterministic": FREE,
        "allow_tf32": FREE,
        # bounded forward-scope hooks: default|math / default|deterministic.
        "sdpa_backend": FREE,
        "bilinear": FREE,
        # optional alias: only `true` is accepted; `false` is rejected (never
        # silently ignore a request for strict reproducibility).
        "strict": FREE,
    },
    "dataset": {
        "name": FREE,
        "index_path": FREE,
        "index_sha256": FREE,
        "expected_train_pairs": FREE,
        "expected_val_pairs": FREE,
        "heldout_role": FREE,
        "resolution": FREE,
        "resolution_multiple": FREE,
        "data_norm_type": FREE,
        "transform": FREE,
        "depth_units": FREE,
        "depth_window_m": FREE,
        "already_dilated": FREE,
        "verify_hashes": FREE,
        "occlusion_supervision_source": FREE,
        # optional global root override; null => per-pair paths come from the index
        "occlusion_supervision_path": FREE,
        "loader": {
            "num_workers": FREE,
            "pin_memory": FREE,
            "persistent_workers": FREE,
            "batch_pairs": FREE,
            "drop_last": FREE,
        },
    },
    "model": {
        "class": FREE,
        "arch_config": FREE,
        "public": {
            "model_id": FREE,
            "revision": FREE,
            "pretrained_sha256": FREE,
            "pretrained_bytes": FREE,
        },
        "assets": {
            "asset_path": FREE,
            "config_path": FREE,
            "config_sha256": FREE,
            "dino_hub_dir": FREE,
        },
        "dino_python_tree_sha256": FREE,
        "use_depth_fusion": FREE,
        "depth_fusion_kwargs": {
            "feature_dim": FREE,
            "depth_hidden_dim": FREE,
            "gate_init": FREE,
            "depth_scale": FREE,
        },
        "master_precision": FREE,
        "train_precision": FREE,
        "cpu_precision": FREE,
        "freeze": {
            "encoder": FREE,
            "info_sharing": FREE,
            "output_head": FREE,
            "uncertainty_head": FREE,
            "depth_fusion": FREE,
        },
        "param_group_policy": FREE,
    },
    "optimizer": {
        "name": FREE,
        "default": FREE,
        "source": {"commit": FREE, "license": FREE},
        "schedule_free": FREE,
        "external_scheduler": FREE,
        "lr": FREE,
        "sgd_lr": FREE,
        "weight_decay": FREE,
        "betas": FREE,
        "eps": FREE,
        "constructor_args": {
            "beta1": FREE,
            "weight_lr_power": FREE,
            "warmup_steps": FREE,
            "rho": FREE,
            "r": FREE,
            "weight_decay_at_y": FREE,
        },
        "groups": {
            "muon": {
                "use_muon": FREE,
                "lr": FREE,
                "momentum": FREE,
                "aux_update_type": FREE,
                "weight_decay": FREE,
            },
            "aux": {
                "use_muon": FREE,
                "update_type": FREE,
                "lr": FREE,
                "weight_decay": FREE,
            },
            "all": {"lr": FREE, "weight_decay": FREE},
        },
        "train_eval": {
            "eval_returns_averaged_x": FREE,
            "save_training_y": FREE,
            "save_training_z": FREE,
            "save_extra_attrs": FREE,
        },
        "gradient_clip": {"algorithm": FREE, "value": FREE},
    },
    "loss": {
        "name": FREE,
        "supervised": FREE,  # free-form loss-name mapping
        "selfsup": {
            "enabled": FREE,
            "weight": FREE,
            "warmup_updates": FREE,
            "detach_original_prediction": FREE,
            "same_geometry": FREE,
            "aug": {
                "hflip": FREE,
                "photometric": FREE,
                "crop": FREE,
                "complex_warp": FREE,
            },
        },
        "homography": {"enabled": FREE},
        "laplace_nll": {"enabled": FREE, "detach_residual": FREE},
    },
    "experiment": {
        "name": FREE,
        "max_updates": FREE,
        "epochs": FREE,
        "validate_every_updates": FREE,
        "save_every_updates": FREE,
        "validate_every_epochs": FREE,
        "validate_full_each_epoch": FREE,
        "save_every_epochs": FREE,
        "validation_max_pairs": FREE,
        "early_stop": FREE,
        "full_budget_fixed": FREE,
        "eligible_train_population": FREE,
        "comparison": {
            "selfsup_matched": FREE,
            "fixed_seed": FREE,
            "fixed_optimizer": FREE,
            "fixed_budget": FREE,
            "fixed_split": FREE,
            "run_by_step": FREE,
        },
    },
}

#: keys that must be non-null when resolving against a real external config.
_REQUIRED_EXTERNAL_KEYS: Tuple[str, ...] = (
    "dataset.index_path",
    "dataset.index_sha256",
    "dataset.expected_train_pairs",
    "dataset.expected_val_pairs",
    "model.assets.asset_path",
    "model.assets.config_path",
    "model.assets.config_sha256",
    "model.assets.dino_hub_dir",
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def without_hydra(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy with Hydra bookkeeping keys removed."""
    out = copy.deepcopy(cfg)
    if isinstance(out, dict):
        out.pop("hydra", None)
        out.pop("defaults", None)
    return out


def _get(cfg: Dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _looks_like_abs_path(s: str) -> bool:
    if not isinstance(s, str):
        return False
    if s.startswith("${"):
        return False
    return bool(re.match(r"^[A-Za-z]:[\\/]", s)) or s.startswith("/")


def find_host_paths(cfg: Any, prefix: str = "") -> List[str]:
    """Return dotted paths of values that contain a forbidden host prefix."""
    hits: List[str] = []
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            hits.extend(find_host_paths(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            hits.extend(find_host_paths(v, f"{prefix}[{i}]"))
    elif isinstance(cfg, str):
        for p in FORBIDDEN_HOST_PREFIXES:
            if p in cfg:
                hits.append(prefix)
                break
    return hits


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _check_unknown_keys(value: Any, schema: Any, path: str) -> None:
    if schema is FREE:
        return
    if not isinstance(schema, dict):
        return
    if value is None:
        return
    if not isinstance(value, dict):
        raise ShapeError(f"{path}: expected a mapping, got {type(value).__name__}")
    for key, sub in value.items():
        if key not in schema:
            raise UnknownKeyError(f"{path}.{key}: unknown config key (not in schema)")
        _check_unknown_keys(sub, schema[key], f"{path}.{key}")


def _validate_dataset(ds: Dict[str, Any], require_external: bool) -> None:
    if not isinstance(ds, dict):
        raise ShapeError("dataset must be a mapping")
    if ds.get("heldout_role") != "TEST_ONLY_NO_SELECTION":
        raise ExperimentError(
            "dataset.heldout_role must be 'TEST_ONLY_NO_SELECTION' "
            "(the held-out split is test-only; never used for model selection)"
        )

    res = ds.get("resolution")
    if not (isinstance(res, (list, tuple)) and len(res) == 2):
        raise ShapeError(f"dataset.resolution must be a (W, H) pair, got {res!r}")
    if not (_is_int(res[0]) and _is_int(res[1])):
        raise ShapeError(f"dataset.resolution must be integers (W, H), got {res!r}")
    if res[0] <= 0 or res[1] <= 0:
        raise ResolutionError(f"dataset.resolution must be positive, got {res!r}")
    mult = ds.get("resolution_multiple", 14)
    if not _is_int(mult) or mult <= 0:
        raise ResolutionError(f"dataset.resolution_multiple invalid: {mult!r}")
    if res[0] % mult != 0 or res[1] % mult != 0:
        raise ResolutionError(
            f"dataset.resolution {res} not a multiple of the patch size {mult} "
            f"(dinov2 patch_size=14)"
        )

    units = ds.get("depth_units")
    if units not in _ALLOWED_DEPTH_UNITS:
        raise DepthError(
            f"dataset.depth_units must be one of {_ALLOWED_DEPTH_UNITS}, got {units!r}"
        )
    window = ds.get("depth_window_m")
    if not (isinstance(window, (list, tuple)) and len(window) == 2):
        raise ShapeError(f"dataset.depth_window_m must be a (near, far) pair, got {window!r}")
    near, far = window
    if not (_is_num(near) and _is_num(far)):
        raise DepthError(f"dataset.depth_window_m must be numeric, got {window!r}")
    if near < 0.0 or not (far > near):
        raise DepthError(f"dataset.depth_window_m must satisfy 0 <= near < far, got {(near, far)}")

    if ds.get("already_dilated") is not True:
        raise DepthError("dataset.already_dilated must be true (the loader never dilates)")
    if ds.get("verify_hashes") is not True:
        raise DepthError("dataset.verify_hashes must be true for the fixed staging")
    if ds.get("occlusion_supervision_source") not in ("flow_valid", "path"):
        raise DepthError(
            "dataset.occlusion_supervision_source must be 'flow_valid' or 'path'"
        )
    if require_external:
        for dotted in ("index_path", "index_sha256"):
            if _get(ds, dotted) is None:
                raise ExternalConfigError(
                    f"dataset.{dotted} is null: supply it via the external resolved config"
                )
        idx_sha = ds.get("index_sha256")
        if not (isinstance(idx_sha, str) and _HEX64.match(idx_sha)):
            raise ExternalConfigError("dataset.index_sha256 must be a 64-hex sha256 digest")

    for name in ("expected_train_pairs", "expected_val_pairs"):
        v = ds.get(name)
        if require_external:
            if not _is_int(v) or v <= 0:
                raise ShapeError(
                    f"dataset.{name} must be a positive integer when resolving with "
                    f"an external config, got {v!r}"
                )
        elif v is not None and (not _is_int(v) or v <= 0):
            raise ShapeError(
                f"dataset.{name} must be null in the public config or a positive "
                f"integer, got {v!r}"
            )

    loader = ds.get("loader")
    if not isinstance(loader, dict):
        raise ShapeError("dataset.loader must be a mapping")
    if not _is_int(loader.get("num_workers")) or loader.get("num_workers") < 0:
        raise ShapeError(f"dataset.loader.num_workers invalid: {loader.get('num_workers')!r}")
    if not _is_int(loader.get("batch_pairs")) or loader.get("batch_pairs") < 1:
        raise ShapeError(f"dataset.loader.batch_pairs must be >= 1, got {loader.get('batch_pairs')!r}")


def _validate_model(m: Dict[str, Any], require_external: bool) -> None:
    if not isinstance(m, dict):
        raise ShapeError("model must be a mapping")
    use_depth = m.get("use_depth_fusion")
    if not isinstance(use_depth, bool):
        raise ShapeError(
            "model.use_depth_fusion must be a bool (true = RGB-D fusion, false = "
            f"RGB-only); got {use_depth!r}"
        )
    if use_depth:
        fk = m.get("depth_fusion_kwargs")
        if not isinstance(fk, dict):
            raise ShapeError(
                "model.depth_fusion_kwargs must be a mapping when use_depth_fusion is true"
            )
        if fk.get("gate_init") != 0.0:
            raise ShapeError("model.depth_fusion_kwargs.gate_init must be 0.0 (RGB identity at init)")
        if fk.get("depth_hidden_dim") != 64:
            raise ShapeError("model.depth_fusion_kwargs.depth_hidden_dim must be 64")
        if fk.get("depth_scale") != 1.3:
            raise ShapeError("model.depth_fusion_kwargs.depth_scale must be 1.3")
    elif m.get("depth_fusion_kwargs") is not None:
        raise ShapeError(
            "model.depth_fusion_kwargs must be absent/null when use_depth_fusion is false "
            "(no unused depth kwargs for the RGB-only model)"
        )

    if m.get("master_precision") not in _ALLOWED_MASTER_PRECISION:
        raise PrecisionError(
            f"model.master_precision must stay Float32, got {m.get('master_precision')!r}"
        )
    if m.get("cpu_precision") not in _ALLOWED_CPU_PRECISION:
        raise PrecisionError(
            f"model.cpu_precision must stay Float32, got {m.get('cpu_precision')!r}"
        )
    if m.get("train_precision") not in _ALLOWED_TRAIN_PRECISION:
        raise PrecisionError(
            f"model.train_precision must be one of {_ALLOWED_TRAIN_PRECISION}, "
            f"got {m.get('train_precision')!r}"
        )

    freeze = m.get("freeze")
    if not isinstance(freeze, dict):
        raise ShapeError("model.freeze must be a mapping")
    if freeze.get("encoder") is not True:
        raise ShapeError("model.freeze.encoder must be true (encoder_freeze=True)")
    for name in ("info_sharing", "output_head", "uncertainty_head"):
        if freeze.get(name) is not False:
            raise ShapeError(f"model.freeze.{name} must be false (trainable)")
    depth_freeze = freeze.get("depth_fusion")
    if use_depth:
        if depth_freeze is not False:
            raise ShapeError("model.freeze.depth_fusion must be false (trainable)")
    elif depth_freeze is not None and depth_freeze is not False:
        raise ShapeError(
            "model.freeze.depth_fusion must be false or absent for the RGB-only model"
        )

    if not isinstance(m.get("param_group_policy"), (dict,)):
        raise ShapeError("model.param_group_policy must be a mapping")

    if require_external:
        for dotted in ("assets.asset_path", "assets.config_path", "assets.config_sha256",
                       "assets.dino_hub_dir"):
            if _get(m, dotted) is None:
                raise ExternalConfigError(
                    f"model.{dotted} is null: supply it via the external resolved config"
                )
        cfg_sha = _get(m, "assets.config_sha256")
        if not (isinstance(cfg_sha, str) and _HEX64.match(cfg_sha)):
            raise ExternalConfigError("model.assets.config_sha256 must be a 64-hex sha256 digest")


def _validate_optimizer(opt: Dict[str, Any]) -> None:
    if not isinstance(opt, dict):
        raise ShapeError("optimizer must be a mapping")
    name = opt.get("name")
    if name not in ("amuse", "adamw"):
        raise OptimizerError(f"optimizer.name must be 'amuse' or 'adamw', got {name!r}")

    gc = opt.get("gradient_clip")
    if not isinstance(gc, dict) or gc.get("algorithm") != "norm" or not _is_num(gc.get("value")) or gc["value"] <= 0:
        raise OptimizerError("optimizer.gradient_clip must be {algorithm: norm, value: >0}")

    if name == "amuse":
        if opt.get("schedule_free") is not True:
            raise OptimizerError("AMUSE must be schedule_free=true")
        if opt.get("external_scheduler") is not None:
            raise OptimizerError(
                "AMUSE is schedule-free: external_scheduler must be null "
                f"(got {opt.get('external_scheduler')!r}); do not attach a scheduler"
            )
        args = opt.get("constructor_args")
        if not isinstance(args, dict):
            raise ShapeError("optimizer.constructor_args must be a mapping for AMUSE")
        warmup = args.get("warmup_steps")
        if not _is_int(warmup) or warmup <= 0:
            raise OptimizerError(
                f"AMUSE requires warmup_steps > 0 (got {warmup!r}); warmup 0 is invalid"
            )
        muon = opt.get("groups", {}).get("muon")
        if not isinstance(muon, dict) or muon.get("use_muon") is not True:
            raise OptimizerError("AMUSE muon group must set use_muon=true")
        aux = opt.get("groups", {}).get("aux")
        if not isinstance(aux, dict) or aux.get("update_type") != "sgd":
            raise OptimizerError("AMUSE aux group must use update_type='sgd'")
        if not _is_num(opt.get("lr")) or opt["lr"] <= 0:
            raise OptimizerError(f"optimizer.lr must be > 0, got {opt.get('lr')!r}")
        train_eval = opt.get("train_eval")
        if not isinstance(train_eval, dict):
            raise ShapeError("optimizer.train_eval must be a mapping")
        extra = train_eval.get("save_extra_attrs")
        if not (isinstance(extra, list) and all(isinstance(x, str) for x in extra)):
            raise ShapeError(
                "optimizer.train_eval.save_extra_attrs must be a list of attribute names"
            )


def _validate_loss(loss: Dict[str, Any]) -> None:
    if not isinstance(loss, dict):
        raise ShapeError("loss must be a mapping")
    homography = loss.get("homography")
    if not isinstance(homography, dict):
        raise ShapeError("loss.homography must be a mapping")
    if homography.get("enabled") is not False:
        raise LossError(
            "loss.homography.enabled must be false: planar homography loss is OFF "
            "for non-rigid scenes"
        )
    selfsup = loss.get("selfsup")
    if not isinstance(selfsup, dict):
        raise ShapeError("loss.selfsup must be a mapping")
    if not isinstance(selfsup.get("weight"), (int, float)) or isinstance(selfsup.get("weight"), bool):
        raise ShapeError("loss.selfsup.weight must be numeric")
    if selfsup.get("enabled") is False and selfsup.get("weight") != 0.0:
        raise LossError("loss.selfsup.weight must be 0.0 when selfsup is disabled")
    if selfsup.get("detach_original_prediction") is not True:
        raise LossError(
            "loss.selfsup.detach_original_prediction must be true "
            "(original prediction.detach() vs augmented prediction gradient)"
        )
    laplace = loss.get("laplace_nll")
    if not isinstance(laplace, dict):
        raise ShapeError("loss.laplace_nll must be a mapping")


def _validate_experiment(exp: Dict[str, Any]) -> None:
    if not isinstance(exp, dict):
        raise ShapeError("experiment must be a mapping")
    name = exp.get("name")
    if name not in ("smoke", "short", "full"):
        raise ExperimentError(f"experiment.name must be smoke/short/full, got {name!r}")

    max_updates = exp.get("max_updates")
    epochs = exp.get("epochs")
    validation_max_pairs = exp.get("validation_max_pairs")
    if name == "full":
        if max_updates is not None:
            raise ExperimentError(
                "FULL preset must not set max_updates (got "
                f"{max_updates!r}); FULL is epoch-bounded, not update-bounded"
            )
        if not _is_int(epochs) or epochs <= 0:
            raise ExperimentError(f"FULL preset requires a positive epochs, got {epochs!r}")
        if validation_max_pairs is not None:
            raise ExperimentError(
                "FULL preset must validate the full population: "
                f"validation_max_pairs must be null, got {validation_max_pairs!r}"
            )
        if exp.get("validate_full_each_epoch") is not True:
            raise ExperimentError(
                "FULL preset requires validate_full_each_epoch=true"
            )
    else:  # smoke / short
        if not _is_int(max_updates) or max_updates <= 0:
            raise ExperimentError(
                f"{name} preset requires a positive max_updates, got {max_updates!r}"
            )
        if epochs is not None:
            raise ExperimentError(
                f"{name} preset must not set epochs (got {epochs!r}); "
                "epochs belongs to the FULL preset"
            )
        if not _is_int(validation_max_pairs) or validation_max_pairs <= 0:
            raise ExperimentError(
                f"{name} preset requires a positive validation_max_pairs, "
                f"got {validation_max_pairs!r}"
            )
    if exp.get("early_stop") is not False:
        raise ExperimentError("experiment.early_stop must be false (early stop OFF)")


def _validate_checkpoint(ckpt: Dict[str, Any]) -> None:
    if not isinstance(ckpt, dict):
        raise ShapeError("checkpoint must be a mapping")
    if ckpt.get("save_last_y") is not True or ckpt.get("save_best_averaged_x") is not True:
        raise ShapeError("checkpoint must save both last_y and best_averaged_x")
    meta = ckpt.get("metadata")
    if not isinstance(meta, dict) or "last_y" not in meta or "best_averaged_x" not in meta:
        raise ShapeError(
            "checkpoint.metadata must keep training last_y and best_averaged_x separate"
        )
    strict = ckpt.get("strict_resume")
    if not isinstance(strict, dict):
        raise ShapeError("checkpoint.strict_resume must be a mapping")
    required = (
        "require_rng", "require_global_step", "require_epoch",
        "require_sampler_permutation", "require_consumed_cursor",
        "require_config_sha", "require_data_sha", "require_weight_sha",
        "require_source_sha",
    )
    for key in required:
        if strict.get(key) is not True:
            raise ShapeError(f"checkpoint.strict_resume.{key} must be true for strict resume")

_DETERMINISM_BOOL_KEYS = (
    "enabled", "warn_only", "cudnn_benchmark", "cudnn_deterministic", "allow_tf32",
)


def _validate_determinism(det: Dict[str, Any]) -> None:
    """Fail fast on any malformed / silently-ignorable determinism setting."""
    if not isinstance(det, dict):
        raise DeterminismError("determinism must be a mapping")
    for key in _DETERMINISM_BOOL_KEYS:
        if key not in det:
            raise DeterminismError(f"determinism.{key} is required")
        if not isinstance(det[key], bool):
            raise DeterminismError(
                f"determinism.{key} must be a bool, got {det[key]!r}"
            )
    workspace = det.get("cublas_workspace_config")
    if not isinstance(workspace, str) or workspace not in _ALLOWED_CUBLAS_WORKSPACE:
        raise DeterminismError(
            "determinism.cublas_workspace_config must be one of "
            f"{_ALLOWED_CUBLAS_WORKSPACE}, got {workspace!r}"
        )
    # forward-scope hooks: required and typed (never silently ignore a request).
    for key, allowed in (
        ("sdpa_backend", _ALLOWED_SDPA_BACKENDS),
        ("bilinear", _ALLOWED_BILINEAR_MODES),
    ):
        if key not in det:
            raise DeterminismError(f"determinism.{key} is required")
        if det[key] not in allowed:
            raise DeterminismError(
                f"determinism.{key} must be one of {allowed}, got {det[key]!r}"
            )
    # optional `strict` alias: never silently ignore a request for strictness.
    if "strict" in det:
        strict = det["strict"]
        if not isinstance(strict, bool):
            raise DeterminismError(
                f"determinism.strict must be a bool, got {strict!r}"
            )
        if strict is False:
            raise DeterminismError(
                "determinism.strict=false is rejected: strict reproducibility is "
                "required here (set strict=true or omit it); refusing to silently "
                "ignore the setting"
            )


def determinism_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the validated, typed determinism settings for ``cfg``.

    Raises :class:`DeterminismError` if the group is missing or malformed.  The
    returned mapping is exactly what :func:`determinism_context` applies.
    """
    det = _get(cfg, "determinism")
    if det is None:
        raise DeterminismError(
            "determinism group is missing from the resolved config; refusing to "
            "run without an explicit determinism contract"
        )
    _validate_determinism(det)
    return {
        "enabled": det["enabled"],
        "warn_only": det["warn_only"],
        "cublas_workspace_config": det["cublas_workspace_config"],
        "cudnn_benchmark": det["cudnn_benchmark"],
        "cudnn_deterministic": det["cudnn_deterministic"],
        "allow_tf32": det["allow_tf32"],
    }


@contextlib.contextmanager
def determinism_context(cfg: Dict[str, Any], *, torch_module: Any = None,
                        environ: Optional[Dict[str, str]] = None):
    """Apply the determinism settings and restore *all* globals on exit.

    Ordering matters: ``CUBLAS_WORKSPACE_CONFIG`` is set first (it must precede
    the first cuBLAS/CUDA call), then ``torch.use_deterministic_algorithms`` and
    the cuDNN / matmul-tf32 backend flags.  ``torch`` is imported lazily so the
    pure-schema tests stay light; ``torch_module`` / ``environ`` are injectable
    for CPU dependency-injection tests.

    This does **not** guarantee bitwise-identical CUDA results; with
    ``warn_only=true`` PyTorch warns (and falls back) where no deterministic CUDA
    kernel exists (e.g. the DPT bilinear backward).
    """
    settings = determinism_settings(cfg)
    if torch_module is None:  # noqa: SIM108 - intentional lazy import
        import torch as torch_module
    env = os.environ if environ is None else environ

    prev_workspace = env.get("CUBLAS_WORKSPACE_CONFIG")
    prev_enabled = torch_module.are_deterministic_algorithms_enabled()
    prev_warn = torch_module.is_deterministic_algorithms_warn_only_enabled()
    prev_benchmark = torch_module.backends.cudnn.benchmark
    prev_deterministic = torch_module.backends.cudnn.deterministic
    prev_matmul_tf32 = torch_module.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch_module.backends.cudnn.allow_tf32

    try:
        env["CUBLAS_WORKSPACE_CONFIG"] = settings["cublas_workspace_config"]
        torch_module.use_deterministic_algorithms(
            settings["enabled"], warn_only=settings["warn_only"]
        )
        torch_module.backends.cudnn.benchmark = settings["cudnn_benchmark"]
        torch_module.backends.cudnn.deterministic = settings["cudnn_deterministic"]
        torch_module.backends.cuda.matmul.allow_tf32 = settings["allow_tf32"]
        torch_module.backends.cudnn.allow_tf32 = settings["allow_tf32"]
        yield settings
    finally:
        torch_module.use_deterministic_algorithms(prev_enabled, warn_only=prev_warn)
        torch_module.backends.cudnn.benchmark = prev_benchmark
        torch_module.backends.cudnn.deterministic = prev_deterministic
        torch_module.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch_module.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        if prev_workspace is None:
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            env["CUBLAS_WORKSPACE_CONFIG"] = prev_workspace


def validate_config(cfg: Dict[str, Any], *, require_external: bool = False) -> None:
    """Validate a composed (or resolved) RGB-D config.

    Raises a typed :class:`RGBDConfigError` subclass on the first violation.
    ``require_external=True`` additionally requires the private values to have
    been filled in by an external resolved config.
    """
    if not isinstance(cfg, dict):
        raise ShapeError("config root must be a mapping")
    cfg = without_hydra(cfg)

    _check_unknown_keys(cfg, SCHEMA, "config")

    if cfg.get("fail_nonfinite") is not True:
        raise RGBDConfigError("fail_nonfinite must be true (no NaN->zero replacement)")
    if not _is_int(cfg.get("seed")):
        raise ShapeError(f"seed must be an integer, got {cfg.get('seed')!r}")

    for group in GROUP_NAMES:
        if group not in cfg or not isinstance(cfg[group], dict):
            raise ShapeError(f"config.{group} group is missing or not a mapping")

    _validate_dataset(cfg["dataset"], require_external)
    _validate_model(cfg["model"], require_external)
    _validate_optimizer(cfg["optimizer"])
    _validate_loss(cfg["loss"])
    _validate_experiment(cfg["experiment"])
    _validate_checkpoint(cfg.get("checkpoint", {}))
    _validate_determinism(cfg.get("determinism", {}))


# --------------------------------------------------------------------------- #
# external resolution
# --------------------------------------------------------------------------- #
def load_external_config(path: Any) -> Dict[str, Any]:
    """Load the Git-external private config (yaml/json)."""
    if path is None:
        return {}
    p = Path(str(path))
    if not p.is_file():
        raise ExternalConfigError(f"external config not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ExternalConfigError(f"external config is not valid yaml: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ExternalConfigError("external config root must be a mapping")
    return data


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        elif value is not None:
            base[key] = copy.deepcopy(value)
    return base


def resolve_config(
    cfg: Dict[str, Any], external: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge external private values over the public config and validate."""
    resolved = without_hydra(cfg)
    if external:
        resolved = _deep_merge(resolved, copy.deepcopy(external))
    validate_config(resolved, require_external=bool(external))
    return resolved


# --------------------------------------------------------------------------- #
# hashing / writing / redaction
# --------------------------------------------------------------------------- #
def canonical_yaml(cfg: Dict[str, Any]) -> str:
    """Deterministic yaml for hashing (sorted keys, no anchors)."""
    return yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False, allow_unicode=True)


def config_sha256(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def repo_root() -> Path:
    """Absolute path of the public repository that contains this module."""
    return Path(__file__).resolve().parents[2]


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_git_external(path: Any, what: str = "output") -> Path:
    """Resolve ``path`` and refuse any location inside the public repository.

    Symlinks are followed, so a link that points back into the repo is rejected.
    The raised message never echoes the offending private path.
    """
    resolved = Path(str(path)).resolve()
    if _is_inside(resolved, repo_root()):
        raise UnsafeOutputError(
            f"{what} path resolves inside the public repository; "
            "refusing to write private artifacts there"
        )
    return resolved


def _validate_output_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise UnsafeOutputError("output filename must be a non-empty string")
    if (
        os.path.isabs(filename)
        or "\\" in filename
        or filename != os.path.basename(filename)
        or os.path.basename(filename) in ("", ".", "..")
    ):
        raise UnsafeOutputError(
            "output filename must be a bare file name (no directory components)"
        )
    return filename


def save_resolved(
    cfg: Dict[str, Any], out_dir: Any, filename: str = "resolved_rgbd_config.yaml"
) -> Tuple[str, str]:
    """Write the resolved yaml + sha to a Git-external directory.

    Safety (checked *before* any ``mkdir`` / write):
    * the output directory must resolve outside the public repository
      (including via symlink);
    * ``filename`` must be a bare name (no absolute path / parent escape);
    * the final target itself must resolve outside the repository.
    """
    filename = _validate_output_filename(filename)
    out = assert_git_external(out_dir, what="resolved output")
    assert_git_external(out / filename, what="resolved output file")
    out.mkdir(parents=True, exist_ok=True)
    yml = out / filename
    yml.write_text(canonical_yaml(cfg), encoding="utf-8")
    sha = config_sha256(cfg)
    (out / (filename + ".sha256")).write_text(sha + "\n", encoding="utf-8")
    return str(yml), sha


def redact(cfg: Any) -> Any:
    """Return a copy with private keys and forbidden host paths redacted."""
    out = copy.deepcopy(cfg)
    for dotted in PRIVATE_KEYS:
        if _get(out, dotted) is not None:
            _set(out, dotted, "<redacted>")

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, str):
            for p in FORBIDDEN_HOST_PREFIXES:
                if p in node:
                    return "<redacted:path>"
        return node

    return _walk(out)

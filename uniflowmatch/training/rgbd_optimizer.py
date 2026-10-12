"""AMUSE optimizer wrapper for the UFM RGB-D fine-tuning (procedure 17B).

This module consumes the procedure-17A Hydra optimizer config
(``configs/rgbd_training/optimizer/amuse.yaml``, AdamW comparison preset in
``adamw.yaml``) and bridges it to the *official, unmodified* AMUSE optimizer
vendored under ``uniflowmatch/training/vendor/amuse/`` (Apache-2.0, pinned
commit recorded in that directory's ``NOTICE``).

Design highlights
-----------------
* **Parameter plan** built from the real ``model.get_parameter_groups()`` and
  ``model.named_parameters()``.  Every trainable parameter is assigned to
  exactly one destination (Muon or auxiliary SGD); shared tensor aliases are
  de-duplicated by object identity; missing / duplicate / unknown modules raise
  a typed error.  ``get_parameter_groups`` is never mutated.
* **Classification** (per the RGB-D param-group policy):
  - frozen modules (default: ``encoder``) are excluded from the optimizer,
  - hidden 2D / conv-4D weights of the Muon modules (default:
    ``info_sharing``, ``depth_fusion``) use ``use_muon=True``,
  - embeddings, output/uncertainty heads, biases, norms, scalars, gates and
    unsupported ``ndim == 3`` tensors fall back to the auxiliary SGD group
    (``use_muon=False``, ``update_type="sgd"``).
* **Checkpoint identity** stores the *post-constructor* AMUSE parameter order
  (the official constructor sorts Muon parameters by size descending) plus
  per-parameter shape / dtype / group / learning rate.  Strict load rejects a
  config-identity / ordered-name / shape / dtype mismatch *before* any state is
  copied, which also rejects a same-shape tensor-name swap.
* **Mode contract**: the optimizer starts in ``train_mode=False``; ``train()``
  is mandatory before ``step()``.  :meth:`eval_mode` switches to the averaged
  ``x`` state and restores the previous mode even if an exception is raised.
  ``state_dict`` records the mode and whether the parameters are the training
  ``last_y`` or the averaged ``best_x``.
* **Fail-fast**: all gradients are checked for finiteness before ``step`` (a
  non-finite gradient performs *no* partial update) and parameters are checked
  after the step; anomalies raise loudly instead of being silently zeroed.

The module deliberately never builds a learning-rate scheduler for AMUSE
(schedule-free).  ``clip_grad_norm_`` / ``grad_norm`` are exposed for the 17C
trainer to apply the configured gradient clipping.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from uniflowmatch.training.vendor.amuse.AMUSE import AMUSE

__all__ = [
    "OptimizerPlanError",
    "MissingParameterError",
    "DuplicateParameterError",
    "UnknownModuleError",
    "MuonShapeError",
    "NonFiniteGradientError",
    "NonFiniteParameterError",
    "AMUSEModeError",
    "StateIdentityMismatch",
    "StateNameOrderMismatch",
    "StateShapeMismatch",
    "StateDtypeMismatch",
    "ParameterPlan",
    "DEFAULT_POLICY",
    "build_parameter_plan",
    "assert_muon_matrix",
    "build_optimizer",
    "AMUSEControlledOptimizer",
    "AdamWControlledOptimizer",
    "vendor_amuse_sha256",
]


# --------------------------------------------------------------------------- #
# typed errors
# --------------------------------------------------------------------------- #
class OptimizerPlanError(ValueError):
    """Base class for RGB-D optimizer parameter-plan violations."""


class MissingParameterError(OptimizerPlanError):
    """A trainable parameter is not covered by any known module group."""


class DuplicateParameterError(OptimizerPlanError):
    """A parameter object is registered more than once inside one group."""


class UnknownModuleError(OptimizerPlanError):
    """A module group is not part of the declared parameter-group policy."""


class MuonShapeError(OptimizerPlanError):
    """A Muon group contains a parameter that is not a 2D/conv-4D matrix."""


class NonFiniteGradientError(FloatingPointError):
    """At least one gradient is NaN/inf; the step was aborted before any copy."""


class NonFiniteParameterError(FloatingPointError):
    """A parameter became NaN/inf after the step (or was already non-finite)."""


class AMUSEModeError(RuntimeError):
    """The optimizer was not in train mode when ``step`` was called."""


class StateIdentityMismatch(RuntimeError):
    """The saved optimizer state does not match the current structural config."""


class StateNameOrderMismatch(StateIdentityMismatch):
    """The saved post-constructor parameter name order differs (e.g. name swap)."""


class StateShapeMismatch(StateIdentityMismatch):
    """A saved parameter shape differs from the current parameter shape."""


class StateDtypeMismatch(StateIdentityMismatch):
    """A saved parameter dtype differs from the current parameter dtype."""


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
#: default RGB-D parameter-group policy (mirrors the public config defaults).
DEFAULT_POLICY: Dict[str, Tuple[str, ...]] = {
    "frozen": ("encoder",),
    "muon_modules": ("info_sharing", "depth_fusion"),
    "exclude_patterns": (
        "embedding",
        "output_head",
        "uncertainty_head",
        "bias",
        "norm",
        "gate",
    ),
    "aux_modules": ("output_head", "uncertainty_head"),
}

#: AMUSE instance attributes that live outside the standard state_dict.
_HYPER_EXTRA_ATTRS: Tuple[str, ...] = (
    "weight_decay_at_y",
    "beta1_init",
    "weight_lr_power",
    "warmup_steps",
    "rho",
    "r",
)
_MUON_NDIM = (2, 4)


@dataclass
class ParameterPlan:
    """Result of partitioning a model's parameters for the RGB-D optimizer."""

    muon: List[Tuple[str, nn.Parameter]] = field(default_factory=list)
    aux: List[Tuple[str, nn.Parameter]] = field(default_factory=list)
    frozen: List[str] = field(default_factory=list)
    aliases: List[Tuple[str, str, str]] = field(default_factory=list)
    group_of: Dict[str, str] = field(default_factory=dict)

    @property
    def trainable_count(self) -> int:
        return len(self.muon) + len(self.aux)


def _normalize_policy(policy: Optional[Mapping[str, Any]]) -> Dict[str, Tuple[str, ...]]:
    if policy is None:
        return dict(DEFAULT_POLICY)
    out = dict(DEFAULT_POLICY)
    for key in ("frozen", "muon_modules", "exclude_patterns", "aux_modules"):
        if policy.get(key) is not None:
            out[key] = tuple(policy[key])
    return out


def _policy_from_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Tuple[str, ...]]:
    if not isinstance(config, Mapping):
        return dict(DEFAULT_POLICY)
    model_cfg = config.get("model")
    pgp = None
    if isinstance(model_cfg, Mapping):
        pgp = model_cfg.get("param_group_policy")
    if pgp is None:
        pgp = config.get("param_group_policy")
    if not isinstance(pgp, Mapping):
        return dict(DEFAULT_POLICY)

    out = dict(DEFAULT_POLICY)
    if pgp.get("frozen") is not None:
        out["frozen"] = tuple(pgp["frozen"])
    mm = pgp.get("muon_hidden_matrix")
    if isinstance(mm, Mapping):
        if mm.get("modules") is not None:
            out["muon_modules"] = tuple(mm["modules"])
        if mm.get("exclude_name_patterns") is not None:
            out["exclude_patterns"] = tuple(mm["exclude_name_patterns"])
    af = pgp.get("aux_fallback")
    if isinstance(af, Mapping) and af.get("modules") is not None:
        out["aux_modules"] = tuple(af["modules"])
    return out


def _classify_group(group_name: str, policy: Mapping[str, Sequence[str]]) -> str:
    if group_name in policy["frozen"]:
        return "frozen"
    if group_name in policy["aux_modules"]:
        return "aux"
    if group_name in policy["muon_modules"]:
        return "muon"
    raise UnknownModuleError(
        f"module group {group_name!r} is not declared in the parameter-group policy "
        f"(frozen={list(policy['frozen'])}, muon={list(policy['muon_modules'])}, "
        f"aux={list(policy['aux_modules'])})"
    )


def build_parameter_plan(
    model: nn.Module, policy: Optional[Mapping[str, Any]] = None
) -> ParameterPlan:
    """Partition every trainable parameter exactly once (no frozen entries).

    Uses the model's ``get_parameter_groups()`` for module membership and
    ``named_parameters()`` for canonical names.  Shared aliases (the same tensor
    object referenced by two groups) are collapsed by identity and reported in
    :attr:`ParameterPlan.aliases`.
    """
    resolved = _normalize_policy(policy)
    groups = model.get_parameter_groups()
    if not isinstance(groups, Mapping):
        raise OptimizerPlanError("get_parameter_groups() must return a mapping")

    named = dict(model.named_parameters())
    name_by_id: Dict[int, str] = {}
    param_by_name: Dict[str, nn.Parameter] = {}
    for n, p in named.items():
        name_by_id.setdefault(id(p), n)
        param_by_name[n] = p

    plan = ParameterPlan()
    module_of: Dict[int, str] = {}

    for group_name, plist in groups.items():
        kind = _classify_group(group_name, resolved)
        local_ids: set = set()
        for p in plist:
            pid = id(p)
            if pid in local_ids:
                raise DuplicateParameterError(
                    f"parameter registered more than once in module group {group_name!r}"
                )
            local_ids.add(pid)
            name = name_by_id.get(pid)
            if name is None:
                raise MissingParameterError(
                    f"a parameter in module group {group_name!r} is not present in "
                    "named_parameters()"
                )
            if pid in module_of:
                plan.aliases.append((module_of[pid], group_name, name))
                continue
            module_of[pid] = group_name

            if kind == "frozen" or not p.requires_grad:
                plan.frozen.append(name)
                plan.group_of[name] = "frozen"
                continue

            if kind == "aux":
                plan.aux.append((name, p))
                plan.group_of[name] = "aux"
                continue

            excluded = any(pat in name for pat in resolved["exclude_patterns"])
            if p.ndim in _MUON_NDIM and not excluded:
                plan.muon.append((name, p))
                plan.group_of[name] = "muon"
            else:
                plan.aux.append((name, p))
                plan.group_of[name] = "aux"

    # coverage: requires_grad params outside frozen modules must be assigned once
    frozen_modules = set(resolved["frozen"])
    module_name_of_param = {}
    for group_name, plist in groups.items():
        for p in plist:
            module_name_of_param.setdefault(id(p), group_name)

    assigned = {n for n, _ in plan.muon} | {n for n, _ in plan.aux}
    for n, p in named.items():
        pid = id(p)
        mod = module_name_of_param.get(pid)
        if mod is None:
            # a trainable parameter that no module group claims
            if p.requires_grad:
                raise MissingParameterError(
                    f"trainable parameter {n!r} is not covered by any module group"
                )
            continue
        if mod in frozen_modules:
            continue
        if p.requires_grad and n not in assigned:
            raise MissingParameterError(
                f"trainable parameter {n!r} is missing from the optimizer groups"
            )
    extra = {n for n in assigned if n not in named}
    if extra:
        raise OptimizerPlanError(
            f"optimizer groups reference unknown parameter names: {sorted(extra)}"
        )

    assert_muon_matrix(plan)
    return plan


def assert_muon_matrix(plan: ParameterPlan) -> None:
    """Reject any Muon-group parameter whose dimensionality is not 2D/conv-4D."""
    for name, p in plan.muon:
        if p.ndim not in _MUON_NDIM:
            raise MuonShapeError(
                f"Muon parameter {name!r} has ndim={p.ndim}; only 2D matrices and "
                "conv-4D weights are supported (non-matrix parameters must use aux SGD)"
            )


def _apply_freeze(model: nn.Module, plan: ParameterPlan) -> None:
    """Concretely apply the freeze policy (``requires_grad_(False)``)."""
    if not plan.frozen:
        return
    frozen = set(plan.frozen)
    for n, p in model.named_parameters():
        if n in frozen:
            p.requires_grad_(False)


# --------------------------------------------------------------------------- #
# vendor identity
# --------------------------------------------------------------------------- #
def vendor_amuse_sha256() -> str:
    """SHA-256 of the vendored (unmodified) official ``AMUSE.py``."""
    path = Path(__file__).resolve().parent / "vendor" / "amuse" / "AMUSE.py"
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
def _optimizer_cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise OptimizerPlanError("optimizer config must be a mapping")
    if isinstance(config.get("optimizer"), Mapping):
        return config["optimizer"]
    if "name" in config:
        return config
    raise OptimizerPlanError("config has no 'optimizer' mapping and no 'name' key")


def _all_params(optimizer: torch.optim.Optimizer) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for g in optimizer.param_groups:
        params.extend(g["params"])
    return params


def _check_grads_finite(params: Iterable[nn.Parameter], where: str = "") -> None:
    bad: List[str] = []
    for i, p in enumerate(params):
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad.append(f"#{i}{where}")
    if bad:
        raise NonFiniteGradientError(
            "non-finite gradient detected before optimizer.step; aborting with no "
            f"partial update (offending parameters: {bad[:8]}{'...' if len(bad) > 8 else ''})"
        )


def _check_params_finite(params: Iterable[nn.Parameter]) -> None:
    bad: List[int] = []
    for i, p in enumerate(params):
        if not torch.isfinite(p).all():
            bad.append(i)
    if bad:
        raise NonFiniteParameterError(
            "non-finite parameter detected after optimizer.step; failing loudly "
            f"instead of silently continuing (offending parameters: {bad[:8]})"
        )


def _compute_grad_norm(params: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return math.sqrt(total)


# --------------------------------------------------------------------------- #
# AMUSE controlled optimizer
# --------------------------------------------------------------------------- #
class AMUSEControlledOptimizer:
    """AMUSE wrapper adding mode control, strict serialization and fail-fast."""

    name = "amuse"

    def __init__(
        self,
        optimizer: AMUSE,
        plan: ParameterPlan,
        *,
        name_by_id: Mapping[int, str],
        source_commit: Optional[str] = None,
        extra_identity: Optional[Mapping[str, Any]] = None,
    ):
        self.optimizer = optimizer
        self.plan = plan
        self._name_by_id = dict(name_by_id)
        self._source_commit = source_commit
        self._extra_identity = dict(extra_identity or {})
        self._param_by_name = {
            self._name_by_id[id(p)]: p
            for g in optimizer.param_groups
            for p in g["params"]
        }
        self.ordered_names = [
            self._name_by_id[id(p)]
            for g in optimizer.param_groups
            for p in g["params"]
        ]
        self.identity = self._build_identity()

    # -- identity ---------------------------------------------------------- #
    def _build_identity(self) -> Dict[str, Any]:
        ordered: List[str] = []
        group_of: Dict[str, str] = {}
        shapes: Dict[str, List[int]] = {}
        dtypes: Dict[str, str] = {}
        lrs: Dict[str, float] = {}
        weight_decays: Dict[str, float] = {}
        for g in self.optimizer.param_groups:
            kind = "muon" if g.get("use_muon") else str(g.get("update_type", "aux"))
            for p in g["params"]:
                n = self._name_by_id[id(p)]
                ordered.append(n)
                group_of[n] = kind
                shapes[n] = list(p.shape)
                dtypes[n] = str(p.dtype)
                lrs[n] = float(g.get("base_lr", g.get("lr", 0.0)))
                weight_decays[n] = float(g.get("weight_decay", 0.0))
        payload: Dict[str, Any] = {
            "optimizer": self.name,
            "source_commit": self._source_commit,
            "ordered_names": ordered,
            "group_of": group_of,
            "shapes": shapes,
            "dtypes": dtypes,
            "lrs": lrs,
            "weight_decays": weight_decays,
        }
        payload.update(self._extra_identity)
        payload["sha"] = _sha256_json(payload)
        return payload

    # -- introspection ----------------------------------------------------- #
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def train_mode(self) -> bool:
        return bool(self.optimizer.train_mode)

    # -- mode -------------------------------------------------------------- #
    def train(self) -> None:
        self.optimizer.train()

    def eval(self) -> None:
        self.optimizer.eval()

    @contextmanager
    def eval_mode(self):
        """Temporarily use the averaged ``x`` state, restoring mode on exit."""
        was_train = bool(self.optimizer.train_mode)
        if was_train:
            self.optimizer.eval()
        try:
            yield self
        finally:
            if was_train:
                self.optimizer.train()

    @contextmanager
    def train_mode_ctx(self):
        """Temporarily force train (``y``) mode, restoring mode on exit."""
        was_train = bool(self.optimizer.train_mode)
        if not was_train:
            self.optimizer.train()
        try:
            yield self
        finally:
            if not was_train:
                self.optimizer.eval()

    # -- update ------------------------------------------------------------ #
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def grad_norm(self) -> float:
        return _compute_grad_norm(_all_params(self.optimizer))

    def clip_grad_norm_(self, max_norm: float) -> float:
        return float(
            torch.nn.utils.clip_grad_norm_(_all_params(self.optimizer), max_norm)
        )

    def step(self, closure=None):
        if not self.optimizer.train_mode:
            raise AMUSEModeError(
                "AMUSE was not in train mode (call .train() before .step())"
            )
        _check_grads_finite(_all_params(self.optimizer))
        result = self.optimizer.step(closure)
        _check_params_finite(_all_params(self.optimizer))
        return result

    # -- serialization ----------------------------------------------------- #
    def state_dict(self) -> Dict[str, Any]:
        sd = dict(self.optimizer.state_dict())
        train_mode = bool(self.optimizer.train_mode)
        extra: Dict[str, Any] = {
            "format_version": 1,
            "optimizer_name": self.name,
            "ordered_names": list(self.ordered_names),
            "config_identity": copy.deepcopy(self.identity),
            "mode": {
                "train_mode": train_mode,
                "param_state": "y" if train_mode else "x",
                "purpose": "last_y_training" if train_mode else "best_averaged_x_eval",
            },
            "train_mode": train_mode,
        }
        for attr in _HYPER_EXTRA_ATTRS:
            extra[attr] = getattr(self.optimizer, attr)
        extra["beta1"] = getattr(self.optimizer, "beta1", None)
        sd["amuse_extra"] = extra
        return sd

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        extra = state_dict.get("amuse_extra")
        if not isinstance(extra, Mapping):
            raise StateIdentityMismatch(
                "checkpoint is missing the AMUSE extra state (train_mode / hyper-params / "
                "identity); refusing to load a bare state_dict"
            )

        saved_identity = extra.get("config_identity")
        if not isinstance(saved_identity, Mapping):
            raise StateIdentityMismatch("checkpoint has no config identity")
        if saved_identity.get("sha") != self.identity["sha"]:
            raise StateIdentityMismatch(
                "optimizer config identity mismatch (group choice / config / shapes / "
                "dtypes); rejecting before any state copy"
            )
        if list(extra.get("ordered_names", [])) != list(self.ordered_names):
            raise StateNameOrderMismatch(
                "optimizer parameter name order mismatch; a same-shape tensor-name swap "
                "is rejected before resume"
            )

        saved_shapes = saved_identity.get("shapes", {})
        saved_dtypes = saved_identity.get("dtypes", {})
        for n in self.ordered_names:
            p = self._param_by_name[n]
            if list(p.shape) != list(saved_shapes.get(n, [])):
                raise StateShapeMismatch(f"shape mismatch for parameter {n!r}")
            if str(p.dtype) != str(saved_dtypes.get(n)):
                raise StateDtypeMismatch(f"dtype mismatch for parameter {n!r}")

        standard = {k: v for k, v in state_dict.items() if k != "amuse_extra"}
        self.optimizer.load_state_dict(standard)

        for attr in _HYPER_EXTRA_ATTRS:
            if attr in extra:
                setattr(self.optimizer, attr, extra[attr])
        if extra.get("beta1") is not None:
            self.optimizer.beta1 = extra["beta1"]
        if "train_mode" in extra:
            self.optimizer.train_mode = bool(extra["train_mode"])


# --------------------------------------------------------------------------- #
# AdamW comparison optimizer
# --------------------------------------------------------------------------- #
class AdamWControlledOptimizer:
    """Thin AdamW comparison wrapper (schedule-free is NOT claimed for AdamW)."""

    name = "adamw"

    def __init__(self, optimizer, plan: ParameterPlan, name_by_id: Mapping[int, str]):
        self.optimizer = optimizer
        self.plan = plan
        self._name_by_id = dict(name_by_id)
        self.ordered_names = [
            self._name_by_id[id(p)]
            for g in optimizer.param_groups
            for p in g["params"]
        ]
        self.identity = {
            "optimizer": self.name,
            "ordered_names": list(self.ordered_names),
            "shapes": {n: list(p.shape) for n, p in self._named()},
            "dtypes": {n: str(p.dtype) for n, p in self._named()},
        }
        self.identity["sha"] = _sha256_json(self.identity)

    def _named(self):
        for g in self.optimizer.param_groups:
            for p in g["params"]:
                yield self._name_by_id[id(p)], p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def train_mode(self) -> bool:
        return True

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    @contextmanager
    def eval_mode(self):
        yield self

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def grad_norm(self) -> float:
        return _compute_grad_norm(_all_params(self.optimizer))

    def clip_grad_norm_(self, max_norm: float) -> float:
        return float(
            torch.nn.utils.clip_grad_norm_(_all_params(self.optimizer), max_norm)
        )

    def step(self, closure=None):
        _check_grads_finite(_all_params(self.optimizer))
        result = self.optimizer.step(closure)
        _check_params_finite(_all_params(self.optimizer))
        return result

    def state_dict(self) -> Dict[str, Any]:
        sd = dict(self.optimizer.state_dict())
        sd["adams_extra"] = {
            "format_version": 1,
            "optimizer_name": self.name,
            "ordered_names": list(self.ordered_names),
            "config_identity": copy.deepcopy(self.identity),
        }
        return sd

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        extra = state_dict.get("adams_extra")
        if not isinstance(extra, Mapping):
            raise StateIdentityMismatch("checkpoint is missing the AdamW extra state")
        saved = extra.get("config_identity", {})
        if saved.get("sha") != self.identity["sha"]:
            raise StateIdentityMismatch("AdamW config identity mismatch")
        if list(extra.get("ordered_names", [])) != list(self.ordered_names):
            raise StateNameOrderMismatch("AdamW parameter name order mismatch")
        standard = {k: v for k, v in state_dict.items() if k != "adams_extra"}
        self.optimizer.load_state_dict(standard)


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _build_amuse(
    model: nn.Module,
    opt_cfg: Mapping[str, Any],
    plan: ParameterPlan,
    name_by_id: Mapping[int, str],
) -> AMUSEControlledOptimizer:
    args = opt_cfg.get("constructor_args", {}) or {}
    warmup = int(args.get("warmup_steps", opt_cfg.get("warmup_steps", 0)) or 0)
    if warmup <= 0:
        raise OptimizerPlanError(
            f"AMUSE requires warmup_steps > 0 (got {warmup!r}); warmup 0 is invalid"
        )
    if opt_cfg.get("schedule_free", True) is not True:
        raise OptimizerPlanError("AMUSE is schedule-free; schedule_free must be true")
    if opt_cfg.get("external_scheduler") is not None:
        raise OptimizerPlanError(
            "AMUSE is schedule-free: external_scheduler must be null; "
            "do not attach a scheduler"
        )

    groups_cfg = opt_cfg.get("groups", {}) or {}
    muon_cfg = groups_cfg.get("muon", {}) or {}
    aux_cfg = groups_cfg.get("aux", {}) or {}

    param_groups = []
    if plan.muon:
        param_groups.append(
            {
                "params": [p for _, p in plan.muon],
                "use_muon": True,
                "lr": float(muon_cfg.get("lr", opt_cfg.get("lr", 2e-2))),
                "momentum": float(muon_cfg.get("momentum", 0.95)),
                "aux_update_type": muon_cfg.get("aux_update_type", "sgd"),
                "weight_decay": float(
                    muon_cfg.get("weight_decay", opt_cfg.get("weight_decay", 0.0))
                ),
            }
        )
    if plan.aux:
        param_groups.append(
            {
                "params": [p for _, p in plan.aux],
                "use_muon": False,
                "update_type": aux_cfg.get("update_type", "sgd"),
                "lr": float(
                    aux_cfg.get("lr", opt_cfg.get("sgd_lr", opt_cfg.get("lr", 1.0)))
                ),
                "weight_decay": float(aux_cfg.get("weight_decay", 0.0)),
            }
        )

    optimizer = AMUSE(
        param_groups,
        weight_decay_at_y=float(args.get("weight_decay_at_y", 0.0)),
        beta1=float(args.get("beta1", 0.9)),
        weight_lr_power=float(args.get("weight_lr_power", 2.0)),
        warmup_steps=warmup,
        rho=float(args.get("rho", 1.0)),
        r=float(args.get("r", 0.0)),
    )
    source = opt_cfg.get("source") or {}
    return AMUSEControlledOptimizer(
        optimizer,
        plan,
        name_by_id=name_by_id,
        source_commit=source.get("commit") if isinstance(source, Mapping) else None,
    )


def _build_adamw(
    opt_cfg: Mapping[str, Any],
    plan: ParameterPlan,
    name_by_id: Mapping[int, str],
) -> AdamWControlledOptimizer:
    params = [p for _, p in plan.muon] + [p for _, p in plan.aux]
    if not params:
        raise OptimizerPlanError("no trainable parameters for the AdamW optimizer")
    groups_cfg = opt_cfg.get("groups", {}) or {}
    all_cfg = groups_cfg.get("all", {}) or {}
    lr = float(all_cfg.get("lr", opt_cfg.get("lr", 1e-4)))
    wd = float(all_cfg.get("weight_decay", opt_cfg.get("weight_decay", 0.0)))
    betas = tuple(opt_cfg.get("betas", (0.9, 0.95)))
    eps = float(opt_cfg.get("eps", 1e-8))
    optimizer = torch.optim.AdamW(
        [{"params": params, "lr": lr, "weight_decay": wd}],
        betas=betas,
        eps=eps,
    )
    return AdamWControlledOptimizer(optimizer, plan, name_by_id)


def build_optimizer(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    policy: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Build the controlled optimizer declared by ``config``.

    ``config`` is either the full resolved RGB-D config (with an ``optimizer``
    and ``model`` mapping) or the optimizer sub-config directly.  ``policy``
    overrides ``model.param_group_policy`` when supplied.
    """
    opt_cfg = _optimizer_cfg(config)
    resolved_policy = (
        _normalize_policy(policy)
        if policy is not None
        else _policy_from_config(config)
    )
    plan = build_parameter_plan(model, resolved_policy)
    name_by_id = {id(p): n for n, p in dict(model.named_parameters()).items()}

    name = str(opt_cfg.get("name", "amuse"))
    if name == "adamw":
        return _build_adamw(opt_cfg, plan, name_by_id)
    if name == "amuse":
        _apply_freeze(model, plan)
        return _build_amuse(model, opt_cfg, plan, name_by_id)
    raise OptimizerPlanError(f"unsupported optimizer name {name!r}")


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def _sha256_json(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

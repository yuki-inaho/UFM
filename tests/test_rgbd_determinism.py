"""CPU focused tests for the RGB-D determinism group (Step 19 attempt-4).

Covers only the *generic* reproducibility contract:

* the ``determinism`` group is composed with the declared defaults and consumed;
* typed validation rejects malformed / silently-ignorable values (including a
  ``strict=false`` alias and non-bool inputs);
* the settings are applied in the right order (cuBLAS env first, then the torch
  backend flags) and before any CUDA / model construction in ``_train``;
* the pre-existing seeding behavior is not touched;
* every global flag + env var is restored in ``finally`` (even on error).

``rgbd_config.py`` is loaded standalone so the heavy UFM stack is never imported;
``torch`` is imported lazily by the context and is only needed by the restore
tests (CPU is sufficient -- no GPU is used here).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO / "configs" / "rgbd_training"
_MODULE_PATH = _REPO / "uniflowmatch" / "training" / "rgbd_config.py"
_TRAIN_SCRIPT = _REPO / "scripts" / "train_rgbd.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("rgbd_config_det", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_module()

_WORKSPACE = ":4096:8"


def _compose(overrides=()):
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="default", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def _public_config():
    return rc.without_hydra(_compose())


# --------------------------------------------------------------------------- #
# config group: present, composed, consumed
# --------------------------------------------------------------------------- #
def test_determinism_group_defaults_present_and_valid():
    cfg = _public_config()
    det = cfg["determinism"]
    assert det["enabled"] is True
    assert det["warn_only"] is True
    assert det["cublas_workspace_config"] == _WORKSPACE
    assert det["cudnn_benchmark"] is False
    assert det["cudnn_deterministic"] is True
    assert det["allow_tf32"] is False
    # the composed config still validates end-to-end.
    rc.validate_config(cfg, require_external=False)


def test_determinism_settings_consumed_typed():
    cfg = _public_config()
    settings = rc.determinism_settings(cfg)
    assert settings == {
        "enabled": True,
        "warn_only": True,
        "cublas_workspace_config": _WORKSPACE,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32": False,
    }


def test_default_train_precision_stays_bf16():
    # float32 precision comes ONLY from an external override; the default is bf16.
    cfg = _public_config()
    assert cfg["model"]["train_precision"] == "bf16"
    cfg_f32 = rc.without_hydra(_compose(["model.train_precision=float32"]))
    assert cfg_f32["model"]["train_precision"] == "float32"
    rc.validate_config(cfg_f32, require_external=False)


# --------------------------------------------------------------------------- #
# typed validation: reject malformed / silently-ignorable values
# --------------------------------------------------------------------------- #
def test_missing_group_rejected():
    cfg = _public_config()
    del cfg["determinism"]
    with pytest.raises(rc.DeterminismError):
        rc.determinism_settings(cfg)


@pytest.mark.parametrize("key,value", [
    ("enabled", "true"),
    ("warn_only", 1),
    ("cudnn_benchmark", "no"),
    ("cudnn_deterministic", None),
    ("allow_tf32", 0),
    ("cublas_workspace_config", 4096),
    ("cublas_workspace_config", ":1:1"),
])
def test_wrong_types_rejected(key, value):
    cfg = _public_config()
    cfg["determinism"][key] = value
    with pytest.raises(rc.DeterminismError):
        rc.validate_config(cfg)
    with pytest.raises(rc.DeterminismError):
        rc.determinism_settings(cfg)


def test_strict_false_alias_rejected_not_silently_ignored():
    cfg = _public_config()
    cfg["determinism"]["strict"] = False
    with pytest.raises(rc.DeterminismError):
        rc.validate_config(cfg)
    cfg = _public_config()
    cfg["determinism"]["strict"] = "yes"
    with pytest.raises(rc.DeterminismError):
        rc.determinism_settings(cfg)
    # strict=true is accepted.
    cfg = _public_config()
    cfg["determinism"]["strict"] = True
    rc.validate_config(cfg)


# --------------------------------------------------------------------------- #
# application: ordering + flags + restore (DI with a fake torch)
# --------------------------------------------------------------------------- #
class _FakeTorch:
    def __init__(self, env, order):
        self._env = env
        self._order = order
        from types import SimpleNamespace as S

        self.backends = S(
            cudnn=S(benchmark=True, deterministic=False, allow_tf32=True),
            cuda=S(matmul=S(allow_tf32=True)),
        )
        self._det = False
        self._warn = False

    def are_deterministic_algorithms_enabled(self):
        return self._det

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self._warn

    def use_deterministic_algorithms(self, enabled, warn_only=False):
        self._order.append(("det", enabled, warn_only, self._env.get("CUBLAS_WORKSPACE_CONFIG")))
        self._det, self._warn = enabled, warn_only


def test_ordering_cublas_env_before_torch_flags():
    env = {}
    order = []
    fake = _FakeTorch(env, order)
    cfg = _public_config()
    with rc.determinism_context(cfg, torch_module=fake, environ=env) as settings:
        assert settings["cublas_workspace_config"] == _WORKSPACE
        # env is already set when use_deterministic_algorithms is called.
        assert order[0] == ("det", True, True, _WORKSPACE)
        assert env["CUBLAS_WORKSPACE_CONFIG"] == _WORKSPACE
        assert fake.backends.cudnn.benchmark is False
        assert fake.backends.cudnn.deterministic is True
        assert fake.backends.cuda.matmul.allow_tf32 is False
        assert fake.backends.cudnn.allow_tf32 is False


def test_settings_applied_before_first_cuda_in_train_script():
    source = _TRAIN_SCRIPT.read_text()
    enter = source.index("with determinism_context(resolved):")
    # the context opens before seeding / cuBLAS-touching calls / model build.
    assert enter < source.index("torch.manual_seed(seed)")
    assert enter < source.index("torch.cuda.manual_seed_all(seed)")
    assert enter < source.index("build_rgbd_model(")
    assert enter < source.index("model.to(args.device)")


# --------------------------------------------------------------------------- #
# seed behavior preserved + flags restored in finally (real CPU torch)
# --------------------------------------------------------------------------- #
def test_seed_behavior_preserved():
    import torch

    source = _TRAIN_SCRIPT.read_text()
    for call in ("random.seed(seed)", "np.random.seed(seed)",
                 "torch.manual_seed(seed)", "torch.cuda.manual_seed_all(seed)"):
        assert call in source

    torch.manual_seed(20261012)
    before = torch.random.get_rng_state().clone()
    with rc.determinism_context(_public_config(), environ={}):
        pass
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_global_flags_restored_in_finally():
    import torch

    env = {"CUBLAS_WORKSPACE_CONFIG": "PREEXISTING"}
    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with pytest.raises(RuntimeError), rc.determinism_context(_public_config(), environ=env):
        assert env["CUBLAS_WORKSPACE_CONFIG"] == _WORKSPACE
        raise RuntimeError("boom")

    assert env["CUBLAS_WORKSPACE_CONFIG"] == "PREEXISTING"
    assert torch.are_deterministic_algorithms_enabled() is False
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_env_var_removed_when_absent_before():
    import torch

    env = {}
    with rc.determinism_context(_public_config(), environ=env):
        assert env["CUBLAS_WORKSPACE_CONFIG"] == _WORKSPACE
    assert "CUBLAS_WORKSPACE_CONFIG" not in env
    torch.use_deterministic_algorithms(False, warn_only=False)

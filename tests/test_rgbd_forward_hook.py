"""CPU focused tests for the bounded deterministic forward hook (Step 19 attempt-10).

Covers the contract of ``deterministic_forward_context`` in
``uniflowmatch/training/rgbd_deterministic_ops.py``:

* ``sdpa_backend='math'`` restricts SDPA to the math kernel for the scope only;
* ``bilinear='deterministic'`` routes ``F.interpolate(mode='bilinear')`` on 4D
  input to :func:`deterministic_bilinear_interpolate`, while every other mode /
  rank is forwarded unchanged;
* the context is a no-op for the ``default`` settings;
* ``F.interpolate`` and *all* SDPA enable flags are restored on exit, on
  exception, and across nesting;
* ``trainer.forward`` consumes the context;
* the config schema / typed settings accept ``default|math`` and
  ``default|deterministic`` and reject wrong values;
* the CLI source-identity module list includes the new module.

No GPU, no model training, no dataset access.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO / "configs" / "rgbd_training"
_MODULE_PATH = _REPO / "uniflowmatch" / "training" / "rgbd_deterministic_ops.py"
_CONFIG_PATH = _REPO / "uniflowmatch" / "training" / "rgbd_config.py"
_TRAIN_SCRIPT = _REPO / "scripts" / "train_rgbd.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ops = _load("rgbd_deterministic_ops_fh", _MODULE_PATH)
rc = _load("rgbd_config_fh", _CONFIG_PATH)
ctx = ops.deterministic_forward_context
DFE = ops.DeterministicForwardError


def _flags():
    c = torch.backends.cuda
    return (
        c.math_sdp_enabled(), c.flash_sdp_enabled(),
        c.mem_efficient_sdp_enabled(), c.cudnn_sdp_enabled(),
    )


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
# SDPA flags: math-only for the scope, restored afterwards
# --------------------------------------------------------------------------- #
def test_context_sets_math_only_flags_and_restores():
    before = _flags()
    with ctx({"determinism": {"sdpa_backend": "math"}}):
        assert _flags() == (True, False, False, False)
    assert _flags() == before


def test_math_only_matches_named_sdpa_kernel():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with ctx({"determinism": {"sdpa_backend": "math"}}):
        got = _flags()
    with sdpa_kernel(SDPBackend.MATH):
        expected = _flags()
    assert got == expected == (True, False, False, False)


def test_default_does_not_touch_flags_or_function():
    original = F.interpolate
    before = _flags()
    with ctx({"determinism": {"sdpa_backend": "default", "bilinear": "default"}}):
        assert F.interpolate is original
        assert _flags() == before
    # a missing determinism group is the same no-op.
    with ctx({}):
        assert F.interpolate is original
        assert _flags() == before
    with ctx({"determinism": {}}):
        assert F.interpolate is original
        assert _flags() == before


# --------------------------------------------------------------------------- #
# bilinear wrapper: reference agreement + untouched other modes
# --------------------------------------------------------------------------- #
def test_bilinear_matches_reference_inside_scope():
    x = torch.randn(2, 3, 5, 7, dtype=torch.float64)
    ref = F.interpolate(x, size=(9, 4), mode="bilinear", align_corners=False)
    with ctx({"determinism": {"bilinear": "deterministic"}}):
        assert F.interpolate is not ops._ORIGINAL_INTERPOLATE
        got = F.interpolate(x, size=(9, 4), mode="bilinear", align_corners=False)
    assert torch.allclose(got, ref, atol=1e-12)
    assert F.interpolate is ops._ORIGINAL_INTERPOLATE


def test_bilinear_align_corners_none_becomes_false():
    x = torch.randn(1, 2, 6, 6, dtype=torch.float64)
    ref = F.interpolate(x, size=(3, 3), mode="bilinear", align_corners=False)
    with ctx({"determinism": {"bilinear": "deterministic"}}):
        got = F.interpolate(x, size=(3, 3), mode="bilinear", align_corners=None)
    assert torch.allclose(got, ref, atol=1e-12)


@pytest.mark.parametrize("mode", ["nearest", "nearest-exact", "area"])
def test_other_modes_delegate_to_original(mode):
    x = torch.randn(2, 3, 8, 8)
    ref = ops._ORIGINAL_INTERPOLATE(x, size=(4, 4), mode=mode)
    with ctx({"determinism": {"bilinear": "deterministic"}}):
        got = F.interpolate(x, size=(4, 4), mode=mode)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("shape", [(3, 6, 6), (1, 1, 2, 6, 6)])
def test_non_4d_bilinear_delegates_to_original(monkeypatch, shape):
    # non-4D bilinear is outside the helper's contract; the wrapper must
    # delegate to the genuine torch implementation (captured at import).
    seen = {}

    def fake(input, size=None, scale_factor=None, mode="nearest",
             align_corners=None, recompute_scale_factor=None, antialias=False):
        seen["mode"] = mode
        seen["dim"] = input.dim()
        return "delegated"

    monkeypatch.setattr(ops, "_ORIGINAL_INTERPOLATE", fake)
    with ctx({"determinism": {"bilinear": "deterministic"}}):
        got = F.interpolate(torch.randn(*shape), size=(3, 3), mode="bilinear",
                            align_corners=False)
    assert got == "delegated"
    assert seen == {"mode": "bilinear", "dim": len(shape)}


def test_bilinear_scale_factor_passthrough():
    x = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    ref = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    with ctx({"determinism": {"bilinear": "deterministic"}}):
        got = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    assert torch.allclose(got, ref, atol=1e-12)


# --------------------------------------------------------------------------- #
# exception + nesting restore
# --------------------------------------------------------------------------- #
def test_exception_restores_function_and_flags():
    original = F.interpolate
    before = _flags()
    with pytest.raises(RuntimeError):
        with ctx({"determinism": {"sdpa_backend": "math", "bilinear": "deterministic"}}):
            assert F.interpolate is not original
            assert _flags() == (True, False, False, False)
            raise RuntimeError("boom")
    assert F.interpolate is original
    assert _flags() == before


def test_nesting_restores_outer_state():
    original = F.interpolate
    before = _flags()
    with ctx({"determinism": {"sdpa_backend": "math", "bilinear": "deterministic"}}):
        outer_fn = F.interpolate
        assert outer_fn is not original
        assert _flags() == (True, False, False, False)
        with ctx({"determinism": {"sdpa_backend": "math"}}):
            # inner still math-only but keeps the outer wrapper active.
            assert F.interpolate is outer_fn
            assert _flags() == (True, False, False, False)
        assert F.interpolate is outer_fn
        assert _flags() == (True, False, False, False)
    assert F.interpolate is original
    assert _flags() == before


# --------------------------------------------------------------------------- #
# trainer.forward consumes the context
# --------------------------------------------------------------------------- #
def _trainer_stub(cfg):
    import sys

    for _p in (_REPO, _REPO / "UniCeption"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from uniflowmatch.training.rgbd_trainer import RGBDTrainer

    trainer = RGBDTrainer.__new__(RGBDTrainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")
    return trainer


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.observations = []

    def forward(self, view1, view2):
        self.observations.append(
            {"interp_patched": F.interpolate is not ops._ORIGINAL_INTERPOLATE,
             "flags": _flags()}
        )
        return "result"


def test_trainer_forward_consumes_context():
    trainer = _trainer_stub(
        {"determinism": {"sdpa_backend": "math", "bilinear": "deterministic"}}
    )
    model = _RecordingModel()
    trainer.model = model
    out = trainer.forward({"a": 1}, {"b": 2})
    assert out == "result"
    obs = model.observations[0]
    assert obs["interp_patched"] is True
    assert obs["flags"] == (True, False, False, False)
    # restored after forward
    assert F.interpolate is ops._ORIGINAL_INTERPOLATE
    assert _flags() == (True, True, True, True)


def test_trainer_forward_default_is_transparent():
    trainer = _trainer_stub({"determinism": {"sdpa_backend": "default", "bilinear": "default"}})
    model = _RecordingModel()
    trainer.model = model
    trainer.forward({}, {})
    obs = model.observations[0]
    assert obs["interp_patched"] is False
    assert obs["flags"] == (True, True, True, True)


# --------------------------------------------------------------------------- #
# config schema / typed settings
# --------------------------------------------------------------------------- #
def test_default_config_has_default_hook_values():
    cfg = _public_config()
    det = cfg["determinism"]
    assert det["sdpa_backend"] == "default"
    assert det["bilinear"] == "default"
    rc.validate_config(cfg, require_external=False)


@pytest.mark.parametrize("sdpa,bilinear", [("math", "default"),
                                           ("default", "deterministic"),
                                           ("math", "deterministic")])
def test_config_override_values_accepted(sdpa, bilinear):
    cfg = rc.without_hydra(_compose([
        f"determinism.sdpa_backend={sdpa}",
        f"determinism.bilinear={bilinear}",
    ]))
    rc.validate_config(cfg, require_external=False)
    assert cfg["determinism"]["sdpa_backend"] == sdpa
    assert cfg["determinism"]["bilinear"] == bilinear


@pytest.mark.parametrize("key,value", [
    ("sdpa_backend", "flash"),
    ("sdpa_backend", True),
    ("sdpa_backend", None),
    ("bilinear", "nearest"),
    ("bilinear", 1),
])
def test_config_wrong_hook_values_rejected(key, value):
    cfg = _public_config()
    cfg["determinism"][key] = value
    with pytest.raises(rc.DeterminismError):
        rc.validate_config(cfg)
    with pytest.raises(rc.DeterminismError):
        rc.determinism_settings(cfg)


@pytest.mark.parametrize("key", ["sdpa_backend", "bilinear"])
def test_config_missing_hook_key_rejected(key):
    cfg = _public_config()
    del cfg["determinism"][key]
    with pytest.raises(rc.DeterminismError):
        rc.validate_config(cfg)


def test_typed_settings_still_exposes_only_runtime_flags():
    # the forward hook is applied by deterministic_forward_context, not by
    # determinism_context, so the typed settings stay exactly the torch flags.
    settings = rc.determinism_settings(_public_config())
    assert "sdpa_backend" not in settings and "bilinear" not in settings


# --------------------------------------------------------------------------- #
# receipt records the hook settings verbatim
# --------------------------------------------------------------------------- #
def test_run_receipt_records_hook_settings(tmp_path):
    import json

    trainer = _trainer_stub(
        {"determinism": {"sdpa_backend": "math", "bilinear": "deterministic",
                         "enabled": True, "warn_only": False}}
    )
    trainer.epoch = 0
    trainer.best_monitor = {"value": None, "global_step": None, "name": "val_teacher_epe"}
    trainer.identity = {"source": "x"}
    receipt = tmp_path / "run_receipt.json"
    trainer.write_run_receipt(receipt, {"status": "finished"})
    payload = json.loads(receipt.read_text())
    assert payload["determinism"]["sdpa_backend"] == "math"
    assert payload["determinism"]["bilinear"] == "deterministic"


# --------------------------------------------------------------------------- #
# CLI source-identity module list
# --------------------------------------------------------------------------- #
def test_cli_source_identity_lists_new_module():
    source = _TRAIN_SCRIPT.read_text()
    assert '"uniflowmatch/training/rgbd_deterministic_ops.py"' in source
    # it must sit inside the `modules = [...]` list used by `_source_identity`.
    modules_block = source[source.index("modules = ["):source.index("]", source.index("modules = ["))]
    assert "rgbd_deterministic_ops.py" in modules_block
    assert "rgbd_trainer.py" in modules_block


def test_cli_identity_modules_exist_on_disk():
    for rel in ("uniflowmatch/training/rgbd_deterministic_ops.py",
                "uniflowmatch/training/rgbd_trainer.py"):
        assert (_REPO / rel).is_file()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

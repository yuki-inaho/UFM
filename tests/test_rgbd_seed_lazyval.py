"""CPU tests for entrypoint seeding + lazy validation reads (step17c fix)."""

from __future__ import annotations

import contextlib
import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "UniCeption"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from uniflowmatch.training.rgbd_trainer import RGBDTrainer  # noqa: E402
from test_rgbd_trainer import _pairs, _trainer  # noqa: E402


def _load_entry():
    spec = importlib.util.spec_from_file_location(
        "ufm_train_rgbd_entry", _REPO / "scripts" / "train_rgbd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stop(RuntimeError):
    pass


def _counting(dataset):
    calls = {"n": 0}

    class _Counting:
        def __len__(self):
            return len(dataset)

        def __getitem__(self, i):
            calls["n"] += 1
            return dataset[i]

    return _Counting(), calls


# --------------------------------------------------------------------------- #
# Fix 1: entrypoint seeds before the first model construction
# --------------------------------------------------------------------------- #
def test_entrypoint_seeds_identical_draws(tmp_path, monkeypatch):
    import uniflowmatch.training.rgbd_trainer as T

    entry = _load_entry()
    cfg_json = tmp_path / "asset_config.json"
    cfg_json.write_text("{}")
    resolved = {
        "seed": 4242,
        "determinism": {
            "enabled": True,
            "warn_only": True,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "allow_tf32": False,
            "sdpa_backend": "default",
            "bilinear": "default",
        },
        "model": {
            "assets": {"config_path": str(cfg_json), "config_sha256": None,
                      "dino_hub_dir": str(tmp_path)},
            "depth_fusion_kwargs": {},
            "dino_python_tree_sha256": "0" * 64,
        },
    }
    args = SimpleNamespace(device="cpu")

    @contextlib.contextmanager
    def _hub(*a, **k):
        yield str(tmp_path)

    monkeypatch.setattr(T, "local_dino_hub", _hub)

    def _collect():
        draws = {}

        def fake_build(*a, **k):
            draws["random"] = random.random()
            draws["numpy"] = float(np.random.rand())
            draws["torch"] = float(torch.rand(1))
            raise _Stop()

        monkeypatch.setattr(T, "build_rgbd_model", fake_build)
        with pytest.raises(_Stop):
            entry._train(resolved, args)
        return draws

    # perturb the external global RNG differently right before each call.
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    random.random(), np.random.rand(), torch.rand(2)
    first = _collect()

    random.seed(999)
    np.random.seed(998)
    torch.manual_seed(997)
    random.random(), np.random.rand(), torch.rand(5)
    second = _collect()

    assert first == second


# --------------------------------------------------------------------------- #
# Fix 2: lazy validation reads
# --------------------------------------------------------------------------- #
def test_val_batches_sized_reads_only_needed():
    data = list(range(64))
    ds, calls = _counting(data)
    batches = RGBDTrainer._val_batches(None, ds, 2)
    assert calls["n"] == 2
    assert batches == [[0], [1]]

    ds_full, calls_full = _counting(list(range(3)))
    batches_full = RGBDTrainer._val_batches(None, ds_full, None)
    assert calls_full["n"] == 3
    assert batches_full == [[0], [1], [2]]


def test_val_batches_iterable_fallback_islice():
    def gen():
        for i in range(10):
            yield i

    assert RGBDTrainer._val_batches(None, gen(), 3) == [[0], [1], [2]]
    assert RGBDTrainer._val_batches(None, gen(), None) == [[i] for i in range(10)]


def test_short_val_bounded_and_population_kept(tmp_path):
    model, opt, trainer = _trainer()
    val, calls = _counting(_pairs(64))
    exp = {"name": "short", "max_updates": 2, "validation_max_pairs": 2,
           "validate_every_updates": 0, "save_every_updates": 0, "early_stop": False}
    res = trainer.run(_pairs(2), val, experiment=exp, checkpoint_dir=tmp_path / "ckpt")
    assert res["status"] == "finished"
    assert calls["n"] == 2
    assert res["val_population"] == 64


def test_full_val_all_items_per_epoch(tmp_path):
    model, opt, trainer = _trainer()
    val, calls = _counting(_pairs(3))
    exp = {"name": "full", "max_updates": None, "epochs": 2,
           "validate_full_each_epoch": True, "early_stop": False,
           "full_budget_fixed": True, "validation_max_pairs": None}
    res = trainer.run(_pairs(2), val, experiment=exp, checkpoint_dir=tmp_path / "ckpt")
    assert res["status"] == "finished"
    assert res["counts"]["val_calls"] == 2
    assert calls["n"] == 6  # 3 reads per epoch x 2 epochs
    assert res["val_population"] == 3

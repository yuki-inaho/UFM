"""CPU contract tests for RGB-D checkpoint / resume / sampler (procedure 17C).

Reuses the tiny real-model fixture from ``tests/test_rgbd_trainer.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "UniCeption", str(Path(__file__).resolve().parent)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import uniflowmatch.training.rgbd_checkpoint as C  # noqa: E402
from uniflowmatch.training.rgbd_optimizer import build_optimizer  # noqa: E402
from uniflowmatch.training.rgbd_trainer import RGBDTrainer  # noqa: E402
from test_rgbd_trainer import _build_tiny_model, _loss_cfg, _opt_cfg, _pairs  # noqa: E402


def _mk(seed=0):
    model = _build_tiny_model()
    cfg = {"seed": seed, "optimizer": _opt_cfg(), "loss": _loss_cfg(False, 0.0),
           "resolved_output_dir": "/tmp/runA", "checkpoint": {"dir": "/tmp/runA/ckpt"}}
    opt = build_optimizer(model, cfg)
    return model, opt, cfg


def _identity(tag="run"):
    return C.build_identity(
        source_sha_list=[["uniflowmatch/training/rgbd_trainer.py", "a" * 64]],
        weight_sha="b" * 64,
        data_sha="c" * 64,
        resume_config_sha="d" * 64,
        uni_ception_pin="pin",
        public_modules=["rgbd_trainer"],
    )


# --------------------------------------------------------------------------- #
# identity / config semantics
# --------------------------------------------------------------------------- #
def test_semantic_config_sha_ignores_output_locations():
    _, _, cfg_a = _mk()
    cfg_b = dict(cfg_a)
    cfg_b["resolved_output_dir"] = "/tmp/other"
    cfg_b["checkpoint"] = {"dir": "/tmp/other/ckpt"}
    cfg_b["external_config"] = "/tmp/other/external.yaml"
    assert C.semantic_config_sha(cfg_a) == C.semantic_config_sha(cfg_b)
    cfg_c = dict(cfg_a)
    cfg_c["loss"] = _loss_cfg(True, 0.1)
    assert C.semantic_config_sha(cfg_a) != C.semantic_config_sha(cfg_c)


def test_resume_identity_mismatch_typed_fail():
    ident = _identity()
    same = _identity()
    C.assert_resume_identity(ident, same)
    for field, bad in (("data_sha", "e" * 64), ("weight_sha", "f" * 64),
                       ("resume_config_sha", "0" * 64)):
        other = _identity()
        other[field] = bad
        other["sha"] = C._sha256_json({k: v for k, v in other.items() if k != "sha"})
        with pytest.raises(C.ResumeIdentityError):
            C.assert_resume_identity(ident, other)


# --------------------------------------------------------------------------- #
# RNG
# --------------------------------------------------------------------------- #
def test_rng_capture_restore_roundtrip():
    torch.manual_seed(123)
    import random
    import numpy as np
    random.seed(1)
    np.random.seed(1)
    state = C.capture_rng_state()
    a = (torch.rand(3), random.random(), float(np.random.rand()))
    C.restore_rng_state(state)
    b = (torch.rand(3), random.random(), float(np.random.rand()))
    assert torch.equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]


# --------------------------------------------------------------------------- #
# sampler cursor
# --------------------------------------------------------------------------- #
def test_sampler_cursor_advances_only_on_commit():
    s = C.StatefulPermutationSampler(5, seed=7)
    it = iter(s)
    first = next(it)
    assert s.cursor == 0  # iterator creation does not advance the cursor
    assert first in s.perm
    s.commit()
    assert s.cursor == 1
    for _ in range(4):
        s.commit()
    assert s.cursor == 0 and s.epoch == 1  # rolled to the next epoch


def test_sampler_state_roundtrip():
    s = C.StatefulPermutationSampler(6, seed=3)
    s.commit(); s.commit()
    sd = s.state_dict()
    s2 = C.StatefulPermutationSampler(6, seed=3)
    s2.load_state_dict(sd)
    assert s2.cursor == s.cursor and s2.epoch == s.epoch and s2.perm == s.perm
    assert list(iter(s2)) == list(iter(s))


# --------------------------------------------------------------------------- #
# checkpoint save / load / resume
# --------------------------------------------------------------------------- #
def test_last_y_roundtrip_and_best_x_resume_rejection(tmp_path):
    model, opt, cfg = _mk()
    ident = _identity()
    tr = RGBDTrainer(cfg, model=model, optimizer=opt, device="cpu", identity=ident)
    tr.train_step(_pairs(2), global_step=1)
    sampler = C.StatefulPermutationSampler(2, seed=1)
    sampler.commit()
    last = tmp_path / "last_y.pt"
    tr.save_last_y(last, sampler_state=sampler.state_dict())
    best = tmp_path / "best_x.pt"
    tr.save_best_x(best, sampler_state=sampler.state_dict())

    payload = C.load_training_checkpoint(last, expected_identity=ident, require_kind="last_y")
    assert payload["kind"] == "last_y" and payload["global_step"] == 1
    assert payload["trainable_mask"] and "model" in payload and "optimizer" in payload

    with pytest.raises(C.ResumeKindError):
        C.load_training_checkpoint(best, require_kind="last_y")
    # the best-x file is still a valid evaluation-only checkpoint
    assert C.load_training_checkpoint(best)["kind"] == "best_averaged_x"


def test_uninterrupted_matches_resume(tmp_path):
    # A: uninterrupted 3 steps
    model_a, opt_a, cfg = _mk(seed=5)
    ident = _identity()
    tr_a = RGBDTrainer(cfg, model=model_a, optimizer=opt_a, device="cpu", identity=ident)
    sampler_a = C.StatefulPermutationSampler(2, seed=1)
    pairs = _pairs(2)
    tr_a.train_step(pairs, global_step=1)
    ckpt = tmp_path / "step1.pt"
    tr_a.save_last_y(ckpt, sampler_state=sampler_a.state_dict())
    tr_a.train_step(pairs, global_step=2)
    tr_a.train_step(pairs, global_step=3)

    # B: fresh trainer, load step1, continue
    model_b, opt_b, cfg_b = _mk(seed=5)
    tr_b = RGBDTrainer(cfg_b, model=model_b, optimizer=opt_b, device="cpu", identity=ident)
    tr_b.load_last_y(ckpt)
    assert tr_b.global_step == 1
    tr_b.train_step(pairs, global_step=2)
    tr_b.train_step(pairs, global_step=3)

    for n, p in model_a.named_parameters():
        assert torch.allclose(p, dict(model_b.named_parameters())[n], atol=0.0, rtol=0.0), n
    assert tr_a.global_step == tr_b.global_step == 3


def test_bad_identity_resume_typed_fail(tmp_path):
    model, opt, cfg = _mk()
    tr = RGBDTrainer(cfg, model=model, optimizer=opt, device="cpu", identity=_identity())
    tr.train_step(_pairs(2), global_step=1)
    sampler = C.StatefulPermutationSampler(2, seed=1)
    ckpt = tmp_path / "last.pt"
    tr.save_last_y(ckpt, sampler_state=sampler.state_dict())

    other = _identity()
    other["data_sha"] = "9" * 64
    other["sha"] = C._sha256_json({k: v for k, v in other.items() if k != "sha"})
    model2, opt2, cfg2 = _mk()
    tr2 = RGBDTrainer(cfg2, model=model2, optimizer=opt2, device="cpu", identity=other)
    with pytest.raises(C.ResumeIdentityError):
        tr2.load_last_y(ckpt)
    # no state was copied: the fresh model still has its own step counter
    assert tr2.global_step == 0


def test_nonfinite_never_advances():
    from uniflowmatch.training.rgbd_losses import NonFiniteLossError

    model, opt, cfg = _mk()
    tr = RGBDTrainer(cfg, model=model, optimizer=opt, device="cpu")
    pairs = _pairs(2)
    # force a non-finite *prediction*: the loss must fail fast and the global
    # step must not advance (no quiet NaN->0, no partial optimizer update).
    original_forward = model.forward

    def bad_forward(*a, **k):
        out = original_forward(*a, **k)
        out.flow.flow_output = out.flow.flow_output.clone()
        out.flow.flow_output[0, 0, 0, 0] = float("inf")
        return out

    model.forward = bad_forward
    with pytest.raises(NonFiniteLossError):
        tr.train_step(pairs, global_step=1)
    assert tr.global_step == 0  # never silently advanced


def test_checkpoint_persists_all_generators(tmp_path):
    model, opt, cfg = _mk()
    tr = RGBDTrainer(cfg, model=model, optimizer=opt, device="cpu")
    tr.train_step(_pairs(2), global_step=1)
    sampler = C.StatefulPermutationSampler(2, seed=1)
    path = tmp_path / "last.pt"
    tr.save_last_y(path, sampler_state=sampler.state_dict())
    payload = C.load_training_checkpoint(path)
    assert payload["extra"].get("aug_generator_state") is not None
    assert payload["extra"].get("dataloader_generator_state") is not None
    assert payload["sampler"]["generator_state"] is not None


# --------------------------------------------------------------------------- #
# run-loop resume: pause after N updates then finish the remaining budget
# --------------------------------------------------------------------------- #
def test_run_pause_then_resume_matches_uninterrupted(tmp_path):
    from test_rgbd_trainer import _build_tiny_model, _loss_cfg, _opt_cfg, _pairs

    def mk():
        # identical init across the three runs (fresh model draws global RNG).
        torch.manual_seed(1234)
        model = _build_tiny_model()
        loss = _loss_cfg(True, 0.1)
        loss["selfsup"]["aug"]["hflip"] = True
        loss["selfsup"]["aug"]["photometric"] = True
        cfg = {"seed": 0, "optimizer": _opt_cfg(), "loss": loss}
        opt = build_optimizer(model, cfg)
        sampler = C.StatefulPermutationSampler(8, seed=0)
        return RGBDTrainer(cfg, model=model, optimizer=opt, device="cpu", seed=0), sampler

    data = _pairs(8)
    val = _pairs(2)
    exp = {"name": "short", "max_updates": 8, "validation_max_pairs": 2,
           "validate_every_updates": 100, "save_every_updates": 100, "early_stop": False}

    tr_a, s_a = mk()
    res_a = tr_a.run(data, val, experiment=exp, checkpoint_dir=tmp_path / "A", sampler=s_a)

    tr_b, s_b = mk()
    res_b = tr_b.run(data, val, experiment=exp, checkpoint_dir=tmp_path / "B",
                     sampler=s_b, pause_after_updates=3)
    assert res_b["status"] == "paused" and res_b["global_step"] == 3
    assert s_b.state_dict()["cursor"] == 3  # sampler cursor advanced with the loop

    tr_c, s_c = mk()
    res_c = tr_c.run(data, val, experiment=exp, checkpoint_dir=tmp_path / "C",
                     sampler=s_c, resume_path=tmp_path / "B" / "last_y.pt")
    assert res_c["status"] == "finished" and res_c["global_step"] == 8
    assert res_c["remaining_at_start"] == 5  # 8 - 3, not a fresh 8

    # index sequence / loss sequence continue exactly where the pause stopped
    assert res_a["index_sequence"] == res_b["index_sequence"] + res_c["index_sequence"]
    la = [round(x, 6) for x in res_a["loss_sequence"]]
    lbc = [round(x, 6) for x in res_b["loss_sequence"] + res_c["loss_sequence"]]
    assert la == lbc

    # model / optimizer / aug-RNG / sampler state all match
    for n, p in tr_a.model.named_parameters():
        assert torch.equal(p, dict(tr_c.model.named_parameters())[n]), n
    assert torch.equal(tr_a._aug_generator.get_state(), tr_c._aug_generator.get_state())
    sa, sc = s_a.state_dict(), s_c.state_dict()
    assert sa["perm"] == sc["perm"] and sa["cursor"] == sc["cursor"] and sa["epoch"] == sc["epoch"]


def test_build_identity_keeps_uniception_pin_and_dino_tree_separate():
    ident = C.build_identity(
        source_sha_list=[["uniflowmatch/training/rgbd_trainer.py", "a" * 64]],
        weight_sha="b" * 64,
        data_sha="c" * 64,
        resume_config_sha="d" * 64,
        uni_ception_pin="1cff08065666d351b2bbdf2db678a4df196afa43",
        dino_python_tree_sha256="e" * 64,
        asset_config_sha="f" * 64,
        asset_sha="b" * 64,
        public_modules=["rgbd_trainer"],
    )
    assert ident["uni_ception_pin"] == "1cff08065666d351b2bbdf2db678a4df196afa43"
    assert ident["dino_python_tree_sha256"] == "e" * 64
    assert ident["uni_ception_pin"] != ident["dino_python_tree_sha256"]
    assert ident["asset_config_sha"] == "f" * 64

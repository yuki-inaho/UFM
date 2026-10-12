"""RGB-D fine-tuning entry (procedure 17C). CPU resolve / preflight / real run.

The config is composed in-process with ``hydra.compose`` (no ``hydra.main``), so
Hydra never auto-creates a run directory.  Every write is routed through a
pre-write output guard that refuses any path inside the public repository.

Modes
-----
* ``--mode resolve``   : compose + resolve + validate + write the resolved yaml
  (Git-external).
* ``--mode preflight`` : resolve, open the real dataset index, assert the
  train/validation counts and the split contract (no model).
* ``--mode train``     : build the real model (pinned local DINO hub), load the
  released weights strictly, and run ``experiment=smoke|short|full``.  **This is
  the real (GPU) run and is executed on the host as procedure 18/19/20.**

Scope note: this unit (17C) only *implements and CPU-tests* the trainer; the real
CUDA run is not started here.

Usage (all private values come from a Git-external config):
    python scripts/train_rgbd.py --mode preflight \
        external_config=/path/private.yaml experiment=short loss=selfsup_on \
        optimizer=amuse
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from uniflowmatch.training.rgbd_config import (
    RGBDConfigError,
    assert_git_external,
    canonical_yaml,
    config_sha256,
    determinism_context,
    load_external_config,
    redact,
    resolve_config,
    save_resolved,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "rgbd_training"


def _compose(overrides):
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="default", overrides=list(overrides))
        return OmegaConf.to_container(cfg, resolve=True)


def _output_guard(resolved) -> Path:
    root = resolved.get("resolved_output_dir")
    out = assert_git_external(root, what="run output")
    return out


def _resolve(args):
    container = _compose(args.overrides)
    external_path = container.get("external_config")
    external = load_external_config(external_path) if external_path else {}
    resolved = resolve_config(container, external)
    _output_guard(resolved)
    return resolved, external


def _preflight(resolved) -> int:
    from uniflowmatch.datasets.rgbd_pairs import RGBDPairDataset

    ds_cfg = resolved["dataset"]
    train = RGBDPairDataset(
        ds_cfg["index_path"],
        split="train",
        data_norm_type=ds_cfg["data_norm_type"],
        transform=ds_cfg["transform"],
        resolution=ds_cfg["resolution"],
        depth_window_m=ds_cfg["depth_window_m"],
        already_dilated=ds_cfg["already_dilated"],
        verify_hashes=False,
        occlusion_supervision_source=ds_cfg["occlusion_supervision_source"],
    )
    val = RGBDPairDataset(
        ds_cfg["index_path"],
        split="val",
        data_norm_type=ds_cfg["data_norm_type"],
        transform=ds_cfg["transform"],
        resolution=ds_cfg["resolution"],
        depth_window_m=ds_cfg["depth_window_m"],
        already_dilated=ds_cfg["already_dilated"],
        verify_hashes=False,
        occlusion_supervision_source=ds_cfg["occlusion_supervision_source"],
    )
    exp_train = ds_cfg.get("expected_train_pairs")
    exp_val = ds_cfg.get("expected_val_pairs")
    if exp_train is not None and len(train) != int(exp_train):
        print(f"PREFLIGHT_ERROR: train pairs {len(train)} != expected {exp_train}", file=sys.stderr)
        return 2
    if exp_val is not None and len(val) != int(exp_val):
        print(f"PREFLIGHT_ERROR: val pairs {len(val)} != expected {exp_val}", file=sys.stderr)
        return 2
    print(f"preflight_ok train={len(train)} val={len(val)} resolution={ds_cfg['resolution']}")
    return 0


#: the *real* UniCeption source commit the run is executed against (verified
#: from the pinned local checkout at run time).
UNICEpTION_PINNED_COMMIT = "1cff08065666d351b2bbdf2db678a4df196afa43"


def _uniception_commit():
    """Read + verify the UniCeption git commit of the pinned local checkout."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    uni = repo / "UniCeption"
    try:
        out = subprocess.run(
            ["git", "-C", str(uni), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        commit = out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        raise RGBDConfigError(f"could not read the UniCeption git commit: {exc}") from exc
    if commit != UNICEpTION_PINNED_COMMIT:
        raise RGBDConfigError(
            f"UniCeption checkout is at {commit}, expected the pinned "
            f"{UNICEpTION_PINNED_COMMIT}"
        )
    return commit


def _source_identity(resolved):
    from uniflowmatch.models.checkpoint import sha256_file
    from uniflowmatch.training import rgbd_checkpoint as C

    repo = Path(__file__).resolve().parents[1]
    modules = [
        "uniflowmatch/training/rgbd_trainer.py",
        "uniflowmatch/training/rgbd_losses.py",
        "uniflowmatch/training/rgbd_checkpoint.py",
        "uniflowmatch/training/rgbd_config.py",
        "uniflowmatch/training/rgbd_optimizer.py",
        "uniflowmatch/training/rgbd_deterministic_ops.py",
        "uniflowmatch/models/ufm.py",
        "uniflowmatch/models/base.py",
        "uniflowmatch/models/depth_fusion.py",
        "uniflowmatch/models/checkpoint.py",
        "uniflowmatch/datasets/rgbd_pairs.py",
        "uniflowmatch/datasets/utils/rgbd_geometry.py",
    ]
    source = [[m, sha256_file(str(repo / m))] for m in modules]
    return C.build_identity(
        source_sha_list=source,
        weight_sha=resolved["model"]["public"]["pretrained_sha256"],
        data_sha=resolved["dataset"]["index_sha256"],
        resume_config_sha=C.semantic_config_sha(resolved),
        # the real UniCeption git commit is the pin; the DINO hub tree is a
        # SEPARATE field (never reused as the pin).
        uni_ception_pin=_uniception_commit(),
        dino_python_tree_sha256=resolved["model"]["dino_python_tree_sha256"],
        asset_config_sha=resolved["model"]["assets"].get("config_sha256"),
        asset_sha=resolved["model"]["public"]["pretrained_sha256"],
        public_modules=[m.split("/")[-1] for m in modules],
    )


def _make_dataset(resolved, split):
    from uniflowmatch.datasets.rgbd_pairs import RGBDPairDataset

    ds = resolved["dataset"]
    return RGBDPairDataset(
        ds["index_path"],
        split=split,
        data_norm_type=ds["data_norm_type"],
        transform=ds["transform"],
        resolution=ds["resolution"],
        depth_window_m=ds["depth_window_m"],
        already_dilated=ds["already_dilated"],
        verify_hashes=ds["verify_hashes"],
        occlusion_supervision_source=ds["occlusion_supervision_source"],
    )


def _train(resolved, args) -> int:
    import random

    import numpy as np
    import torch
    from uniflowmatch.models.checkpoint import sha256_file, verify_file_identity
    from uniflowmatch.models.ufm import UniFlowMatchConfidence
    from uniflowmatch.training import rgbd_checkpoint as C
    from uniflowmatch.training.rgbd_optimizer import build_optimizer
    from uniflowmatch.training.rgbd_trainer import (
        EventsLogger,
        RGBDTrainer,
        build_rgbd_model,
        freeze_encoder,
        load_released_rgbd_weights,
        local_dino_hub,
    )

    # reproducibility: set the cuBLAS workspace env + the deterministic torch
    # backend flags BEFORE the first CUDA use / model construction.  The
    # context restores every global flag in `finally`.
    with determinism_context(resolved):
        assets = resolved["model"]["assets"]
        # verify the asset config identity BEFORE consuming its kwargs.
        expected_config_sha = assets.get("config_sha256")
        actual_config_sha = sha256_file(str(assets["config_path"]))
        if expected_config_sha is not None and actual_config_sha.lower() != str(expected_config_sha).lower():
            print(f"CONFIG_ERROR: asset config sha mismatch {actual_config_sha} != {expected_config_sha}",
                  file=sys.stderr)
            return 2
        with open(assets["config_path"]) as f:
            model_kwargs = json.load(f)

        seed = int(resolved["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # ordering contract: CPU construct -> strict released load -> to(device) ->
        # freeze -> build_optimizer (so AMUSE momentum buffers are created on the
        # model device from the start).
        fusion = resolved["model"]["depth_fusion_kwargs"]
        with local_dino_hub(
            assets["dino_hub_dir"], "dinov2_vitl14",
            resolved["model"]["dino_python_tree_sha256"], pretrained=False,
        ):
            model = build_rgbd_model(model_kwargs, fusion_kwargs=fusion, model_class=UniFlowMatchConfidence)
        report = load_released_rgbd_weights(
            model, assets["asset_path"],
            expected_sha256=resolved["model"]["public"]["pretrained_sha256"],
            expected_bytes=resolved["model"]["public"].get("pretrained_bytes"),
        )
        model.to(args.device)
        freeze_encoder(model)
        optimizer = build_optimizer(model, resolved)

        ckpt_dir = Path(resolved["checkpoint"]["dir"])
        trainer = RGBDTrainer(
            resolved, model=model, optimizer=optimizer, device=args.device,
            identity=_source_identity(resolved),
            events_path=ckpt_dir / "events.jsonl",
        )
        train_ds = _make_dataset(resolved, "train")
        val_ds = _make_dataset(resolved, "val")
        sampler = C.StatefulPermutationSampler(len(train_ds), seed=int(resolved.get("seed", 0)))
        deadline = None
        if os.environ.get("UFM_RGBD_DEADLINE_S"):
            deadline = float(os.environ["UFM_RGBD_DEADLINE_S"])
        trainer.trainable = C.trainable_mask(model)
        result = trainer.run(
            train_ds, val_ds, experiment=resolved["experiment"], checkpoint_dir=ckpt_dir,
            sampler=sampler, deadline_s=deadline, resume_path=args.resume,
            pause_after_updates=args.pause_after_updates,
            receipt_path=ckpt_dir / "run_receipt.json",
        )
        print(f"run_status={result['status']} global_step={result['global_step']} "
              f"migration_missing={report.get('genuine_missing')} gate={report.get('gate_value')}")
        return 0 if result["status"] == "finished" else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--mode", choices=["resolve", "preflight", "train"], default="preflight")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None, help="Git-external last_y checkpoint to resume")
    ap.add_argument("--pause-after-updates", type=int, default=None,
                    help="short-only: pause after N updates (status=paused)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    try:
        resolved, _ = _resolve(args)
    except RGBDConfigError as exc:
        print(f"CONFIG_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # hydra compose failure
        print(f"CONFIG_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.mode == "resolve":
        try:
            path, sha = save_resolved(resolved, resolved["resolved_output_dir"])
        except RGBDConfigError as exc:
            print(f"CONFIG_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(f"resolved_yaml={path}\nconfig_sha256={sha}")
        return 0
    if args.mode == "preflight":
        return _preflight(resolved)
    return _train(resolved, args)


if __name__ == "__main__":
    raise SystemExit(main())

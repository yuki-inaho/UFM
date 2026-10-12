"""CPU tests for the RGB-D Hydra config schema/resolver (UFM_STEP17A_HYDRA_CONFIG).

These tests exercise the *config contract only* (compose / validate / resolve /
redact / safe-write).  17B (AMUSE wrapper) and 17C (trainer) are not implemented
here, so nothing in this file claims that training runs; it only proves the
config resolver and its write-safety guards.

``uniflowmatch/training/rgbd_config.py`` is loaded standalone (importlib path)
so the heavy UFM model stack is never imported for the pure-schema tests.  The
CLI entry is additionally smoke-tested via a subprocess with the real env.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO / "configs" / "rgbd_training"
_MODULE_PATH = _REPO / "uniflowmatch" / "training" / "rgbd_config.py"
_SCRIPT = _REPO / "scripts" / "resolve_rgbd_config.py"

if not _MODULE_PATH.exists():
    raise FileNotFoundError(f"RED: missing actual API {_MODULE_PATH}")


def _load_module():
    spec = importlib.util.spec_from_file_location("rgbd_config_standalone", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_module()

_HEX_A = hashlib.sha256(b"a").hexdigest()
_HEX_B = hashlib.sha256(b"b").hexdigest()

# synthetic (public-test) pair counts -- never the real private counts.
_SYNTH_TRAIN = 4
_SYNTH_VAL = 2


# --------------------------------------------------------------------------- #
# compose helpers
# --------------------------------------------------------------------------- #
def _compose(overrides):
    hydra = pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="default", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def _public_config():
    return rc.without_hydra(_compose([]))


def _external_fixture(tmp_path: Path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"schema_version": 1, "already_dilated": True, "pairs": []}))
    occ = tmp_path / "occ"
    occ.mkdir()
    asset = tmp_path / "model.safetensors"
    asset.write_bytes(b"x")
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    hub = tmp_path / "hub"
    hub.mkdir()
    return {
        "dataset": {
            "index_path": str(index),
            "index_sha256": _HEX_A,
            "expected_train_pairs": _SYNTH_TRAIN,
            "expected_val_pairs": _SYNTH_VAL,
            "occlusion_supervision_path": str(occ),
        },
        "model": {
            "assets": {
                "asset_path": str(asset),
                "config_path": str(cfg),
                "config_sha256": _HEX_B,
                "dino_hub_dir": str(hub),
            }
        },
    }


# --------------------------------------------------------------------------- #
# compose: every group / preset / CLI override
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("experiment", ["smoke", "short", "full"])
@pytest.mark.parametrize("optimizer", ["amuse", "adamw"])
@pytest.mark.parametrize("loss", ["selfsup_off", "selfsup_on"])
def test_compose_all_groups_and_presets(experiment, optimizer, loss):
    cfg = rc.without_hydra(_compose([f"experiment={experiment}", f"optimizer={optimizer}", f"loss={loss}"]))
    rc.validate_config(cfg, require_external=False)
    for group in rc.GROUP_NAMES:
        assert group in cfg
    assert cfg["experiment"]["name"] == experiment
    assert cfg["optimizer"]["name"] == optimizer
    assert cfg["seed"] == 20261012
    assert cfg["fail_nonfinite"] is True
    assert cfg["dataset"]["resolution"] == [448, 336]
    assert cfg["dataset"]["loader"]["num_workers"] == 0
    # generic (non-leaking) held-out role; real pair counts stay external.
    assert cfg["dataset"]["heldout_role"] == "TEST_ONLY_NO_SELECTION"
    assert cfg["dataset"]["expected_train_pairs"] is None
    assert cfg["dataset"]["expected_val_pairs"] is None
    assert cfg["loss"]["homography"]["enabled"] is False
    assert cfg["loss"]["laplace_nll"]["enabled"] is False


def test_default_preset_is_smoke_and_amuse():
    cfg = rc.without_hydra(_compose([]))
    assert cfg["experiment"]["name"] == "smoke"
    assert cfg["optimizer"]["name"] == "amuse"
    assert cfg["optimizer"]["default"] is True


def test_selfsup_on_off_matched_comparison_fields():
    on = rc.without_hydra(_compose(["loss=selfsup_on"]))
    off = rc.without_hydra(_compose(["loss=selfsup_off"]))
    assert on["loss"]["selfsup"]["enabled"] is True
    assert on["loss"]["selfsup"]["weight"] == pytest.approx(0.1)
    assert off["loss"]["selfsup"]["enabled"] is False
    assert off["loss"]["selfsup"]["weight"] == 0.0
    for cfg in (on, off):
        cmp = cfg["experiment"]["comparison"]
        assert cmp["selfsup_matched"] is True
        assert cmp["fixed_seed"] == 20261012
        assert cmp["fixed_optimizer"] is True
        assert cmp["fixed_budget"] is True
        assert cmp["fixed_split"] is True
        assert cmp["run_by_step"] == 21


def test_preset_table_matches_requirements_and_configs():
    smoke = rc.without_hydra(_compose(["experiment=smoke"]))["experiment"]
    short = rc.without_hydra(_compose(["experiment=short"]))["experiment"]
    full = rc.without_hydra(_compose(["experiment=full"]))["experiment"]

    # smoke: 1 update AND validate/save cadence 1 (consistent), 2 val pairs
    assert smoke["max_updates"] == 1
    assert smoke["validate_every_updates"] == 1
    assert smoke["save_every_updates"] == 1
    assert smoke["validation_max_pairs"] == 2
    # short: 8 updates, cadence 4, 4 val pairs
    assert short["max_updates"] == 8
    assert short["validate_every_updates"] == 4
    assert short["save_every_updates"] == 4
    assert short["validation_max_pairs"] == 4
    # full: epoch-bounded, full validation each epoch, no pair cap
    assert full["max_updates"] is None
    assert full["epochs"] == 12
    assert full["validate_full_each_epoch"] is True
    assert full["validation_max_pairs"] is None
    assert full["early_stop"] is False
    assert full["full_budget_fixed"] is False

    # the preset constants table mirrors the composed configs.
    assert rc.EXPERIMENT_PRESETS["smoke"]["validate_every_updates"] == 1
    assert rc.EXPERIMENT_PRESETS["smoke"]["validation_max_pairs"] == 2
    assert rc.EXPERIMENT_PRESETS["short"]["validation_max_pairs"] == 4
    assert rc.EXPERIMENT_PRESETS["full"]["validation_max_pairs"] is None


def test_checkpoint_contract_fields_present():
    cfg = _public_config()
    ckpt = cfg["checkpoint"]
    assert ckpt["save_last_y"] is True and ckpt["save_best_averaged_x"] is True
    assert "last_y" in ckpt["metadata"] and "best_averaged_x" in ckpt["metadata"]
    assert ckpt["metadata"]["last_y"]["param_state"] == "y"
    assert ckpt["metadata"]["best_averaged_x"]["param_state"] == "x"
    strict = ckpt["strict_resume"]
    for key in (
        "require_rng", "require_global_step", "require_epoch",
        "require_sampler_permutation", "require_consumed_cursor",
        "require_config_sha", "require_data_sha", "require_weight_sha",
        "require_source_sha",
    ):
        assert strict[key] is True


def test_amuse_save_extra_attrs_contract():
    cfg = rc.without_hydra(_compose(["optimizer=amuse"]))
    extra = cfg["optimizer"]["train_eval"]["save_extra_attrs"]
    assert extra == [
        "train_mode",
        "weight_decay_at_y",
        "beta1_init",
        "weight_lr_power",
        "warmup_steps",
        "rho",
        "r",
    ]
    # group-state scalars must NOT be listed as extra instance attrs.
    for group_state in ("k", "weight_sum", "beta1", "r_t", "ckp1"):
        assert group_state not in extra


def test_hydra_chdir_false_and_output_git_external():
    raw = yaml.safe_load((_CONFIG_DIR / "default.yaml").read_text())
    hydra_cfg = raw["hydra"]
    assert hydra_cfg["job"]["chdir"] is False
    assert hydra_cfg["output_subdir"] is None
    assert "ufm_rgbd_runs" in hydra_cfg["run"]["dir"]
    assert "ufm_rgbd_runs" in hydra_cfg["sweep"]["dir"]
    assert "oc.env" in hydra_cfg["run"]["dir"]


# --------------------------------------------------------------------------- #
# typed fail-fast
# --------------------------------------------------------------------------- #
def _valid_cfg():
    return _public_config()


def test_unknown_key_fail_fast():
    cfg = _valid_cfg()
    cfg["dataset"]["not_a_real_key"] = 1
    with pytest.raises(rc.UnknownKeyError):
        rc.validate_config(cfg)


def test_top_level_unknown_key_fail_fast():
    cfg = _valid_cfg()
    cfg["mystery"] = True
    with pytest.raises(rc.UnknownKeyError):
        rc.validate_config(cfg)


def test_invalid_shape_fail_fast():
    cfg = _valid_cfg()
    cfg["dataset"]["resolution"] = [448]
    with pytest.raises(rc.ShapeError):
        rc.validate_config(cfg)
    cfg = _valid_cfg()
    cfg["dataset"]["depth_window_m"] = [0.0]
    with pytest.raises(rc.ShapeError):
        rc.validate_config(cfg)


def test_resolution_not_multiple_of_14_fail_fast():
    cfg = _valid_cfg()
    cfg["dataset"]["resolution"] = [448, 340]
    with pytest.raises(rc.ResolutionError):
        rc.validate_config(cfg)


def test_invalid_depth_units_fail_fast():
    cfg = _valid_cfg()
    cfg["dataset"]["depth_units"] = "mm"
    with pytest.raises(rc.DepthError):
        rc.validate_config(cfg)


def test_invalid_depth_window_fail_fast():
    cfg = _valid_cfg()
    cfg["dataset"]["depth_window_m"] = [1.3, 0.0]
    with pytest.raises(rc.DepthError):
        rc.validate_config(cfg)


def test_invalid_precision_fail_fast():
    for key, value in (
        ("master_precision", "float16"),
        ("cpu_precision", "bf16"),
        ("train_precision", "fp16"),
    ):
        cfg = _valid_cfg()
        cfg["model"][key] = value
        with pytest.raises(rc.PrecisionError):
            rc.validate_config(cfg)


def test_warmup_zero_fail_fast():
    cfg = _valid_cfg()
    cfg["optimizer"]["constructor_args"]["warmup_steps"] = 0
    with pytest.raises(rc.OptimizerError):
        rc.validate_config(cfg)


def test_external_scheduler_fail_fast():
    cfg = _valid_cfg()
    cfg["optimizer"]["external_scheduler"] = "cosine"
    with pytest.raises(rc.OptimizerError):
        rc.validate_config(cfg)


def test_save_extra_attrs_type_fail_fast():
    cfg = _valid_cfg()
    cfg["optimizer"]["train_eval"]["save_extra_attrs"] = "beta1"
    with pytest.raises(rc.ShapeError):
        rc.validate_config(cfg)


def test_homography_on_fail_fast():
    cfg = _valid_cfg()
    cfg["loss"]["homography"]["enabled"] = True
    with pytest.raises(rc.LossError):
        rc.validate_config(cfg)


def test_full_max_updates_confusion_fail_fast():
    cfg = _valid_cfg()
    cfg["experiment"]["name"] = "full"
    cfg["experiment"]["max_updates"] = 5
    cfg["experiment"]["epochs"] = 12
    with pytest.raises(rc.ExperimentError):
        rc.validate_config(cfg)

    cfg = _valid_cfg()
    cfg["experiment"]["name"] = "smoke"
    cfg["experiment"]["max_updates"] = 1
    cfg["experiment"]["epochs"] = 12
    with pytest.raises(rc.ExperimentError):
        rc.validate_config(cfg)


def test_full_rejects_validation_pair_cap():
    cfg = _valid_cfg()
    cfg["experiment"]["name"] = "full"
    cfg["experiment"]["max_updates"] = None
    cfg["experiment"]["epochs"] = 12
    cfg["experiment"]["validate_full_each_epoch"] = True
    cfg["experiment"]["validation_max_pairs"] = 4
    with pytest.raises(rc.ExperimentError):
        rc.validate_config(cfg)


def test_smoke_requires_positive_validation_pairs():
    cfg = _valid_cfg()
    cfg["experiment"]["name"] = "smoke"
    cfg["experiment"]["validation_max_pairs"] = None
    with pytest.raises(rc.ExperimentError):
        rc.validate_config(cfg)


def test_fail_nonfinite_required():
    cfg = _valid_cfg()
    cfg["fail_nonfinite"] = False
    with pytest.raises(rc.RGBDConfigError):
        rc.validate_config(cfg)


# --------------------------------------------------------------------------- #
# external resolution / counts
# --------------------------------------------------------------------------- #
def test_public_counts_are_null():
    cfg = _public_config()
    assert cfg["dataset"]["expected_train_pairs"] is None
    assert cfg["dataset"]["expected_val_pairs"] is None
    # compose-only validation must accept the public null counts.
    rc.validate_config(cfg, require_external=False)


def test_resolve_with_external_writes_git_external(tmp_path):
    public = _public_config()
    external = _external_fixture(tmp_path)
    resolved = rc.resolve_config(public, external)
    assert resolved["dataset"]["index_path"] == str(tmp_path / "index.json")
    assert resolved["dataset"]["expected_train_pairs"] == _SYNTH_TRAIN
    assert resolved["dataset"]["expected_val_pairs"] == _SYNTH_VAL
    assert resolved["model"]["assets"]["config_sha256"] == _HEX_B

    out = tmp_path / "resolved"
    path, sha = rc.save_resolved(resolved, out)
    assert Path(path).is_file()
    assert Path(path + ".sha256").read_text().strip() == sha
    assert sha == rc.config_sha256(resolved)
    loaded = yaml.safe_load(Path(path).read_text())
    assert loaded["dataset"]["index_path"] == str(tmp_path / "index.json")


def test_resolve_requires_external_values(tmp_path):
    public = _public_config()
    with pytest.raises(rc.ExternalConfigError):
        rc.resolve_config(public, {"dataset": {"index_path": str(tmp_path / "x")}})


@pytest.mark.parametrize("count", [None, 0, -3, "many"])
def test_resolve_requires_positive_counts(tmp_path, count):
    public = _public_config()
    external = _external_fixture(tmp_path)
    external["dataset"]["expected_train_pairs"] = count
    with pytest.raises(rc.ShapeError):
        rc.resolve_config(public, external)


def test_resolve_rejects_bad_index_sha(tmp_path):
    public = _public_config()
    external = _external_fixture(tmp_path)
    external["dataset"]["index_sha256"] = "not-a-sha"
    with pytest.raises(rc.ExternalConfigError):
        rc.resolve_config(public, external)


def test_load_external_config_roundtrip(tmp_path):
    public = _public_config()
    external = _external_fixture(tmp_path)
    ext_path = tmp_path / "private.yaml"
    ext_path.write_text(yaml.safe_dump(external))
    loaded = rc.load_external_config(str(ext_path))
    resolved = rc.resolve_config(public, loaded)
    assert resolved["dataset"]["index_sha256"] == _HEX_A


# --------------------------------------------------------------------------- #
# privacy / redaction
# --------------------------------------------------------------------------- #
def test_public_config_tree_has_no_host_paths():
    hits = []
    for yml in _CONFIG_DIR.rglob("*.yaml"):
        data = yaml.safe_load(yml.read_text())
        hits.extend((str(yml), h) for h in rc.find_host_paths(data))
    assert hits == [], f"public config leaked host paths: {hits}"


def test_public_config_files_contain_no_forbidden_prefix():
    for yml in _CONFIG_DIR.rglob("*.yaml"):
        text = yml.read_text()
        for prefix in rc.FORBIDDEN_HOST_PREFIXES:
            assert prefix not in text, f"{yml} contains forbidden prefix {prefix}"


def test_public_config_describes_dataset_generically():
    # Public source must have no knowledge of any concrete dataset identity:
    # only a generic name and held-out role may appear, while the real index
    # location/sha and pair counts stay external (null in the public config).
    cfg = _public_config()
    ds = cfg["dataset"]
    assert ds["name"] == "paired_rgbd"
    assert ds["heldout_role"] == "TEST_ONLY_NO_SELECTION"
    assert ds["index_path"] is None
    assert ds["index_sha256"] is None
    assert ds["expected_train_pairs"] is None
    assert ds["expected_val_pairs"] is None


def test_redact_hides_private_paths(tmp_path):
    public = _public_config()
    external = _external_fixture(tmp_path)
    resolved = rc.resolve_config(public, external)
    red = rc.redact(resolved)
    assert red["dataset"]["index_path"] == "<redacted>"
    assert red["model"]["assets"]["asset_path"] == "<redacted>"
    assert red["dataset"]["index_sha256"] == "<redacted>"
    # private pair counts and absolute output/input locations are redacted too.
    assert red["dataset"]["expected_train_pairs"] == "<redacted>"
    assert red["dataset"]["expected_val_pairs"] == "<redacted>"
    assert red["resolved_output_dir"] == "<redacted>"
    if resolved.get("external_config"):
        assert red["external_config"] == "<redacted>"
    dumped = rc.canonical_yaml(red)
    for prefix in rc.FORBIDDEN_HOST_PREFIXES:
        assert prefix not in dumped


# --------------------------------------------------------------------------- #
# save_resolved write safety
# --------------------------------------------------------------------------- #
def _committed_cfg():
    return _public_config()


def test_save_resolved_accepts_git_external_tmp_path(tmp_path):
    out = tmp_path / "safe_out"
    path, sha = rc.save_resolved(_committed_cfg(), out)
    assert Path(path).is_file()
    assert Path(path).parent == out


def test_save_resolved_rejects_repo_inside_path():
    target = _REPO / "private_out_should_not_exist"
    with pytest.raises(rc.UnsafeOutputError):
        rc.save_resolved(_committed_cfg(), target)
    assert not target.exists()


def test_save_resolved_rejects_repo_root_itself():
    with pytest.raises(rc.UnsafeOutputError):
        rc.save_resolved(_committed_cfg(), _REPO)


def test_save_resolved_rejects_symlink_into_repo(tmp_path):
    link = tmp_path / "repo_link"
    link.symlink_to(_REPO, target_is_directory=True)
    with pytest.raises(rc.UnsafeOutputError):
        rc.save_resolved(_committed_cfg(), link)


@pytest.mark.parametrize("filename", ["../escape.yaml", "sub/dir.yaml", "/abs/evil.yaml", ".."])
def test_save_resolved_rejects_unsafe_filename(tmp_path, filename):
    out = tmp_path / "safe_out"
    with pytest.raises(rc.UnsafeOutputError):
        rc.save_resolved(_committed_cfg(), out, filename=filename)
    assert not out.exists()


def test_assert_git_external_message_does_not_leak_path():
    try:
        rc.assert_git_external(_REPO / "secret_dir")
    except rc.UnsafeOutputError as exc:
        assert str(_REPO) not in str(exc)
    else:
        raise AssertionError("expected UnsafeOutputError")


# --------------------------------------------------------------------------- #
# CLI entry (subprocess, real env)
# --------------------------------------------------------------------------- #
def _cli_env(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO), str(_REPO / "UniCeption")])
    env["UFM_RGBD_RUN_ROOT"] = str(tmp_path / "runs")
    env["UFM_RGBD_RESOLVED_ROOT"] = str(tmp_path / "resolved")
    return env


def test_cli_dry_run_exit0_and_redacted(tmp_path):
    ext = tmp_path / "private.yaml"
    ext.write_text(yaml.safe_dump(_external_fixture(tmp_path)))
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "dry_run=true", f"external_config={ext}",
         "experiment=short", "loss=selfsup_on"],
        cwd=str(_REPO), env=_cli_env(tmp_path), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "config_sha256=" in r.stdout
    for prefix in rc.FORBIDDEN_HOST_PREFIXES:
        assert prefix not in r.stdout, "dry-run leaked a private host path"
    # the external config path is redacted (not printed).
    assert str(ext) not in r.stdout


def test_cli_resolve_writes_outside_git(tmp_path):
    ext = tmp_path / "private.yaml"
    ext.write_text(yaml.safe_dump(_external_fixture(tmp_path)))
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), f"external_config={ext}", "experiment=smoke"],
        cwd=str(_REPO), env=_cli_env(tmp_path), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    resolved_root = tmp_path / "resolved"
    yamls = list(resolved_root.rglob("resolved_rgbd_config.yaml"))
    assert len(yamls) == 1
    assert (yamls[0].parent / "resolved_rgbd_config.yaml.sha256").is_file()


def test_cli_rejects_repo_inside_output_before_writing(tmp_path):
    ext = tmp_path / "private.yaml"
    ext.write_text(yaml.safe_dump(_external_fixture(tmp_path)))
    inside = _REPO / "tmp_private_out_should_not_exist"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), f"external_config={ext}",
         f"resolved_output_dir={inside}", "experiment=smoke"],
        cwd=str(_REPO), env=_cli_env(tmp_path), capture_output=True, text=True,
    )
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "CONFIG_ERROR" in r.stderr
    assert not inside.exists(), "CLI wrote into the public repo before rejecting"
    # on rejection nothing (no path, no private counts) is printed to stdout
    assert r.stdout == ""
    assert str(_REPO) not in r.stderr


def test_cli_fail_fast_exit2(tmp_path):
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "dry_run=true",
         "optimizer.constructor_args.warmup_steps=0"],
        cwd=str(_REPO), env=_cli_env(tmp_path), capture_output=True, text=True,
    )
    assert r.returncode == 2, r.stderr
    assert "CONFIG_ERROR" in r.stderr


# --------------------------------------------------------------------------- #
# existing config defaults preserved
# --------------------------------------------------------------------------- #
def test_existing_root_config_defaults_preserved():
    root = yaml.safe_load((_REPO / "configs" / "default.yaml").read_text())
    assert root["seed"] == 42
    assert root["defaults"][0] == {"dataset": "???"}
    assert root["defaults"][1] == {"loss": "???"}
    assert root["defaults"][2] == {"model": "uniflowmatch_covisibility"}


def test_existing_rgbd_model_config_preserved():
    model = yaml.safe_load((_REPO / "configs" / "model" / "uniflowmatch_rgbd.yaml").read_text())
    assert model["use_depth_fusion"] is True
    assert model["depth_fusion_kwargs"]["gate_init"] == 0.0
    assert model["depth_fusion_kwargs"]["depth_hidden_dim"] == 64


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

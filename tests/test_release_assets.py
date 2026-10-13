"""Unit tests for the public release asset helpers (offline, no downloads)."""

import hashlib
import json

import pytest
import torch

from uniflowmatch.models.checkpoint import CheckpointStateError
from uniflowmatch.models.release_assets import (
    ReleaseAssetError,
    load_release_checkpoint,
    parse_sha256sums,
    release_asset_url,
    validate_release_config,
    verify_release_assets,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_release_asset_url_is_public_github_download_url():
    url = release_asset_url("owner/repo", "tag-1", "model.safetensors")
    assert url == "https://github.com/owner/repo/releases/download/tag-1/model.safetensors"


def test_parse_sha256sums_accepts_comments_and_stars():
    digest = "a" * 64
    text = f"# a comment\n{digest}  model.safetensors\n{'b' * 64} *other.bin\n\n"
    sums = parse_sha256sums(text)
    assert sums == {"model.safetensors": digest, "other.bin": "b" * 64}


@pytest.mark.parametrize(
    "text",
    [
        "",  # no entries
        "not-a-digest  file\n",  # invalid sha
        "abc  file\n",  # short sha
        "a" * 64 + "  file\n" + "b" * 64 + "  file\n",  # duplicate
        "a" * 64 + "  file\n" + "extra\n",  # malformed line
    ],
)
def test_parse_sha256sums_rejects_malformed_input(text):
    with pytest.raises(ReleaseAssetError):
        parse_sha256sums(text)


def _write_assets(directory, *, corrupt_digest=False, omit_name=None):
    payloads = {"config.json": b"{}", "manifest.json": b"{}", "weights.bin": b"weights"}
    lines = []
    for name, data in payloads.items():
        digest = "0" * 64 if corrupt_digest and name == "weights.bin" else _sha256(data)
        lines.append(f"{digest}  {name}")
        if name != omit_name:
            (directory / name).write_bytes(data)
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payloads


def test_verify_release_assets_passes_on_consistent_directory(tmp_path):
    payloads = _write_assets(tmp_path)
    sums = verify_release_assets(tmp_path)
    assert sums["weights.bin"] == _sha256(payloads["weights.bin"])


def test_verify_release_assets_detects_digest_mismatch(tmp_path):
    _write_assets(tmp_path, corrupt_digest=True)
    with pytest.raises(ReleaseAssetError):
        verify_release_assets(tmp_path)


def test_verify_release_assets_detects_missing_file(tmp_path):
    _write_assets(tmp_path, omit_name="weights.bin")
    with pytest.raises(ReleaseAssetError):
        verify_release_assets(tmp_path)


def test_validate_release_config_refuses_depth_fusion():
    with pytest.raises(ReleaseAssetError):
        validate_release_config({"encoder_str": "dinov2", "head_type": "dpt", "use_depth_fusion": True})


def test_validate_release_config_requires_architecture_fields():
    with pytest.raises(ReleaseAssetError):
        validate_release_config({"head_type": "dpt"})
    with pytest.raises(ReleaseAssetError):
        validate_release_config({"encoder_str": "dinov2"})


def test_validate_release_config_forces_rgb_only_default():
    kwargs = validate_release_config({"encoder_str": "dinov2", "head_type": "dpt"})
    assert kwargs["use_depth_fusion"] is False


def _tiny_model() -> torch.nn.Module:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model[0].weight.data.zero_()
    model[0].bias.data.zero_()
    return model


def _save_state(path, model):
    from safetensors.torch import save_file

    save_file(dict(model.state_dict()), str(path))


def test_load_release_checkpoint_strict_roundtrip(tmp_path):
    model = _tiny_model()
    state_path = tmp_path / "weights.safetensors"
    _save_state(state_path, model)

    fresh = _tiny_model()
    with torch.no_grad():
        fresh[0].weight.data.fill_(1.0)
    report = load_release_checkpoint(fresh, state_path)
    assert report["unexpected"] == []
    assert torch.equal(fresh[0].weight, model[0].weight)


def test_load_release_checkpoint_rejects_unexpected_tensors(tmp_path):
    from safetensors.torch import save_file

    model = _tiny_model()
    state_path = tmp_path / "weights.safetensors"
    state = {f"0.{key}": value for key, value in model.state_dict().items()}
    state["0.not_a_real_weight"] = torch.zeros(1)
    save_file(state, str(state_path))

    with pytest.raises(CheckpointStateError):
        load_release_checkpoint(model, state_path)


def test_validate_release_config_matches_published_release(tmp_path):
    """The published config (if the assets are cached locally) is loadable."""
    import os
    from pathlib import Path

    from uniflowmatch.models.release_assets import DEFAULT_RELEASE_TAG

    env_dir = os.environ.get("UFM_RELEASE_DIR")
    if env_dir:
        release_root = Path(env_dir).expanduser()
    else:
        release_root = (
            Path(os.environ.get("UFM_RELEASE_CACHE", "~/.cache/ufm_release")).expanduser() / DEFAULT_RELEASE_TAG
        )
    config_path = release_root / "config.json"
    if not config_path.is_file():
        pytest.skip(f"release assets not cached at {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    kwargs = validate_release_config(config)
    assert kwargs["use_depth_fusion"] is False
    assert kwargs["encoder_str"] == "dinov2"

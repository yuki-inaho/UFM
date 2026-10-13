"""
Public GitHub-release asset handling for UFM checkpoints.

This module downloads, verifies and loads the model packages published on a
GitHub *release* page (``config.json`` + ``manifest.json`` + ``SHA256SUMS`` +
``*.safetensors``).  It builds on the strict checkpoint helpers in
:mod:`uniflowmatch.models.checkpoint`:

* every downloaded byte is verified against the release's ``SHA256SUMS`` and the
  weight file is additionally cross-checked against the ``manifest.json``
  digest/size *before* the tensors are read;
* the model is constructed from the release ``config.json`` with
  ``use_depth_fusion=False`` (a release that claims depth fusion is rejected, not
  silently downgraded);
* the weights are loaded with the alias-aware strict loader: zero unexpected
  keys, zero shape mismatches and only the benign storage aliases of a
  deduplicated checkpoint are tolerated.

Only public identifiers (repository, tag, asset names) are used here; nothing
private is embedded.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from uniflowmatch.models.checkpoint import (
    CheckpointError,
    load_released_state,
    load_safetensors_state,
    sha256_file,
    verify_file_identity,
)

# public defaults: the research-prerelease RGB-only package of the fork
DEFAULT_RELEASE_REPO = "yuki-inaho/UFM"
DEFAULT_RELEASE_TAG = "rgb-only-off-20261013"

CONFIG_NAME = "config.json"
MANIFEST_NAME = "manifest.json"
SUMS_NAME = "SHA256SUMS"
WEIGHTS_NAME = "ufm-rgb-only-selfsup-off.safetensors"
# assets that must be listed in SHA256SUMS for a loadable release package
REQUIRED_ASSET_NAMES: Tuple[str, ...] = (CONFIG_NAME, MANIFEST_NAME, WEIGHTS_NAME)

__all__ = [
    "ReleaseAssetError",
    "DEFAULT_RELEASE_REPO",
    "DEFAULT_RELEASE_TAG",
    "REQUIRED_ASSET_NAMES",
    "release_asset_url",
    "parse_sha256sums",
    "validate_release_config",
    "build_release_model",
    "load_release_checkpoint",
    "download_release_assets",
    "verify_release_assets",
    "load_release_model",
]


class ReleaseAssetError(Exception):
    """A release asset is missing, mismatched, or of an unexpected format."""


# --------------------------------------------------------------------------- #
# URLs / checksums
# --------------------------------------------------------------------------- #
def release_asset_url(repo: str, tag: str, name: str) -> str:
    """Public download URL of one asset of a GitHub release."""
    return f"https://github.com/{repo}/releases/download/{tag}/{name}"


def parse_sha256sums(text: str) -> Dict[str, str]:
    """Parse ``SHA256SUMS`` (``<64 hex>  <name>`` per line) into a mapping."""
    sums: Dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ReleaseAssetError(f"malformed SHA256SUMS line {lineno}: {raw!r}")
        digest, name = parts[0].lower(), parts[1].lstrip("*")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ReleaseAssetError(f"SHA256SUMS line {lineno} has no valid SHA256: {raw!r}")
        if name in sums:
            raise ReleaseAssetError(f"SHA256SUMS lists {name!r} twice")
        sums[name] = digest
    if not sums:
        raise ReleaseAssetError("SHA256SUMS contains no entries")
    return sums


# --------------------------------------------------------------------------- #
# config -> model kwargs
# --------------------------------------------------------------------------- #
def validate_release_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a release ``config.json`` and return safe constructor kwargs.

    The released RGB-only package never consumes depth: a config that requests
    ``use_depth_fusion=true`` is rejected instead of being silently downgraded.
    """
    kwargs = dict(config)
    if kwargs.get("use_depth_fusion", False):
        raise ReleaseAssetError(
            "release config requests use_depth_fusion=true; this RGB-only loader "
            "refuses to silently disable a documented model capability"
        )
    kwargs["use_depth_fusion"] = False
    if not kwargs.get("encoder_str"):
        raise ReleaseAssetError("release config is missing 'encoder_str'")
    if not kwargs.get("head_type"):
        raise ReleaseAssetError("release config is missing 'head_type'")
    return kwargs


def build_release_model(config_path: Union[str, Path]):
    """Build a ``UniFlowMatchConfidence`` from a release ``config.json`` (CPU, fp32)."""
    from uniflowmatch.models.ufm import UniFlowMatchConfidence

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    kwargs = validate_release_config(config)
    return UniFlowMatchConfidence(**kwargs)


def load_release_checkpoint(model: torch.nn.Module, weights_path: Union[str, Path], *, verify_fp32: bool = True):
    """Strict-load the release safetensors state into ``model``; returns the load report."""
    state = load_safetensors_state(str(weights_path))
    return load_released_state(model, state, require_fp32=verify_fp32)


# --------------------------------------------------------------------------- #
# download / verify
# --------------------------------------------------------------------------- #
def _download_file(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    """Stream ``url`` to ``dest`` atomically (``.part`` file + ``os.replace``)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "uniflowmatch-release-loader"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(part, "wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        seen = 0
        reported = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            seen += len(chunk)
            if total and seen - reported >= (100 << 20):
                reported = seen
                print(f"  … {dest.name}: {seen / 1e9:.2f} / {total / 1e9:.2f} GB")
    os.replace(part, dest)
    return dest


@dataclass(frozen=True)
class ReleaseAssets:
    """Verified local copy of a release asset set."""

    repo: str
    tag: str
    directory: Path
    config: Path
    manifest: Path
    sums: Path
    weights: Path
    sha256: Mapping[str, str]


def _sums_path(directory: Path) -> Path:
    return directory / SUMS_NAME


def verify_release_assets(directory: Union[str, Path]) -> Dict[str, str]:
    """Verify every asset present in ``directory`` against its ``SHA256SUMS``.

    Every file listed in ``SHA256SUMS`` must exist and match its digest (the
    reverse direction is enforced by :func:`download_release_assets`, which only
    fetches known assets).  Returns ``{name: sha256}`` on success; raises
    :class:`ReleaseAssetError` on the first violation.
    """
    directory = Path(directory)
    sums_file = _sums_path(directory)
    if not sums_file.is_file():
        raise ReleaseAssetError(f"missing {SUMS_NAME}: {sums_file}")
    sums = parse_sha256sums(sums_file.read_text(encoding="utf-8"))
    for name, expected in sums.items():
        path = directory / name
        if not path.is_file():
            raise ReleaseAssetError(f"missing release asset listed in {SUMS_NAME}: {path}")
        try:
            verify_file_identity(str(path), expected_sha256=expected)
        except CheckpointError as exc:
            raise ReleaseAssetError(str(exc)) from exc
    return sums


def download_release_assets(
    directory: Union[str, Path],
    *,
    tag: str = DEFAULT_RELEASE_TAG,
    repo: str = DEFAULT_RELEASE_REPO,
    verify: bool = True,
) -> ReleaseAssets:
    """Download the complete asset set into ``directory`` and verify its digests.

    Existing files are re-used (their digest is still verified), so the call is
    safe to repeat.  ``verify=False`` skips the digest check for the *existing*
    files as well; download-time verification of ``SHA256SUMS`` is always done
    because the published sums are the root of trust here.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    sums_file = _sums_path(directory)
    if not sums_file.is_file():
        print(f"[release] fetching {SUMS_NAME} for {repo}@{tag}")
        _download_file(release_asset_url(repo, tag, SUMS_NAME), sums_file)
    sums = parse_sha256sums(sums_file.read_text(encoding="utf-8"))

    missing_required = [name for name in REQUIRED_ASSET_NAMES if name not in sums]
    if missing_required:
        raise ReleaseAssetError(f"{SUMS_NAME} does not list the required assets: {missing_required}")

    # every file the release publishes (per SHA256SUMS) is fetched and verified
    for name in sums:
        path = directory / name
        if not path.is_file():
            print(f"[release] downloading {name} ({repo}@{tag})")
            _download_file(release_asset_url(repo, tag, name), path)

    if verify:
        verified = verify_release_assets(directory)
    else:
        verified = dict(sums)

    assets = ReleaseAssets(
        repo=repo,
        tag=tag,
        directory=directory,
        config=directory / CONFIG_NAME,
        manifest=directory / MANIFEST_NAME,
        sums=sums_file,
        weights=directory / WEIGHTS_NAME,
        sha256=verified,
    )
    _cross_check_manifest(assets)
    return assets


def _cross_check_manifest(assets: ReleaseAssets) -> None:
    """Cross-check the weight digest/size in ``manifest.json`` against the file."""
    if not assets.manifest.is_file():
        raise ReleaseAssetError(f"missing {MANIFEST_NAME}: {assets.manifest}")
    manifest = json.loads(assets.manifest.read_text(encoding="utf-8"))
    expected_sha = manifest.get("sha256")
    expected_bytes = manifest.get("bytes")
    try:
        identity = verify_file_identity(
            str(assets.weights), expected_sha256=expected_sha, expected_bytes=expected_bytes
        )
    except CheckpointError as exc:
        raise ReleaseAssetError(f"manifest.json disagrees with the weight file: {exc}") from exc
    published = assets.sha256.get(assets.weights.name)
    if published is not None and identity["sha256"].lower() != published.lower():
        raise ReleaseAssetError(
            f"manifest.json and {SUMS_NAME} disagree on {assets.weights.name}: " f"{identity['sha256']} vs {published}"
        )


# --------------------------------------------------------------------------- #
# high-level loader
# --------------------------------------------------------------------------- #
def load_release_model(
    directory: Union[str, Path],
    *,
    tag: str = DEFAULT_RELEASE_TAG,
    repo: str = DEFAULT_RELEASE_REPO,
    device: Union[str, torch.device] = "cpu",
    download_missing: bool = True,
):
    """Load a verified release model on ``device``.

    Existing (verified) assets in ``directory`` are re-used; anything missing is
    downloaded from the public release page when ``download_missing=True``.

    Returns:
        ``(model, info)`` where ``info`` records the verified identity, the load
        report and the release config (architecture, inference resolution).
    """
    directory = Path(directory)
    if download_missing:
        assets = download_release_assets(directory, tag=tag, repo=repo)
    else:
        sums = verify_release_assets(directory)
        assets = ReleaseAssets(
            repo=repo,
            tag=tag,
            directory=directory,
            config=directory / CONFIG_NAME,
            manifest=directory / MANIFEST_NAME,
            sums=directory / SUMS_NAME,
            weights=directory / WEIGHTS_NAME,
            sha256=sums,
        )
        _cross_check_manifest(assets)

    model = build_release_model(assets.config)
    report = load_release_checkpoint(model, assets.weights)
    model = model.eval().to(device)

    with open(assets.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    info = {
        "repo": assets.repo,
        "tag": assets.tag,
        "directory": str(assets.directory),
        "weights": assets.weights.name,
        "weights_sha256": assets.sha256[assets.weights.name],
        "weights_bytes": assets.weights.stat().st_size,
        "config": config,
        "load_report": report,
    }
    return model, info

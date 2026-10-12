"""Runnable entry: compose + validate + resolve the RGB-D training config (17A).

Safe compose CLI: the config is composed in-process with ``hydra.compose``
(no ``hydra.main``), so Hydra never auto-creates a run/sweep output directory.
All writes go through :func:`save_resolved`, which refuses any path inside the
public repository (including via symlink) and any unsafe filename *before*
creating a file.

Usage (all private values come from a Git-external config):
    python scripts/resolve_rgbd_config.py \
        external_config=/path/private.yaml \
        experiment=short optimizer=amuse loss=selfsup_on

Dry-run (validate + print a redacted config, write nothing):
    python scripts/resolve_rgbd_config.py experiment=full dry_run=true

stdout is always redacted: no private host path, index sha, resolved output
dir, or external config path is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from uniflowmatch.training.rgbd_config import (
    RGBDConfigError,
    canonical_yaml,
    config_sha256,
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


def main(argv=None) -> int:
    overrides = list(sys.argv[1:] if argv is None else argv)
    try:
        container = _compose(overrides)
    except Exception as exc:  # hydra/omegaconf compose failure (bad override)
        print(f"CONFIG_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    external_path = container.get("external_config")
    dry_run = bool(container.get("dry_run"))

    try:
        external = load_external_config(external_path) if external_path else {}
        resolved = resolve_config(container, external)
    except RGBDConfigError as exc:
        print(f"CONFIG_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if dry_run:
        print("=== resolved RGB-D config (redacted) ===")
        print(canonical_yaml(redact(resolved)))
        print(f"[dry-run] config_sha256={config_sha256(resolved)}")
        return 0

    try:
        path, sha = save_resolved(resolved, resolved["resolved_output_dir"])
    except RGBDConfigError as exc:
        # never echo the private output path to stdout/stderr
        print(f"CONFIG_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"resolved_yaml={path}")
    print(f"config_sha256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

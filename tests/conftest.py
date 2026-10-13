"""Shared pytest fixtures for the UFM test-suite.

The end-to-end suites (:mod:`tests.test_release_model_e2e`,
:mod:`tests.test_onnx_e2e`) need the published release package.  Set
``UFM_RELEASE_DIR`` to an existing asset directory, otherwise the assets are
downloaded into ``$UFM_RELEASE_CACHE/<tag>`` (default
``~/.cache/ufm_release/<tag>``) on first use.
"""

import os
from pathlib import Path

import pytest

from uniflowmatch.models.release_assets import DEFAULT_RELEASE_TAG


def release_cache_dir() -> Path:
    env_dir = os.environ.get("UFM_RELEASE_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    cache_root = Path(os.environ.get("UFM_RELEASE_CACHE", "~/.cache/ufm_release")).expanduser()
    return cache_root / DEFAULT_RELEASE_TAG


@pytest.fixture(scope="session")
def release_dir() -> Path:
    """Local directory holding the (verified) release assets."""
    return release_cache_dir()


@pytest.fixture(scope="session")
def release_model(release_dir):
    """Verified release model on CPU plus its identity/load report."""
    from uniflowmatch.models.release_assets import load_release_model

    model, info = load_release_model(release_dir, tag=release_dir.name, device="cpu")
    return model, info

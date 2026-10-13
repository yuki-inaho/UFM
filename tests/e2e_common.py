"""Shared constants/helpers for the end-to-end test modules."""

from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIR_DIR = REPO_ROOT / "examples" / "image_pairs"
FIRE_PAIR = ("fire_academy_0.png", "fire_academy_1.png")

# the fine-tuned operating point documented in the release model card
RELEASE_INFERENCE_RESOLUTION = (448, 336)  # (width, height)
ONNX_INPUT_SHAPE = (336, 448)  # (height, width)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    assert image is not None, f"Could not read image: {path}"
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Released checkpoints, ONNX export and ONNX Runtime (optional mode)

This document covers the public **GitHub release** checkpoints of this
repository (e.g. `rgb-only-off-20261013`), how to run them on a single GPU such
as an RTX 2070 (8 GB), and how to use the optional ONNX pipeline (`onnx`,
`onnxsim`, `onnxruntime-gpu`).

Nothing here is required for the default installation: the release loader is
plain Python (stdlib + `torch` + `safetensors`) and the ONNX toolchain lives in
optional extras.

---

## 1. Release packages and the verification contract

A release package contains:

| Asset | Purpose |
|-------|---------|
| `config.json` | model architecture kwargs for `UniFlowMatchConfidence` |
| `manifest.json` | weight digest + size + whitelisted training facts |
| `SHA256SUMS` | digests of every published file |
| `*.safetensors` | model state tensors only (strict-loadable) |
| `model_card.md` | human-readable usage notes |

`uniflowmatch.models.release_assets` enforces the following before any tensor
is read:

1. every file listed in `SHA256SUMS` is present and matches its digest;
2. the weight file's SHA256 **and byte size** match `manifest.json`, and
   `manifest.json` / `SHA256SUMS` agree with each other;
3. `config.json` is built with `use_depth_fusion=False`; a release that claims
   depth fusion is **rejected**, never silently downgraded;
4. the state is loaded with the alias-aware strict loader
   (`uniflowmatch.models.checkpoint.load_released_state`): zero unexpected
   keys, zero shape mismatches, benign storage aliases of a deduplicated
   checkpoint tolerated.

```python
from uniflowmatch.models.release_assets import load_release_model

model, info = load_release_model("~/.cache/ufm_release/rgb-only-off-20261013")
print(info["weights_sha256"], info["load_report"]["loaded"])
```

## 2. Running the released checkpoint

```bash
# Downloads + verifies the assets on first use, runs one image pair, writes
# flow / covisibility / warped visualizations and run_info.json.
python scripts/release_infer.py -o outputs/release_torch

# explicit assets dir, custom operating resolution, custom pair
python scripts/release_infer.py \
    --release-dir ~/.cache/ufm_release/rgb-only-off-20261013 \
    --width 448 --height 336 \
    --source path/to/a.png --target path/to/b.png -o outputs/release_torch
```

Useful environment variables for tests/scripts:

* `UFM_RELEASE_DIR` – existing asset directory (skips the default cache path);
* `UFM_RELEASE_CACHE` – cache root (default `~/.cache/ufm_release`).

### Device handling

`predict_correspondences_batched` follows the **model** device: passing CPU
`uint8` tensors (`torch.from_numpy(np.array(PIL.Image.open(...)))`) to a model
that was moved to `cuda` works as written in the release model card. Assigning
`model.inference_resolution = (width, height)` **rebuilds the image scaler**, so
the override is effective after construction.

### Observed on an RTX 2070 (8 GB, 2026-10, fp32, 448x336)

* peak GPU memory ≈ 1.9 GiB for a 1080x580 pair;
* ~0.3 s per pair (PyTorch, autocast fp16 prediction path).

These numbers describe this machine and are not a quality claim; the released
model card carries the training-time monitor metric and its caveats.

## 3. ONNX export, onnxsim optimization and ONNX Runtime

```bash
pip install -e '.[onnx-gpu]'          # toolchain + CUDA runtime

# export the verified release model at its fine-tuned resolution,
# simplify with onnxsim, then verify against PyTorch
python scripts/export_onnx.py \
    --release-tag rgb-only-off-20261013 \
    --height 336 --width 448 \
    -o exports/ufm_rgb_only_336x448.onnx \
    --simplify --verify
```

What the script does:

1. builds + strict-loads the release checkpoint (same verification as above);
2. exports the **core network** (encoder + info sharing + heads) with a static
   input shape; normalization/resizing and the mapping of predictions back to
   the original image space stay in Python (`uniflowmatch.utils.onnx`);
3. `--simplify` runs [onnxsim](https://github.com/daquexian/onnx-simplifier)
   (shape inference + constant folding, `onnxsim>=0.8`); with
   `--simplify-check-n N` onnxsim additionally re-verifies the outputs through
   ONNX Runtime;
4. `--verify` runs the full pipeline through ONNX Runtime and compares against
   the PyTorch pipeline (the CUDA execution provider is used when available).

### Measured (RTX 2070, opset 18, 336x448 input)

| Artifact | Nodes | Initializers | Total size |
|----------|------:|-------------:|-----------:|
| exported (`--output`) | 1453 | 650 | 1716.5 MB (3.4 MB graph + 1711 MB external data) |
| after `--simplify` | 1411 | 642 | 1715.5 MB (single file) |

onnxsim removes redundant nodes/initializers and folds constants; the total
weight bytes dominate, so the size delta is small (−0.06%) and **numerical
equality is preserved** (verified: max flow difference 2.4e-4 px, max
covisibility difference 6e-5 between the original and the simplified graph).
Weight-format changes (fp16 quantization) are deliberately out of scope of the
`--simplify` step.

Parity notes:

* ONNX Runtime (CUDA) vs PyTorch **fp32**: max flow difference ≈ 2.4e-4 px,
  max covisibility difference ≈ 6e-5.
* ONNX Runtime is fp32; PyTorch inference on CUDA uses an autocast fp16
  prediction wrapper, so ONNX-vs-GPU-PyTorch differences are slightly larger
  (max ≈ 0.1 px). The ONNX graph is not less precise; it just does not run the
  fp16 autocast wrapper.

## 4. Optional dependency modes

| Mode | Install | Contents |
|------|---------|----------|
| export/simplify only | `pip install -e '.[onnx]'` | `onnx`, `onnxscript`, `onnxsim` |
| CUDA inference | `pip install -e '.[onnx-gpu]'` | the above + `onnxruntime-gpu` |
| CPU inference | `pip install -e '.[onnx]' && pip install onnxruntime` | no CUDA dependency |

Never install `onnxruntime` and `onnxruntime-gpu` into the same environment.
Without any runtime, `scripts/export_onnx.py` (export / simplify) still works;
only `--verify` and `scripts/release_infer.py --backend onnx` need a runtime.

```bash
# ONNX Runtime backend end to end
python scripts/release_infer.py \
    --backend onnx --onnx-model exports/ufm_rgb_only_336x448_simplified.onnx \
    --width 448 --height 336 -o outputs/release_onnx
```

The `--backend` flag is the "mode" switch: `torch` (default) uses the PyTorch
model, `onnx` routes the core network through ONNX Runtime
(`CUDAExecutionProvider` when available) while preprocessing/postprocessing stay
in the shared Python implementation.

## 5. Automated tests

```bash
# offline unit tests (no downloads)
python -m pytest -m "not e2e"

# end-to-end: release verification, strict load, inference, ONNX export,
# onnxsim simplification and ONNX Runtime (CUDA) parity
UFM_RELEASE_DIR=~/.cache/ufm_release/rgb-only-off-20261013 \
    python -m pytest -m e2e -v -s
```

The e2e tests download the public release assets (≈1.77 GB) when the cache is
empty and skip the ONNX tests when `CUDAExecutionProvider` is unavailable.

## 6. Limitations

* The ONNX graph has a **static input shape**; height/width must be divisible by
  the patch size 14 (the scripts use the model resolution by default).
* Dynamic shapes, TensorRT/OpenVINO execution providers and fp16 quantization
  are not part of this pipeline.
* onnxsim is a graph-level optimizer: expect few removed nodes and a near-zero
  size delta on weight-bound models.

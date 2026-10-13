# RGB and RGB-D fine-tuning and evaluation for UFM

This document explains how to fine-tune and evaluate the released UFM model in
**two modes that share one code path**: a **generic RGB-only** mode
(`use_depth_fusion: false`, no depth consumed) and an **RGB-D** mode (zero-gated
post-encoder depth residual, gate init `0.0`).  It is written for a generic user
who has their own captures already prepared, with or without depth.  The doc
covers both modes throughout and names which mode each contract applies to.  No
dataset identifier, host path, calibration array, or result number is embedded
here; every private value is supplied at run time through a Git-external
descriptor.

The RGB-D support is **additive**: the RGB API and the default RGB path are
**preserved**, and the RGB-D functionality is added through **optional hooks**
(a post-encoder depth residual plus new dataset / trainer / CLI modules) rather
than by rewriting the RGB code.  With no depth, `use_depth_fusion=false`, or a
gate value of `0`, the RGB-D model is numerically identical to the RGB-only
model (see *RGB parity* below).

> Scope note: this file documents the interface and contracts enforced by the
> current source.  It makes no claim about model quality or about any completed
> training run.

---

## 1. Concepts and layout

| Piece | Location |
|-------|----------|
| Paired RGB-D loader | `uniflowmatch/datasets/rgbd_pairs.py` |
| Pixel geometry helpers | `uniflowmatch/datasets/utils/rgbd_geometry.py` |
| 3x3 min-valid depth dilation | `uniflowmatch/utils/depth.py` |
| Zero-gated depth fusion module | `uniflowmatch/models/depth_fusion.py` |
| Model hook (post-encoder residual) | `uniflowmatch/models/ufm.py` |
| Trainer | `uniflowmatch/training/rgbd_trainer.py` |
| Losses (supervised + self-supervised) | `uniflowmatch/training/rgbd_losses.py` |
| Optimizer wrapper (AMUSE / AdamW) | `uniflowmatch/training/rgbd_optimizer.py` |
| Checkpointing / strict resume | `uniflowmatch/training/rgbd_checkpoint.py` |
| Config schema / external resolution | `uniflowmatch/training/rgbd_config.py` |
| Determinism runtime ops | `uniflowmatch/training/rgbd_deterministic_ops.py` |
| Hydra config tree | `configs/rgbd_training/` |
| Train entry point | `scripts/train_rgbd.py` |
| Resolve / dry-run entry point | `scripts/resolve_rgbd_config.py` |
| Evaluation runner | `scripts/evaluate_rgbd.py` |
| Pure evaluation metrics | `uniflowmatch/evaluation/rgbd_evaluation.py` |
| Model-grid -> native scene-flow adapter | `uniflowmatch/evaluation/scene_flow_adapter.py` |
| Vendored AMUSE (Apache-2.0) | `uniflowmatch/training/vendor/amuse/` |

### Environment

Install the package as usual, and make the bundled `UniCeption` checkout
importable (it provides `uniception.models.encoders...`):

```bash
pip install -e .
export PYTHONPATH="$PWD/UniCeption:$PYTHONPATH"
python -c "import uniflowmatch, uniception; print('ok')"
```

`torch` is required for training/evaluation; the pure geometry helpers
(`rgbd_geometry.py`, `depth.py`) only need `numpy`/`cv2` (and lazily `torch`).

---

## 2. The paired RGB-D index

The loader reads a single **JSON index** describing all pairs.  There is no
fixed on-disk directory layout: each file is referenced by a path relative to
the index file's directory (`root` defaults to the index's parent, and every
path is resolved against it).  A synthetic example (IDs are placeholders):

```json
{
  "schema_version": 1,
  "already_dilated": true,
  "dataset_name": "example_paired_rgbd",
  "depth_window_m": [0.0, 1.3],
  "pairs": [
    {
      "pair_id": "pair_0001",
      "capture": "scene000",
      "split": "train",
      "is_synthetic": false,
      "views": [
        {
          "frame_index": 0,
          "rgb_path": "scene000/rgb/000000.png",
          "raw_depth_mm_path": "scene000/depth_raw/000000.png",
          "dilated3_depth_mm_path": "scene000/depth_dilated3/000000.png",
          "raw_depth_mask_path": "scene000/mask_raw/000000.png",
          "dilated3_depth_mask_path": "scene000/mask_dilated3/000000.png",
          "depth_provenance_path": "scene000/depth_prov/000000.png",
          "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
          "K_known": true
        },
        {
          "frame_index": 1,
          "rgb_path": "scene000/rgb/000001.png",
          "raw_depth_mm_path": "scene000/depth_raw/000001.png",
          "dilated3_depth_mm_path": "scene000/depth_dilated3/000001.png",
          "raw_depth_mask_path": "scene000/mask_raw/000001.png",
          "dilated3_depth_mask_path": "scene000/mask_dilated3/000001.png",
          "depth_provenance_path": "scene000/depth_prov/000001.png",
          "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
          "K_known": true
        }
      ],
      "flow_fwd": {
        "flow_path": "scene000/flow/0001_fwd.npz",
        "valid_path": "scene000/flow/0001_fwd_valid.png",
        "covis_path": "scene000/flow/0001_fwd_covis.png",
        "reference_hw": [224, 288]
      },
      "flow_bwd": {
        "flow_path": "scene000/flow/0001_bwd.npz",
        "valid_path": "scene000/flow/0001_bwd_valid.png",
        "covis_path": "scene000/flow/0001_bwd_covis.png",
        "reference_hw": [224, 288]
      },
      "sha256": {
        "scene000/rgb/000000.png": "<64 lowercase hex>",
        "scene000/flow/0001_fwd.npz": "<64 lowercase hex>"
      }
    }
  ]
}
```

### Required fields and types

Top level (validated by `validate_index`):

* `schema_version` — must be exactly `1`.
* `already_dilated` — must be `true`.  The loader **never dilates** depth; a
  doubly-dilated store is forbidden.
* `pairs` — a non-empty list.  Each pair requires `pair_id`, `capture`,
  `split` (one of `train`, `val`, `test`), `views` (exactly two), and both
  `flow_fwd` / `flow_bwd`.
* optional `depth_window_m` — a `(near, far)` metre pair with `near >= 0` and
  `far > near`.
* optional `native_hw` — `[H, W]`; when absent it is derived per pair from the
  RGB shape.

Each **view** requires: `frame_index`, `rgb_path`, `raw_depth_mm_path`,
`dilated3_depth_mm_path`, `raw_depth_mask_path`, `dilated3_depth_mask_path`,
`depth_provenance_path`, `K`.  `K` is a `3x3` intrinsics array; `K_known`
(defaults `true`) marks a known finite `K`.  When `K_known=false`, `K` is
replaced by all-`NaN` and the intrinsics are treated as unknown.

Each **flow direction** requires `flow_path` (a `.npz` with a key `flow` of
shape `(2, H, W)`), `valid_path`, `covis_path`, and `reference_hw` (`[H, W]`).

### Units and numeric conventions

* **Depth files are `uint16` millimetres**, single channel.  If `verify_hashes`
  is on (default) **every file the sample actually loads** must have a `sha256`
  entry (64 lowercase hex) in the pair's `sha256` table, and the bytes must
  match.  That covers each per-view RGB / depth / mask / provenance path that is
  read and every `flow_fwd` / `flow_bwd` `*.npz`.  The synthetic snippet above
  is abbreviated and **omits most of those entries**; a real index must list all
  of them.
* The converter uses a fixed factor `mm * 0.001 = metres`.  The sentinel invalid
  values are `0` and `65535`; both become `0.0` m and are flagged invalid.
* **Masks are read as bool** (`pixel > 0`).  Depth, masks and provenance are
  warped with **nearest** sampling only — no interpolated (fabricated) depth is
  ever created.  RGB alone may be resampled bilinearly.
* Flow masks (`valid`, `covis`) are bool and must match `reference_hw`.

### Fail-fast behaviour

Malformed or non-finite samples raise typed errors instead of being skipped:
`RGBDIndexError` (index contract), `SplitLeakError` (a frame shared across
splits), `HashMismatchError` (bad/missing sha), `ZeroValidLabelsError` (a
direction with zero valid flow labels), and `RGBDSampleError` (missing/ corrupt
file, shape mismatch, non-finite value at a valid label).  Silent skipping is
forbidden (`on_zero_valid` must stay `"error"`).

---

## 3. Geometry and depth preparation

All helpers live in `uniflowmatch/datasets/utils/rgbd_geometry.py` and follow a
single convention: a `3x3` affine `A` maps homogeneous **source** pixels
`(u, v, 1)` to destination pixels, and the transformed intrinsic is
`K' = A @ K`.

* **Raw vs dilated depth.**  The raw mapped depth and the pre-computed
  `3x3 min-valid dilated` depth are stored and returned separately.  The loader
  reads the dilated arrays; it does not dilate.  The reference implementation of
  the dilation is `min_valid_depth_dilate` in `uniflowmatch/utils/depth.py`:
  for every pixel it takes the **minimum** (closest-to-camera) valid depth in
  the `3x3` neighbourhood, never a maximum.  Out-of-image neighbours are treated
  as invalid, and invalid inputs (`0`/`65535` for `uint16`, or `0`/ non-finite /
  negative for metric floats) are never used as sources or propagated.
* **Raw valid vs window.**  `raw_numeric_valid` comes from the sanitiser; the
  declared measurement mask is ANDed with it, so a declared-valid pixel can
  never resurrect a numeric-invalid one and a declared-invalid pixel can never
  pass.  The input window `near < z < far` (an explicit configurable, default
  `0 < z < 1.3` m) is preserved as a **separate** mask distinct from the raw
  measurement mask.  The window is evaluated in float64 metres so an exact
  `1.3` boundary is excluded.
* **Two-stage flow resize.**  Native RGB / raw depth live at the capture
  resolution while the teacher flow already lives at `flow_fwd.reference_hw`.
  `plan_affines` first resizes images/depth/K from native to the reference frame
  (`stage1`) and then applies the augmentation in that frame (`stage2`).  Flow
  is transformed by `stage2` only, so an already-resized flow is **never scaled
  twice**.  Resize uses the `align_corners=False` half-pixel convention.
* **Augmentation.**  Order is resize -> crop -> hflip -> vflip.  A horizontal
  flip reflects RGB pixels while keeping camera coordinates: `fx` becomes
  negative and `cx' = (W-1) - cx`; it negates `dx` and keeps `dy`, and the
  source/target roles are **never swapped**.
* **`K` / `A` provenance.**  One affine is applied to RGB, raw depth, dilated
  depth, masks, provenance **and** `K`.  `K` therefore records the intrinsic of
  the transformed image (`K' = A @ K`).  `K` is known finite when `K_known`, and
  all-`NaN` otherwise.
* **Source/target geometry.**  Each pair yields two views (source, target).
  `view[0]` carries the forward teacher flow and `view[1]` the backward teacher
  flow; a fov mask (`valid & endpoint-in-bounds`) and a non-occluded mask
  (`covis & endpoint-in-bounds`) are produced in the same geometry.

---

## 4. Model: post-encoder gated depth residual

`uniflowmatch/models/depth_fusion.py` implements

```
h_fused = h_rgb + g * (masked_depth_adapter(depth, depth_mask))
```

* The residual is added **after** the RGB encoder (outside it), so gradients to
  the adapter and gate are never severed by a frozen encoder.
* The gate `g` is a single learnable scalar initialised to exactly `0.0`.
* `depth_mask` is mandatory whenever depth is supplied; invalid pixels are
  zeroed before the stem and masked on the output grid, so an all-invalid mask
  gives exactly `0` contribution even with a non-zero adapter bias.
* Non-finite depth, wrong shape/dtype/device, or a missing mask fail fast with
  typed `DepthFusion*Error` exceptions.
* The depth stem treats depth as a **per-pixel scalar feature map**.  **It does
  not unproject depth to 3D rays and does not consume the intrinsics.**  The
  loader keeps `K` and the forward API carries `camera_intrinsics` through for
  the loader/trainer, but `K` is informational for the depth stem.

### RGB parity

Two independent guarantees make the RGB-D path reduce exactly to RGB:

1. **Gate 0:** with `g == 0` the residual is `0 * finite == 0`, so the fused
   feature equals `h_rgb` bit-exactly — even if depth is present.
2. **Depth absent / all-invalid mask:** with no depth the fusion is the
   identity; with an all-invalid mask the contribution is exactly `0`.

The evaluation runner checks this directly: `rgb_pretrained` and
`rgbd_pretrained_init` (the released RGB weights migrated to RGB-D with the gate
at `0`) must match to `atol = 0` (`PARITY_ATOL`).  Depth priority read by the
model is `depth` -> `dilated_depth_m` -> `depthmap`; mask priority is
`depth_mask` -> `dilated_depth_window_mask` -> `valid_mask`.

---

## 5. Pins live in an external descriptor

Public source never embeds private asset locations.  The public model config
(`configs/rgbd_training/model/ufm_rgbd.yaml`) records only public identity
(Hugging Face model id / revision / weight SHA / byte size, the DINO python-tree
SHA, and the architecture kwargs).  The sibling `configs/rgbd_training/model/ufm_rgb_only.yaml`
records the same public identity for the **true RGB-only** mode.  The following are **PRIVATE** and must come
from a Git-external resolved config (or, for evaluation, a Git-external JSON
descriptor):

* `dataset.index_path`, `dataset.index_sha256`,
  `dataset.expected_train_pairs`, `dataset.expected_val_pairs`,
  `dataset.occlusion_supervision_path`
* `model.assets.asset_path`, `model.assets.config_path`,
  `model.assets.config_sha256`, `model.assets.dino_hub_dir`

These map, for evaluation, onto the descriptor keys `dataset.*`, `assets.*`
(`asset_path`, `asset_sha256`, `config_path`, `config_sha256`, `dino_hub_dir`,
`dino_python_tree_sha256`), `model.*`, and `variants.*`.

The **loader** (`RGBDPairDataset`) only reads the RGB-D index and the
referenced RGB / depth / mask / flow files; it does **not** construct any
pretrained model.  The **pretrained model construction** is what consumes the
**pinned local** `facebookresearch/dinov2` hub tree (`local_dino_hub`, verified
aggregate python-tree SHA, `pretrained=False`); a moving `github/main` download
is never used.  Released weights are verified by SHA256/byte size before the
bytes are read.

The **source pin** is the real `UniCeption` submodule commit of the pinned local
checkout, read at run time and verified against the constant in the training
entry point.  It is advanced together with the upstream integration
(`ee7fa0b50d8295b3f0485e42c1d133fb534bfce7`); the previous pin is kept only as
history.  Because the pin is part of the strict resume identity, a checkpoint
written under the previous source/pin is rejected before any state is copied --
the source-SHA / pin guard is intentionally **never loosened**.  Resuming such a
checkpoint therefore requires **both the original public source and the original
`UniCeption` checkout** it was produced with (the original source SHA list and
the original submodule commit); the guard must **not** be relaxed to push an old
checkpoint into the new source.

---

## 6. Hydra composition and presets

The config is composed in-process with `hydra.compose` (never `hydra.main`), so
no run/sweep directory is auto-created.  Groups: `dataset`, `model`,
`optimizer`, `loss`, `experiment`.

Presets (`configs/rgbd_training/experiment/`):

| Preset | Updates | Validation | Save | Notes |
|--------|---------|-----------|------|-------|
| `smoke` | `max_updates: 1` | every update, `validation_max_pairs: 2` | every update | fast gate |
| `short` | `max_updates: 8` | every 4 updates, `validation_max_pairs: 4` | every 4 updates | bounded gate |
| `full` | epoch-bounded, no `max_updates` | full population every epoch | every epoch | `full_budget_fixed` must be `true` first |

The **canonical FULL budget is explicitly 10 epochs** over the entire eligible
train population, fixed from the approved matched capacity measurement and
applied as an external Hydra override.  The `epochs` value shipped inside
`experiment/full` is only a **public placeholder** and must never be read as the
measured budget; the resolved budget is what the training entry point records in
the resume identity.

### Model presets (generic RGB / RGB-D modes)

The modality is selected by the model preset and composes with any dataset /
optimizer / loss / experiment group:

| Preset | Mode |
|--------|------|
| `configs/rgbd_training/model/ufm_rgb_only.yaml` | **true RGB-only** (`use_depth_fusion: false`, no depth kwargs) |
| `configs/rgbd_training/model/ufm_rgbd.yaml` | **RGB-D** (`use_depth_fusion: true`, zero-gated depth residual) |

Both presets use the same `UniFlowMatchConfidence` architecture and the same
released RGB weights; the true RGB-only model consumes no depth field at all.

Validation rules (fail-fast):
* `smoke`/`short` require a positive `max_updates` and an explicit positive
  `validation_max_pairs`.
* `full` must not set `max_updates`; it requires a positive `epochs`,
  `validate_full_each_epoch: true`, `validation_max_pairs: null`, and
  `full_budget_fixed: true` (set only after the capacity measurement).
* `early_stop` must be `false`.
* `depth_units` must be `metre(.s)`, `already_dilated`/`verify_hashes` `true`,
  resolution a multiple of the patch size `14`, gate init `0.0`.

Example overrides:

```bash
python scripts/resolve_rgbd_config.py dry_run=true experiment=full
python scripts/resolve_rgbd_config.py \
    external_config=../private/resolved.yaml \
    experiment=short optimizer=amuse loss=selfsup_on
```

`resolve` + `preflight` are CPU-only; the real training run requires CUDA.

---

## 7. Vendored AMUSE (Apache-2.0 provenance)

The AMUSE optimizer is vendored **unmodified** under
`uniflowmatch/training/vendor/amuse/` (sources: `AMUSE.py`, `LICENSE`,
`NOTICE`).  Provenance is recorded in that directory's `NOTICE`: upstream
project AMUSE, source repository, the pinned commit, commit date, and the
Apache License 2.0.  Byte-exact SHA-256 digests for `AMUSE.py` and `LICENSE`
are listed there.  All RGB-D wrapper behaviour (parameter grouping, train/eval
mode control, strict serialization) lives in the sibling wrapper module
`uniflowmatch/training/rgbd_optimizer.py`, **not** in `AMUSE.py`.

AMUSE is schedule-free: `schedule_free: true`, `external_scheduler: null`, and
`warmup_steps > 0` are enforced.  An AdamW comparison preset is also provided.

---

## 8. Backbone frozen vs the FULL budget

The encoder is frozen (`requires_grad_(False)` + `eval()`, re-applied after
every `model.train()` because `model.train()` would otherwise flip it back).
The info-sharing module, output head, uncertainty head and depth fusion are
trainable.

Importantly, **`experiment=full` is not merely a "backbone run"**: it trains
**all trainable weights** (info-sharing + heads + depth adapter/gate) over the
**entire eligible train population** for the configured epochs, with full
validation each epoch and no `max_updates` cap.  The FULL budget is fixed from
a capacity measurement before the run starts; a shortened run must never be
reported as FULL.

---

## 9. Checkpoints: averaged-x evaluation vs last-y resume

Checkpointing cleanly separates two artifacts:

* `last_y` — the **training iterate**, used **only** to resume.
* `best_averaged_x` — the **averaged iterate**, used **only** for evaluation.

AMUSE is schedule-free and exposes an averaged evaluation iterate `x` alongside
the training iterate `y`.  Validation runs the averaged `x` state
(`eval_returns_averaged_x`); the training `y`+`z` are saved for resume.

Resume is **strict**: a semantic identity (public source SHA list, released
weight SHA, dataset-index SHA, resume-config SHA, UniCeption pin; the local DINO
python-tree SHA and asset config SHA are recorded as separate fields) is
compared **before** any state is copied.  The UniCeption pin is the real
submodule commit (`ee7fa0b50d8295b3f0485e42c1d133fb534bfce7`) read from the
pinned local checkout; advancing the pin changes the identity, so an old-resume
guarantee is **not** silently carried over -- a checkpoint from the previous
source/pin is rejected rather than partially applied.  Resuming it would require
the **original source and the original `UniCeption` checkout**, and the guard is
**never** loosened to accept an old checkpoint under the new source.  RNG (python/numpy/torch/CUDA),
`global_step`, `epoch`, the sampler permutation and its consumed cursor, the
dedicated augmentation and dataloader generators, and the best-monitor are all
persisted.  Loading a `best_averaged_x` checkpoint for resume is rejected
(`ResumeKindError`); a mismatched identity raises `ResumeIdentityError` without
partially mutating the live model.

Training checkpoints are written atomically (temp file + `os.replace`), so a
crash never leaves a half-written last-good file.

---

## 10. Data guards: split contract and sealed validation

* Cross-split frame sharing (train/val/test leakage) is rejected at load time.
* The held-out split is declared `TEST_ONLY_NO_SELECTION` and the evaluation
  runner refuses to **select** a checkpoint/variant on a test-only role
  (`SelectionPolicyError`).  Measuring on test is allowed; selecting is not.
* The descriptor's real `role` must match the CLI `--split`, removing any
  `role=test` + `--split val` bypass.
* Validation uses the averaged `x` state and writes a **sealed** monitor value;
  the evaluation runner can re-check a stored sealed value against a freshly
  measured full-population number.

---

## 11. Teacher vs ground truth, common cohort, failure handling

* Metrics are **pseudo-teacher** metrics, not ground truth
  (`PSEUDO_TEACHER_NOT_GROUND_TRUTH = true`).  A low EPE means "agrees with the
  dataset's own teacher", never "correct".
* Variants are compared on a **common cohort**: the intersection of the normal
  (finite, uncontaminated) pair ids across all variants.  An empty intersection
  yields `None`, never a fabricated ranking.
* **Missing / empty support** gives `epe = None` (`no_data`), never a fabricated
  `0`.  A non-finite prediction or target **inside the valid region** is
  retained and counted as a failure; the primary metric for that row becomes
  `None` rather than a NaN-dropped partial mean.  The raw counts
  (`n_valid`, `n_nonfinite_pred`, `n_nonfinite_target`, `epe_sum`, support) are
  kept.
* The flow mask is supplied by the dataset as a genuine `bool` tensor; a
  float/int mask (e.g. a confidence heatmap) is rejected, so support is never
  selected by predicted confidence.  Covisibility supervision comes from
  `occlusion_supervision_mask` and is independent of flow confidence.

---

## 12. Determinism (no tolerance widening)

The determinism contract is applied and restored around the whole run
(`determinism_context`) and around each forward (`deterministic_forward_context`):

* `CUBLAS_WORKSPACE_CONFIG` is set **before** the first CUDA/cuBLAS call.
* `torch.use_deterministic_algorithms(enabled, warn_only=...)`; `cudnn.benchmark
  = false`; `cudnn.deterministic = true`; TF32 disabled.
* `warn_only=true` is the **default-compatibility mode only**: PyTorch has no
  deterministic CUDA kernel for the DPT bilinear backward, so strict mode would
  abort.  It keeps the warning visible rather than hidden, but it does **not**
  guarantee bitwise determinism -- it warns and falls back wherever no
  deterministic CUDA kernel exists.
* The **strict GPU path** is `warn_only=false` combined with the bounded
  forward-scope hooks `sdpa_backend: math` (restricts SDPA to the math backend)
  and `bilinear: deterministic` (routes bilinear `F.interpolate` through the
  deterministic helper).  `default` changes nothing.  A generic strict override:

  ```bash
  python scripts/resolve_rgbd_config.py dry_run=true experiment=short \
      determinism.warn_only=false \
      determinism.sdpa_backend=math \
      determinism.bilinear=deterministic
  ```

**No tolerance is widened and no failure is replaced.** `fail_nonfinite` must
be `true`: a non-finite loss/gradient aborts the update **before** the optimizer
step (`NonFiniteTrainingError`), leaving the last-good checkpoint intact; NaN is
never replaced by zero.

---

## 13. Self-supervised augmentation (independent implementation)

The self-supervision is an **independent implementation** of augmentation
consistency; no license-restricted code was copied.

* The original prediction is **detached** (`detach_original_prediction: true`);
  only the augmented prediction carries gradient.
* The same geometry flags drive RGB / depth / masks / `K` / flow, so all fields
  share one geometry (`same_geometry: true`).  Augmentation is `hflip` +
  `photometric`; `crop` and `complex_warp` are **off** (and rejected if enabled).
* Planar **homography is off** (`loss.homography.enabled: false`), appropriate
  for non-rigid scenes.
* Optional Laplace NLL is **off** by default; it may not be enabled without a
  model flow covariance (sigma must not be faked).
* The weight ramps linearly (`weight * min(1, step/warmup)`, strictly `> 0` at
  step 1).  `loss=selfsup_off` performs no extra forward; `loss=selfsup_on`
  enables it with the ON weight candidate — a matched OFF/ON comparison keeps
  seed, optimizer, budget and split identical.

---

## 14. Running training

`--mode resolve|preflight|train`; all private values come from an external config.

```bash
# Compose + validate + write the resolved yaml to a Git-external path (CPU).
python scripts/resolve_rgbd_config.py \
    external_config=../private/private.yaml \
    experiment=short optimizer=amuse loss=selfsup_on

# Resolve + open the real index + assert split contract and pair counts (CPU, no model).
python scripts/train_rgbd.py --mode preflight \
    external_config=../private/private.yaml \
    experiment=short loss=selfsup_on optimizer=amuse

# The real CUDA run (smoke first, then short, then FULL once the budget is fixed).
python scripts/train_rgbd.py --mode train --device cuda \
    external_config=../private/private.yaml \
    experiment=smoke loss=selfsup_on optimizer=amuse

# Resume from a Git-external last_y checkpoint (strict identity).
python scripts/train_rgbd.py --mode train --device cuda \
    --resume ../private/checkpoints/last_y.pt \
    external_config=../private/private.yaml \
    experiment=short loss=selfsup_on optimizer=amuse

# short-only pause (writes status=paused and keeps last_y).
python scripts/train_rgbd.py --mode train --device cuda --pause-after-updates 4 \
    external_config=../private/private.yaml \
    experiment=short loss=selfsup_on optimizer=amuse
```

Exit codes: `0` ok, `2` config/contract error, `3` run not finished (e.g.
paused/stopped).  A wall-clock deadline (`UFM_RGBD_DEADLINE_S`) stops gracefully
with status `stopped` and keeps last-y; it is never reported as a completed
FULL run.

---

## 15. Running evaluation

`scripts/evaluate_rgbd.py` scores a set of variants over **all** pairs of one
split with an identical pair order / input / mask population.  There is a
**canonical set of 4** and an **optional set of 2**:

| Variant | Set | Meaning |
|---------|-----|---------|
| `rgb_pretrained` | canonical | released UFM-Base, RGB encoder only (no fusion) |
| `rgbd_pretrained_init` | canonical | released base migrated to RGB-D (gate `0`) |
| `rgbd_off` | canonical | trained RGB-D checkpoint, self-supervision OFF |
| `rgbd_on` | canonical | trained RGB-D checkpoint, self-supervision ON |
| `rgb_off` | optional | trained **RGB** (depth-free) checkpoint, self-supervision OFF |
| `rgb_on` | optional | trained **RGB** (depth-free) checkpoint, self-supervision ON |

`--variants` selects the subset; it defaults to the canonical 4, and the optional
2 are added explicitly (for example `rgb_off,rgb_on`).

Every value comes from a Git-external JSON **descriptor**; no private dataset
identifier, path, count, calibration or result is embedded.

```bash
python scripts/evaluate_rgbd.py \
    --descriptor ../private/eval_descriptor.json \
    --dataset-index ../private/index.json \
    --split val \
    --output-dir ../private/eval_out \
    --variants rgb_pretrained,rgbd_pretrained_init,rgbd_off,rgbd_on \
    --device cuda --gpu 0

# Add the two optional RGB-trained variants (depth-free):
python scripts/evaluate_rgbd.py \
    --descriptor ../private/eval_descriptor.json \
    --dataset-index ../private/index.json \
    --split val \
    --output-dir ../private/eval_out \
    --variants rgb_pretrained,rgbd_pretrained_init,rgbd_off,rgbd_on,rgb_off,rgb_on \
    --device cuda --gpu 0
```

The descriptor requires the top-level objects `dataset`, `assets`, `model`,
`variants`, `determinism`, and a `role`.  `dataset.resolution` is `[W, H]`;
`dataset.depth_units` must be metres (a millimetre unit is rejected because the
loader converts `uint16` mm with a fixed factor).

The `variants` object must define the **canonical 4** (`rgb_pretrained`,
`rgbd_pretrained_init`, `rgbd_off`, `rgbd_on`); the **optional 2** (`rgb_off`,
`rgb_on`) are defined **only when used** and are never required.  Each defined
variant needs a `kind`:

* `rgb` -- released UFM-Base, depth-free (no fusion).
* `rgbd_init` -- released base migrated to RGB-D (zero gate).
* `checkpoint` -- an RGB-D trained checkpoint; **requires `checkpoint_path`**.
* `rgb_checkpoint` -- a depth-free RGB trained checkpoint; **also requires
  `checkpoint_path`**.

Both checkpoint-backed kinds (`checkpoint` and `rgb_checkpoint`) therefore
**require `checkpoint_path`** -- the validator rejects either kind when
`checkpoint_path` is absent.  `rgb` / `rgb_checkpoint` are depth-free (no
fusion), while `rgbd_init` / `checkpoint` use the zero-gated RGB-D module.

Contracts enforced: the staged index grid (`teacher_reference_hw` and every
pair's `reference_hw`) must equal the model target `(H, W)` so nothing is
re-scaled twice; every variant must expose the same pair id/order; each model
stays FP32; the released asset identity is verified before its bytes are read;
and each model runs under the determinism context with `model.eval()` +
`torch.inference_mode`.

Outputs (all Git-external): per-pair `*.npz` predictions, a `comparison.json`
(per-variant and common-cohort metrics, RGB/RGB-D-init parity, optional sealed
monitor check), an `events.jsonl`, and a `run_receipt.json`.

The pure metric core and the pure model-grid -> native scene-flow adapter
(`uniflowmatch/evaluation/scene_flow_adapter.py`) carry no private identifiers:
frame ids, query ids, timestamps and calibration all arrive from the caller.

---

## 16. Outputs and privacy

* Every write is routed through a Git-external guard (`assert_git_external`,
  symlinks followed).  Any resolved output path inside the public repository —
  including via a symlink — is refused **before** a directory is created or a
  byte is written.  The same guard rejects absolute/escaping filenames.
* Stdout from the resolve/dry-run path is **redacted**: private host paths, the
  index SHA, the resolved output directory and the external config path are
  never printed (`redact`).
* Public docs, configs and tests must not contain private dataset names,
  archive names, calibration, experimental numbers, place/personal names, local
  absolute paths, secrets, or real-data fingerprints.  External examples use
  generic relative paths and synthetic IDs (as above).

A pre-stage privacy guard is available at
`scripts/check_public_stage.py` (inspects staged index blobs or `--paths`).

---

## 17. Released RGB weights: strict load and `predict_correspondences_batched`

This section describes loading the **original** released UFM-Base asset and, for
reference, the **separate** OFF model-only release asset.  They are different
files with different identities and are **not** interchangeable.  Both are
loaded strictly -- never through the ModelHub mixin's silent `strict=False`.

### Original UFM-Base asset (deduplicated, shared-storage aliases)

The original released asset stores its state **deduplicated**: its real
safetensors file has **613 canonical entries**, and the sharing is recorded in
the file's `__metadata__` (16 alias names, each mapped to a canonical name).
Building the UFM-Base `UniFlowMatchConfidence` yields a `model.state_dict()` of
**629** entries because those **16 shared-storage aliases expand** the 613
canonical entries.  This 613 -> 629 expansion is a property of **this UFM-Base
structure** and is **not** a claim about every UFM model.

The alias-aware helper `uniflowmatch.models.checkpoint.load_released_state`
resolves those aliases: **613** keys are copied by canonical name and the **16**
missing keys are accepted only because each shares storage with a loaded key.
Any genuine missing key, unexpected key or shape mismatch fails fast.  The asset
identity (SHA256 / byte size) is verified **before** the weight bytes are read,
and the model is built against the pinned **local** `facebookresearch/dinov2`
hub tree under `pretrained=False` (never a moving `github/main` download).

### OFF model-only release asset (materialised, all keys)

The published **OFF** model-only release asset
(`.../releases/tag/rgb-only-off-20261013`) is a **separate** artifact: it
materialises every alias into its own contiguous tensor, so it contains **all
629 keys** for this UFM-Base structure and loads with an ordinary
`model.load_state_dict(load_file(...), strict=True)` -- no alias handling
required.  It is a depth-free (`use_depth_fusion=false`) RGB model.  Do not mix
its identity with the original asset above.

The snippet below uses generic run-time paths and the true RGB-only preset; it
makes no claim about any trained checkpoint.  The expected size / digest are
**generic variables** you replace after independently verifying them, and the
file is read from a Git-external location.

```python
import json
import torch
from uniflowmatch.models.checkpoint import (
    verify_file_identity, load_checkpoint, extract_state_dict, load_released_state,
)
from uniflowmatch.models.ufm import UniFlowMatchConfidence
from uniflowmatch.training.rgbd_trainer import local_dino_hub

ASSET = "<path-to>/ufm_base_pretrained/model.safetensors"   # Git-external
CONFIG = "<path-to>/ufm_base_pretrained/config.json"        # Git-external
DINO_HUB = "<path-to>/torch_hub/hub/facebookresearch_dinov2_main"

EXPECTED_WEIGHT_BYTES = 0                 # replace with the verified size
EXPECTED_WEIGHT_SHA256 = "<verified-sha256>"

model_kwargs = json.load(open(CONFIG))
for k in ("pretrained_checkpoint_path", "pretrained_backbone_checkpoint_path",
          "use_depth_fusion", "depth_fusion_kwargs"):
    model_kwargs.pop(k, None)

with local_dino_hub(DINO_HUB, "dinov2_vitl14", "<dino_python_tree_sha256>",
                    pretrained=False):
    model = UniFlowMatchConfidence(**model_kwargs)   # true RGB-only, depth-free

verify_file_identity(ASSET, expected_sha256=EXPECTED_WEIGHT_SHA256,
                     expected_bytes=EXPECTED_WEIGHT_BYTES)
state = load_checkpoint(ASSET, weights_only=True, map_location="cpu")
load_released_state(model, extract_state_dict(state), require_fp32=True)

source_image = torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8)
target_image = torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8)
with torch.inference_mode():
    result = model.predict_correspondences_batched(
        source_image=source_image,
        target_image=target_image,
    )
    flow = result.flow.flow_output            # (B, 2, H, W)
    covisibility = result.covisibility.mask   # (B, H, W)
```

For the **OFF model-only release asset** the load is the ordinary strict load,
because the release asset is materialised and complete:

```python
import json
import torch
from safetensors.torch import load_file
from uniflowmatch.models.ufm import UniFlowMatchConfidence

OFF_ASSET = "<path-to>/ufm-rgb-only-selfsup-off.safetensors"  # Git-external
OFF_CONFIG = "<path-to>/config.json"                          # Git-external

kwargs = json.loads(open(OFF_CONFIG).read())
kwargs["use_depth_fusion"] = False
model = UniFlowMatchConfidence(**kwargs)
model.load_state_dict(load_file(OFF_ASSET), strict=True)
model.eval()

source_image = torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8)
target_image = torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8)
with torch.inference_mode():
    result = model.predict_correspondences_batched(source_image, target_image)
    flow = result.flow.flow_output            # (B, 2, H, W)
    covisibility = result.covisibility.mask   # (B, H, W)
```

`predict_correspondences_batched(self, source_image, target_image,
data_norm_type=None, source_depth=None, target_depth=None, depth_mask=None,
camera_intrinsics=None, source_depth_mask=None, target_depth_mask=None)` takes
`uint8` or `float32` image tensors in `BHWC` / `BCHW`; `uint8` inputs are
normalised internally, while `float32` inputs must pass a `data_norm_type`.  The
true RGB-only model consumes **no** depth or camera inputs, so those keyword
arguments are simply omitted.

> Note: the upstream CLI (`ufm infer`, see `example_inference.py`) additionally
> exposes **DINO-init** variants (`base-dinov2l-init` / `base-dinov2g-init`)
> that are DINOv2-initialised UFM models.  They are distinct from the trainer's
> current RGB / RGB-D presets, which both build `UniFlowMatchConfidence` from the
> released **UFM-Base** weights (`configs/rgbd_training/model/*.yaml`).  Pick the
> preset (and therefore the weight asset) that matches what you actually load.

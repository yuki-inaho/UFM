
<div align="center">
<h1>UFM: A Simple Path towards Unified Dense Correspondence with Flow</h1>
<a href="https://uniflowmatch.github.io/assets/UFM.pdf"><img src="https://img.shields.io/badge/paper-blue" alt="Paper"></a>
<a href="https://arxiv.org/abs/2506.09278"><img src="https://img.shields.io/badge/arXiv-2506.09278-b31b1b" alt="arXiv"></a>
<a href="https://uniflowmatch.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/spaces/infinity1096/UFM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>


**Carnegie Mellon University**

[Yuchen Zhang](https://infinity1096.github.io/), [Nikhil Keetha](https://nik-v9.github.io/), [Chenwei Lyu](https://www.linkedin.com/in/chenwei-lyu/), [Bhuvan Jhamb](https://www.linkedin.com/in/bhuvanjhamb/), [Yutian Chen](https://www.yutianchen.blog/about/), [Yuheng Qiu](https://haleqiu.github.io), [Jay Karhade](https://jaykarhade.github.io/), [Shreyas Jha](https://www.linkedin.com/in/shreyasjha/), [Yaoyu Hu](http://www.huyaoyu.com/), [Deva Ramanan](https://www.cs.cmu.edu/~deva/), [Sebastian Scherer](https://theairlab.org/team/sebastian/), [Wenshan Wang](http://www.wangwenshan.com/)
</div>

<p align="center">
    <img src="assets/teaser.jpg" alt="example" width=80%>
    <br>
    <em>UFM unifies the tasks of Optical Flow Estimation and Wide Baseline Matching and provides accurate dense correspondences for in-the-wild images at significantly fast inference speeds.</em>
</p>

## Welcome to the Training Branch
If you reach here, you are interested in building upon UFM, thank you for your effort contributing to generalizable correspondence prediction!

This branch contains the data downloading and processing instructions, example and full training scripts used to create UFM.

## ✅ Data Processing Is Ready

We have completed cleaning the data processing, all data should be ready. Please let us know if something does not work.

## Updates
- [2026/04/03] UFM-G version initialized from DINOv2-G weights. More robust and accurate!
- [2026/02/20] Smaller UFM initialized from DINOv2 weights, faster at similar performance.
- [2025/10/26] Complete data source and processing scripts.
- [2025/10/21] Complete training and most data processing scripts. (branch: train)
- [2025/10/20] Benchmark & data script for primary results. (branch: benchmark)
- [2025/10/08] Released 980 resolution models.
- [2025/06/10] Initial release of model checkpoint and inference code.

## What's Included

- **Data downloading and processing scripts** for recreate entire UFM data.
- **Example and complete training code** for recreating UFM training.
- **Logging and visualization support** to monitor training.

## Stay Tuned for the Upcoming Updates!
- UFM-Tiny for real-time applications such as robotics.

## Models

| Checkpoint | Typical Runtime (ms, RTX 5090) | Parameters | HuggingFace |
|---|---|---|---|
| UFM-Base | 33  | 0.4B | [infinity1096/UFM-Base](https://huggingface.co/infinity1096/UFM-Base) |
| UFM-Refine | 44 | 0.4B | [infinity1096/UFM-Refine](https://huggingface.co/infinity1096/UFM-Refine) |
| UFM-Base-980 | 77 | 0.4B | [infinity1096/UFM-Base-980](https://huggingface.co/infinity1096/UFM-Base-980) |
| UFM-Refine-980 | 96 | 0.4B | [infinity1096/UFM-Refine-980](https://huggingface.co/infinity1096/UFM-Refine-980) |
| UFM-Base-DINOv2L-init | 28 | 0.3B | [infinity1096/UFM-Base-DINOv2L-init](https://huggingface.co/infinity1096/UFM-Base-DINOv2L-init) |
| UFM-Base-DINOv2G-init | 55 | 1B | [infinity1096/UFM-Base-DINOv2G-init](https://huggingface.co/infinity1096/UFM-Base-DINOv2G-init) |

## Overview

> **RGB and RGB-D fine-tuning and evaluation:** see [`docs/rgbd_training.md`](docs/rgbd_training.md).

## RGB and RGB-D training & evaluation

The training branch supports both a **generic RGB-only** mode and an **RGB-D**
mode from one code path, selected by the model preset:

| Preset | Mode |
|--------|------|
| `configs/rgbd_training/model/ufm_rgb_only.yaml` | true RGB-only (`use_depth_fusion: false`, no depth inputs) |
| `configs/rgbd_training/model/ufm_rgbd.yaml` | RGB-D (zero-gated depth residual, gate init `0.0`) |

Key points:

* **Self-supervision OFF/ON** are matched presets (`loss=selfsup_off` /
  `loss=selfsup_on`); a matched comparison keeps seed, optimizer, budget and
  split identical.  **AMUSE** (schedule-free) is the default optimizer, with an
  AdamW comparison preset.
* **Explicit 10-epoch FULL budget** over the entire eligible train population
  (fixed from the capacity measurement and applied as an external override; the
  shipped `experiment/full` value is only a public placeholder).
* **Strict source pin.**  Resume identity pins the public source SHA list, the
  released-weight SHA, the dataset-index SHA, the resume-config SHA and the
  `UniCeption` submodule commit
  (`ee7fa0b50d8295b3f0485e42c1d133fb534bfce7`), verified before any state is
  copied.  The source-SHA / pin guard is never loosened, so a checkpoint from a
  previous source or pin is rejected rather than silently resumed.
* **Released RGB weights load strictly** (alias-aware, never `strict=False`).
  The **original UFM-Base asset** stores a deduplicated state whose real file has
  **613 canonical entries** (16 shared-storage aliases recorded in the file
  metadata); building the UFM-Base model yields a `model.state_dict()` of **629**
  entries because those 16 aliases expand -- a property of *this* UFM-Base
  structure, not a claim about every UFM model.  The separate **OFF model-only
  release asset** instead materialises every alias and contains **all 629 keys**,
  loading with an ordinary `strict=True`.  Either way there are zero unexpected
  keys and zero shape mismatches, and the asset identity (SHA256 / byte size) is
  verified before any weight bytes are read.  Model construction uses the pinned
  **local** `facebookresearch/dinov2` hub tree with `pretrained=False` (never a
  moving download).
* **Evaluation variants.**  `scripts/evaluate_rgbd.py` exposes a canonical set of
  4 (`rgb_pretrained`, `rgbd_pretrained_init`, `rgbd_off`, `rgbd_on`) plus an
  optional set of 2 (`rgb_off`, `rgb_on`) selected via `--variants` (defaults to
  the canonical 4); the optional 2 are defined in the descriptor only when used.
  Both checkpoint-backed kinds -- `checkpoint` (RGB-D) and the additive
  `rgb_checkpoint` (depth-free RGB) -- **require `checkpoint_path`**.  See the doc
  for the correct `predict_correspondences_batched` usage against the
  original vs release assets.

The OFF-mode released weights are published at
<https://github.com/yuki-inaho/UFM/releases/tag/rgb-only-off-20261013>.

## Getting Started
### Setup Environment

You will need to `pip install -e .` again if you switched to here from the other branches.


### Setup Data

Please refer to `preprocess_datasets/README.md` for a complete guide. We provided preprocessed blendedmvs dataset for running the example training scripts.

### Setup Configuration

You need to setup data paths for your machine. please modify `configs/machine/your_machine.yaml` to another name and modify `root_data_dir` to point to the root folder of all `*_processed` folders. 

The `machines` configuration allows the same code to run on multiple places seamlessly. You can see how they work at the top of the example training scripts. 

### Example Training

Modify the logic for setting the `machine` environment variable in the file, and run: 

```bash
bash bash_scripts/examples/blendedmvs.sh
```

## Functional Components

|Component | Code location | Documentation |
|-------------|---------------|---------------|
|Base Model Architecture| `uniflowmatch/models/ufm.py` line 474 | paper Section `3.1`|
|Refinement Model Architecture| `uniflowmatch/models/ufm.py` line 710 | paper Section `3.1`|
|Computing Covisible Masks| `uniflowmatch/datasets/base/flow_postprocessing.py` line `544`| paper Appendix `A`|
|Robust EPE Loss|`uniflowmatch/loss/epe.py` | paper Section `3.4`, Appendix `H`|
|Refinement Loss|`uniflowmatch/loss/refinement_cross_entropy[_efficient].py`| paper Appendix `D`|
|Visualizations|`uniflowmatch/utils/viz.py`| |

## Entry Points
| Script | Function |
|--------|----------|
| `bash_scripts/examples/blendedmvs.sh` | Example training of the UFM base model |
| `bash_scripts/examples/refinement.sh` | Example training of the refinement network |
| `bash_scripts/training/megatraining_560.sh` | Train `UFM 560` |
| `bash_scripts/training/megatraining_980.sh` | Train `UFM 980` from `UFM 560` |
| `bash_scripts/training/megatraining_refinement_560.sh` | Train `UFM 560 Refine` from `UFM 560` |
| `bash_scripts/training/megatraining_refinement_980.sh` | Train `UFM 980 Refine` from `UFM 560` |

## Verify Installation

Verify your installation by running the basic model test:

```bash
# Test installation
ufm test

# Or run the basic model test
python uniflowmatch/models/ufm.py
```

Verify that `ufm_output.png` looks like `examples/example_ufm_output.png`.

### Command Line Interface

UFM provides a convenient CLI for common tasks:

```bash
# Test installation
ufm test

# Launch interactive demo
ufm demo

# Launch demo with specific settings
ufm demo --port 8080 --share --model refine

# Run inference on image pair
ufm infer source.jpg target.jpg --output results/

# Run inference with refinement model
ufm infer img1.png img2.png --model refine --output ./output
```

### Python API

```python
import cv2
import torch

# Load the base model (for general use)
from uniflowmatch.models.ufm import UniFlowMatchConfidence
model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")

# Or load the refinement model (for higher accuracy)
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement
model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")

# Choose from
# UFM-Base, UFM-Refine, UFM-Base-980, UFM-Refine-980, UFM-Base-DINOv2L-init, UFM-Base-DINOv2G-init

# Set the model to evaluation mode
model.eval()

# Load images using cv2 or PIL
source_image = cv2.imread("path/to/source.jpg")
target_image = cv2.imread("path/to/target.jpg")
source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Convert to torch tensors (uint8 or float32)
# Forward call takes care of normalizing uint8 images appropriate to the UFM model
source_image = torch.from_numpy(source_rgb)  # Shape: (H, W, 3)
target_image = torch.from_numpy(target_rgb)  # Shape: (H, W, 3)

# Predict correspondences
with torch.no_grad():
    result = model.predict_correspondences_batched(
        source_image=source_image,
        target_image=target_image,
    )

    flow = result.flow.flow_output[0].cpu().numpy()
    covisibility = result.covisibility.mask[0].cpu().numpy()
```

## Interactive Demo

### Online Demo

Try our online demo without installation: [🤗 Hugging Face Demo](https://huggingface.co/spaces/infinity1096/UFM)

### Local Gradio Demo

Run the interactive Gradio demo locally to visualize UFM outputs:

```bash
# Using the CLI (recommended)
ufm demo

# Or run directly
python gradio_demo.py

# Advanced options
ufm demo --port 8080 --share --model refine
```

## License

This code is licensed under a fully open-source [BSD-3-Clause license](LICENSE). The pre-trained UFM model checkpoints inherit the licenses of the underlying training datasets and as result, may not be used for commercial purposes (CC BY-NC-SA 4.0). Please refer to the respective training dataset licenses for more details.

Based on community interest, we can look into releasing an Apache 2.0 licensed version of the model in the future. Please upvote the issue [here](https://github.com/UniFlowMatch/UFM/issues/1#issue-3135416718) if you would like to see this happen.

## Acknowledgements

We thank the folowing projects for their open-source code: [DUSt3R](https://github.com/naver/dust3r), [MASt3R](https://github.com/naver/mast3r), [RoMA](https://github.com/Parskatt/RoMa), and [DINOv2](https://github.com/facebookresearch/dinov2).

## Citation
If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@inproceedings{zhang2025ufm,
 title={UFM: A Simple Path towards Unified Dense Correspondence with Flow},
 author={Zhang, Yuchen and Keetha, Nikhil and Lyu, Chenwei and Jhamb, Bhuvan and Chen, Yutian and Qiu, Yuheng and Karhade, Jay and Jha, Shreyas and Hu, Yaoyu and Ramanan, Deva and Scherer, Sebastian and Wang, Wenshan},
 booktitle={arXiV},
 year={2025}
}
```

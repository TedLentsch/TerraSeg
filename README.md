# <div align="center">**🌍 TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR**</div>

<p align="center">
    <a href="https://scholar.google.com/citations?user=54NWkMoAAAAJ&hl=en">Ted Lentsch</a><sup>1</sup>,
    <a href="https://www.santimontiel.eu/">Santiago Montiel-Marín</a><sup>2</sup>,
    <a href="https://sites.google.com/it-caesar.de/homepage/">Holger Caesar</a><sup>2</sup>, and
    <a href="https://scholar.google.com/citations?user=wQU1dJAAAAAJ&hl=en">Dariu Gavrila</a><sup>1</sup>
</p>

<p align="center" style="font-size: 0.9em; font-style: italic;">
  <sup>1</sup> Technical University of Delft,
  <sup>2</sup> University of Alcalá
</p>

<p align="center">
    📝 <a href="https://arxiv.org/abs/2603.27344">arXiv</a> ·
    🤗 <a href="https://huggingface.co/TedLentsch/TerraSeg">Hugging Face</a> ·
    📚 <a href="#️-citation">Cite us!</a>
</p>


## 📰 News

* [2026-06-05] *Poster presented at CVPR 2026!*
* [2026-05-29] *Code released on GitHub!*
* [2026-05-28] *Model weights released on Hugging Face!*
* [2026-03-28] *Paper released on arXiv (v1)!*
* [2026-02-21] *TerraSeg has been accepted to CVPR 2026!*


## 🎥 Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=A-WTTnsplmo">
    <img src="https://img.youtube.com/vi/A-WTTnsplmo/maxresdefault.jpg" alt="TerraSeg video" width="640">
  </a>
</p>
<p align="center"><i>▶︎ 90 seconds video on YouTube showing qualitative results on our three main benchmarks</i></p>


## ⭐ Highlights

TerraSeg is the first self-supervised, domain-agnostic model for LiDAR ground segmentation. It is trained on the **OmniLiDAR** dataset (~22M raw scans aggregated from 12 public autonomous-driving benchmarks across 15 distinct LiDAR sensors) using pseudo-labels produced by our self-supervised **PseudoLabeler**, and uses an adapted **Point Transformer v3** backbone with dataset-specific normalization disabled.

The released `terraseg_s.pth` and `terraseg_b.pth` checkpoints are re-trained with the cleaned, public code release on OmniLiDAR **excluding View-of-Delft** (license-restricted; cannot be redistributed as part of OmniLiDAR). Performance below is the mean mIoU across the three evaluation splits used in the paper (nuScenes val, SemanticKITTI val, and Waymo Perception val):

| Checkpoint | Params | Throughput (A100) | Mean mIoU (val) |
| --- | --- | --- | --- |
| `terraseg_s.pth` | ~12M | 17 - 50 Hz | **93.43** |
| `terraseg_b.pth` | ~46M | 10 - 28 Hz | **94.02** |

These are obtained without any manual annotations during training (self-supervised). For full per-dataset results and ablations, see Tables 3 - 5 of the [paper](https://arxiv.org/abs/2603.27344).


## 🚀 Quick start: Use TerraSeg in your own Python project

Add the library and its PTv3 backbone to your project:

```bash
uv add "ptv3 @ git+https://github.com/TedLentsch/TerraSeg.git#subdirectory=ptv3"
uv add "terraseg @ git+https://github.com/TedLentsch/TerraSeg.git#subdirectory=terraseg_lib"
```

Run inference on a point cloud. The trained weights are downloaded once and cached automatically from [Hugging Face](https://huggingface.co/TedLentsch/TerraSeg):

```python
import torch
from terraseg import TerraSegPredictor

predictor = TerraSegPredictor(
    variant="S",  # "S" for Small (~12M) or "B" for Base (~46M)
    checkpoint_path="hf://TedLentsch/TerraSeg/terraseg_s.pth",
)

coord = torch.randn(50_000, 3, device="cuda")  # Your (N, 3) point cloud in meters.
labels = predictor.predict(coord=coord)        # Shape (N,) with datatype uint8. Labels: 0 = ground, 1 = non-ground.
```

That is the entire integration. TerraSeg runs in FP32 (sparse-conv stability). The predictor enables TF32 tensor-core matmuls by default (`tf32=True`) for a small speedup on Ampere+ GPUs; pass `tf32=False` to reproduce the paper's exact FP32 numerics. It also accepts `compile_model=True` to wrap the model with `torch.compile`, though the gain is limited because PTv3's spconv path is opaque to the compiler.

*Note: TerraSegPredictor expects point cloud to be in TerraSeg-standardized frame (z = 0 approximately ground-aligned, +x forward).*


## 🤖 Quick start: Use TerraSeg as a ROS2 node

The ROS2 node ships with a [pixi](https://pixi.sh) environment that provides ROS2 Humble and the PyTorch CUDA stack in a single Python interpreter, so `rclpy` and `torch` share one interpreter and no system ROS2 or separate virtualenv is needed. Install pixi (`curl -fsSL https://pixi.sh/install.sh | bash`), then:

```bash
git clone https://github.com/TedLentsch/TerraSeg.git && cd TerraSeg
pixi install                 # One-time: ROS2 Humble + torch stack (large download).
pixi shell                   # Enter the env (ros2 and colcon are on PATH).
pixi run build               # The colcon build --packages-select terraseg_ros2 --symlink-install.
source install/setup.bash
ros2 launch terraseg_ros2 terraseg.launch.py
```

The node subscribes to a `sensor_msgs/PointCloud2` topic (default `/lidar/points`, sensor-data QoS) and publishes a labeled `sensor_msgs/PointCloud2` on `/terraseg/segmented` with an added `uint8 label` field (0 = ground, 1 = non-ground). By default it transforms each incoming cloud into `base_link` via TF, so the model receives a ground-aligned cloud regardless of sensor mounting, and it loads weights directly from [Hugging Face](https://huggingface.co/TedLentsch/TerraSeg) out of the box. See [`TerraSeg_ros2/README.md`](TerraSeg_ros2/README.md) for the topic API, the full configuration reference (`target_frame`, `tf32`, `compile_model`), frame handling, and measured performance.


## 📦 What's in this repository

This repository is organized as a `uv` workspace plus a sibling ROS2 package:

* [`terraseg_lib/`](terraseg_lib/): The shared TerraSeg library: model definition, BatchNorm → GroupNorm swap, feature engineering, and the `TerraSegPredictor` class used for single-frame deployment. **This is the package you `uv add` into your own project.**
* [`ptv3/`](ptv3/): Vendored Point Transformer v3 backbone, pinned to upstream commit `3229e9b7de1770c8ad17c316f8e349982de509f8`. See [`ptv3/README.md`](ptv3/README.md) for the one-time vendoring step.
* [`TerraSeg_scripts/`](TerraSeg_scripts/): Training and offline-evaluation scripts for both the TerraSeg-B (~46M params) and TerraSeg-S (~12M params) variants. A single `VARIANT` constant switches between them. The `terraseg_test.py` evaluation script accepts checkpoints either by local path or by `hf://` URI.
* [`TerraSeg_ros2/`](TerraSeg_ros2/): The ament_python ROS2 package that wraps `TerraSegPredictor` and exposes TerraSeg as a `sensor_msgs/PointCloud2` filter. Built with `colcon` inside the bundled `pixi` environment. See [`TerraSeg_ros2/README.md`](TerraSeg_ros2/README.md) for the full topic API and configuration.
* [`OmniLiDAR_scripts/`](OmniLiDAR_scripts/): Dataset converters that aggregate 12 public autonomous-driving datasets into the unified OmniLiDAR format.
* [`PseudoLabeler_scripts/`](PseudoLabeler_scripts/): The self-supervised PseudoLabeler module that produces ground / non-ground pseudo-labels on every OmniLiDAR scan, plus the ablation studies from the paper.
* [`utils/`](utils/): Shared dataset, evaluation, and split utilities.


## 🧠 Pre-trained weights

The released TerraSeg-B and TerraSeg-S checkpoints are hosted on Hugging Face at [TedLentsch/TerraSeg](https://huggingface.co/TedLentsch/TerraSeg) as `terraseg_b.pth` and `terraseg_s.pth`. Both checkpoints bundle the model weights, the tuned decision threshold, and training metadata.

You almost never need to download these manually: both the Python API (`TerraSegPredictor`) and the ROS2 node accept either a local filesystem path or a Hugging Face URI of the form `hf://<user>/<repo>/<filename>`. The file is fetched once and cached locally by `huggingface_hub`.

The weights are released under **CC BY-NC-SA 4.0** (non-commercial, share-alike). This reflects the most restrictive terms of the upstream datasets that contributed to OmniLiDAR (most notably the Waymo Open Dataset, nuScenes, SemanticKITTI, Argoverse 2, and MAN TruckScenes). See the [Hugging Face model card](https://huggingface.co/TedLentsch/TerraSeg) for the full licensing notice, upstream-dataset attributions, and the Waymo-specific *Derivative IP* restriction. The source code in this repository is released separately under the Apache License 2.0 (see the License section below).


## 🧪 Reproduce the paper (train + evaluate from scratch)

Researchers and developers who want to retrain TerraSeg, evaluate on the OmniLiDAR validation splits, or run the published ablation studies should refer to the dedicated sub-project READMEs. The high-level flow is:

1. Build the unified OmniLiDAR dataset with the converter scripts under [`OmniLiDAR_scripts/`](OmniLiDAR_scripts/).
2. Generate ground / non-ground pseudo-labels on every OmniLiDAR scan with [`PseudoLabeler_scripts/`](PseudoLabeler_scripts/).
3. Train and evaluate TerraSeg-B and TerraSeg-S with the scripts under [`TerraSeg_scripts/`](TerraSeg_scripts/).

Training takes ~10 epochs on a single GPU and uses the balanced multi-dataset sampler matching paper section A.1. The released checkpoints in this repository were produced by exactly this flow on OmniLiDAR minus VoD (the only dataset we cannot redistribute). See the [model card](https://huggingface.co/TedLentsch/TerraSeg) for the full reproduction notes.


### 🛠️ GPU compatibility

| Hardware | Examples | Status |
| --- | --- | --- |
| **Volta CUDA** (sm_70) | V100, V100S | ❌ flash-attn requires Ampere+ |
| **Turing CUDA** (sm_75) | RTX 20xx, T4 | ❌ flash-attn requires Ampere+ |
| **Ampere CUDA** (sm_80, sm_86) | RTX 30xx, A40, A6000, A100 | ✅ Verified |
| **Hopper CUDA** (sm_90) | H100, H200 | ✅ Expected to work |
| **Blackwell CUDA** (sm_120) | RTX 50xx, RTX PRO, B100, B200 | ❌ Needs pin override (planned) |
| **CPU only** | any | ❌ spconv `MaskImplicitGemm` is CUDA-only |

The default stack (`torch 2.4.1` + `cu124`, `spconv-cu124`, `torch-scatter pt24cu124`, `flash-attn 2.8.3 cu12torch2.4`) is verified on Ampere A100 hardware. A future release will add Blackwell support!


## 🖊️ Citation

<p align="justify">
If TerraSeg is useful to your research, please kindly recognize our contributions by citing our paper.
</p>


```
@inproceedings{lentsch2026terraseg,
  title={TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR},
  author={Lentsch, Ted and Montiel-Marín, Santiago and Caesar, Holger and Gavrila, Dariu M},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## 📄 License

This project is released under the Apache-2.0 License.
See [LICENSE](LICENSE) for details.

## 🫂 Acknowledgements

This research has been conducted as part of the EVENTS project, which is funded by the European Union, under grant agreement No 101069614. Views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or European Commission. Neither the European Union nor the granting authority can be held responsible for them. This work has also been supported by project PID2024-161576OB-I00, funded by Spanish MICIU/AEI/10.13039/501100011033 and co-funded by the European Regional Development Fund (ERDF, “A way of making Europe”).

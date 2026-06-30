# TerraSeg ROS2 node


This package wraps **TerraSeg** as a real-time ROS2 node. It subscribes to a
`sensor_msgs/PointCloud2` LiDAR topic, runs the trained TerraSeg-B or TerraSeg-S model, and
publishes a labeled `PointCloud2` with an added `uint8 label` field (0 = ground, 1 = non-ground).

The node reuses the same model definition and inference pipeline as the training scripts,
through the shared `terraseg` library.

The runtime environment is provided by **[pixi](https://pixi.sh)**: ROS2 Humble and the
PyTorch CUDA stack live in a single environment with a single Python interpreter. This is the
only supported setup, and it is deliberate. Mixing a system ROS2 (one Python) with a separate
virtualenv (another Python) leads to import failures and brittle `PYTHONPATH` workarounds.
With pixi, `rclpy` and `torch` share one interpreter, `colcon` runs under that same
interpreter, and the node's entry point resolves both ROS2 and `terraseg` without any
shebang patching.


## 🧠 Topic API


* **Input:** `<input_topic>`, `sensor_msgs/PointCloud2` (default `/lidar/points`).
  Subscribed with the **sensor-data QoS profile** (BEST_EFFORT reliability, depth 5), which
  matches how LiDAR drivers and recorded bags publish. This matters: a RELIABLE subscriber
  receives nothing from a BEST_EFFORT publisher, so a plain depth-only QoS would silently
  drop every message.

* **Output:** `<output_topic>`, `sensor_msgs/PointCloud2` (default `/terraseg/segmented`).
  Carries fields `x`, `y`, `z` (float32, meters) and `label` (uint8: `0 = ground`,
  `1 = non-ground`), published in `target_frame` (see Frame handling). Downstream nodes can
  filter by `label` to isolate the ground or non-ground subset for free-space estimation,
  object discovery, and so on.


## 📐 Frame handling


TerraSeg expects the cloud roughly ground-aligned: `z` near 0 at the ground and the positive
`x`-axis pointing forward. A roof-mounted or otherwise-offset sensor frame is neither, so
feeding raw sensor-frame points shifts the model's height feature by the full mount height and
can flip the forward axis, which degrades the labels.

The node handles this with TF. Set `target_frame` (default `base_link`); the node looks up the
static transform from the incoming cloud's frame into `target_frame` once, caches it, applies
it to every scan before inference, and republishes in `target_frame`. For a standard vehicle
this puts the ground at `z = 0` and `x` forward automatically, with no per-sensor tuning.

* If your cloud is already ground-aligned, set `target_frame: ""` to skip the transform and
  run in the cloud's native frame.
* The transform must be available **before** the first scan is processed. In practice: start
  the node first, then start the sensor driver or bag playback, and make sure `/tf` and
  `/tf_static` are being published. Scans arriving before the transform is cached are skipped
  with a throttled warning, and processing begins once the transform appears.


## ⚙️ Step 1: Prerequisites


* **pixi.** Install with `curl -fsSL https://pixi.sh/install.sh | bash`, then restart your
  shell (or `source ~/.bashrc`).
* **A CUDA GPU, Ampere or newer** (RTX 30xx/40xx, A40, A6000, A100, ...). See the GPU
  compatibility table in the [root README](../README.md). FlashAttention requires Ampere+.
* **A trained checkpoint**, either a local `best.pth` from
  `TerraSeg_scripts/terraseg_train.py`, or the released Hugging Face weights
  (`hf://TedLentsch/TerraSeg/terraseg_s.pth` or `hf://TedLentsch/TerraSeg/terraseg_b.pth`),
  which are downloaded and cached automatically on the first launch.

You do **not** need a separate ROS2 installation, and you do **not** need `uv`. pixi provides
ROS2 Humble, the torch stack, and the `terraseg` library together.


## 🛠️ Step 2: Build


From the repository root:

```bash
pixi install                 # One-time: ROS2 Humble + torch stack (large download).
pixi shell                   # Enter the env (ros2 and colcon are on PATH).
pixi run build               # The colcon build --packages-select terraseg_ros2 --symlink-install.
source install/setup.bash
```

`--symlink-install` (run by `pixi run build`) lets you edit the Python sources in place
without rebuilding. Confirm the single interpreter sees everything:

```bash
python -c "import torch, rclpy, terraseg; print(torch.__version__, torch.cuda.is_available())"
```

This should print the torch version and `True`.


## 🛠️ Step 3: Configure


Edit `config/terraseg.yaml`:

* `variant`: `"S"` (~12M) or `"B"` (~46M). Must match the checkpoint.
* `checkpoint_path`: a local filesystem path or an `hf://<user>/<repo>/<file>` URI.
* `hf_revision`: optional Hugging Face branch / tag / commit; empty tracks the default branch.
* `decision_threshold`: a negative value defers to the checkpoint's tuned threshold (recommended).
* `grid_size`: PTv3 voxel size in meters (default `0.05`).
* `input_topic` / `output_topic`.
* `target_frame`: frame to transform into before inference (default `base_link`). Set `""` to
  disable. See Frame handling.
* `tf32`: enable TF32 tensor-core matmuls (default `true`). Faster on Ampere+ at a slight
  matmul-precision cost versus strict FP32. Set `false` to reproduce the paper's exact FP32
  numerics. Safe no-op on pre-Ampere GPUs and CPU.
* `compile_model`: wrap the model with `torch.compile` (default `false`). PTv3's spconv path is
  opaque to the compiler, so the gain is small and it adds a one-time compile cost; eager is
  recommended. If enabled and compilation fails, the node logs a warning and falls back to
  eager rather than crashing.


## 🚀 Step 4: Run


Inside the pixi shell, with `install/setup.bash` sourced:

```bash
ros2 launch terraseg_ros2 terraseg.launch.py
```

Override the config file at launch time:

```bash
ros2 launch terraseg_ros2 terraseg.launch.py config:=/absolute/path/to/terraseg.yaml
```

**Bag replay** during development, in a second pixi shell. Include `/tf` and `/tf_static` so
the target-frame transform resolves, and remap your LiDAR topic onto the input topic:

```bash
ros2 bag play your_lidar_bag \
    --topics /your/lidar/topic /tf /tf_static \
    --remap /your/lidar/topic:=/lidar/points
```

Start the node first, then start playback, so the static transform is cached before the first
cloud arrives.

**Visualize in RViz:** set Fixed Frame to your `target_frame` (default `base_link`), add a
`PointCloud2` display on `/terraseg/segmented`, and set Color Transformer to the `label` field.
Ground appears as one colour, obstacles as the other.


## ⚡ Performance


The node runs in Python on PyTorch in **FP32**. Lower-precision dtypes (BF16, FP16) are not
supported: PTv3's sparse-convolution path is numerically unstable at half precision. The two
relevant speed knobs are `tf32` (default on) and `compile_model` (default off); see Step 3.

Measured on a single **NVIDIA RTX A6000** (Ampere), eager, `tf32: true`, on a ~119k-point
Ouster top-LiDAR scan: about **125 ms per scan (~8 Hz)** end-to-end, including the TF
transform and the host/device copies. That is slightly under a 10 Hz real-time stream on this
card; a stronger GPU or the same model on a less dense scan runs faster.

For reference, the paper reports the following on an NVIDIA A100 with the authors' setup
(Tables 3-5):

* TerraSeg-S: 17 - 50 Hz across the three benchmark datasets.
* TerraSeg-B: 10 - 28 Hz across the three benchmark datasets.

Real throughput depends heavily on the GPU, the point count per scan, and the variant, so
profile on your own hardware rather than assuming the A100 figures. A quick no-ROS benchmark
is to construct `TerraSegPredictor` directly and time `predict()` on a representative cloud.

**Why eager is the default.** A profile of TerraSeg-S shows the runtime spread across many
small kernels (linear layers, GroupNorm, LayerNorm, GELU) plus un-fusable custom ops (spconv,
FlashAttention, `torch_scatter`). `torch.compile` fuses only the glue and graph-breaks at every
spconv call, so on dynamic per-scan point counts the net gain is small and can even be
negative. TF32 targets the dominant matmul share directly with negligible risk, which is why
it is the default speed knob instead.


## ❓ Why not TensorRT or a C++ rewrite?


We deliberately keep the ROS2 node in Python:

* **PTv3's compute graph is unfriendly to TensorRT.** It relies on sparse convolutions
  (spconv) and FlashAttention. Neither has a stock ONNX / TensorRT translation, so a TRT port
  would require writing and maintaining custom plugins.
* **Python is not the bottleneck.** Single-stream LiDAR processing is GPU-bound; the Python
  interpreter and the GIL are inactive while CUDA kernels run. A C++ rewrite with `libtorch`
  would shave a few hundred microseconds of per-call CPU overhead, which is invisible at LiDAR
  scan rates.
* **The available wins are simpler.** TF32 (default on) captures the matmul speedup with one
  flag, and the single-interpreter pixi setup removes the integration pain, without the
  maintenance cost of a TRT or C++ port.


## 🧰 Troubleshooting


* **`python` not found, or it points at the wrong interpreter inside `pixi shell`.** A conda
  `base` environment activating on top of pixi can shadow the pixi interpreter. Disable
  conda's auto-activation once with `conda config --set auto_activate_base false`, open a fresh
  shell, then `pixi shell` again. Do not run `conda deactivate` from inside a pixi shell.
* **`No transform '<frame>' -> '<target_frame>' yet; skipping scan`.** TF was not available
  when the cloud arrived. Make sure `/tf` and `/tf_static` are being published, and start the
  node before playback. If your cloud is already ground-aligned, set `target_frame: ""`.
* **The subscriber receives nothing.** Check the publisher's QoS. The node subscribes
  BEST_EFFORT (sensor-data profile); a mismatch with a strictly RELIABLE publisher is the
  usual cause. Confirm flow with `ros2 topic hz /terraseg/segmented`.


## 📂 Repository structure


* `package.xml`: ROS2 package manifest.
* `setup.py` / `setup.cfg`: ament_python build configuration.
* `terraseg_ros2/terraseg_node.py`: the node implementation (subscriber, TF transform, publisher).
* `terraseg_ros2/pointcloud_conversion.py`: PointCloud2 ↔ torch.Tensor utilities.
* `launch/terraseg.launch.py`: default launch file.
* `config/terraseg.yaml`: default node parameters.

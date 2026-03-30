# OmniLiDAR dataset converters


This repository contains the standalone Python scripts used to convert various public autonomous driving datasets into the unified OmniLiDAR format.

To ensure absolute reproducibility and avoid the notorious dependency conflicts associated with different dataset devkits (e.g. Waymo requires TensorFlow, nuScenes requires old numpy version), this project strictly utilizes [uv](https://github.com/astral-sh/uv) as its package manager.

By using `uv`'s inline script metadata, each converter script runs in its own **isolated environment**. You do not need to manually create multiple Conda environments!


## 🧠 What do the converters do?


Running a converter script on a dataset performs three main standardization tasks:

1. **Temporal subsampling:** Autonomous driving datasets are recorded at wildly different frequencies (10Hz, 20Hz, etc.). These scripts standardise the data by downsampling all LiDAR sequences to a target frequency of **0.2 Hz** (one sweep every 5 seconds) to create a small, diverse subset (if labels are available).

2. **Unified semantic labels:** For datasets that provide point-level semantic segmentation, the scripts automatically extract and remap the native dataset classes into a unified 3-class OmniLiDAR format:
    * `0`: Ground (road, sidewalk, terrain, etc.)
    * `1`: Non-Ground (vehicles, pedestrians, buildings, vegetation, etc.)
    * `2`: Ignore (Ego-vehicle, noise, undefined)

3. **Sensor calibration & metadata:** The scripts extract sensor calibration data and automatically generate two crucial JSON files for every dataset:
    * `transforms/`: Contains the 4x4 transformation matrices required to bring the LiDAR data from the sensor frame into a standardized OmniLiDAR frame.
    * `egoremovalradii/`: Defines the specific radius (in meters) required to safely crop out the ego-vehicle from the point cloud. Each LiDAR has its own radius.

Note: You do need to download each dataset yourself! We only provide the converter code.


## ⚙️ Step 1: Prerequisites & setup


1. **Install `uv`** (if you haven't already). For Windows/Mac instructions, see the official `uv` documentation.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Sync the Base Environment**:
Navigate to this `OmniLiDAR_scripts` directory and sync the base project environment. This reads the provided `pyproject.toml` and `uv.lock` files to instantly build the base `.venv`.
```bash
uv sync
```


## 🛠️ Step 2: Configure the scripts


Before running a converter, you must edit the script to point to your local file system.

Open the target converter script (e.g. `dataset_converter__nuScenes.py`) and locate the target and source root variables near the top of the file. Replace the placeholder text with your actual directory paths:

```python
omnilidar_root = Path('PUT_YOUR_DIRECTORY_HERE/OmniLiDAR')
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/Dataset')   # E.g. nuScenes root
```

*Note 1: Ensure you leave the `.mkdir()` commands intact so the script can build the proper subdirectories.*
*Note 2: You need to create a folder named `OmniLiDAR` and choose the location yourself.*


## 🚀 Step 3: Run the converters


Do **not** use the standard `python script.py` command. Instead, use `uv run` so the package manager can dynamically build the isolated environment required for that specific dataset.

For standard datasets (e.g. nuScenes), simply run:

```bash
uv run dataset_converter__nuScenes.py
```

`uv` will instantly resolve the required devkit, cache the dependencies, and execute the conversion.


### ⚠️ Special case: Waymo Open Dataset


The Waymo devkit relies on an older versions of TensorFlow and JAX that conflict with standard indices. To run the Waymo converter, you must pass specific flags to bypass the security index lock and locate archived wheels:

```bash
uv run --index-strategy unsafe-best-match --find-links https://storage.googleapis.com/jax-releases/jax_releases.html dataset_converter__WaymoPerception.py
```


## 📂 Included converters

* `dataset_converter__AevaScenes.py`
* `dataset_converter__Argoverse2Lidar.py`
* `dataset_converter__KITTI360.py`
* `dataset_converter__Lyft.py`
* `dataset_converter__nuScenes.py`
* `dataset_converter__ONCE.py`
* `dataset_converter__PandaSet.py`
* `dataset_converter__SemanticKITTI.py`
* `dataset_converter__TruckScenes.py`
* `dataset_converter__ViewOfDelft.py`
* `dataset_converter__WaymoPerception.py`
* `dataset_converter__Zenseact.py`

# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "lyft-dataset-sdk",
#     "numpy==1.26.4",
#     "tqdm==4.66.5",
# ]
# ///



import json
import numpy as np
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from pathlib import Path
from tqdm import tqdm
from typing import List



TARGET_FREQUENCY = 0.2   # Hz.
TARGET_PERIOD = 1.0 / TARGET_FREQUENCY   # Sec.



omnilidar_root = Path('PUT_YOUR_DIRECTORY_HERE/OmniLiDAR')
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/Lyft')
target_root    = omnilidar_root / 'Lyft'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/Lyft', print('Directory to Lyft dataset. Change to directory in your file system!')

omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)

print(target_root)



# No point-level labels available.



## Helper function.
def get_downsampled_tokens(
        tokens: List,
        timestamps: List,
        period: float,
        ) -> List:
    """
    Get downsampled tokens based on irregular timestamps and a target period.

    Args:
        tokens (list) : List of tokens.
        timestamps (list) : List or array of timestamps corresponding to the tokens.
        period (float) : Target period for downsampling.
    
    Returns:
        downsampled_tokens (list) : Downsampled list of tokens.
    """
    ts = np.asarray(timestamps)
    ids, t = [0], ts[0] + period
    while (i := np.searchsorted(ts, t)) < len(ts):
        ids.append(i)
        t = ts[i] + period
    downsampled_tokens = np.asarray(tokens)[ids].tolist()
    return downsampled_tokens


## Train split part 1 (OmniLiDAR train).
folder_name = 'train_scans__part1'
sensor_names = ['LIDAR_TOP', 'LIDAR_FRONT_RIGHT', 'LIDAR_FRONT_LEFT',]
sensor_frequencies = {'LIDAR_TOP': 5, 'LIDAR_FRONT_RIGHT': 5, 'LIDAR_FRONT_LEFT': 5,}   # Hz.
devkit = LyftDataset(data_path=str(source_root), json_path=str(source_root / 'train_data'), verbose=False)

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

multi_lidar_sequences = [0, 4, 6, 12, 25, 30, 32, 41, 45, 46, 48, 61, 70, 76, 79, 80, 83, 84, 87, 104, 108, 114, 120, 130, 134, 141, 142, 146, 153, 174, 175, 176]
selected_scenes = [scene for idx, scene in enumerate(devkit.scene) if idx in multi_lidar_sequences]

for seq_num, scene_token in tqdm(list(enumerate(selected_scenes)), desc='Scenes'):
    sample_tokens = []
    token = selected_scenes[seq_num]['first_sample_token']
    while token:
        sample_tokens.append(token)
        token = devkit.get(table_name='sample', token=token)['next']

    timestamps = [devkit.get(table_name='sample', token=token)['timestamp'] / 1e6 for token in sample_tokens]
    downsampled_tokens = get_downsampled_tokens(tokens=sample_tokens, timestamps=timestamps, period=TARGET_PERIOD)

    for sensor_name in sensor_names:
        downsampled_lidar_tokens = [devkit.get(table_name='sample', token=token)['data'][sensor_name] for token in downsampled_tokens]

        for scan_num, lidar_token in enumerate(downsampled_lidar_tokens):
            lidar_record = devkit.get(table_name='sample_data', token=lidar_token)

            data_root = Path(source_root)
            pc_lidar = LidarPointCloud.from_file(file_name=data_root / lidar_record['filename']).points[:3].T   # Shape (N,3).

            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            # No labels.


## Train split part 2 (OmniLiDAR train).
folder_name = 'train_scans__part2'
sensor_names = ['LIDAR_TOP',]
sensor_frequencies = {'LIDAR_TOP': 5,}   # Hz.
devkit = LyftDataset(data_path=str(source_root), json_path=str(source_root / 'train_data'), verbose=False)

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

multi_lidar_sequences = [0, 4, 6, 12, 25, 30, 32, 41, 45, 46, 48, 61, 70, 76, 79, 80, 83, 84, 87, 104, 108, 114, 120, 130, 134, 141, 142, 146, 153, 174, 175, 176]
selected_scenes = [scene for idx, scene in enumerate(devkit.scene) if idx not in multi_lidar_sequences]

for seq_num, scene_token in tqdm(list(enumerate(selected_scenes)), desc='Scenes'):
    sample_tokens = []
    token = selected_scenes[seq_num]['first_sample_token']
    while token:
        sample_tokens.append(token)
        token = devkit.get(table_name='sample', token=token)['next']

    timestamps = [devkit.get(table_name='sample', token=token)['timestamp'] / 1e6 for token in sample_tokens]
    downsampled_tokens = get_downsampled_tokens(tokens=sample_tokens, timestamps=timestamps, period=TARGET_PERIOD)

    for sensor_name in sensor_names:
        downsampled_lidar_tokens = [devkit.get(table_name='sample', token=token)['data'][sensor_name] for token in downsampled_tokens]

        for scan_num, lidar_token in enumerate(downsampled_lidar_tokens):
            lidar_record = devkit.get(table_name='sample_data', token=lidar_token)

            data_root = Path(source_root)
            pc_lidar = LidarPointCloud.from_file(file_name=data_root / lidar_record['filename']).points[:3].T   # Shape (N,3).

            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            # No labels.


## Test split (OmniLiDAR train).
folder_name = 'test_scans'
sensor_names = ['LIDAR_TOP',]
sensor_frequencies = {'LIDAR_TOP': 5,}   # Hz.
devkit = LyftDataset(data_path=str(source_root), json_path=str(source_root / 'test_data'), verbose=False)

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

selected_scenes = devkit.scene

for seq_num, scene_token in tqdm(list(enumerate(selected_scenes)), desc='Scenes'):
    sample_tokens = []
    token = selected_scenes[seq_num]['first_sample_token']
    while token:
        sample_tokens.append(token)
        token = devkit.get(table_name='sample', token=token)['next']

    timestamps = [devkit.get(table_name='sample', token=token)['timestamp'] / 1e6 for token in sample_tokens]
    downsampled_tokens = get_downsampled_tokens(tokens=sample_tokens, timestamps=timestamps, period=TARGET_PERIOD)

    for sensor_name in sensor_names:
        downsampled_lidar_tokens = [devkit.get(table_name='sample', token=token)['data'][sensor_name] for token in downsampled_tokens]

        for scan_num, lidar_token in enumerate(downsampled_lidar_tokens):
            lidar_record = devkit.get(table_name='sample_data', token=lidar_token)

            data_root = Path(source_root)
            pc_lidar = LidarPointCloud.from_file(file_name=data_root / lidar_record['filename']).points[:3].T   # Shape (N,3).

            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            # No labels.



# Create transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'Lyft_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['LIDAR_TOP', 'LIDAR_FRONT_RIGHT', 'LIDAR_FRONT_LEFT',]

T1 = np.array([[-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 1., 1.80], [0., 0., 0., 1.]], dtype=np.float32)
T2 = np.array([[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., -1., 0.45], [0., 0., 0., 1.]], dtype=np.float32)
T3 = np.array([[0., -1., 0., 0.], [-1., 0., 0., 0.], [0., 0., -1., 0.45], [0., 0., 0., 1.]], dtype=np.float32)

serializable = {
    sensors[0]: T1.flatten().tolist(),
    sensors[1]: T2.flatten().tolist(),
    sensors[2]: T3.flatten().tolist(),
}

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'Lyft_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['LIDAR_TOP', 'LIDAR_FRONT_RIGHT', 'LIDAR_FRONT_LEFT',]
radii = [2.5, 3.0, 3.0,]

serializable = {sensor: radius for sensor, radius in zip(sensors, radii)}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

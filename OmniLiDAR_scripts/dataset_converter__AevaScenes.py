# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "numpy==1.26.4",
#     "tqdm==4.66.5",
# ]
# ///



import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List



TARGET_FREQUENCY = 0.2   # Hz.
TARGET_PERIOD = 1.0 / TARGET_FREQUENCY   # Sec.



omnilidar_root = Path('PUT_YOUR_DIRECTORY_HERE/OmniLiDAR')
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/AevaScenes')
target_root    = omnilidar_root / 'AevaScenes'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/AevaScenes', print('Directory to AevaScenes Perception dataset. Change to directory in your file system!')

omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)

print(target_root)



ground_cls = ['road', 'lane_boundary', 'road_marking', 'reflective_marker', 'sidewalk', 'other_ground']
nonground_cls = ['car', 'bus', 'truck', 'trailer', 'vehicle_on_rails', 'other_vehicle', 'bicycle', 'motorcycle', 'motorcyclist', 'bicyclist', 'pedestrian', 'animal', 'traffic_item', 'traffic_sign', 'pole_trunk', 'building', 'other_structure', 'vegetation']
ignore_cls = ['unknown']



## Helper function.
def get_downsampled_lidar_dirs(
        lidar_dirs: List,
        timestamps: List,
        period: float,
        ) -> List:
    """
    Get downsampled LiDAR directories based on irregular timestamps and a target period.

    Args:
        lidar_dirs (list) : List of LiDAR directories.
        timestamps (list) : List or array of timestamps corresponding to the LiDAR directories.
        period (float) : Target period for downsampling.
    
    Returns:
        downsampled_lidar_dirs (list) : Downsampled list of LiDAR directories.
    """
    ts = np.asarray(timestamps)
    ids, t = [0], ts[0] + period
    while (i := np.searchsorted(ts, t)) < len(ts):
        ids.append(i)
        t = ts[i] + period
    downsampled_lidar_dirs = np.asarray(lidar_dirs)[ids].tolist()
    return downsampled_lidar_dirs


## Data (OmniLiDAR train without labels).
metadata = json.load(open(source_root / 'metadata.json', 'r'))
context_names = metadata['sequence_uuids']

folder_name = 'train_scans'
sensor_names = ['front_narrow_lidar', 'front_wide_lidar', 'left_lidar', 'rear_narrow_lidar', 'rear_wide_lidar', 'right_lidar',]
sensor_frequencies = {'front_narrow_lidar': 10, 'front_wide_lidar': 10, 'left_lidar': 10, 'rear_narrow_lidar': 10, 'rear_wide_lidar': 10, 'right_lidar': 10}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

for seq_num, context_name in tqdm(list(enumerate(context_names)), desc='Scenes'):
    lidar_folder_dir = source_root / context_name / 'pointcloud_compensated'

    for sensor_name in sensor_names:
        lidar_dirs = sorted(lidar_folder_dir.glob(f'{sensor_name}*.npz'))

        timestamps = np.array([float(str(p.stem).split('_')[-1]) for p in lidar_dirs]) / 1e9
        timestamps += -timestamps[0]
        timestamps = timestamps.tolist()

        downsampled_lidar_dirs = get_downsampled_lidar_dirs(lidar_dirs=lidar_dirs, timestamps=timestamps, period=TARGET_PERIOD)

        for scan_num, lidar_dir in enumerate(downsampled_lidar_dirs):
            pc_data = np.load(lidar_dir, allow_pickle=True)

            pc_lidar = pc_data['xyz']
            labels = pc_data['semantic_labels'].ravel()

            mapped_labels = np.zeros((labels.shape[0],), dtype=np.uint8)
            mapped_labels[np.isin(labels, ground_cls)] = 0
            mapped_labels[np.isin(labels, nonground_cls)] = 1
            mapped_labels[np.isin(labels, ignore_cls)] = 2
            
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            np.save(file=label_dir / f'labels__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=mapped_labels)



# Get transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'AevaScenes_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['front_narrow_lidar', 'front_wide_lidar', 'left_lidar', 'rear_narrow_lidar', 'rear_wide_lidar', 'right_lidar',]
lidar_heights = {
    'front_narrow_lidar': 1.98,
    'front_wide_lidar': 2.00,
    'left_lidar': 1.72,
    'rear_narrow_lidar': 1.45,
    'rear_wide_lidar': 1.80,
    'right_lidar': 1.62,
}

serializable = {}
for sensor in sensors:
    T = np.eye(4)
    T[2,3] = lidar_heights[sensor]
    serializable[sensor] = T.flatten().tolist()

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'AevaScenes_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['front_narrow_lidar', 'front_wide_lidar', 'left_lidar', 'rear_narrow_lidar', 'rear_wide_lidar', 'right_lidar',]

serializable = {sensor: 0.0 for sensor in sensors}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

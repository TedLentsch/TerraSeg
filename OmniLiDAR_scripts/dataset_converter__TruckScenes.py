# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "truckscenes-devkit",
#     "numpy==1.26.4",
#     "pyquaternion>=0.9.9",
#     "tqdm==4.66.5",
# ]
# ///



import json
import numpy as np
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud
from truckscenes.utils.geometry_utils import transform_matrix
from truckscenes.utils.splits import train, val
from pathlib import Path
from pyquaternion import Quaternion
from tqdm import tqdm
from typing import List



TARGET_FREQUENCY = 0.2   # Hz.
TARGET_PERIOD = 1.0 / TARGET_FREQUENCY   # Sec.



omnilidar_root = Path('PUT_YOUR_DIRECTORY_HERE/OmniLiDAR')
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/truckscenes')
target_root    = omnilidar_root / 'TruckScenes'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/TruckScenes', print('Directory to TruckScenes dataset. Change to directory in your file system!')

omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)
    
print(target_root)



# No point-level labels available.



## Helper function.
def get_downsampled_lidar_tokens(
        lidar_tokens: List,
        timestamps: List,
        period: float,
        ) -> List:
    """
    Get downsampled LiDAR tokens based on irregular timestamps and a target period.

    Args:
        lidar_tokens (list) : List of LiDAR tokens.
        timestamps (list) : List or array of timestamps corresponding to the LiDAR tokens.
        period (float) : Target period for downsampling.
    
    Returns:
        downsampled_lidar_tokens (list) : Downsampled list of LiDAR tokens.
    """
    ts = np.asarray(timestamps)
    ids, t = [0], ts[0] + period
    while (i := np.searchsorted(ts, t)) < len(ts):
        ids.append(i)
        t = ts[i] + period
    downsampled_lidar_tokens = np.asarray(lidar_tokens)[ids].tolist()
    return downsampled_lidar_tokens


## Train split (OmniLiDAR train).
folder_name = 'train_scans'
sensor_names = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR',]
sensor_frequencies = {'LIDAR_LEFT': 10, 'LIDAR_RIGHT': 10, 'LIDAR_TOP_FRONT': 10, 'LIDAR_TOP_LEFT': 10, 'LIDAR_TOP_RIGHT': 10, 'LIDAR_REAR': 10,}
devkit = TruckScenes(version='v1.0-trainval', dataroot=str(source_root), verbose=False)

scene_name2token = {s['name']: s['token'] for s in devkit.scene}

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

for seq_num, scene_name in tqdm(list(enumerate(train)), desc='Scenes'):
    scene_token = scene_name2token[scene_name]
    scene_record = devkit.get(table_name='scene', token=scene_token)
    first_sample_token = scene_record['first_sample_token']

    for sensor_name in sensor_names:
        token = devkit.get(table_name='sample', token=first_sample_token)['data'][sensor_name]

        lidar_tokens = []
        while token:
            lidar_tokens.append(token)
            lidar_record = devkit.get('sample_data', token)
            token = lidar_record['next']

        timestamps = []
        for scan_num, lidar_token in enumerate(lidar_tokens):
            lidar_record = devkit.get(table_name='sample_data', token=lidar_token)
            timestamps.append(lidar_record['timestamp'] / 1e6)
        downsampled_lidar_tokens = get_downsampled_lidar_tokens(lidar_tokens=lidar_tokens, timestamps=timestamps, period=TARGET_PERIOD)

        for scan_num, lidar_token in enumerate(downsampled_lidar_tokens):
            lidar_record = devkit.get(table_name='sample_data', token=lidar_token)
            
            data_root = Path(source_root)
            pc_lidar = LidarPointCloud.from_file(file_name=str(data_root / lidar_record['filename'])).points[:3].T   # Shape (N,3).
            
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            # No labels.



# Create transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'TruckScenes_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR',]
extra_rotations = {'LIDAR_LEFT': 0, 'LIDAR_RIGHT': 0, 'LIDAR_TOP_FRONT': 0, 'LIDAR_TOP_LEFT': -90, 'LIDAR_TOP_RIGHT': 90, 'LIDAR_REAR': 180,}   # Degrees.
devkit = TruckScenes(version='v1.0-mini', dataroot=str(source_root), verbose=False)

preprocess_transforms = {key: np.eye(4) for key in sensors}
for sensor_name in sensors:
    if sensor_name in sensors:#['LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT']:
        first_sample_token = devkit.scene[0]['first_sample_token']
        sample_record = devkit.get('sample', first_sample_token)
        
        lidar_token = sample_record['data'][sensor_name]
        lidar_record = devkit.get('sample_data', lidar_token)
        
        lidar_sensorpose_token = lidar_record['calibrated_sensor_token']
        lidar_sensorpose_record = devkit.get('calibrated_sensor', lidar_sensorpose_token)
        
        T_vehicle_lidar = transform_matrix(lidar_sensorpose_record['translation'], Quaternion(lidar_sensorpose_record['rotation']), inverse=False).astype(np.float32)
        lidar_height = T_vehicle_lidar[2,3].item()   # Meters.

        T_lidaraligned_lidar = np.eye(4, dtype=np.float32)
        T_lidaraligned_lidar[:3, :3] = T_vehicle_lidar[:3, :3]
        T_lidaraligned_lidar[2, 3] = lidar_height

        extra_rotation = extra_rotations[sensor_name]

        if extra_rotation == 0:
            T_lidaraligned2_lidaraligned = np.eye(4, dtype=np.float32)
        else:
            c = np.cos(np.radians(extra_rotation))
            s = np.sin(np.radians(extra_rotation))
            T_lidaraligned2_lidaraligned = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        preprocess_transforms[sensor_name] = T_lidaraligned2_lidaraligned @ T_lidaraligned_lidar

serializable = {}
for sensor, T in preprocess_transforms.items():
    serializable[sensor] = T.flatten().tolist()
    
with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'TruckScenes_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR',]
radii = [4.0, 4.0, 3.0, 3.0, 3.0, 0.5]

serializable = {sensor: radius for sensor, radius in zip(sensors, radii)}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

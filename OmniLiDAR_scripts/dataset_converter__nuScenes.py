# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "nuscenes-devkit",
#     "numpy==1.26.4",
#     "pyquaternion>=0.9.9",
#     "tqdm==4.66.5",
# ]
# ///



import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import train, val
from pathlib import Path
from pyquaternion import Quaternion
from tqdm import tqdm
from typing import List



TARGET_FREQUENCY = 0.2   # Hz.
TARGET_PERIOD = 1.0 / TARGET_FREQUENCY   # Sec.



omnilidar_root = Path('PUT_YOUR_DIRECTORY_HERE/OmniLiDAR')
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/nuScenes')
target_root    = omnilidar_root / 'nuScenes'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/nuScenes', print('Directory to nuScenes dataset. Change to directory in your file system!')

omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)

print(target_root)



# Map to ``ID = 0``.
ground_cls = {
    'flat.driveable_surface': 24,
    'flat.sidewalk':          26,
    'flat.terrain':           27,
    'flat.other':             25,
}

# Map to ``ID = 1``.
nonground_cls = {
    'animal':                                1,
    'human.pedestrian.adult':                2,
    'human.pedestrian.child':                3,
    'human.pedestrian.construction_worker':  4,
    'human.pedestrian.personal_mobility':    5,
    'human.pedestrian.police_officer':       6,
    'human.pedestrian.stroller':             7,
    'human.pedestrian.wheelchair':           8,
    'movable_object.barrier':                9,
    'movable_object.debris':                10,
    'movable_object.pushable_pullable':     11,
    'movable_object.trafficcone':           12,
    'static_object.bicycle_rack':           13,
    'vehicle.bicycle':                      14,
    'vehicle.bus.bendy':                    15,
    'vehicle.bus.rigid':                    16,
    'vehicle.car':                          17,
    'vehicle.construction':                 18,
    'vehicle.emergency.ambulance':          19,
    'vehicle.emergency.police':             20,
    'vehicle.motorcycle':                   21,
    'vehicle.trailer':                      22,
    'vehicle.truck':                        23,
    'static.manmade':                       28,
    'static.other':                         29,
    'static.vegetation':                    30,
}

# Map to ``ID = 2``.
ignore_cls = {
    'noise':        0,
    'vehicle.ego': 31,
}

# Create mapping.
mapping = {v: 0 for v in ground_cls.values()}
mapping.update({v: 1 for v in nonground_cls.values()})
mapping.update({v: 2 for v in ignore_cls.values()})



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
sensor_names = ['LIDAR_TOP',]
sensor_frequencies = {'LIDAR_TOP': 20,}   # Hz.
devkit = NuScenes(version='v1.0-trainval', dataroot=str(source_root), verbose=False)

scene_name2token = {s['name']: s['token'] for s in devkit.scene}
lookup_table = np.zeros((256,), dtype=np.uint8)
for old, new in mapping.items():
    lookup_table[old] = new
    
split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

for seq_num, scene_name in tqdm(list(enumerate(train)), desc='Scenes'):
    scene_token = scene_name2token[scene_name]

    sample_tokens = []
    token = devkit.get(table_name='scene', token=scene_token)['first_sample_token']
    while token:
        sample_tokens.append(token)
        token = devkit.get(table_name='sample', token=token)['next']
        
    for sensor_name in sensor_names:
        downsampling_factor = 10   # Annotations are 2 Hz, we want 0.2 Hz.
        downsampled_sample_tokens = sample_tokens[::downsampling_factor]
        
        for scan_num, sample_token in enumerate(downsampled_sample_tokens):
            lidar_token = devkit.get(table_name='sample', token=sample_token)['data'][sensor_name]
            lidar_record = devkit.get(table_name='sample_data', token=lidar_token)
            
            data_root = Path(source_root)
            pc_lidar = LidarPointCloud.from_file(file_name=str(data_root / lidar_record['filename'])).points[:3].T   # Shape (N,3).
            
            raw_labels = np.fromfile(file=Path(devkit.dataroot) / devkit.get(table_name='lidarseg', token=lidar_record['token'])['filename'], dtype=np.uint8)
            mapped_labels = lookup_table[raw_labels]
            
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            np.save(file=label_dir / f'labels__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=mapped_labels)


## Val split (OmniLiDAR test).
folder_name = 'val_scans'
sensor_names = ['LIDAR_TOP',]
sensor_frequencies = {'LIDAR_TOP': 2,}   # Hz.
devkit = NuScenes(version='v1.0-trainval', dataroot=str(source_root), verbose=False)

scene_name2token = {s['name']: s['token'] for s in devkit.scene}
lookup_table = np.zeros((256,), dtype=np.uint8)
for old, new in mapping.items():
    lookup_table[old] = new
    
split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

for seq_num, scene_name in tqdm(list(enumerate(val)), desc='Scenes'):
    scene_token = scene_name2token[scene_name]

    sample_tokens = []
    token = devkit.get(table_name='scene', token=scene_token)['first_sample_token']
    while token:
        sample_tokens.append(token)
        token = devkit.get(table_name='sample', token=token)['next']
        
    for sensor_name in sensor_names:
        downsampling_factor = 1   # Keep all scans because this is OmniLIDAR test data.
        downsampled_sample_tokens = sample_tokens[::downsampling_factor]

        for scan_num, sample_token in enumerate(downsampled_sample_tokens):
            lidar_token = devkit.get(table_name='sample', token=sample_token)['data'][sensor_name]
            lidar_record = devkit.get(table_name='sample_data', token=lidar_token)
            
            data_root = Path(source_root)
            pc_lidar = LidarPointCloud.from_file(file_name=str(data_root / lidar_record['filename'])).points[:3].T   # Shape (N,3).
            
            raw_labels = np.fromfile(file=Path(devkit.dataroot) / devkit.get(table_name='lidarseg', token=lidar_record['token'])['filename'], dtype=np.uint8)
            mapped_labels = lookup_table[raw_labels]
            
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            np.save(file=label_dir / f'labels__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=mapped_labels)



# Get LiDAR height above ground.
devkit = NuScenes(version='v1.0-mini', dataroot=str(source_root), verbose=False)
first_sample_token = devkit.scene[0]['first_sample_token']
first_sample_record = devkit.get('sample', first_sample_token)
first_sample_lidar_record = devkit.get('sample_data', first_sample_record['data']['LIDAR_TOP'])
lidar_sensorpose_token = first_sample_lidar_record['calibrated_sensor_token']
lidar_sensorpose_record = devkit.get('calibrated_sensor', lidar_sensorpose_token)
T_vehicle_lidar = transform_matrix(lidar_sensorpose_record['translation'], Quaternion(lidar_sensorpose_record['rotation']), inverse=False).astype(np.float32)
lidar_height = float(T_vehicle_lidar[2,3])   # Meters.


# Create transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'nuScenes_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['LIDAR_TOP',]

T = np.array([[ 0., 1., 0., 0.], [ -1.,  0., 0., 0.], [ 0.,  0., 1., lidar_height], [ 0.,  0., 0., 1.]], dtype=np.float32)
serializable = {sensor: T.flatten().tolist() for sensor in sensors}

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'nuScenes_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['LIDAR_TOP',]

serializable = {sensor: 3.0 for sensor in sensors}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

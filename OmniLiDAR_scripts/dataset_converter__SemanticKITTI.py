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
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/SemanticKITTI')
target_root    = omnilidar_root / 'SemanticKITTI'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/SemanticKITTI', print('Directory to SemanticKITTI dataset. Change to directory in your file system!')

omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)

print(target_root)



# Map to ``ID = 0``.
ground_cls = {
    'road':         40,
    'parking':      44,
    'sidewalk':     48,
    'other-ground': 49,
    'lane-marking': 60,
    'terrain':      72,
}

# Map to ``ID = 1``.
nonground_cls = {
    'car':                   10,
    'bicycle':               11,
    'bus':                   13,
    'motorcycle':            15,
    'on-rails':              16,
    'truck':                 18,
    'other-vehicle':         20,
    'person':                30,
    'bicyclist':             31,
    'motorcyclist':          32,
    'building':              50,
    'fence':                 51,
    'other-structure':       52,
    'vegetation':            70,
    'trunk':                 71,
    'pole':                  80,
    'traffic-sign':          81,
    'other-object':          99,
    'moving-car':           252,
    'moving-bicyclist':     253,
    'moving-person':        254,
    'moving-motorcyclist':  255,
    'moving-on-rails':      256,
    'moving-bus':           257,
    'moving-truck':         258,
    'moving-other-vehicle': 259,
}

# Map to ``ID = 2``.
ignore_cls = {
    'unlabeled': 0,
    'outlier':   1,
}

# Create mapping.
mapping = {v: 0 for v in ground_cls.values()}
mapping.update({v: 1 for v in nonground_cls.values()})
mapping.update({v: 2 for v in ignore_cls.values()})



## Splits.
train_seq_ids = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10,]
val_seq_ids = [8,]
test_seq_ids = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,]


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


## Train split (OmniLiDAR train).
folder_name = 'train_scans'
sensor_names = ['velodyne',]
sensor_frequencies = {'velodyne': 10,}   # Hz.

lookup_table = np.zeros(max(mapping) + 1, dtype=np.uint8)
for old, new in mapping.items():
    lookup_table[old] = new
    
split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

seq_dirs = [seq_dir for seq_dir in sorted(Path(source_root, 'dataset', 'sequences').glob('*')) if int(seq_dir.name) in train_seq_ids]
for seq_num, seq_dir in tqdm(enumerate(seq_dirs), desc='Sequences'):
    seq_name = seq_dir.name

    for sensor_name in sensor_names:
        velodyne_folder_dir = seq_dir / sensor_name
        sem_label_folder_dir = seq_dir / 'labels'

        lidar_dirs = sorted(velodyne_folder_dir.glob('*.bin'))

        timestamps = np.loadtxt(seq_dir / 'times.txt', dtype=float).tolist()
        downsampled_lidar_dirs = get_downsampled_lidar_dirs(lidar_dirs=lidar_dirs, timestamps=timestamps, period=TARGET_PERIOD)
        
        for scan_num, lidar_dir in enumerate(downsampled_lidar_dirs):
            pc_lidar = np.fromfile(lidar_dir, dtype=np.float32).reshape(-1,4)[:,:3]
            
            label_file = sem_label_folder_dir / lidar_dir.with_suffix('.label').name
            raw_labels = np.fromfile(label_file, dtype=np.uint32) & 0xFFFF
            mapped_labels = lookup_table[raw_labels]
            
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            np.save(file=label_dir / f'labels__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=mapped_labels)


## Val split (OmniLiDAR test).
folder_name = 'val_scans'
sensor_names = ['velodyne',]
sensor_frequencies = {'velodyne': 10,}   # Hz.

lookup_table = np.zeros(max(mapping) + 1, dtype=np.uint8)
for old, new in mapping.items():
    lookup_table[old] = new
    
split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

seq_dirs = [seq_dir for seq_dir in sorted(Path(source_root, 'dataset', 'sequences').glob('*')) if int(seq_dir.name) in val_seq_ids]
for seq_num, seq_dir in tqdm(enumerate(seq_dirs), desc='Sequences'):
    seq_name = seq_dir.name
    
    for sensor_name in sensor_names:
        velodyne_folder_dir = seq_dir / sensor_name
        sem_label_folder_dir = seq_dir / 'labels'

        lidar_dirs = sorted(velodyne_folder_dir.glob('*.bin'))
        
        downsampled_lidar_dirs = lidar_dirs   # Keep all scans because this is OmniLIDAR test data.
        
        for scan_num, lidar_dir in enumerate(downsampled_lidar_dirs):
            pc_lidar = np.fromfile(lidar_dir, dtype=np.float32).reshape(-1,4)[:,:3]
            
            label_file = sem_label_folder_dir / lidar_dir.with_suffix('.label').name
            raw_labels = np.fromfile(label_file, dtype=np.uint32) & 0xFFFF
            mapped_labels = lookup_table[raw_labels]
            
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            np.save(file=label_dir / f'labels__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=mapped_labels)



# Create transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'SemanticKITTI_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['velodyne',]

lidar_height = 1.73   # Meters.

T = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., lidar_height], [0., 0., 0., 1.],], dtype=np.float32)
serializable = {sensor: T.flatten().tolist() for sensor in sensors}

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'SemanticKITTI_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['velodyne',]

serializable = {sensor: 3.0 for sensor in sensors}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "numpy==1.26.4",
#     "pandas>=2.0",
#     "tqdm==4.66.5",
# ]
# ///



import gzip
import json
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import List



TARGET_FREQUENCY = 0.2   # Hz.
TARGET_PERIOD = 1.0 / TARGET_FREQUENCY   # Sec.



omnilidar_root = Path('PUT_YOUR_DIRECTORY_HERE/OmniLiDAR')
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/PandaSet')
target_root    = omnilidar_root / 'PandaSet'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/PandaSet', print('Directory to PandaSet dataset. Change to directory in your file system!')
omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)

print(target_root)



# PandaSet label IDs.
PANDASET_ID2NAME = {
    0: '',
    1: 'Smoke',
    2: 'Exhaust',
    3: 'Spray or rain',
    4: 'Reflection',
    5: 'Vegetation',
    6: 'Ground',
    7: 'Road',
    8: 'Lane Line Marking',
    9: 'Stop Line Marking',
    10: 'Other Road Marking',
    11: 'Sidewalk',
    12: 'Driveway',
    13: 'Car',
    14: 'Pickup Truck',
    15: 'Medium-sized Truck',
    16: 'Semi-truck',
    17: 'Towed Object',
    18: 'Motorcycle',
    19: 'Other Vehicle - Construction Vehicle',
    20: 'Other Vehicle - Uncommon',
    21: 'Other Vehicle - Pedicab',
    22: 'Emergency Vehicle',
    23: 'Bus',
    24: 'Personal Mobility Device',
    25: 'Motorized Scooter',
    26: 'Bicycle',
    27: 'Train',
    28: 'Trolley',
    29: 'Tram / Subway',
    30: 'Pedestrian',
    31: 'Pedestrian with Object',
    32: 'Animals - Bird',
    33: 'Animals - Other',
    34: 'Pylons',
    35: 'Road Barriers',
    36: 'Signs',
    37: 'Cones',
    38: 'Construction Signs',
    39: 'Temporary Construction Barriers',
    40: 'Rolling Containers',
    41: 'Building',
    42: 'Other Static Object',
}

# Map to ``ID = 0``.
ground_cls = {
    'Ground': 6,
    'Road': 7,
    'Lane Line Marking': 8,
    'Stop Line Marking': 9,
    'Other Road Marking': 10,
    'Sidewalk': 11,
    'Driveway': 12,
}

# Map to ``ID = 2``.
ignore_cls = {
    'Smoke': 1,
    'Exhaust': 2,
    'Spray or rain': 3,
    'Reflection': 4,
}

# Map to ``ID = 1``.
nonground_cls = {
    name: idx
    for idx, name in PANDASET_ID2NAME.items()
    if idx not in set(ground_cls.values()) | set(ignore_cls.values())
}

# Create mapping.
mapping = {v: 0 for v in ground_cls.values()}
mapping.update({v: 1 for v in nonground_cls.values()})
mapping.update({v: 2 for v in ignore_cls.values()})



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


# Splits.
unlabeled_IDs = [4, 6, 8, 12, 14, 18, 20, 45, 47, 48, 50, 51, 55, 59, 62, 63, 68, 74, 79, 85, 86, 91, 92, 93, 99, 100, 104]

sequence_dirs = sorted(source_root.glob('*'))
train_sequence_dirs = [sequence_dir for sequence_dir in sequence_dirs if int(sequence_dir.name) not in unlabeled_IDs]
test_sequence_dirs = [sequence_dir for sequence_dir in sequence_dirs if int(sequence_dir.name) in unlabeled_IDs]


## Train split (OmniLiDAR train).
folder_name = 'train_scans'
sensor_names = ['Pandar64', 'PandarGT']
sensor_frequencies = {'lidar': 10}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

lidar_folder_name = 'lidar'
label_folder_name = 'annotations/semseg'
unlabeled = []
for seq_num, sequence_dir in tqdm(list(enumerate(train_sequence_dirs)), desc='Scenes'):
    with open(sequence_dir / lidar_folder_name / 'timestamps.json', 'r') as f:
        timestamps = json.load(f)

    lidar_dirs = sorted((sequence_dir / lidar_folder_name).glob('*.pkl.gz'))
    downsampled_lidar_dirs = get_downsampled_lidar_dirs(lidar_dirs=lidar_dirs, timestamps=timestamps, period=TARGET_PERIOD)

    label_dirs = [sequence_dir / label_folder_name / lidar_dir.name for lidar_dir in downsampled_lidar_dirs]
    
    for scan_num, (lidar_dir, source_label_dir) in enumerate(zip(downsampled_lidar_dirs, label_dirs)):
        with gzip.open(lidar_dir, 'rb') as f:
            pc_data = pickle.load(f)

        xyz = pc_data[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        lidar_id = pc_data['d'].to_numpy(dtype=np.int16)

        pc1_lidar = xyz[lidar_id == 0]
        pc2_lidar = xyz[lidar_id == 1]
        
        with gzip.open(source_label_dir, 'rb') as f:
            label_data = pickle.load(f)

        raw_labels = label_data['class'].to_numpy(dtype=np.int32)
        mapped_labels = np.vectorize(lambda x: mapping.get(int(x), 2))(raw_labels).astype(np.int32)

        mapped_labels1 = mapped_labels[lidar_id == 0]
        mapped_labels2 = mapped_labels[lidar_id == 1]

        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[0]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc1_lidar)
        np.save(file=label_dir / f'labels__sensor_{sensor_names[0]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=mapped_labels1)

        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[1]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc2_lidar)
        np.save(file=label_dir / f'labels__sensor_{sensor_names[1]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=mapped_labels2)


## Test split (OmniLiDAR train).
folder_name = 'test_scans'
sensor_names = ['Pandar64', 'PandarGT']
sensor_frequencies = {'lidar': 10}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

lidar_folder_name = 'lidar'
label_folder_name = 'annotations/semseg'
unlabeled = []
for seq_num, sequence_dir in tqdm(list(enumerate(test_sequence_dirs)), desc='Scenes'):
    with open(sequence_dir / lidar_folder_name / 'timestamps.json', 'r') as f:
        timestamps = json.load(f)

    lidar_dirs = sorted((sequence_dir / lidar_folder_name).glob('*.pkl.gz'))
    downsampled_lidar_dirs = get_downsampled_lidar_dirs(lidar_dirs=lidar_dirs, timestamps=timestamps, period=TARGET_PERIOD)

    label_dirs = [sequence_dir / label_folder_name / lidar_dir.name for lidar_dir in downsampled_lidar_dirs]
    
    for scan_num, (lidar_dir, source_label_dir) in enumerate(zip(downsampled_lidar_dirs, label_dirs)):
        with gzip.open(lidar_dir, 'rb') as f:
            pc_data = pickle.load(f)

        xyz = pc_data[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        lidar_id = pc_data['d'].to_numpy(dtype=np.int16)

        pc1_lidar = xyz[lidar_id == 0]
        pc2_lidar = xyz[lidar_id == 1]

        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[0]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc1_lidar)
        # No labels.

        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[1]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc2_lidar)
        # No labels.



# Create transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'PandaSet_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['Pandar64', 'PandarGT']

lidar_height = 0.30   # Meters.

T = np.array([[ 0., 1., 0., 0.], [ -1.,  0., 0., 0.], [ 0.,  0., 1., lidar_height], [ 0.,  0., 0., 1.],], dtype=np.float32)   # No rotation because LiDAR orientation seems to be random.
serializable = {sensor: T.flatten().tolist() for sensor in sensors}

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'PandaSet_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['Pandar64', 'PandarGT']
radii = [2.25, 0.0]

serializable = {sensor: radius for sensor, radius in zip(sensors, radii)}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

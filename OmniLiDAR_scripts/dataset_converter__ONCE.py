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
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/ONCE')
target_root    = omnilidar_root / 'ONCE'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/ONCE', print('Directory to ONCE dataset. Change to directory in your file system!')

omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)

print(target_root)



# No point-level labels available.



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
sensor_names = ['lidar']
sensor_frequencies = {'lidar': 10}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

sequence_folder_name = 'data'
sequence_root = Path(source_root) / sequence_folder_name
sequence_dirs = sorted(sequence_root.glob('*'))

sensor_name = sensor_names[0]
lidar_folder_name = 'lidar_roof'
for seq_num, sequence_dir in tqdm(list(enumerate(sequence_dirs)), desc='Scenes'):
    lidar_dirs = sorted((sequence_dir / lidar_folder_name).glob('*.bin'))

    timestamps = [int(lidar_dir.name.split('.bin')[0])/1e3 for lidar_dir in lidar_dirs]
    downsampled_lidar_dirs = get_downsampled_lidar_dirs(lidar_dirs=lidar_dirs, timestamps=timestamps, period=TARGET_PERIOD)

    for scan_num, lidar_dir in enumerate(downsampled_lidar_dirs):
        pc_lidar = np.fromfile(lidar_dir, dtype=np.float32).reshape(-1,4)[:,:3]

        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
        # No labels.



# Create transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'ONCE_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['lidar',]

lidar_height = 1.80   # Meters.

T = np.array([[ 0., 1., 0., 0.], [ -1.,  0., 0., 0.], [ 0.,  0., 1., lidar_height], [ 0.,  0., 0., 1.],], dtype=np.float32)
serializable = {sensor: T.flatten().tolist() for sensor in sensors}

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'ONCE_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['lidar',]

serializable = {sensor: 2.5 for sensor in sensors}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

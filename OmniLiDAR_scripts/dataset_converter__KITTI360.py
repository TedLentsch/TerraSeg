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
from typing import List
from tqdm import tqdm



TARGET_FREQUENCY = 0.2   # Hz.
TARGET_PERIOD = 1.0 / TARGET_FREQUENCY   # Sec.



omnilidar_root = Path('PUT_YOUR_DIRECTORY_HERE/OmniLiDAR')
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/KITTI360')
target_root    = omnilidar_root / 'KITTI360'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/KITTI360', print('Directory to KITTI-360 dataset. Change to directory in your file system!')

omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)

print(target_root)



# No point-level labels available (per sweep).



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
train = [
    '2013_05_28_drive_0000_sync',
    '2013_05_28_drive_0002_sync',
    '2013_05_28_drive_0003_sync',
    '2013_05_28_drive_0004_sync',
    '2013_05_28_drive_0005_sync',
    '2013_05_28_drive_0006_sync',
    '2013_05_28_drive_0007_sync',
    '2013_05_28_drive_0009_sync',
    '2013_05_28_drive_0010_sync',
]

test = [
    '2013_05_28_drive_0008_sync',
    '2013_05_28_drive_0018_sync',
]


## Train split (OmniLiDAR train).
data_3d_folder = 'data_3d_raw'
folder_name = 'train_scans'
sensor_names = ['velodyne_points',]
sensor_frequencies = {'velodyne_points': 10,}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

for seq_num, seq_folder_name in tqdm(enumerate(train), desc='Sequences'):
    seq_dir = source_root / data_3d_folder / seq_folder_name

    for sensor_name in sensor_names:
        velodyne_folder_dir = seq_dir / sensor_name / 'data'
        
        lidar_dirs = sorted(velodyne_folder_dir.glob('*.bin'))

        timestamps_encoded = np.loadtxt(seq_dir / sensor_name / 'timestamps.txt', dtype=str)
        timestamps_encoded2 = np.array([f"{d}T{t}" for d, t in timestamps_encoded])
        dt64 = timestamps_encoded2.astype('datetime64[ns]')
        timestamps = ((dt64 - dt64[0]) / np.timedelta64(1, 's')).tolist()
        downsampled_lidar_dirs = get_downsampled_lidar_dirs(lidar_dirs=lidar_dirs, timestamps=timestamps[:len(lidar_dirs)], period=TARGET_PERIOD)

        for scan_num, lidar_dir in enumerate(downsampled_lidar_dirs):
            pc_lidar = np.fromfile(lidar_dir, dtype=np.float32).reshape(-1, 4)[:,:3]

            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            # No labels available.


## Test split (OmniLiDAR train).
data_3d_folder = 'data_3d_raw'
folder_name = 'test_scans'
sensor_names = ['velodyne_points',]
sensor_frequencies = {'velodyne_points': 10,}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

for seq_num, seq_folder_name in tqdm(enumerate(test), desc='Sequences'):
    seq_dir = source_root / data_3d_folder / seq_folder_name

    for sensor_name in sensor_names:
        velodyne_folder_dir = seq_dir / sensor_name / 'data'

        lidar_dirs = sorted(velodyne_folder_dir.glob('*.bin'))
        
        timestamps = (np.arange(len(lidar_dirs)) * (1 / sensor_frequencies[sensor_name])).round(5).tolist()
        downsampled_lidar_dirs = get_downsampled_lidar_dirs(lidar_dirs=lidar_dirs, timestamps=timestamps[:len(lidar_dirs)], period=TARGET_PERIOD)

        for scan_num, lidar_dir in enumerate(downsampled_lidar_dirs):
            pc_lidar = np.fromfile(lidar_dir, dtype=np.float32).reshape(-1, 4)[:,:3]

            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            # No labels available.



# Create transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'KITTI360_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['velodyne_points',]

lidar_height = 1.73   # Meters.

T = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., lidar_height], [0., 0., 0., 1.],], dtype=np.float32)
serializable = {sensor: T.flatten().tolist() for sensor in sensors}

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'KITTI360_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['velodyne_points',]

serializable = {sensor: 2.5 for sensor in sensors}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

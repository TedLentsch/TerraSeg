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



TARGET_FREQUENCY = 0.2   # Hz.
TARGET_PERIOD = 1.0 / TARGET_FREQUENCY   # Sec.



omnilidar_root = Path('PUT_YOUR_DIRECTORY_HERE/OmniLiDAR')
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/VoD')
target_root    = omnilidar_root / 'VoD'

assert str(omnilidar_root)!='PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)!='PUT_YOUR_DIRECTORY_HERE/VoD', print('Directory to KITTI-360 dataset. Change to directory in your file system!')

omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)

print(target_root)



# No point-level labels available (per sweep).



## Sequence meta data.
SCENES = {
    'train': {
        '01': {'start_frame': '00544', 'end_frame': '01311'},
        '02': {'start_frame': '01312', 'end_frame': '01802'},
        '03': {'start_frame': '01803', 'end_frame': '02199'},
        '04': {'start_frame': '02200', 'end_frame': '02531'},
        '07': {'start_frame': '03277', 'end_frame': '03574'},
        '09': {'start_frame': '03610', 'end_frame': '04047'},
        '10': {'start_frame': '04049', 'end_frame': '04386'},
        '11': {'start_frame': '04387', 'end_frame': '04651'},
        '15': {'start_frame': '06759', 'end_frame': '07542'},
        '19': {'start_frame': '08481', 'end_frame': '08748'},
        '20': {'start_frame': '08749', 'end_frame': '09095'},
        '21': {'start_frame': '09518', 'end_frame': '09775'},
        '22': {'start_frame': '09776', 'end_frame': '09930'},
    },
    'val': {
        '00': {'start_frame': '00000', 'end_frame': '00543'},
        '08': {'start_frame': '03575', 'end_frame': '03609'},
        '12': {'start_frame': '04652', 'end_frame': '05085'},
        '18': {'start_frame': '08198', 'end_frame': '08480'},
    },
    'test': {
        '05': {'start_frame': '02532', 'end_frame': '02797'},
        '06': {'start_frame': '02798', 'end_frame': '03276'},
        '13': {'start_frame': '06334', 'end_frame': '06570'},
        '14': {'start_frame': '06571', 'end_frame': '06758'},
        '16': {'start_frame': '07543', 'end_frame': '07899'},
        '17': {'start_frame': '07900', 'end_frame': '08197'},
        '21': {'start_frame': '09096', 'end_frame': '09517'},
    },
}


## Train split (OmniLiDAR train without labels).
folder_name = 'train_scans'
sensor_names = ['velodyne',]
sensor_frequencies = {'velodyne': 10,}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

sensor_name = sensor_names[0]
for seq_num, (scene_id, scene_info) in tqdm(enumerate(SCENES['train'].items()), desc='Scenes'):
    start_frame = int(scene_info['start_frame'])
    end_frame   = int(scene_info['end_frame'])

    lidar_files = [f'{frame_idx:05d}.bin' for frame_idx in range(start_frame, end_frame + 1)]
    
    downsampling_factor = int(sensor_frequencies[sensor_name] / TARGET_FREQUENCY)
    downsampled_lidar_files = lidar_files[::downsampling_factor]

    for scan_num, lidar_filename in enumerate(downsampled_lidar_files):
        lidar_dir = source_root / 'lidar' / 'training' / 'velodyne' / lidar_filename

        pc_lidar = np.fromfile(lidar_dir, dtype=np.float32).reshape(-1,4)[:,:3]

        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
        # No labels.


## Val split (OmniLiDAR train without labels).
folder_name = 'val_scans'
sensor_names = ['velodyne',]
sensor_frequencies = {'velodyne': 10,}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

sensor_name = sensor_names[0]
for seq_num, (scene_id, scene_info) in tqdm(enumerate(SCENES['val'].items()), desc='Scenes'):
    start_frame = int(scene_info['start_frame'])
    end_frame   = int(scene_info['end_frame'])

    lidar_files = [f'{frame_idx:05d}.bin' for frame_idx in range(start_frame, end_frame + 1)]
    
    downsampling_factor = int(sensor_frequencies[sensor_name] / TARGET_FREQUENCY)
    downsampled_lidar_files = lidar_files[::downsampling_factor]

    for scan_num, lidar_filename in enumerate(downsampled_lidar_files):
        lidar_dir = source_root / 'lidar' / 'training' / 'velodyne' / lidar_filename

        pc_lidar = np.fromfile(lidar_dir, dtype=np.float32).reshape(-1,4)[:,:3]

        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
        # No labels.


## Test split (OmniLiDAR train without labels).
folder_name = 'test_scans'
sensor_names = ['velodyne',]
sensor_frequencies = {'velodyne': 10,}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

sensor_name = sensor_names[0]
for seq_num, (scene_id, scene_info) in tqdm(enumerate(SCENES['test'].items()), desc='Scenes'):
    start_frame = int(scene_info['start_frame'])
    end_frame   = int(scene_info['end_frame'])

    lidar_files = [f'{frame_idx:05d}.bin' for frame_idx in range(start_frame, end_frame + 1)]
    
    downsampling_factor = int(sensor_frequencies[sensor_name] / TARGET_FREQUENCY)
    downsampled_lidar_files = lidar_files[::downsampling_factor]

    for scan_num, lidar_filename in enumerate(downsampled_lidar_files):
        lidar_dir = source_root / 'lidar' / 'training' / 'velodyne' / lidar_filename

        pc_lidar = np.fromfile(lidar_dir, dtype=np.float32).reshape(-1,4)[:,:3]

        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
        # No labels.



# Create transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'VoD_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['velodyne',]

lidar_height = 1.73   # Meters.

T = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., lidar_height], [0., 0., 0., 1.],], dtype=np.float32)
serializable = {sensor: T.flatten().tolist() for sensor in sensors}

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'VoD_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['velodyne',]

serializable = {sensor: 2.25 for sensor in sensors}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

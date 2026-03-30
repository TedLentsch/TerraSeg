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
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/ZOD')
target_root    = omnilidar_root / 'ZOD'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/ZOD', print('Directory to ZOD dataset. Change to directory in your file system!')

omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)

print(target_root)



# No point-level labels available.



## Train split (OmniLiDAR train).
folder_name = 'train_scans'
sensor_names = ['velodyne_main', 'velodyne_left', 'velodyne_right']
sensor_frequencies = {'velodyne_main': 10, 'velodyne_left': 10, 'velodyne_right': 10}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

sequence_folder_name = 'single_frames'
sequence_root = Path(source_root) / sequence_folder_name
sequence_dirs = sorted(sequence_root.glob('*'))

lidar_folder_name = 'lidar_velodyne'
for seq_num, sequence_dir in tqdm(list(enumerate(sequence_dirs)), desc='Scenes'):
    lidar_dirs = sorted((sequence_dir / lidar_folder_name).glob('*.npy'))

    frame_lidar_dir = lidar_dirs[0]   # Single frames. Note: We only download the center frames!
    data_lidar = np.load(frame_lidar_dir)

    combi_pc_lidar = np.stack([data_lidar['x'], data_lidar['y'], data_lidar['z']], axis=1)   # Shape (N,3).

    diode_ids = data_lidar['diode_index']
    main_bool = (diode_ids <= 128)
    right_bool = (diode_ids > 128) & (diode_ids <= 128 + 16)
    left_bool = (diode_ids > 128 + 16) & (diode_ids <= 128 + 16 + 16)

    main_pc_lidar = combi_pc_lidar[main_bool]
    right_pc_lidar = combi_pc_lidar[right_bool]
    left_pc_lidar = combi_pc_lidar[left_bool]

    np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[0]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(0).zfill(8)}.npy', arr=main_pc_lidar)
    # No labels.

    np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[1]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(0).zfill(8)}.npy', arr=left_pc_lidar)
    # No labels.

    np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[2]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(0).zfill(8)}.npy', arr=right_pc_lidar)
    # No labels.



# Create transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'ZOD_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['velodyne_main', 'velodyne_left', 'velodyne_right']

lidar_height = 2.18   # Meters.

T = np.array([[ 0., 1., 0., 0.], [ -1.,  0., 0., 0.], [ 0.,  0., 1., lidar_height], [ 0.,  0., 0., 1.],], dtype=np.float32)
serializable = {sensor: T.flatten().tolist() for sensor in sensors}

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'ZOD_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['velodyne_main', 'velodyne_left', 'velodyne_right']

serializable = {sensor: 2.75 for sensor in sensors}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

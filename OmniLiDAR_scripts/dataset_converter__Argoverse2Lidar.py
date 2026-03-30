# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "numpy==1.26.4",
#     "pyarrow==16.0.0",
#     "tqdm==4.66.5",
# ]
# ///



import json
import numpy as np
import pyarrow.feather as feather
from pathlib import Path
from tqdm import tqdm
from typing import List



TARGET_FREQUENCY = 0.2   # Hz.
TARGET_PERIOD = 1.0 / TARGET_FREQUENCY   # Sec.



omnilidar_root = Path('PUT_YOUR_DIRECTORY_HERE/OmniLiDAR')
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/AV2_lidar')
target_root    = omnilidar_root / 'AV2_Lidar'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/AV2_lidar', print('Directory to Argoverse 2 Lidar dataset. Change to directory in your file system!')

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
EMPTY_PC = np.empty((0, 3), dtype=np.float32)

split_name = 'train'
folder_name = 'train_scans'
sensor_names = ['Lidar1', 'Lidar2',]
sensor_frequencies = {'Lidar1': 10, 'Lidar2': 10,}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

seq_dirs = sorted([p for p in (source_root / split_name).iterdir() if p.is_dir()])
for seq_num, seq_dir in tqdm(list(enumerate(seq_dirs)), desc='Sequences'):
    lidar_dirs = sorted((seq_dir / 'sensors/lidar').rglob('*.feather'))

    timestamps = [int(lidar_dir.stem) / 1e9 for lidar_dir in lidar_dirs]

    downsampled_lidar_dirs = get_downsampled_lidar_dirs(lidar_dirs=lidar_dirs, timestamps=timestamps, period=TARGET_PERIOD)

    for scan_num, lidar_dir in enumerate(downsampled_lidar_dirs):
        table = feather.read_table(lidar_dir, columns=['x', 'y', 'z'], memory_map=True)
        
        xcol = table.column('x')
        ycol = table.column('y')
        zcol = table.column('z')
        
        # Lidar1 (no labels available).
        x1 = xcol.chunk(0).to_numpy()
        y1 = ycol.chunk(0).to_numpy()
        z1 = zcol.chunk(0).to_numpy()
        pc1_lidar = np.column_stack((x1, y1, z1)).astype(np.float32, copy=False)
        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[0]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc1_lidar, allow_pickle=False)
        # No labels.

        # Lidar2 (no labels available).
        num_chunks = min(xcol.num_chunks, ycol.num_chunks, zcol.num_chunks)
        if num_chunks >= 2:
            x2 = xcol.chunk(1).to_numpy()
            y2 = ycol.chunk(1).to_numpy()
            z2 = zcol.chunk(1).to_numpy()
            pc2_lidar = np.column_stack((x2, y2, z2)).astype(np.float32, copy=False)
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[1]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc2_lidar, allow_pickle=False)

        else:
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[1]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=EMPTY_PC, allow_pickle=False)


## Val split (OmniLiDAR train).
EMPTY_PC = np.empty((0, 3), dtype=np.float32)

split_name = 'val'
folder_name = 'val_scans'
sensor_names = ['Lidar1', 'Lidar2',]
sensor_frequencies = {'Lidar1': 10, 'Lidar2': 10,}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

seq_dirs = sorted([p for p in (source_root / split_name).iterdir() if p.is_dir()])
for seq_num, seq_dir in tqdm(list(enumerate(seq_dirs)), desc='Sequences'):
    lidar_dirs = sorted((seq_dir / 'sensors/lidar').rglob('*.feather'))

    timestamps = [int(lidar_dir.stem) / 1e9 for lidar_dir in lidar_dirs]

    downsampled_lidar_dirs = get_downsampled_lidar_dirs(lidar_dirs=lidar_dirs, timestamps=timestamps, period=TARGET_PERIOD)

    for scan_num, lidar_dir in enumerate(downsampled_lidar_dirs):
        table = feather.read_table(lidar_dir, columns=['x', 'y', 'z'], memory_map=True)
        
        xcol = table.column('x')
        ycol = table.column('y')
        zcol = table.column('z')
        
        # Lidar1 (no labels available).
        x1 = xcol.chunk(0).to_numpy()
        y1 = ycol.chunk(0).to_numpy()
        z1 = zcol.chunk(0).to_numpy()
        pc1_lidar = np.column_stack((x1, y1, z1)).astype(np.float32, copy=False)
        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[0]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc1_lidar, allow_pickle=False)
        # No labels.

        # Lidar2 (no labels available).
        num_chunks = min(xcol.num_chunks, ycol.num_chunks, zcol.num_chunks)
        if num_chunks >= 2:
            x2 = xcol.chunk(1).to_numpy()
            y2 = ycol.chunk(1).to_numpy()
            z2 = zcol.chunk(1).to_numpy()
            pc2_lidar = np.column_stack((x2, y2, z2)).astype(np.float32, copy=False)
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[1]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc2_lidar, allow_pickle=False)

        else:
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[1]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=EMPTY_PC, allow_pickle=False)


# Test split (OmniLiDAR train).
EMPTY_PC = np.empty((0, 3), dtype=np.float32)

split_name = 'test'
folder_name = 'test_scans'
sensor_names = ['Lidar1', 'Lidar2',]
sensor_frequencies = {'Lidar1': 10, 'Lidar2': 10,}   # Hz.

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

seq_dirs = sorted([p for p in (source_root / split_name).iterdir() if p.is_dir()])
for seq_num, seq_dir in tqdm(list(enumerate(seq_dirs)), desc='Sequences'):
    lidar_dirs = sorted((seq_dir / 'sensors/lidar').rglob('*.feather'))

    timestamps = [int(lidar_dir.stem) / 1e9 for lidar_dir in lidar_dirs]

    downsampled_lidar_dirs = get_downsampled_lidar_dirs(lidar_dirs=lidar_dirs, timestamps=timestamps, period=TARGET_PERIOD)

    for scan_num, lidar_dir in enumerate(downsampled_lidar_dirs):
        table = feather.read_table(lidar_dir, columns=['x', 'y', 'z'], memory_map=True)
        
        xcol = table.column('x')
        ycol = table.column('y')
        zcol = table.column('z')
        
        # Lidar1 (no labels available).
        x1 = xcol.chunk(0).to_numpy()
        y1 = ycol.chunk(0).to_numpy()
        z1 = zcol.chunk(0).to_numpy()
        pc1_lidar = np.column_stack((x1, y1, z1)).astype(np.float32, copy=False)
        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[0]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc1_lidar, allow_pickle=False)
        # No labels.

        # Lidar2 (no labels available).
        num_chunks = min(xcol.num_chunks, ycol.num_chunks, zcol.num_chunks)
        if num_chunks >= 2:
            x2 = xcol.chunk(1).to_numpy()
            y2 = ycol.chunk(1).to_numpy()
            z2 = zcol.chunk(1).to_numpy()
            pc2_lidar = np.column_stack((x2, y2, z2)).astype(np.float32, copy=False)
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[1]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc2_lidar, allow_pickle=False)
            
        else:
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_names[1]}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=EMPTY_PC, allow_pickle=False)



# Create transforms.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'AV2_lidar_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['Lidar1', 'Lidar2',]

lidar_height = 0.30   # Meters.

T = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., lidar_height], [0., 0., 0., 1.],], dtype=np.float32)
serializable = {sensor: T.flatten().tolist() for sensor in sensors}

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'AV2_lidar_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['Lidar1', 'Lidar2',]

serializable = {sensor: 2.0 for sensor in sensors}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

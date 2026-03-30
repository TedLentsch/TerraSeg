# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "dask==2023.3.1",
#     "jaxlib==0.4.13",
#     "numpy==1.23.5",
#     "pandas==1.5.3",
#     "pyarrow==16.0.0",
#     "tensorflow==2.13.0",
#     "tqdm==4.66.5",
#     "setuptools==67.6.0",
#     "waymo-open-dataset-tf-2-12-0==1.6.7",
# ]
# ///



import dask.dataframe as dd
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from typing import List
from waymo_open_dataset import dataset_pb2, v2



TARGET_FREQUENCY = 0.2   # Hz.
TARGET_PERIOD = 1.0 / TARGET_FREQUENCY   # Sec.



omnilidar_root = Path('PUT_YOUR_DIRECTORY_HERE/OmniLiDAR')
source_root    = Path('PUT_YOUR_DIRECTORY_HERE/WaymoPerception')
target_root    = omnilidar_root / 'WaymoPerception'

assert str(omnilidar_root) != 'PUT_YOUR_DIRECTORY_HERE/OmniLiDAR', print('Folder for OmniLiDAR dataset. Change to directory in your file system!')
assert str(source_root)    != 'PUT_YOUR_DIRECTORY_HERE/WaymoPerception', print('Directory to Waymo Open Perception dataset. Change to directory in your file system!')

omnilidar_root.mkdir(exist_ok=True, parents=True)
target_root.mkdir(exist_ok=True, parents=True)

print(f'Target: {target_root}')

# %%
# Ground classes ``ID = 0``.
ground_cls = {
    'curb':         17,
    'road':         18,
    'lane_marker':  19,
    'other_ground': 20,
    'walkable':     21,
    'sidewalk':     22,
}

# Non-ground classes ``ID = 1``.
nonground_cls = {
    'car':                1,
    'truck':              2,
    'bus':                3,
    'other_vehicle':      4,
    'motorcyclist':       5,
    'bicyclist':          6,
    'pedestrian':         7,
    'sign':               8,
    'traffic_light':      9,
    'pole':              10,
    'construction_cone': 11,
    'bicycle':           12,
    'motorcycle':        13,
    'building':          14,
    'vegetation':        15,
    'tree_trunk':        16,
}

# Ignore classes ``ID = 2``.
ignore_cls = {
    'undefined': 0,
}

# Create mapping.
mapping = {v: 0 for v in ground_cls.values()}
mapping.update({v: 1 for v in nonground_cls.values()})
mapping.update({v: 2 for v in ignore_cls.values()})



## Helper functions.
def read(source_root: Path, split_name: str, tag_name: str, context_name: str):
    file_dir = source_root / split_name / tag_name / context_name
    df = dd.read_parquet(file_dir)
    return df


def get_downsampled_row_nums(
        row_nums: List,
        timestamps: List,
        period: float,
        ) -> List:
    """
    Get downsampled row numbers based on irregular timestamps and a target period.

    Args:
        row_nums (list) : List of row numbers.
        timestamps (list) : List or array of timestamps corresponding to the row numbers.
        period (float) : Target period for downsampling.
    
    Returns:
        downsampled_row_nums (list) : Downsampled list of row numbers.
    """
    ts = np.asarray(timestamps)
    ids, t = [0], ts[0] + period
    while (i := np.searchsorted(ts, t)) < len(ts):
        ids.append(i)
        t = ts[i] + period
    downsampled_row_nums = np.asarray(row_nums)[ids].tolist()
    return downsampled_row_nums


def convert_range_image_to_point_cloud_labels_v2(
        range_image: v2.perception.lidar.RangeImage,
        segmentation_label: v2.perception.segmentation.LiDARSegmentationRangeImage,
        ) -> tf.Tensor:
    """
    Convert range image segmentation labels to point labels.
    
    Args:
        range_image (waymo_open_dataset.v2.perception.lidar.RangeImage) : A RangeImage object containing the range image data.
        segmentation_label (waymo_open_dataset.v2.perception.segmentation.LiDARSegmentationRangeImage) : A LiDARSegmentationRangeImage object containing the segmentation labels.

    Returns:
        segmentation_label_points_tensor (tf.Tensor) : A tensor of shape (N,) containing the segmentation labels for each point in the point cloud.
    """
    range_image_tensor = range_image.tensor
    range_image_mask = range_image_tensor[..., 0] > 0
    segmentation_label_tensor = segmentation_label.tensor
    segmentation_label_points_tensor = tf.gather_nd(segmentation_label_tensor, tf.where(range_image_mask))
    return segmentation_label_points_tensor


## Train split (OmniLiDAR train): part 1.
folder_name = 'train_scans__TOP'
sensor_names = ['TOP']
sensor_name2id = {name: getattr(dataset_pb2.LaserName, name) for name in sensor_names}
sensor_id2name = {v: k for k, v in sensor_name2id.items()}

lookup_table = np.zeros((256,), dtype=np.uint8)
for old, new in mapping.items():
    lookup_table[old] = new

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

split_name = 'training'
sensor_name = sensor_names[0]

lidar_tag_name = 'lidar'
label_tag_name = 'lidar_segmentation'
calib_tag_name = 'lidar_calibration'

context_names = sorted([Path(context_dir).name for context_dir in glob.glob(str(source_root / split_name / lidar_tag_name / '*.parquet'))])

for seq_num, context_name in tqdm(enumerate(context_names), desc='Contexts'):
    lidar_df = read(source_root=source_root, split_name=split_name, tag_name=lidar_tag_name, context_name=context_name)
    label_df = read(source_root=source_root, split_name=split_name, tag_name=label_tag_name, context_name=context_name)
    calib_df = read(source_root=source_root, split_name=split_name, tag_name=calib_tag_name, context_name=context_name)

    merged1_df = v2.merge(lidar_df, calib_df)
    df = v2.merge(merged1_df, label_df)

    df_pd = df.compute()
    df_pd = df_pd[df_pd['key.laser_name'] == sensor_name2id[sensor_name]]
    df_pd = df_pd.sort_values('key.frame_timestamp_micros').reset_index(drop=True)

    timestamps_sec = (df_pd['key.frame_timestamp_micros'].to_numpy() * 1e-6).tolist()   # Unit: seconds.
    row_nums = np.arange(len(df_pd)).tolist()   # Unit: 1.
    downsampled_row_nums = get_downsampled_row_nums(row_nums=row_nums, timestamps=timestamps_sec, period=TARGET_PERIOD-1e-3)

    for scan_num, row_idx in enumerate(downsampled_row_nums):
        data_row   = df_pd.iloc[row_idx]

        lidar_data  = v2.LiDARComponent.from_dict(data_row)
        lidar_calib = v2.LiDARCalibrationComponent.from_dict(data_row)
        lidar_label = v2.LiDARSegmentationLabelComponent.from_dict(data_row)

        pc_lidar = v2.convert_range_image_to_point_cloud(range_image=lidar_data.range_image_return1, calibration=lidar_calib, keep_polar_features=False).numpy()
        
        raw_labels = convert_range_image_to_point_cloud_labels_v2(range_image=lidar_data.range_image_return1, segmentation_label=lidar_label.range_image_return1).numpy()[...,1]
        mapped_labels = lookup_table[raw_labels]

        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
        np.save(file=label_dir / f'labels__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=mapped_labels)


## Train split (OmniLiDAR train): part 2.
folder_name = 'train_scans__OTHERS'
sensor_names = ['FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
sensor_name2id = {name: getattr(dataset_pb2.LaserName, name) for name in sensor_names}
sensor_id2name = {v: k for k, v in sensor_name2id.items()}

split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

split_name = 'training'

lidar_tag_name = 'lidar'
calib_tag_name = 'lidar_calibration'

context_names = sorted([Path(context_dir).name for context_dir in glob.glob(str(source_root / split_name / lidar_tag_name / '*.parquet'))])

for seq_num, context_name in tqdm(enumerate(context_names), desc='Contexts'):
    lidar_df_keys = dd.read_parquet(source_root / split_name / lidar_tag_name / context_name, columns=['key.laser_name', 'key.frame_timestamp_micros'])
    pdf = lidar_df_keys.astype({'key.laser_name': 'int64'}).compute()
    pdf['sensor_name'] = pdf['key.laser_name'].map(sensor_id2name)
    pdf = pdf[pdf['sensor_name'].isin(sensor_names)].sort_values(['sensor_name', 'key.frame_timestamp_micros']).reset_index(drop=True)

    selected = []
    for sensor_name in sensor_names:
        sensor_pdf = pdf[pdf['sensor_name'] == sensor_name]
        if not sensor_pdf.empty:
            timestamps = (sensor_pdf['key.frame_timestamp_micros'].to_numpy() / 1e6).tolist()
            row_ids = get_downsampled_row_nums(row_nums=list(range(len(sensor_pdf))), timestamps=timestamps, period=TARGET_PERIOD)
            selected.append(sensor_pdf.iloc[row_ids][['key.laser_name', 'key.frame_timestamp_micros']])
    selected_keys = pd.concat(selected, ignore_index=True)
    
    lidar_df_full = read(source_root=source_root, split_name=split_name, tag_name=lidar_tag_name, context_name=context_name)
    calib_df_full = read(source_root=source_root, split_name=split_name, tag_name=calib_tag_name, context_name=context_name)
    df = v2.merge(lidar_df_full, calib_df_full)
    df_selected = df.merge(selected_keys, on=['key.laser_name', 'key.frame_timestamp_micros'], how='inner').compute()
    df_selected['sensor_name'] = df_selected['key.laser_name'].astype(int).map(sensor_id2name)
    df_selected = df_selected.sort_values(['sensor_name', 'key.frame_timestamp_micros']).reset_index(drop=True)
    
    for sensor_name, group in df_selected.groupby('sensor_name'):
        for scan_num, (_, data_row) in enumerate(group.iterrows()):
            lidar_data = v2.LiDARComponent.from_dict(data_row)
            lidar_calib = v2.LiDARCalibrationComponent.from_dict(data_row)

            pc_lidar = v2.convert_range_image_to_point_cloud(range_image=lidar_data.range_image_return1, calibration=lidar_calib, keep_polar_features=False).numpy()
            
            np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
            # No labels.


## Val split (OmniLiDAR test).
folder_name = 'val_scans'
sensor_names = ['TOP']
sensor_name2id = {name: getattr(dataset_pb2.LaserName, name) for name in sensor_names}
sensor_id2name = {v: k for k, v in sensor_name2id.items()}

lookup_table = np.zeros((256,), dtype=np.uint8)
for old, new in mapping.items():
    lookup_table[old] = new
    
split_root = Path(target_root) / folder_name
(pc_dir := split_root / 'pointcloud').mkdir(exist_ok=True, parents=True)
(label_dir := split_root / 'labels').mkdir(exist_ok=True, parents=True)

split_name = 'validation'
sensor_name = sensor_names[0]

lidar_tag_name = 'lidar'
label_tag_name = 'lidar_segmentation'
calib_tag_name = 'lidar_calibration'

context_names = sorted([Path(context_dir).name for context_dir in glob.glob(str(source_root / split_name / lidar_tag_name / '*.parquet'))])

for seq_num, context_name in tqdm(enumerate(context_names), desc='Contexts'):
    lidar_df = read(source_root=source_root, split_name=split_name, tag_name=lidar_tag_name, context_name=context_name)
    label_df = read(source_root=source_root, split_name=split_name, tag_name=label_tag_name, context_name=context_name)
    calib_df = read(source_root=source_root, split_name=split_name, tag_name=calib_tag_name, context_name=context_name)

    merged1_df = v2.merge(lidar_df, calib_df)
    df = v2.merge(merged1_df, label_df)
    
    for scan_num, (_, data_row) in enumerate(iter(df.iterrows())):
        lidar_data = v2.LiDARComponent.from_dict(data_row)
        lidar_calib = v2.LiDARCalibrationComponent.from_dict(data_row)
        lidar_label = v2.LiDARSegmentationLabelComponent.from_dict(data_row)

        pc_lidar = v2.convert_range_image_to_point_cloud(range_image=lidar_data.range_image_return1, calibration=lidar_calib, keep_polar_features=False).numpy()
        
        raw_labels = convert_range_image_to_point_cloud_labels_v2(range_image=lidar_data.range_image_return1, segmentation_label=lidar_label.range_image_return1).numpy()[...,1]
        mapped_labels = lookup_table[raw_labels]

        np.save(file=pc_dir / f'pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=pc_lidar)
        np.save(file=label_dir / f'labels__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy', arr=mapped_labels)



# Get transform.
transforms_root = omnilidar_root / 'transforms'
transforms_file_dir = transforms_root / 'WaymoPerception_transforms.json'
transforms_root.mkdir(exist_ok=True, parents=True)

sensors = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR',]
yaw_rot_degs = {
    'TOP':         0.0,
    'FRONT':       0.0,
    'SIDE_LEFT':  -np.pi / 2,
    'SIDE_RIGHT':  np.pi / 2,
    'REAR':        np.pi,
}
translations = {
    'TOP':         [ 0.0,  0.0, 0.0],
    'FRONT':       [-2.8,  0.1, 0.0],
    'SIDE_LEFT':   [ 0.0,  3.2, 0.0],
    'SIDE_RIGHT':  [ 0.0, -3.3, 0.0],
    'REAR':        [ 0.0,  0.0, 0.0],
}

serializable = {}
for sensor_name in sensors:
    yaw = yaw_rot_degs[sensor_name]
    c, s = np.cos(yaw), np.sin(yaw)
    x, y, z = translations[sensor_name]
    T = np.array([[ c, -s, 0., x], [ s,  c, 0., y], [ 0., 0., 1., z], [ 0., 0., 0., 1.],], dtype=float)
    serializable[sensor_name] = T.reshape(-1).tolist()

with open(transforms_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)



# Create ego removal radius file.
egoremovalradius_root = omnilidar_root / 'egoremovalradii'
egoremovalradius_file_dir = egoremovalradius_root / 'WaymoPerception_egoremovalradii.json'
egoremovalradius_root.mkdir(exist_ok=True, parents=True)

sensors = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR',]

serializable = {sensor: 6.0 if sensor == 'TOP' else 0.0 for sensor in sensors}

with open(egoremovalradius_file_dir, 'w') as f:
    json.dump(serializable, f, indent=4)

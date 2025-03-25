# nuscenes_extractor.py

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import numpy as np
import os


def get_car_tracks(scene_name='scene-0061', dataroot='./nuscenes', min_track_length=20):
    """
    Extracts vehicle tracks and ego poses from a nuScenes scene.
    Args:
        scene_name: name of the scene to extract
        dataroot: path to nuScenes dataset root
        min_track_length: minimum number of frames a car must appear in
    Returns:
        dict with:
            'ego_poses': list of (x, y, yaw) ego poses in world frame
            'tracks': dict of {instance_token: list of (timestamp, [x, y, yaw]) in ego frame}
    """
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)

    # Find the desired scene
    scene = next(s for s in nusc.scene if s['name'] == scene_name)
    sample_token = scene['first_sample_token']

    tracks = {}
    ego_poses = []
    timestamps = []

    while sample_token:
        sample = nusc.get('sample', sample_token)
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose_data = nusc.get('ego_pose', sample_data['ego_pose_token'])

        # Ego pose in world frame
        ego_translation = np.array(ego_pose_data['translation'][:2])
        ego_rotation = Quaternion(ego_pose_data['rotation'])
        ego_yaw = ego_rotation.yaw_pitch_roll[0]
        ego_pose = np.array([ego_translation[0], ego_translation[1], ego_yaw])
        ego_poses.append(ego_pose)
        timestamps.append(sample_data['timestamp'])

        # Process all annotations in the sample
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            if not ann['category_name'].startswith('vehicle.car'):
                continue

            instance_token = ann['instance_token']
            box_translation = np.array(ann['translation'][:2])
            box_yaw = Quaternion(ann['rotation']).yaw_pitch_roll[0]

            # Transform annotation from world frame to ego frame
            T_world_ego = transform_matrix(ego_pose_data['translation'],
                                           Quaternion(ego_pose_data['rotation']), inverse=True)
            pos_world = np.array(ann['translation'] + [1.0])
            pos_ego = T_world_ego @ pos_world
            theta_ego = box_yaw - ego_yaw

            if instance_token not in tracks:
                tracks[instance_token] = []

            tracks[instance_token].append((sample_data['timestamp'],
                                           np.array([pos_ego[0], pos_ego[1], theta_ego])))

        # Move to next sample
        sample_token = sample['next']

    # Filter out short tracks
    filtered_tracks = {k: v for k, v in tracks.items() if len(v) >= min_track_length}

    return {
        'ego_poses': ego_poses,
        'timestamps': timestamps,
        'tracks': filtered_tracks
    }


if __name__ == "__main__":
    data = get_car_tracks()
    print(f"Extracted {len(data['ego_poses'])} ego poses.")
    print(f"Found {len(data['tracks'])} car tracks with sufficient length.")
    for instance, track in list(data['tracks'].items())[:1]:
        print(f"Example track {instance}, length {len(track)}:")
        for t, obs in track[:5]:
            print(f"  t={t}: z_ego = {obs}")

    # dataroot = /home/thh3/data/datasets/Nuscenes/

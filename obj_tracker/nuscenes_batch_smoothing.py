
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from pprint import pprint
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# ---------------------------
# Extract from nuScenes
# ---------------------------

def get_car_tracks(scene_name='scene-0061', dataroot='./nuscenes', min_track_length=20):
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
    scene = next(s for s in nusc.scene if s['name'] == scene_name)
    sample_token = scene['first_sample_token']

    tracks = {}
    ego_poses = []
    timestamps = []

    while sample_token:
        sample = nusc.get('sample', sample_token)
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose_data = nusc.get('ego_pose', sample_data['ego_pose_token'])

        ego_translation = np.array(ego_pose_data['translation'][:2])
        ego_rotation = Quaternion(ego_pose_data['rotation'])
        ego_yaw = ego_rotation.yaw_pitch_roll[0]
        ego_pose = np.array([ego_translation[0], ego_translation[1], ego_yaw])
        ego_poses.append(ego_pose)
        timestamps.append(sample_data['timestamp'])

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            if not ann['category_name'].startswith('vehicle.car'):
                continue

            instance_token = ann['instance_token']
            box_yaw = Quaternion(ann['rotation']).yaw_pitch_roll[0]

            T_world_ego = transform_matrix(ego_pose_data['translation'],
                                           Quaternion(ego_pose_data['rotation']), inverse=True)
            pos_world = np.array(ann['translation'] + [1.0])
            pos_ego = T_world_ego @ pos_world
            theta_ego = box_yaw - ego_yaw

            if instance_token not in tracks:
                tracks[instance_token] = []

            tracks[instance_token].append((sample_data['timestamp'],
                                           np.array([pos_ego[0], pos_ego[1], theta_ego])))

        sample_token = sample['next']

    filtered_tracks = {k: v for k, v in tracks.items() if len(v) >= min_track_length}
    return {'ego_poses': ego_poses, 'timestamps': timestamps, 'tracks': filtered_tracks}


def prepare_smoothing_input(track_data, ego_poses, timestamps):
    ego_pose_map = {ts: pose for ts, pose in zip(timestamps, ego_poses)}
    observations_ego, ego_poses_aligned = [], []

    for ts, obs in track_data:
        if ts in ego_pose_map:
            observations_ego.append(obs)
            ego_poses_aligned.append(ego_pose_map[ts])

    return np.array(observations_ego), np.array(ego_poses_aligned)


def transform_ego_to_world(z_ego, ego_pose):
    x_e, y_e, phi_e = ego_pose
    R = np.array([
        [np.cos(phi_e), -np.sin(phi_e)],
        [np.sin(phi_e),  np.cos(phi_e)]
    ])
    pos_world = R @ z_ego[:2] + np.array([x_e, y_e])
    theta_world = z_ego[2] + phi_e
    return np.array([pos_world[0], pos_world[1], theta_world])


def process_model(x_k, dt):
    x, y, theta, v, a, omega = x_k
    x_next = x + v * np.cos(theta) * dt + 0.5 * a * np.cos(theta) * dt**2
    y_next = y + v * np.sin(theta) * dt + 0.5 * a * np.sin(theta) * dt**2
    theta_next = theta + omega * dt
    v_next = v + a * dt
    return np.array([x_next, y_next, theta_next, v_next, a, omega])


def measurement_model(x_k):
    return x_k[:3]


def build_batch_problem(observations_ego, ego_poses, dt, P0, Q, R, x0_init, max_iter=5):
    K, n, m = len(observations_ego), 6, 3
    z_world = np.array([transform_ego_to_world(observations_ego[k], ego_poses[k]) for k in range(K)])
    X = np.zeros((K, n))
    X[0] = x0_init
    for k in range(1, K):
        X[k] = process_model(X[k - 1], dt)

    for _ in range(max_iter):
        H = lil_matrix((K * n, K * n))
        b = np.zeros(K * n)

        for k in range(K):
            i0, i1 = k * n, (k + 1) * n
            z_pred = measurement_model(X[k])
            r_meas = z_world[k] - z_pred
            H_meas = np.zeros((m, n))
            H_meas[:, :3] = np.eye(3)
            W_meas = np.linalg.inv(R)
            H[i0:i1, i0:i1] += H_meas.T @ W_meas @ H_meas
            b[i0:i1] += H_meas.T @ W_meas @ r_meas

            if k > 0:
                ip0, ip1 = (k - 1) * n, k * n
                x_pred = process_model(X[k - 1], dt)
                r_proc = X[k] - x_pred
                eps = 1e-5
                F = np.zeros((n, n))
                for i in range(n):
                    dx = np.zeros(n)
                    dx[i] = eps
                    F[:, i] = (process_model(X[k - 1] + dx, dt) - x_pred) / eps
                I = np.eye(n)
                W_proc = np.linalg.inv(Q)
                H[i0:i1, i0:i1] += I.T @ W_proc @ I
                H[i0:i1, ip0:ip1] += -I.T @ W_proc @ F
                H[ip0:ip1, i0:i1] += -F.T @ W_proc @ I
                H[ip0:ip1, ip0:ip1] += F.T @ W_proc @ F
                b[i0:i1] += I.T @ W_proc @ r_proc
                b[ip0:ip1] += -F.T @ W_proc @ r_proc

        W_prior = np.linalg.inv(P0)
        H[0:n, 0:n] += W_prior
        b[0:n] += W_prior @ (x0_init - X[0])
        dx = spsolve(csr_matrix(H), b)
        for k in range(K):
            X[k] += dx[k * n:(k + 1) * n]
    return X, z_world


def plot_results(true_states, estimated_states, observations_world):
    plt.figure(figsize=(10, 6))
    plt.plot(observations_world[:, 0], observations_world[:, 1], 'rx', label='Observations')
    plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'b--', label='Smoothed')
    plt.xlabel('X [m]'); plt.ylabel('Y [m]')
    plt.title('Smoothed Vehicle Trajectory in World Frame')
    plt.axis('equal'); plt.grid(True); plt.legend(); plt.show()


if __name__ == '__main__':    
    datarootpath='/home/thh3/data/sets/nuscenes'
    data = get_car_tracks(dataroot=datarootpath)
    instance_token, track = next(iter(data['tracks'].items()))
    print(f"Selected vehicle: {instance_token}, {len(track)} observations")

    observations_ego, aligned_ego_poses = prepare_smoothing_input(
        track, data['ego_poses'], data['timestamps']
    )

    dt = 0.5
    P0 = 0.5 * np.eye(6)
    Q = 0.05 * np.eye(6)
    R = 0.1 * np.eye(3)
    x0_init = np.zeros(6)
    x0_init[:3] = transform_ego_to_world(observations_ego[0], aligned_ego_poses[0])

    estimated_states, z_world = build_batch_problem(
        observations_ego, aligned_ego_poses, dt, P0, Q, R, x0_init, max_iter=5
    )
    plot_results(None, estimated_states, z_world)

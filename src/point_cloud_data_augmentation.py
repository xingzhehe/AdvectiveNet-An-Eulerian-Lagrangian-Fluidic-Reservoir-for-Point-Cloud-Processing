import torch
import numpy as np


def random_points(pc, sample_rate=0.9):
    num_points = pc.shape[0]
    idx = np.arange(num_points)
    np.random.shuffle(idx)
    pc = pc[idx, :]
    threshold = int(np.random.uniform(0.1, sample_rate) * num_points)
    idx = np.random.randint(0, threshold, num_points - threshold)
    pc[threshold:, :] = pc[idx]
    return pc


def gauss_sample_points(pc, mean=1024, std=128, clip_min=256, clip_high=1024):
    num_points = pc.shape[0]
    idx = np.arange(num_points)
    np.random.shuffle(idx)
    pc = pc[idx, :]
    threshold = int(np.clip(np.random.normal(mean, std), a_min=clip_min, a_max=clip_high))
    idx = np.random.randint(0, threshold, num_points - threshold)
    pc[threshold:, :] = pc[idx]
    return pc


def random_points_seg(pc, seg, sample_rate=0.8):
    num_points = pc.shape[0]
    idx = np.arange(num_points)
    np.random.shuffle(idx)
    pc = pc[idx, :]
    seg = seg[idx]
    threshold = int(np.random.uniform(1 - sample_rate, 1) * num_points)
    idx = np.random.randint(0, threshold, num_points - threshold)
    pc[threshold:, :] = pc[idx]
    seg[threshold:] = seg[idx]
    return pc, seg


def gauss_sample_points_seg(pc, seg, mean=2048, std=256, clip_min=512, clip_high=2048):
    num_points = pc.shape[0]
    idx = np.arange(num_points)
    np.random.shuffle(idx)
    pc = pc[idx, :]
    seg = seg[idx]
    threshold = int(np.clip(np.random.normal(mean, std), a_min=clip_min, a_max=clip_high))
    idx = np.random.randint(0, threshold, num_points - threshold)
    pc[threshold:, :] = pc[idx]
    seg[threshold:] = seg[idx]
    return pc, seg


def rotate_point_cloud_Y(pc, normal=False):
    rotation_matrix = torch.FloatTensor(pc.shape[0], 3, 3).to(pc)
    for i in range(pc.shape[0]):
        #rotation_angle = np.random.choice([np.pi / 2, np.pi, np.pi / 2 * 3, 2 * np.pi])
        rotation_angle = np.random.uniform(0, 2*np.pi)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix[i] = torch.FloatTensor([[cosval, 0, sinval],
                                                [0, 1, 0],
                                                [-sinval, 0, cosval]]).to(pc)
    pc[:, :, :3] = pc[:, :, :3].bmm(rotation_matrix)
    if normal:
        pc[:, :, 3:6] = pc[:, :, 3:6].bmm(rotation_matrix)
    return pc


def rotate_point_cloud_Z(pc, normal=False):
    rotation_matrix = torch.FloatTensor(pc.shape[0], 3, 3).to(pc)
    for i in range(pc.shape[0]):
        rotation_angle = np.random.choice([np.pi / 2, np.pi, np.pi / 2 * 3, 2 * np.pi])
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix[i] = torch.FloatTensor([[cosval, -sinval, 0],
                                                [sinval, cosval, 0],
                                                [0, 0, 1]]).to(pc)
    pc[:, :, :3] = pc[:, :, :3].bmm(rotation_matrix)
    if normal:
        pc[:, :, 3:6] = pc[:, :, 3:6].bmm(rotation_matrix)
    return pc


def rotate_perturbation_point_cloud(pc, angle_sigma=0.06, angle_clip=0.18, normal=False):
    rotation_rx = torch.FloatTensor(pc.shape[0], 3, 3).to(pc)
    rotation_ry = torch.FloatTensor(pc.shape[0], 3, 3).to(pc)
    rotation_rz = torch.FloatTensor(pc.shape[0], 3, 3).to(pc)
    angles = np.clip(np.random.normal(0, angle_sigma, (pc.shape[0], 3)), a_min=-angle_clip, a_max=angle_clip)
    for i in range(pc.shape[0]):
        rotation_rx[i] = torch.FloatTensor([[1, 0, 0],
                                            [0, np.cos(angles[i, 0]), -np.sin(angles[i, 0])],
                                            [0, np.sin(angles[i, 0]), np.cos(angles[i, 0])]]).to(pc)
        rotation_ry[i] = torch.FloatTensor([[np.cos(angles[i, 1]), 0, np.sin(angles[i, 1])],
                                            [0, 1, 0],
                                            [-np.sin(angles[i, 1]), 0, np.cos(angles[i, 1])]]).to(pc)
        rotation_rz[i] = torch.FloatTensor([[np.cos(angles[i, 2]), -np.sin(angles[i, 2]), 0],
                                            [np.sin(angles[i, 2]), np.cos(angles[i, 2]), 0],
                                            [0, 0, 1]]).to(pc)
    pc[:, :, :3] = pc[:, :, :3].bmm(rotation_rz).bmm(rotation_ry).bmm(rotation_rx)
    if normal:
        pc[:, :, 3:6] = pc[:, :, 3:6].bmm(rotation_rz).bmm(rotation_ry).bmm(rotation_rx)
    return pc


def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    pc[:, :, :3] = pc[:, :, :3] * torch.FloatTensor(pc.shape[0], 1, 1).uniform_(scale_low, scale_high).to(pc)
    return pc


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    pc[:, :, :3] = pc[:, :, :3] + torch.clamp(torch.randn(pc[:, :, :3].shape).to(pc) * sigma, min=-clip, max=clip)
    return pc


def random_shift_point_cloud(pc, shift_range=0.1):
    pc[:, :, :3] = pc[:, :, :3] + torch.FloatTensor(pc.shape[0], 1, 3).uniform_(-shift_range, shift_range).to(pc)
    return pc


def center_rescale(pc):
    """
    :param pc: (batch_size, dim, num_points)
    :return: (batch_size, dim, num_points)
    """
    pc = pc - pc.min(-1)[0].unsqueeze(-1)
    pc = pc / pc.max(dim=-1)[0].unsqueeze(-1)
    pc = pc * 2 - 1
    return pc


def data_augment(pc, classification=True, semantic=False, normal=False):
    if classification:
        pc = rotate_point_cloud_Y(pc, normal=normal)
        pc = random_scale_point_cloud(pc)
        pc = random_shift_point_cloud(pc)
        pc = jitter_point_cloud(pc)
    pc = rotate_perturbation_point_cloud(pc, normal=normal)
    if semantic:
        pc = rotate_point_cloud_Z(pc, normal=normal)
        pc = jitter_point_cloud(pc, sigma=0.005, clip=0.025)
    else:
        pass#pc = jitter_point_cloud(pc)
    return pc

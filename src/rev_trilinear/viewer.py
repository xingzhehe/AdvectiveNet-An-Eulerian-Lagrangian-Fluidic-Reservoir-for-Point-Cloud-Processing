import matplotlib.pyplot as plt
from sklearn import datasets
import torch
import torch.nn as nn
import numpy as np
from src.gauss_interp import *
from src.nn_utils import one_hot_encoding
import torch.nn.functional as F
from src.point_cloud_data_augmentation import center_rescale
from src.viewer_utils import *
from src.rev_bilinear.rev_bilinear_interp import RevBilinear


def seg2color(seg):
    if len(np.unique(seg)) == 3:
        return one_hot_encoding(seg)
    elif len(np.unique(seg)) == 2:
        one_hot = np.zeros((len(seg), 3))
        one_hot[np.arange(len(seg)), seg] = 1
        return one_hot


if __name__ == '__main__':
    grid_size = 32
    sigma = 1/grid_size
    pc, seg = blobs = datasets.make_blobs(n_samples=1024, random_state=0)
    pc_color = seg2color(seg)
    plt.figure()
    plt.scatter(pc[:, 0].reshape(-1), pc[:, 1].reshape(-1), s=10, c=pc_color)
    pc = torch.from_numpy(pc.astype(np.float32)).transpose(0, 1).unsqueeze(0)
    pc = center_rescale(pc)
    pc_color = torch.from_numpy(pc_color).transpose(0, 1).unsqueeze(0).float()
    grid = make_grid(grid_size, dim=2).unsqueeze(0).expand(1, grid_size ** 2, 2).transpose(1, 2)
    grid_color = pc_grid(pc, pc_color, grid, grid_size, sigma, dim=2)
    grid_color = grid_color.reshape(3, grid_size ** 2).transpose(0, 1).reshape(grid_size, grid_size, 3)
    plt.figure()
    plt.imshow(np.transpose(grid_color.numpy(), (1, 0, 2)), origin='lower')

    grid_color = RevBilinear.apply(pc.cuda(), pc_color.cuda(), grid_size)
    grid_color = grid_color.cpu().reshape(3, grid_size ** 2).transpose(0, 1).reshape(grid_size, grid_size, 3)
    grid_color[grid_color==grid_color.max()]=1
    print(grid_color.min(), grid_color.max())
    plt.figure()
    plt.imshow(np.transpose(grid_color.numpy(), (1, 0, 2)), origin='lower')

    grid_color = grid_color.reshape(1, grid_size ** 2, 3).transpose(1, 2)
    pc_color = grid_pc(pc, grid_color, grid, grid_size, sigma, dim=2)
    pc_color = pc_color.transpose(1, 2).squeeze(0)
    plt.figure()
    plt.scatter(pc[:, 0].reshape(-1), pc[:, 1].reshape(-1), s=10, c=pc_color)

    grid_color = grid_color.reshape(1, 3, grid_size, grid_size).transpose(2, 3)
    pc_color = F.grid_sample(grid_color, pc.transpose(1, 2).unsqueeze(1)).squeeze(2)
    pc_color = pc_color.transpose(1, 2).squeeze(0)
    plt.figure()
    plt.scatter(pc[:, 0].reshape(-1), pc[:, 1].reshape(-1), s=10, c=pc_color)

    plt.show()


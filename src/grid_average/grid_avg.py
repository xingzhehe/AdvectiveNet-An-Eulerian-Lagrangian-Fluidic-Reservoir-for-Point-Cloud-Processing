import torch
from torch.autograd import gradcheck
import torch.nn.functional as F
import grid_average


def grid_avg(pc, grid_size):
    return grid_average.cal_pc_relative(pc, grid_size)


def grid_avg_feature(pc, feature, grid_size):
    return grid_average.cal_feature_relative(pc, feature, grid_size)


if __name__ == '__main__':
    pc = torch.rand(1, 3, 1024, dtype=torch.float32, requires_grad=False).cuda() * 2 - 1
    feature = torch.rand(1, 32, 1024, dtype=torch.float32, requires_grad=False).cuda() * 2 - 1
    relative_pc = grid_avg(pc, 16)
    relative_feature = grid_avg_feature(pc, feature, 16)
    print(pc.abs().mean())
    print(relative_pc.abs().mean())
    print(relative_feature.shape)

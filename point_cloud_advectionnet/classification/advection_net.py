import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import math
from src.nn_utils import *
from src.gauss_interp import *
from src.viewer_utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.point_cloud_data_augmentation import center_rescale
from src.rev_trilinear.rev_trilinear_interp import RevTrilinear
from src.grid_average.grid_avg import grid_avg
import os


class Advection(nn.Module):
    def __init__(self, in_channel=2, grid_size=4, pic_flip=0.3, batch_norm=True):
        super(Advection, self).__init__()
        self.pic_flip = pic_flip
        self.grid_size = grid_size
        self.fc = nn.Sequential(Conv1dBatchRelu(in_channel, 32, batch_norm=batch_norm))
        self.conv = nn.Sequential(Conv3dBatchRelu(32, 32, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm),
                                  Conv3dBatchRelu(32, 32, batch_norm=batch_norm),
                                  )
        self.gen_feature = nn.Sequential(Conv3dBatchRelu(32, 32, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm))
        self.gen_velocity = nn.Sequential(nn.Conv3d(32, 3, kernel_size=1, stride=1, padding=0))

    def forward(self, pc, value, velocity):
        value1 = self.fc(value)
        field = RevTrilinear.apply(pc, value1, self.grid_size)
        field = self.conv(field).permute(0, 1, 4, 3, 2)
        pc_ = pc.transpose(1, 2).unsqueeze(1).unsqueeze(1).contiguous()
        feature = F.grid_sample(self.gen_feature(field), pc_).squeeze(2).squeeze(2)
        new_velocity = self.gen_velocity(field)
        flip_diff = new_velocity - RevTrilinear.apply(pc, velocity, self.grid_size)
        pic_velocity = F.grid_sample(new_velocity, pc_).squeeze(2).squeeze(2) / self.grid_size
        velocity = (velocity + flip_diff) * self.pic_flip + pic_velocity * (1 - self.pic_flip)
        pc = pc + velocity
        return pc, torch.cat((value, feature, value1), dim=1), velocity


class AdvectionNet(nn.Module):
    def __init__(self, num_class=50, grid_size=6, drop_rate=0, sigma=0.1, time_step=4, pic_flip=0.3, batch_norm=True):
        super(AdvectionNet, self).__init__()
        self.num_class = num_class
        self.grid_size = grid_size
        self.grid_sizes = [12, 10, 8, 6, 4, 3, 2]
        self.time_step = time_step
        self.indep_conv1 = nn.Sequential(Conv1dBatchRelu(3 + 6 * len(self.grid_sizes), 64, batch_norm=batch_norm),
                                         Conv1dBatchRelu(64, 64, batch_norm=batch_norm))

        self.init_velocity = nn.Sequential(Conv1dBatchRelu(64, 64, batch_norm=batch_norm),
                                           nn.Conv1d(64, 3, 1))

        self.advection = nn.ModuleList([Advection(in_channel=64 * (i + 1), grid_size=grid_size,
                                                  pic_flip=pic_flip, batch_norm=batch_norm)
                                        for i in range(time_step)])

        self.indep_conv3 = nn.Sequential(Conv1dBatchRelu(64 * (time_step + 1), 512, batch_norm=batch_norm),
                                         Conv1dBatchRelu(512, 1024, batch_norm=batch_norm))

        self.fc = nn.Sequential(LinearBatchRelu(1024, 512, batch_norm=batch_norm),
                                LinearBatchRelu(512, 256, batch_norm=batch_norm),
                                nn.Dropout(drop_rate),
                                nn.Linear(256, num_class))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, pc, label):
        pc = pc.transpose(1, 2)
        batch_size = pc.shape[0]
        num_points = pc.shape[2]
        value = torch.cat([pc] + [grid_avg(pc, grid_size) for grid_size in self.grid_sizes], dim=1)
        pc = center_rescale(pc)
        value = self.indep_conv1(value)

        velocity = self.init_velocity(value) / self.grid_size
        pc = pc + velocity

        for i, advection in enumerate(self.advection):
            pc, value, velocity = advection(pc, value, velocity)
        del velocity

        value = self.indep_conv3(value)
        value = value.max(dim=2)[0]
        value = self.fc(value)
        value = F.log_softmax(value, 1)

        penalty = 0
        batch_centers = []
        for label_i in label.unique():
            pc_i = pc[label == label_i]
            centers_i = pc_i.mean(dim=-1)
            for j in range(pc_i.shape[0]):
                penalty += max(pc_i[j].norm(dim=0).max() - 1, 0) * 20
            penalty += (centers_i - centers_i.mean(dim=0).unsqueeze(0)).norm().mean()
            batch_centers.append(centers_i.mean(dim=0))
        for i in range(len(batch_centers)):
            for j in range(i + 1, len(batch_centers)):
                penalty += max(1 - (batch_centers[i] - batch_centers[j]).norm(), 0) / 2
        penalty = penalty / batch_size

        return value, penalty

    def forward_with_view(self, pc):
        pc = pc.transpose(1, 2)
        batch_size = pc.shape[0]
        num_points = pc.shape[2]
        value = torch.cat([pc] + [grid_avg(pc, grid_size) for grid_size in self.grid_sizes], dim=1)
        value = self.indep_conv1(value)

        velocity = self.init_velocity(value) / self.grid_size
        print(velocity.norm(dim=1).mean(dim=-1).mean())
        pc = pc + velocity

        for i, advection in enumerate(self.advection):
            pc, value, velocity = advection(pc, value, velocity)
            print(velocity.norm(dim=1).mean(dim=-1).mean())
            draw_pc(pc.cpu().numpy())
        del velocity


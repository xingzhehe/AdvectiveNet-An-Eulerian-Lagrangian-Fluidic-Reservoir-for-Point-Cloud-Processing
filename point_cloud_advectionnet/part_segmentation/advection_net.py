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
        self.sigma = sigma
        self.time_step = time_step
        self.indep_conv1 = nn.Sequential(Conv1dBatchRelu(3+6*len(self.grid_sizes), 64, batch_norm=batch_norm),
                                         Conv1dBatchRelu(64, 64, batch_norm=batch_norm))

        self.init_velocity = nn.Sequential(Conv1dBatchRelu(64, 64, batch_norm=batch_norm),
                                           nn.Conv1d(64, 3, kernel_size=1, stride=1, padding=0))

        self.advection = nn.ModuleList([Advection(in_channel=64 * (i + 1), grid_size=grid_size,
                                                  pic_flip=pic_flip, batch_norm=batch_norm)
                                        for i in range(time_step)])

        self.indep_conv2 = Conv1dBatchRelu(64 * (time_step + 1), 1024, batch_norm=batch_norm)

        self.fc = nn.Sequential(LinearBatchRelu(64 * (time_step + 1) + 1024, 512, batch_norm=batch_norm),
                                LinearBatchRelu(512, 256, batch_norm=batch_norm),
                                LinearBatchRelu(256, 256, batch_norm=batch_norm),
                                LinearBatchRelu(256, 128, batch_norm=batch_norm),
                                nn.Dropout(drop_rate),
                                nn.Linear(128, num_class))

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
        for i in range(time_step):
            self.state_dict()['advection.'+str(i)+'.gen_velocity.0.weight'] /= self.grid_size**3
        self.state_dict()['init_velocity.1.weight'] /= self.grid_size**3

    def forward(self, pc, seg):
        batch_size, num_points, _ = pc.shape
        pc = pc.transpose(1, 2)
        value = torch.cat([pc] + [grid_avg(pc, grid_size) for grid_size in self.grid_sizes], dim=1)
        value = self.indep_conv1(value)

        velocity = self.init_velocity(value)
        penalty = 0
        pc = pc + velocity

        for i, advection in enumerate(self.advection):
            pc, value, velocity = advection(pc, value, velocity)
        del velocity

        global_value = self.indep_conv2(value)
        global_value = global_value.max(dim=2)[0].unsqueeze(2).repeat(1, 1, num_points)

        value = value.transpose(1, 2).reshape((batch_size * num_points, -1))
        global_value = global_value.transpose(1, 2).reshape((batch_size * num_points, -1))
        value = torch.cat((value, global_value), dim=1)
        del global_value
        value = self.fc(value)
        value = F.log_softmax(value, 1)

        pc_ = pc.transpose(1, 2)
        for i in range(seg.shape[0]):
            seg_i = []
            for seg_index in seg[i].unique():
                seg_i.append(seg[i] == seg_index)
            for j in range(len(seg_i)):
                penalty += max((pc_[i][seg_i[j]].mean(dim=0).unsqueeze(0) - pc_[i][seg_i[j]]).norm(dim=1).mean(), 0.1)#*10
                penalty += max(pc_[i][seg_i[j]].mean(dim=0).norm() - 1, 0) #* 20
                for k in range(j + 1, len(seg_i)):
                    penalty += max(1.5 - (pc_[i][seg_i[j]].mean(dim=0) - pc_[i][seg_i[k]].mean(dim=0)).norm(), 0)
        penalty = penalty / seg.shape[0]

        return value, penalty

    def forward_with_view(self, pc, seg):
        if not os.path.exists("../shapenet_log/imgs/"):
            os.makedirs("../shapenet_log/imgs/")

        pc = pc.transpose(1, 2)
        batch_size = pc.shape[0]
        num_points = pc.shape[2]
        grid = self.grid.to(pc)
        #pc = center_rescale(pc)
        value = self.indep_conv1(pc)
        value = self.interp_conv1(pc, value, grid)

        velocity = self.init_velocity(value) / self.grid_size ** 2

        for i, advection in enumerate(self.advection):
            pc, velocity = advection(pc, value, grid, velocity)
            #pc = center_rescale(pc)
            value = self.interp_conv[i](pc, value, grid)
            draw_pc(pc.cpu().numpy(), seg, save=f"../shapenet_log/imgs/{i + 1}.png")
        del velocity

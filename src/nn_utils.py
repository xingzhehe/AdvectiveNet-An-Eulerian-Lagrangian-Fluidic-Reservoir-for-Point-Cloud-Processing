import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.cosine_annealing import CosineAnnealingWithRestartsLR
from src.adamw import AdamW


class ScheduledOptimizer:
    """
    If otype is adam, then optimizer will take regular step to decay learning rate
    If otype is sgd, then optimizer will take irregular step to decay learning rate, which can be specified
    If otype is sgdr, then optimizer will take cosine annealing with restart to decay learning rate
    If otype is adamw, then optimizer will take regular step to decay learning rate
    """

    def __init__(self, parameters, otype, lr, lr_scheduler_step_size, lr_gamma,
                 lr_clip=1e-5, lr_weight_decay=1e-3, lr_sgd_momentum=0.9, T_mult=2):
        if otype == "adam":
            self.optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=lr_weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_scheduler_step_size, gamma=lr_gamma)
        elif otype == "sgd":
            self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=lr_sgd_momentum, weight_decay=lr_weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_scheduler_step_size, gamma=lr_gamma)
        elif otype == "sgdr":
            self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=lr_sgd_momentum, weight_decay=lr_weight_decay)
            self.scheduler = CosineAnnealingWithRestartsLR(self.optimizer, T_max=lr_scheduler_step_size,
                                                           eta_min=lr_clip, last_epoch=-1, T_mult=T_mult)
        elif otype == "adamw":
            self.optimizer = AdamW(parameters, lr=lr, weight_decay=lr_weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_scheduler_step_size, gamma=lr_gamma)

        self.lr_clip = lr_clip

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def scheduler_step(self):
        self.scheduler.step()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.lr_clip)


class BNMomentumScheduler:
    def __init__(self, net, init_momentum=0.1, step_size=40, gamma=0.5, clip=0.01):
        self.step_size = step_size
        self.gamma = gamma
        self.clip = clip
        self.epoch = 0
        self.net = net

        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = init_momentum

    def scheduler_step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.momentum = max(m.momentum*self.gamma, self.clip)


def label_smooth_loss(log_prob, label, confidence=0.9):
    """
    :param log_prob: log probability
    :param label: one hot encoded
    :param confidence: we replace one (in the one hot) with confidence. 0 <= confidence <= 1.
    :return:
    """
    N = log_prob.size(0)
    C = log_prob.size(1)
    smoothed_label = torch.full(size=(N, C), fill_value=(1-confidence) / (C - 1)).to(log_prob)
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(label, dim=1), value=confidence)
    loss = - torch.sum(log_prob * smoothed_label) / N
    return loss


def cal_label_weight(label):
    num_label = np.unique(label).shape[0]
    label_weight = np.zeros(num_label).astype(np.float32)
    for i in range(num_label):
        label_weight[i] = (label == i).sum()
    label_weight = label_weight / label_weight.sum()
    label_weight = 1 / np.log(1 + label_weight)
    label_weight = label_weight / label_weight.sum()
    label_weight = torch.from_numpy(label_weight)
    return label_weight


class Conv1dBatchRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=0, batch_norm=True):
        super(Conv1dBatchRelu, self).__init__()
        if batch_norm:
            self.block = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class Conv2dBatchRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=True):
        super(Conv2dBatchRelu, self).__init__()
        if batch_norm:
            self.block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class Conv3dBatchRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=True):
        super(Conv3dBatchRelu, self).__init__()
        if batch_norm:
            self.block = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class CatDilationConv2dBatchRelu(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=(1, 2), batch_norm=True):
        super(CatDilationConv2dBatchRelu, self).__init__()
        self.dilation_num = len(dilation)
        assert out_channel % self.dilation_num == 0
        self.block = nn.ModuleList([Conv2dBatchRelu(in_channel, int(out_channel/self.dilation_num), batch_norm=batch_norm,
                                                    kernel_size=3, stride=1, padding=dilation[i], dilation=dilation[i])
                                    for i in range(self.dilation_num)])

    def forward(self, x):
        x = torch.cat([self.block[i](x) for i in range(self.dilation_num)], dim=1)
        return x


class CatDilationConv3dBatchRelu(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=(1, 2), batch_norm=True):
        super(CatDilationConv3dBatchRelu, self).__init__()
        self.dilation_num = len(dilation)
        assert out_channel % self.dilation_num == 0
        self.block = nn.ModuleList([Conv3dBatchRelu(in_channel, int(out_channel/self.dilation_num), batch_norm=batch_norm,
                                                    kernel_size=3, stride=1, padding=dilation[i], dilation=dilation[i])
                                    for i in range(self.dilation_num)])

    def forward(self, x):
        x = torch.cat([self.block[i](x) for i in range(self.dilation_num)], dim=1)
        return x


class LinearBatchRelu(nn.Module):
    def __init__(self, in_feature, out_feature, batch_norm=True):
        super(LinearBatchRelu, self).__init__()
        if batch_norm:
            self.block = nn.Sequential(
                nn.Linear(in_features=in_feature, out_features=out_feature),
                nn.BatchNorm1d(out_feature),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                nn.Linear(in_features=in_feature, out_features=out_feature),
                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


def one_hot_encoding(x):
    one_hot = np.zeros((len(x), x.max()+1))
    one_hot[np.arange(len(x)), x] = 1
    return one_hot


def torch_one_hot_encoding(x):
    x = torch.LongTensor(x).view(-1, 1)
    one_hot = torch.zeros((len(x), x.max() + 1))
    one_hot.scatter_(dim=1, index=x, value=1)
    return one_hot


if __name__ == '__main__':
    x = torch.tensor([1,2,1,3])
    print(torch_one_hot_encoding(x))
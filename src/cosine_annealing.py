from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class CosineAnnealingWithRestartsLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min, last_epoch=-1, T_mult=1.501):
        self.T_max = T_max
        self.T_mult = T_mult
        self.restart_every = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.restarted_at = 0
        super().__init__(optimizer, last_epoch)

    def restart(self):
        self.restart_every *= self.T_mult
        self.restarted_at = self.last_epoch

    def cosine(self, base_lr):
        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step_n / self.restart_every)) / 2

    @property
    def step_n(self):
        return self.last_epoch - self.restarted_at

    def get_lr(self):
        if self.step_n >= self.restart_every:
            self.restart()
        return [self.cosine(base_lr) for base_lr in self.base_lrs]


if __name__ == '__main__':
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=15, eta_min=1e-5, last_epoch=-1, T_mult=1.501)

    epoch = 4

    plt.figure()
    x = list(range(200))
    y = []

    for epoch in range(200):
        scheduler.step()
        lr = scheduler.get_lr()
        # print(epoch, scheduler.get_lr()[0])
        y.append(scheduler.get_lr()[0])

    plt.plot(x, y)
    plt.show()

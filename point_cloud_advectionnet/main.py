import sys
sys.path.append("../../")
import argparse
from classification.classifier import Classifier
from classification10.classifier10 import Classifier10
from part_segmentation.part_segmentor import PartSegmentor
from semantic.SementicSegmentor import SemanticSegmentor
import torch
import numpy as np
import os
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default="1", help='index of the job')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--num_points', type=int, default=512, help='input point cloud size')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--max_epoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-2, help='base learning rate')
    parser.add_argument('--lr_scheduler_step_size', type=str, default='20', help='learning rate decay step')
    parser.add_argument('--lr_gamma', type=float, default=0.8, help='learning rate decay rate')
    parser.add_argument('--lr_sgd_momentum', type=float, default=0.9, help='sgd momentum')
    parser.add_argument('--lr_clip', type=float, default=1e-4, help='minimum learning rate')
    parser.add_argument('--lr_weight_decay', type=float, default=1e-3, help='learning penalty')
    parser.add_argument('--sgdr_T_mult', type=float, default=2, help='T_mult of sgdr')
    parser.add_argument('--bn_momentum', type=float, default=0.5, help='initial bn momentum')
    parser.add_argument('--bn_gamma', type=float, default=0.5, help='decay rate of bn momentum')
    parser.add_argument('--bn_clip', type=float, default=0.1, help='smallest bn momentum allowed')
    parser.add_argument('--bn_step_size', type=int, default=40, help='steps needed for bn to decay')
    parser.add_argument('--optimizer', type=str, default="adamw", help='optimizer')
    parser.add_argument('--net', type=str, default='advection_net', help='net type')
    parser.add_argument('--grid_size', type=int, default=3, help='size of grid')
    parser.add_argument('--time_step', type=int, default=1, help='number of advection layer')
    parser.add_argument('--pic_flip', type=float, default=0.5, help='pic/flip ratio')
    parser.add_argument('--sigma', type=float, default=1/12, help='standard deviation of gauss interp')
    parser.add_argument('--name', type=str, default='part_shapenet', help='output folder')
    parser.add_argument('--manual_seed', type=int, default=0, help='manual seed')
    parser.add_argument('--device', type=str, default='cuda:1', help='cuda')
    parser.add_argument('--drop_rate', type=float, default=0, help='dropout rate')
    parser.add_argument('--label_smooth', type=bool, default=False, help='whether to use label smoothing')
    parser.add_argument('--penalty', type=float, default=0.1, help='the coefficient of penalty term')
    parser.add_argument('--oversampling', type=bool, default=False, help='whether to use oversampling')
    parser.add_argument('--batch_norm', type=bool, default=False, help='whether to use BatchNorm')

    hp = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(hp.device[-1]))
        hp.device = torch.device(hp.device)
    else:
        hp.device = torch.device('cpu')

    if hp.optimizer == 'sgd':
        hp.lr_scheduler_step_size = hp.lr_scheduler_step_size[1:-1].split(',')
        for i in range(len(hp.lr_scheduler_step_size)):
            hp.lr_scheduler_step_size[i] = int(hp.lr_scheduler_step_size[i])
    elif hp.optimizer == 'adam' or hp.optimizer == 'sgdr' or hp.optimizer == 'adamw':
        hp.lr_scheduler_step_size = int(hp.lr_scheduler_step_size)

    hp.manual_seed = random.randint(1, 10000)  # fix seed
    random.seed(hp.manual_seed)
    np.random.seed(hp.manual_seed)
    torch.manual_seed(hp.manual_seed)

    if hp.name == 'modelnet':
        hp.num_class = 40
        if not os.path.exists("modelnet_log"):
            os.makedirs("modelnet_log")
        hp.name = os.path.join("modelnet_log", hp.name + hp.index)
        trainer = Classifier(hp)
    elif hp.name == 'modelnet10':
        hp.num_class = 10
        if not os.path.exists("modelnet10_log"):
            os.makedirs("modelnet10_log")
        hp.name = os.path.join("modelnet10_log", hp.name + hp.index)
        trainer = Classifier10(hp)
    elif hp.name == 'part_shapenet':
        if not os.path.exists("part_shapenet_log"):
            os.makedirs("part_shapenet_log")
        hp.name = os.path.join("part_shapenet_log", hp.name + hp.index)
        trainer = PartSegmentor(hp)
    elif hp.name == 's3dis':
        hp.num_class = 13
        if not os.path.exists("s3dis_log"):
            os.makedirs("s3dis_log")
        hp.name = os.path.join("s3dis_log", hp.name + hp.index)
        trainer = SemanticSegmentor(hp)

    with open(hp.name+".txt", 'w') as f:
        f.write('num_points: {0}\n'.format(hp.num_points))
        f.write('lr: {0}\n'.format(hp.lr))
        f.write('lr_scheduler_step_size: {0}\n'.format(hp.lr_scheduler_step_size))
        f.write('max_epoch: {0}\n'.format(hp.max_epoch))
        if hp.optimizer != 'sgdr':
            f.write('lr_gamma: {0}\n'.format(hp.lr_gamma))
        else:
            f.write('sgdr_T_mult: {0}\n'.format(hp.sgdr_T_mult))
        f.write('lr_sgd_momentum: {0}\n'.format(hp.lr_sgd_momentum))
        f.write('lr_weight_decay: {0}\n'.format(hp.lr_weight_decay))
        f.write('bn_momentum: {0}\n'.format(hp.bn_momentum))
        f.write('bn_gamma: {0}\n'.format(hp.bn_gamma))
        f.write('bn_clip: {0}\n'.format(hp.bn_clip))
        f.write('bn_step_size: {0}\n'.format(hp.bn_step_size))
        f.write('optimizer: {0}\n'.format(hp.optimizer))
        f.write('net: {0}\n'.format(hp.net))
        f.write('grid_size: {0}\n'.format(hp.grid_size))
        f.write('time_step: {0}\n'.format(hp.time_step))
        f.write('pic_flip: {0}\n'.format(hp.pic_flip))
        f.write('sigma: {0}\n'.format(hp.sigma))
        f.write('manual_seed: {0}\n'.format(hp.manual_seed))
        f.write('device: {0}\n'.format(hp.device))
        f.write('batch_size: {0}\n'.format(hp.batch_size))
        f.write('drop_rate: {0}\n'.format(hp.drop_rate))
        f.write('label_smooth: {0}\n'.format(hp.label_smooth))
        f.write('penalty: {0}\n'.format(hp.penalty))
        f.write('oversampling: {0}\n'.format(hp.oversampling))
        f.write('batch_norm: {0}\n'.format(hp.batch_norm))

    trainer.train()

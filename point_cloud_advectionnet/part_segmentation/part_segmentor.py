import torch.utils.data
import torch.nn.functional as F
from part_segmentation.part_shapenet import PartShapeNetDataset
from part_segmentation.advection_net import AdvectionNet
import numpy as np
from src.point_cloud_data_augmentation import data_augment
from src.nn_utils import *


class PartSegmentor:
    def __init__(self, hp):
        self.hp = hp
        num_classes = {0: 4, 1: 2, 2: 2, 3: 4, 4: 4, 5: 3, 6: 3, 7: 2, 8: 4, 9: 2, 10: 6, 11: 2, 12: 3, 13: 3, 14: 3, 15: 3}
        if self.hp.net == 'advection_net':
            self.net = AdvectionNet(num_class=num_classes[int(self.hp.index)], sigma=self.hp.sigma, time_step=self.hp.time_step,
                                    grid_size=self.hp.grid_size, drop_rate=self.hp.drop_rate, pic_flip=self.hp.pic_flip)
        self.net.to(self.hp.device)

    def train(self):
        log_file = open(self.hp.name+".txt", 'a')
        log_file.close()
        train_data = torch.utils.data.DataLoader(PartShapeNetDataset("train", num_points=self.hp.num_points, type_id=int(self.hp.index)),
                                                 batch_size=self.hp.batch_size,
                                                 shuffle=True,
                                                 num_workers=self.hp.workers,
                                                 pin_memory=True)
        test_data = torch.utils.data.DataLoader(PartShapeNetDataset("test", num_points=2048, type_id=int(self.hp.index)),
                                                batch_size=8,
                                                num_workers=self.hp.workers,
                                                pin_memory=True)
        self.hp.num_class = np.unique(train_data.dataset.seg).shape[0]

        optimizer = ScheduledOptimizer(self.net.parameters(), otype=self.hp.optimizer, lr=self.hp.lr, T_mult=self.hp.sgdr_T_mult,
                                       lr_scheduler_step_size=self.hp.lr_scheduler_step_size, lr_gamma=self.hp.lr_gamma,
                                       lr_clip=self.hp.lr_clip, lr_weight_decay=self.hp.lr_weight_decay, lr_sgd_momentum=self.hp.lr_sgd_momentum)

        bn_scheduler = BNMomentumScheduler(self.net, init_momentum=self.hp.bn_momentum, clip=self.hp.bn_clip,
                                           step_size=self.hp.bn_step_size, gamma=self.hp.bn_gamma)
        highest_mIoU = 0

        for epoch in range(self.hp.max_epoch):
            total_loss = []
            self.net.train()
            optimizer.scheduler_step()
            bn_scheduler.scheduler_step()
            train_correct = 0
            train_sample = 0
            for batch_index, data_batch in enumerate(train_data):
                pc, seg, label = data_batch
                pc = pc.to(self.hp.device)
                pc = data_augment(pc, classification=False)
                seg = seg.to(self.hp.device)
                optimizer.zero_grad()
                scores, penalty = self.net(pc, seg)
                seg = seg.reshape(-1)
                if self.hp.label_smooth:
                    loss = label_smooth_loss(scores, seg) + penalty * self.hp.penalty
                else:
                    loss = F.nll_loss(scores, seg) + penalty * self.hp.penalty
                loss.backward()
                total_loss.append(loss.detach().cpu())
                optimizer.step()
                pred = scores.max(1)[1]
                train_correct += (pred == seg).sum().cpu().item()
                train_sample += pc.shape[0] * pc.shape[1]
                del pc, seg, scores, loss, pred, penalty
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            train_acc = train_correct * 1.0 / train_sample
            train_acc = round(train_acc, 4)
            train_loss = round(torch.tensor(total_loss).mean().detach().item(), 6)
            with open(self.hp.name + ".txt", 'a') as f:
                f.write("Epoch: {0} | Train Loss: {1} | Train Acc: {2} | ".format(epoch + 1, train_loss, train_acc))

            if (epoch + 1) % 1 == 0:
                self.net.eval()
                with torch.no_grad():
                    IoUs = []
                    test_correct = 0
                    test_sample = 0
                    for batch_index, data_batch in enumerate(test_data):
                        pc, seg, label = data_batch
                        pc = pc.to(self.hp.device)
                        seg = seg.to(self.hp.device)
                        scores, penalty = self.net(pc, seg)
                        seg = seg.reshape(-1)
                        pred = scores.max(1)[1]
                        test_correct += (pred == seg).sum().cpu().item()
                        test_sample += pc.shape[0] * pc.shape[1]
                        pred = pred.cpu().reshape(pc.shape[0], -1).numpy()
                        seg = seg.cpu().reshape(pc.shape[0], -1).numpy()
                        for i in range(pc.shape[0]):
                            IoUs_i = []
                            label_i = label[i].cpu().item()
                            for seg_idx in test_data.dataset.id_seg_separate[label_i]:
                                I = np.logical_and(pred[i] == seg_idx, seg[i] == seg_idx).sum()
                                U = np.logical_or(pred[i] == seg_idx, seg[i] == seg_idx).sum()
                                IoUs_i.append(1 if U == 0 else I / float(U))
                            IoUs.append(np.mean(IoUs_i))
                        del pc, seg, scores, pred, penalty
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    test_acc = test_correct * 1.0 / test_sample
                    test_acc = round(test_acc, 4)
                    avg_mIoU = np.mean(IoUs)
                    with open(self.hp.name + ".txt", 'a') as f:
                        f.write("Test Acc: {0} | mIoU: {1}\n".format(test_acc, avg_mIoU))
                    if avg_mIoU > highest_mIoU:
                        highest_mIoU = avg_mIoU
                        with open(self.hp.name + "_evaluation.txt", 'a') as f:
                            f.write("Highest mIoU: {0}\n".format(avg_mIoU))
                        torch.save(self.net.state_dict(), self.hp.name + ".pt")

        with open(self.hp.name + ".txt", 'a') as f:
            f.write("Final highest mIoU: {0}\n".format(highest_mIoU))

    def evaluation(self):
        self.net.load_state_dict(torch.load(self.hp.name + ".pt", map_location=lambda storage, location: storage))
        self.net.to(self.hp.device)
        torch.set_grad_enabled(False)
        self.net.eval()
        test_data = PartShapeNetDataset("test", num_points=2048, type_id=int(self.hp.index))

        mIoU = []
        instance_IoUs = []
        label = int(self.hp.index)

        for i in range(test_data.__len__()):
            pc, seg = test_data.__getitem__(i)
            pc = torch.from_numpy(pc)
            pc = pc.unsqueeze(0)
            pc = pc.to(self.hp.device)
            seg = seg.to(self.hp.device)
            scores, penalty = self.net(pc, seg)
            seg = seg.reshape(-1)
            pred = scores.max(1)[1].cpu().numpy()
            IoUs_i = []
            for seg_idx in test_data.id_seg_separate[label]:
                I = np.logical_and(pred == seg_idx, seg == seg_idx).sum()
                U = np.logical_or(pred == seg_idx, seg == seg_idx).sum()
                IoUs_i.append(1 if U == 0 else I/float(U))
            instance_IoUs = instance_IoUs + IoUs_i
            mIoU.append(np.mean(IoUs_i))

        with open(self.hp.name + '_evaluation.txt', 'w') as f:
            f.write('class {0}\n'.format(test_data.id_name[label]))
            f.write('class average mIoU {0} class size {1}\n'.format(np.mean(mIoU), len(mIoU)))
            f.write('instance average mIoU {0} instance num {1}\n'.format(np.mean(instance_IoUs), len(instance_IoUs)))

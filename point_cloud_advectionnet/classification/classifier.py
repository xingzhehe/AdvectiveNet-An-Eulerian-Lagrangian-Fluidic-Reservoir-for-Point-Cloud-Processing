import h5py
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from classification.modelnet import ModelNetDataset
from classification.advection_net import AdvectionNet
from src.nn_utils import *
from src.point_cloud_data_augmentation import data_augment


class Classifier:
    def __init__(self, hp):
        self.hp = hp
        if self.hp.net == 'advection_net':
            self.net = AdvectionNet(num_class=self.hp.num_class, pic_flip=self.hp.pic_flip,
                                    grid_size=self.hp.grid_size, drop_rate=self.hp.drop_rate)
        self.net.to(self.hp.device)

    def train(self):
        log_file = open(self.hp.name+".txt", 'a')
        log_file.close()
        train_data = torch.utils.data.DataLoader(ModelNetDataset("train", num_points=self.hp.num_points,),
                                                 batch_size=self.hp.batch_size,
                                                 shuffle=True,
                                                 num_workers=self.hp.workers)
        test_data = torch.utils.data.DataLoader(ModelNetDataset("test", num_points=self.hp.num_points),
                                                batch_size=self.hp.batch_size,
                                                num_workers=self.hp.workers)
        optimizer = ScheduledOptimizer(self.net.parameters(), otype=self.hp.optimizer, lr=self.hp.lr, T_mult=self.hp.sgdr_T_mult,
                                       lr_scheduler_step_size=self.hp.lr_scheduler_step_size, lr_gamma=self.hp.lr_gamma,
                                       lr_clip=self.hp.lr_clip, lr_weight_decay=self.hp.lr_weight_decay, lr_sgd_momentum=self.hp.lr_sgd_momentum)

        bn_scheduler = BNMomentumScheduler(self.net, init_momentum=self.hp.bn_momentum, clip=self.hp.bn_clip,
                                           step_size=self.hp.bn_step_size, gamma=self.hp.bn_gamma)

        highest_accuracy = 0

        for epoch in range(self.hp.max_epoch):
            total_loss = []
            self.net.train()
            optimizer.scheduler_step()
            train_correct = 0
            train_sample = 0
            for batch_index, data_batch in enumerate(train_data):
                data, label = data_batch
                data = data.to(self.hp.device)
                data = data_augment(data, classification=True)
                optimizer.zero_grad()
                scores, penalty = self.net(data, label.to(self.hp.device))
                if self.hp.label_smooth:
                    loss = label_smooth_loss(scores, label.to(self.hp.device)) + self.hp.penalty * penalty
                else:
                    loss = F.nll_loss(scores, label.to(self.hp.device)) + self.hp.penalty * penalty
                loss.backward()
                total_loss.append(loss.detach().cpu().item())
                optimizer.step()
                pred = scores.max(1)[1].detach()
                train_correct += (pred.cpu() == label.detach()).sum().detach().item()
                train_sample += data.shape[0]
                del data, label, scores, loss, pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            train_acc = train_correct * 1.0 / train_sample
            train_acc = round(train_acc, 4)
            train_loss = round(torch.tensor(total_loss).mean().detach().item(), 6)
            with open(self.hp.name + ".txt", 'a') as f:
                f.write("Epoch: {0} | Train Loss: {1} | Train Acc: {2} ".format(epoch + 1, train_loss, train_acc))

            if (epoch + 1) % 1 == 0:
                self.net.eval()
                with torch.no_grad():
                    test_correct = 0
                    test_sample = 0
                    for batch_index, data_batch in enumerate(test_data):
                        data, label = data_batch
                        scores, penalty = self.net(data.to(self.hp.device), label.to(self.hp.device))
                        pred = scores.max(1)[1].detach()
                        test_correct += (pred.cpu() == label.detach()).sum().item()
                        test_sample += data.shape[0]
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    test_acc = test_correct * 1.0 / test_sample
                    test_acc = round(test_acc, 4)
                    with open(self.hp.name + ".txt", 'a') as f:
                        f.write(" | Test Acc: {0}\n".format(test_acc))
                    if test_acc > highest_accuracy:
                        highest_accuracy = test_acc
                        torch.save(self.net.state_dict(), self.hp.name + ".pt")

        with open(self.hp.name+".txt", 'a') as f:
            f.write("Final highest accuracy: {0}\n".format(highest_accuracy))
        return highest_accuracy

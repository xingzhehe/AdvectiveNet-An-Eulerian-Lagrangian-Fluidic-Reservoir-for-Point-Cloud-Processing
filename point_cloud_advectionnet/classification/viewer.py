import torch
from segmentation.shapenet import *
from classification.modelnet import *
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.point_cloud_data_augmentation import *
from classification.advection_net import AdvectionNet


def draw_pc(pc):
    ax = plt.subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], cmap='viridis', s=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.figure()


def find_pc(dataset, shape, count):
    current_count = 0
    for i in range(1000):
        if dataset.id_name[dataset.label[i]] == shape:
            current_count += 1
            if current_count == count:
                return i


def draw_multi_pc(dataset, count, num):
    num += count
    for i in range(2467):
        if count == num:
            break
        if dataset.id_name[dataset.label[i]] == shape:
            count += 1
            draw_pc(dataset.data[i])


if __name__ == '__main__':
    data_type = "test"
    data_rand = True
    data_aug = True
    dataset = ModelNetDataset(data_type, num_points=1024)
    shape = "airplane"
    index = find_pc(dataset, shape, 1)
    print(dataset.id_name[dataset.label[index]])
    data = random_points(torch.tensor(dataset.data[index])) if data_rand else dataset.data[index]
    data = data_augment(data.unsqueeze(0)).squeeze(0).numpy() if data_aug else data.numpy()
    draw_pc(data)

    plt.show()


import os
import h5py
import numpy as np
import torch.utils.data
from src.point_cloud_data_augmentation import random_points, gauss_sample_points


def load_data_label(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_type, num_points):
        self.num_points = num_points
        self.data_type = data_type
        self.data_augmentation = True if data_type == "train" else False

        root = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "..", "..",  "data", "modelnet40")

        self.id_name = {}
        count = 0
        for shape_name in open(os.path.join(root, "shape_names.txt")):
            self.id_name[count] = shape_name.rstrip()
            count += 1

        data = []
        label = []
        for file_name in open(os.path.join(root, data_type + "_files.txt")):
            one_data, one_label = load_data_label(os.path.join(root, "..", "..", file_name.rstrip()))
            data.append(one_data)
            label.append(one_label)
        self.data = np.concatenate(data).astype(np.float32)
        self.label = np.concatenate(label).astype(np.int64).reshape(-1)

        if num_points < 2048:
            self.data = self.data[:, 0:num_points, :]

    def __getitem__(self, index):
        #data = random_points(self.data[index]) if self.data_augmentation else self.data[index]
        data = gauss_sample_points(self.data[index], mean=self.num_points, std=int(self.num_points/8),
                                   clip_min=int(self.num_points/2), clip_high=self.num_points)
        return data, self.label[index]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    a = ModelNetDataset('test', 1024)

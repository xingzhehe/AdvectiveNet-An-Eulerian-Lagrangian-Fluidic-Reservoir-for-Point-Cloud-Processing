import os
import sys
sys.path.append("../../../")
import numpy as np
import h5py
import torch.utils.data
from src.point_cloud_data_augmentation import gauss_sample_points_seg, jitter_point_cloud
from src.nn_utils import one_hot_encoding, cal_label_weight
import json


def load_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return data, label, seg


class PartShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_type, num_points, type_id=0):
        self.num_points = num_points
        self.data_type = data_type
        self.data_augmentation = True if data_type == "train" else False

        root = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "..", "..", "data", "shapenet")
        data = []
        label = []
        seg = []
        for file_name in open(os.path.join(root, data_type + "_hdf5_file_list.txt")):
            one_data, one_label, one_seg = load_data_label_seg(os.path.join(root, file_name.rstrip()))
            data.append(one_data)
            label.append(one_label)
            seg.append(one_seg)
        if data_type == 'train':
            for file_name in open(os.path.join(root, "val_hdf5_file_list.txt")):
                one_data, one_label, one_seg = load_data_label_seg(os.path.join(root, file_name.rstrip()))
                data.append(one_data)
                label.append(one_label)
                seg.append(one_seg)
        self.data = np.concatenate(data).astype(np.float32)
        self.label = np.concatenate(label).astype(np.int64).reshape(-1)
        self.seg = np.concatenate(seg).astype(np.int64)
        if num_points < 2048:
            self.data = self.data[:, 0:num_points, :]
            self.seg = self.seg[:, 0:num_points]

        chosen = self.label == type_id
        self.data = self.data[chosen, :, :]
        self.label = self.label[chosen]
        self.seg = self.seg[chosen]
        seg_min = self.seg.min()
        self.seg = self.seg - seg_min

        self.id_name = {}
        file_id = {}
        count = 0
        for line in open(os.path.join(root, "all_object_categories.txt")):
            self.id_name[count] = line.split('\t')[0]
            file_id[line.split('\t')[1].rstrip()] = count
            count += 1

        self.id_seg = {}
        with open(os.path.join(root, "catid_partid_to_overallid.json")) as json_file:
            temp_file_seg = json.load(json_file)
        for key in temp_file_seg.keys():
            self.id_seg[file_id[key[0:8]]] = []
        for key, value in temp_file_seg.items():
            self.id_seg[file_id[key[0:8]]].append(value)

        self.id_seg_separate = {}
        for i in range(16):
            seg_i = np.array(self.id_seg[i])
            self.id_seg_separate[i] = list(seg_i - seg_i.min())

        self.seg_color = {}
        for line in open(os.path.join(root, "color_partid_catid_map.txt")):
            id_num, color = line.split(':')
            self.seg_color[int(id_num)] = (float(color[7:11]), float(color[13:17]), float(color[19:23]))

    def __getitem__(self, index):
        if self.data_augmentation:
            pc, seg = gauss_sample_points_seg(self.data[index], self.seg[index],
                                              mean=self.num_points, std=self.num_points/8,
                                              clip_min=self.num_points/4, clip_high=self.num_points)
        else:
            pc = self.data[index]
            seg = self.seg[index]
        return pc, seg, self.label[index]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    a = PartShapeNetDataset('test', 1024)
    print(one_hot_encoding(np.array([2,1,0,4])))
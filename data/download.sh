#!/usr/bin/env bash

wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip
rm modelnet40_ply_hdf5_2048.zip
mv modelnet40_ply_hdf5_2048 modelnet40

wget https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip
unzip shapenet_part_seg_hdf5_data.zip
rm shapenet_part_seg_hdf5_data.zip
mv shapenet_part_seg_hdf5_data shapenet

wget https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip
unzip indoor3d_sem_seg_hdf5_data.zip
rm indoor3d_sem_seg_hdf5_data.zip

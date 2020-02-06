# AdvectiveNet: An Eulerian-Lagrangian Fluidic Reservoir for Point Cloud Processing
We propose a new neural network module inspired by advection in CFD to deal with the high-level tasks on point clouds including classification and segmentation.

[[Paper]](https://arxiv.org/abs/2002.00118)

## Citation
Please cite this paper if you want to use it in your work,

	@inproceedings{
    He2020AdvectiveNet:,
    title={AdvectiveNet: An Eulerian-Lagrangian Fluidic Reservoir for Point Cloud Processing     },
    author={Xingzhe He and Helen Lu Cao and Bo Zhu},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=H1eqQeHFDS}
    }
    
    
## Install
To make it run, one need to first compile the cuda codes.
```
  cd src/grid_average
  python setup.py install
  cd ../rev_trilinear
  python setup.py install
  ```


## License
MIT License

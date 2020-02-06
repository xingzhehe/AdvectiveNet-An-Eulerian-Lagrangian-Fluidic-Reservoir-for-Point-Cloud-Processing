#include <torch/extension.h>
#include <iostream>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor cal_pc_relative(torch::Tensor pc, int grid_size);

torch::Tensor cal_feature_relative(torch::Tensor pc, torch::Tensor feature, int grid_size);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cal_pc_relative", &cal_pc_relative, "relative particle position to average of particles in a grid cell");
  m.def("cal_feature_relative", &cal_feature_relative, "relative feature to average of particles in a grid cell");
}
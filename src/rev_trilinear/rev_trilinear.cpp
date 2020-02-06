#include <torch/extension.h>
#include <iostream>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor cal_pc_grid_index(torch::Tensor pc, int grid_size);

torch::Tensor cal_weight_sum(torch::Tensor pc, torch::Tensor pc_grid_index, int grid_size);

torch::Tensor cal_grid_value(torch::Tensor pc, torch::Tensor pc_value, torch::Tensor pc_grid_index, int grid_size);

torch::Tensor cal_pc_grad(torch::Tensor grad_grid_value, torch::Tensor grid_value, torch::Tensor pc, torch::Tensor grid_weight_sum,
                                       torch::Tensor pc_value, torch::Tensor pc_grid_index, int grid_size);

torch::Tensor cal_pc_value_grad(torch::Tensor grad_grid_value, torch::Tensor pc, torch::Tensor weight_sum, torch::Tensor pc_grid_index, int grid_size);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cal_pc_grid_index", &cal_pc_grid_index, "trilinear_cal_pc_grid_index");
  m.def("cal_weight_sum", &cal_weight_sum, "trilinear_cal_weight_sum");
  m.def("cal_grid_value", &cal_grid_value, "trilinear_cal_grid_value");
  m.def("cal_pc_grad", &cal_pc_grad, "trilinear_cal_pc_grad");
  m.def("cal_pc_value_grad", &cal_pc_value_grad, "trilinear_cal_pc_value_grad");
}
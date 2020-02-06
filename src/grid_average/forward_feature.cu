#include <torch/script.h>
#include <cuda.h>
#include <cuda_runtime.h>


//// blockIdx.x: num_points
//// blockIdx.y: num_channel
//// threadIdx.x: batch_size

__global__ void cal_feature_sum_kernel(
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc, //// (batch, 3, num_points)
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> feature, //// (batch, num_channel, num_points)
    const int grid_size,
    torch::PackedTensorAccessor<float,5,torch::RestrictPtrTraits,size_t> sum_feature,  //// (batch, num_channel, grid_size, grid_size, grid_size)
    torch::PackedTensorAccessor<int32_t,4,torch::RestrictPtrTraits,size_t> count_pc   //// (batch, grid_size, grid_size, grid_size)
    )
{
  float dx=1.0/grid_size;
  float x = pc[threadIdx.x][0][blockIdx.x];
  float y = pc[threadIdx.x][1][blockIdx.x];
  float z = pc[threadIdx.x][2][blockIdx.x];
  int cell_x = __float2int_rd(x/dx);
  int cell_y = __float2int_rd(y/dx);
  int cell_z = __float2int_rd(z/dx);
  cell_x = min(max(cell_x, 0), grid_size-1);
  cell_y = min(max(cell_y, 0), grid_size-1);
  cell_z = min(max(cell_z, 0), grid_size-1);

  atomicAdd(&(sum_feature[threadIdx.x][blockIdx.y][cell_x][cell_y][cell_z]), feature[threadIdx.x][blockIdx.y][blockIdx.x]);
  atomicAdd(&(count_pc[threadIdx.x][cell_x][cell_y][cell_z]), 1);
}

__global__ void cal_feature_relative_kernel(
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc, //// (batch, 3, num_points)
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> feature, //// (batch, num_channel, num_points)
    const torch::PackedTensorAccessor<float,5,torch::RestrictPtrTraits,size_t> sum_feature, //// (batch, 3, grid_size, grid_size, grid_size)
    const torch::PackedTensorAccessor<int32_t,4,torch::RestrictPtrTraits,size_t> count_pc, //// (batch, 3, grid_size, grid_size, grid_size)
    const int grid_size,
    torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> relative_feature   //// (batch, 2*num_channel, num_points)
    )
{
  float dx=1.0/grid_size;
  float x = pc[threadIdx.x][0][blockIdx.x];
  float y = pc[threadIdx.x][1][blockIdx.x];
  float z = pc[threadIdx.x][2][blockIdx.x];
  int cell_x = __float2int_rd(x/dx);
  int cell_y = __float2int_rd(y/dx);
  int cell_z = __float2int_rd(z/dx);
  cell_x = min(max(cell_x, 0), grid_size-1);
  cell_y = min(max(cell_y, 0), grid_size-1);
  cell_z = min(max(cell_z, 0), grid_size-1);

  int count = max(count_pc[threadIdx.x][cell_x][cell_y][cell_z], 1);

  relative_feature[threadIdx.x][blockIdx.y][blockIdx.x]=sum_feature[threadIdx.x][blockIdx.y][cell_x][cell_y][cell_z]/count;
  relative_feature[threadIdx.x][blockIdx.y+blockDim.y][blockIdx.x]=feature[threadIdx.x][blockIdx.y][blockIdx.x]-relative_feature[threadIdx.x][blockIdx.y][blockIdx.x];
}


torch::Tensor cal_feature_relative(torch::Tensor pc, torch::Tensor feature, int grid_size)
{
int batch_size = pc.size(0);
int num_points = pc.size(2);
int num_channel = feature.size(1);
auto sum_feature = torch::zeros({batch_size, num_channel, grid_size, grid_size, grid_size}).to(pc);
auto count_pc = torch::zeros({batch_size, grid_size, grid_size, grid_size}).to(pc).to(at::kInt);
auto relative_feature = torch::zeros({batch_size, num_channel*2, num_points}).to(pc);
pc = (pc + 1) / 2;

const int threads = batch_size;
const dim3 blocks(num_points, num_channel);

cal_feature_sum_kernel<<<blocks, threads>>>(
pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
feature.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
grid_size,
sum_feature.packed_accessor<float,5,torch::RestrictPtrTraits,size_t>(),
count_pc.packed_accessor<int32_t,4,torch::RestrictPtrTraits,size_t>()
);

cal_feature_relative_kernel<<<blocks, threads>>>(
pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
feature.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
sum_feature.packed_accessor<float,5,torch::RestrictPtrTraits,size_t>(),
count_pc.packed_accessor<int32_t,4,torch::RestrictPtrTraits,size_t>(),
grid_size,
relative_feature.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>()
);


return relative_feature;
}
#include <torch/script.h>
#include <cuda.h>
#include <cuda_runtime.h>


//// blockIdx.x: num_points
//// blockIdx.y: 4
//// threadIdx.x: batch_size

__global__ void cal_pc_grad_kernel(
    const torch::PackedTensorAccessor<float,5,torch::RestrictPtrTraits,size_t> grad_grid_value,   //// (batch, channel, grid_size, grid_size, grid_size)
    const torch::PackedTensorAccessor<float,5,torch::RestrictPtrTraits,size_t> grid_value,   //// (batch, channel, grid_size, grid_size, grid_size)
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc, //// (batch, 3, num_points)
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> weight_sum, //// (batch, grid_size, grid_size, grid_size)
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc_value, //// (batch, channel, num_points)
    const torch::PackedTensorAccessor<int32_t,3,torch::RestrictPtrTraits,size_t> pc_grid_index,   //// (batch, 6, num_points)
    const int grid_size,
    const int num_channel,
    torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> grad_pc   //// (batch, 2, num_points)
    )
{
  float dx=1.0/(grid_size-1);
float x = pc[threadIdx.x][0][blockIdx.x];
float y = pc[threadIdx.x][1][blockIdx.x];
float z = pc[threadIdx.x][2][blockIdx.x];
int cell_x0 = pc_grid_index[threadIdx.x][0][blockIdx.x], cell_x1 = pc_grid_index[threadIdx.x][1][blockIdx.x];
int cell_y0 = pc_grid_index[threadIdx.x][2][blockIdx.x], cell_y1 = pc_grid_index[threadIdx.x][3][blockIdx.x];
int cell_z0 = pc_grid_index[threadIdx.x][4][blockIdx.x], cell_z1 = pc_grid_index[threadIdx.x][5][blockIdx.x];
float x0=cell_x0*dx, x1=cell_x1*dx, y0=cell_y0*dx, y1=cell_y1*dx, z0=cell_z0*dx, z1=cell_z1*dx;
if(x0<x && x1>x && y0<y && y1>y && z0<z && z1>z){
  int cell_x=cell_x0, cell_y=cell_y0, cell_z=cell_z0; float w1=-(y1-y)*(z1-z), w2=-(x1-x)*(z1-z), w3=-(x1-x)*(y1-y);
  switch(blockIdx.y){
    case 1: {cell_x=cell_x0; cell_y=cell_y0; cell_z=cell_z1; w1=-(y1-y)*(z-z0); w2=-(x1-x)*(z-z0); w3=(x1-x)*(y1-y); break;}
    case 2: {cell_x=cell_x0; cell_y=cell_y1; cell_z=cell_z0; w1=-(y-y0)*(z1-z); w2=(x1-x)*(z1-z); w3=-(x1-x)*(y-y0); break;}
    case 3: {cell_x=cell_x1; cell_y=cell_y0; cell_z=cell_z0; w1=(y1-y)*(z1-z); w2=-(x-x0)*(z1-z); w3=-(x-x0)*(y1-y); break;}
    case 4: {cell_x=cell_x0; cell_y=cell_y1; cell_z=cell_z1; w1=-(y-y0)*(z-z0); w2=(x1-x)*(z-z0); w3=(x1-x)*(y-y0); break;}
    case 5: {cell_x=cell_x1; cell_y=cell_y0; cell_z=cell_z1; w1=(y1-y)*(z-z0); w2=-(x-x0)*(z-z0); w3=(x-x0)*(y1-y); break;}
    case 6: {cell_x=cell_x1; cell_y=cell_y1; cell_z=cell_z0; w1=(y-y0)*(z1-z); w2=(x-x0)*(z1-z); w3=-(x-x0)*(y-y0); break;}
    case 7: {cell_x=cell_x1; cell_y=cell_y1; cell_z=cell_z1; w1=(y-y0)*(z-z0); w2=(x-x0)*(z-z0); w3=(x-x0)*(y-y0); break;}
    default:break;
  }

  float dLdw=0;
  for(int channel_i=0;channel_i<num_channel;channel_i++){
    dLdw += grad_grid_value[threadIdx.x][channel_i][cell_x][cell_y][cell_z] * (pc_value[threadIdx.x][channel_i][blockIdx.x]-grid_value[threadIdx.x][channel_i][cell_x][cell_y][cell_z]);
  }
  dLdw /= weight_sum[threadIdx.x][cell_x][cell_y][cell_z];
  atomicAdd(&(grad_pc[threadIdx.x][0][blockIdx.x]), dLdw*w1/2.0);
  atomicAdd(&(grad_pc[threadIdx.x][1][blockIdx.x]), dLdw*w2/2.0);
  atomicAdd(&(grad_pc[threadIdx.x][2][blockIdx.x]), dLdw*w3/2.0);
  }
}


torch::Tensor cal_pc_grad(torch::Tensor grad_grid_value, torch::Tensor grid_value, torch::Tensor pc, torch::Tensor weight_sum,
                                       torch::Tensor pc_value, torch::Tensor pc_grid_index, int grid_size)
{
    int batch_size = pc.size(0);
    int num_points = pc.size(2);
    int num_channel = grad_grid_value.size(1);
    auto grad_pc = torch::zeros({batch_size, 3, num_points}).to(pc);
    pc = (pc + 1) / 2;

    const int threads = batch_size;
    const dim3 blocks(num_points, 8);

    cal_pc_grad_kernel<<<blocks, threads>>>(
        grad_grid_value.packed_accessor<float,5,torch::RestrictPtrTraits,size_t>(),
        grid_value.packed_accessor<float,5,torch::RestrictPtrTraits,size_t>(),
        pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        weight_sum.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
        pc_value.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        pc_grid_index.packed_accessor<int32_t,3,torch::RestrictPtrTraits,size_t>(),
        grid_size,
        num_channel,
        grad_pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>()
        );

    return grad_pc;
}


//// blockIdx.x: num_points
//// blockIdx.y: num_channel
//// threadIdx.x: batch_size
__global__ void cal_pc_value_grad_kernel(
  const torch::PackedTensorAccessor<float,5,torch::RestrictPtrTraits,size_t> grad_grid_value,   //// (batch, channel, grid_size, grid_size, grid_size)
  const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc, //// (batch, 3, num_points)
  const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> weight_sum, //// (batch, grid_size, grid_size, grid_size)
  const torch::PackedTensorAccessor<int32_t,3,torch::RestrictPtrTraits,size_t> pc_grid_index,   //// (batch, 6, num_points)
  const int grid_size,
  torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> grad_pc_value   //// (batch, 3, num_points)
  )
{
float dx=1.0/(grid_size-1);
float x = pc[threadIdx.x][0][blockIdx.x];
float y = pc[threadIdx.x][1][blockIdx.x];
float z = pc[threadIdx.x][2][blockIdx.x];
int cell_x0 = pc_grid_index[threadIdx.x][0][blockIdx.x], cell_x1 = pc_grid_index[threadIdx.x][1][blockIdx.x];
int cell_y0 = pc_grid_index[threadIdx.x][2][blockIdx.x], cell_y1 = pc_grid_index[threadIdx.x][3][blockIdx.x];
int cell_z0 = pc_grid_index[threadIdx.x][4][blockIdx.x], cell_z1 = pc_grid_index[threadIdx.x][5][blockIdx.x];
float x0=cell_x0*dx, x1=cell_x1*dx, y0=cell_y0*dx, y1=cell_y1*dx, z0=cell_z0*dx, z1=cell_z1*dx;
if(x0<x && x1>x && y0<y && y1>y && z0<z && z1>z){

float w000=(x1-x) * (y1-y) * (z1-z) / weight_sum[threadIdx.x][cell_x0][cell_y0][cell_z0];
float w001=(x1-x) * (y1-y) * (z-z0) / weight_sum[threadIdx.x][cell_x0][cell_y0][cell_z1];
float w010=(x1-x) * (y-y0) * (z1-z) / weight_sum[threadIdx.x][cell_x0][cell_y1][cell_z0];
float w100=(x-x0) * (y1-y) * (z1-z) / weight_sum[threadIdx.x][cell_x1][cell_y0][cell_z0];
float w011=(x1-x) * (y-y0) * (z-z0) / weight_sum[threadIdx.x][cell_x0][cell_y1][cell_z1];
float w101=(x-x0) * (y1-y) * (z-z0) / weight_sum[threadIdx.x][cell_x1][cell_y0][cell_z1];
float w110=(x-x0) * (y-y0) * (z1-z) / weight_sum[threadIdx.x][cell_x1][cell_y1][cell_z0];
float w111=(x-x0) * (y-y0) * (z-z0) / weight_sum[threadIdx.x][cell_x1][cell_y1][cell_z1];

grad_pc_value[threadIdx.x][blockIdx.y][blockIdx.x] = w000*grad_grid_value[threadIdx.x][blockIdx.y][cell_x0][cell_y0][cell_z0]
                                                   + w001*grad_grid_value[threadIdx.x][blockIdx.y][cell_x0][cell_y0][cell_z1]
                                                   + w010*grad_grid_value[threadIdx.x][blockIdx.y][cell_x0][cell_y1][cell_z0]
                                                   + w100*grad_grid_value[threadIdx.x][blockIdx.y][cell_x1][cell_y0][cell_z0]
                                                   + w011*grad_grid_value[threadIdx.x][blockIdx.y][cell_x0][cell_y1][cell_z1]
                                                   + w101*grad_grid_value[threadIdx.x][blockIdx.y][cell_x1][cell_y0][cell_z1]
                                                   + w110*grad_grid_value[threadIdx.x][blockIdx.y][cell_x1][cell_y1][cell_z0]
                                                   + w111*grad_grid_value[threadIdx.x][blockIdx.y][cell_x1][cell_y1][cell_z1];

}
}


torch::Tensor cal_pc_value_grad(torch::Tensor grad_grid_value, torch::Tensor pc, torch::Tensor weight_sum, torch::Tensor pc_grid_index, int grid_size)
{
int batch_size = pc.size(0);
int num_points = pc.size(2);
int num_channel = grad_grid_value.size(1);
auto grad_pc_value = torch::zeros({batch_size, num_channel, num_points}).to(pc);
pc = (pc + 1) / 2;

const int threads = batch_size;
const dim3 blocks(num_points, num_channel);

cal_pc_value_grad_kernel<<<blocks, threads>>>(
grad_grid_value.packed_accessor<float,5,torch::RestrictPtrTraits,size_t>(),
pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
weight_sum.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
pc_grid_index.packed_accessor<int32_t,3,torch::RestrictPtrTraits,size_t>(),
grid_size,
grad_pc_value.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>()
);

return grad_pc_value;
}
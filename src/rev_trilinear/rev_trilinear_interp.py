import torch
from torch.autograd import gradcheck
import torch.nn.functional as F
import rev_trilinear


class RevTrilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pc, pc_value, grid_size):
        pc_grid_index = rev_trilinear.cal_pc_grid_index(pc, grid_size)
        weight_sum = rev_trilinear.cal_weight_sum(pc, pc_grid_index, grid_size)
        torch.cuda.synchronize()
        weight_sum[weight_sum == 0] = 1
        grid_value = rev_trilinear.cal_grid_value(pc, pc_value, pc_grid_index, grid_size)
        torch.cuda.synchronize()
        grid_value = grid_value / weight_sum.unsqueeze(1)
        ctx.save_for_backward(pc, pc_value, grid_value, weight_sum, pc_grid_index)
        ctx.grid_size = grid_size
        return grid_value

    @staticmethod
    def backward(ctx, grad_grid_value):
        pc, pc_value, grid_value, weight_sum, pc_grid_index = ctx.saved_tensors
        grad_pc = grad_pc_value = None

        if ctx.needs_input_grad[1]:
            grad_pc_value = rev_trilinear.cal_pc_value_grad(grad_grid_value, pc, weight_sum, pc_grid_index, ctx.grid_size)
            torch.cuda.synchronize()
        if ctx.needs_input_grad[0]:
            grad_pc = rev_trilinear.cal_pc_grad(grad_grid_value, grid_value, pc, weight_sum, pc_value, pc_grid_index, ctx.grid_size)
            torch.cuda.synchronize()

        return grad_pc, grad_pc_value, None


if __name__ == '__main__':
    pc = torch.rand(1, 3, 6, dtype=torch.float32, requires_grad=True).cuda()
    pc_value = torch.rand(1, 1, 6, dtype=torch.float32, requires_grad=True).cuda()
    #pc = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).unsqueeze(0).transpose(1, 2).cuda()
    #pc_value = torch.tensor([1.0, -1.0]).reshape(1, 1, 2).cuda()
    pc.requires_grad_(True)
    pc_value.requires_grad_(True)

    grid_value = RevTrilinear.apply(pc, pc_value, 3)

    input = (pc, pc_value, 3)
    test = gradcheck(RevTrilinear.apply, input, eps=1e-3, atol=1e-3)
    print(test)
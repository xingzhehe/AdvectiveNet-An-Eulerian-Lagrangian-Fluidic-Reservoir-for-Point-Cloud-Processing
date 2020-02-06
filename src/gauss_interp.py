import torch
from torch.autograd import gradcheck


class GaussInterp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pc1, pc2, pc1_value, sigma):
        ctx.save_for_backward(pc1, pc2, pc1_value, torch.tensor(sigma))
        weight = torch.exp(-(pc2.unsqueeze(3) - pc1.unsqueeze(2)).norm(p=2, dim=1)**2 / (2*sigma**2))
        weight = weight / (weight.sum(dim=-1).unsqueeze(2) + 1e-6)
        return pc1_value.bmm(weight.transpose(1, 2))

    @staticmethod
    def backward(ctx, grad_output):
        pc1, pc2, pc1_value, sigma = ctx.saved_tensors
        grad_pc1 = grad_pc2 = grad_pc1_value = None
        diff = pc2.unsqueeze(3) - pc1.unsqueeze(2)
        weight = torch.exp(-diff.norm(p=2, dim=1)**2 / (2 * sigma ** 2))
        weight = weight / (weight.sum(dim=-1).unsqueeze(2) + 1e-6)

        if ctx.needs_input_grad[2]:
            grad_pc1_value = grad_output.bmm(weight)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            d_weight_pc = - diff / (sigma ** 2) * weight.unsqueeze(1)
            d_loss_weight = grad_output.transpose(1, 2).bmm(pc1_value).unsqueeze(1)
            del diff
            if ctx.needs_input_grad[0]:
                d_weight_pc1 = - (weight.unsqueeze(1) * d_loss_weight).sum(dim=-1, keepdim=True)
                d_weight_pc1 = (d_weight_pc1 + d_loss_weight) * d_weight_pc
                grad_pc1 = - d_weight_pc1.sum(dim=-2)
                del d_weight_pc1
            if ctx.needs_input_grad[1]:
                d_weight_pc2 = d_weight_pc - weight.unsqueeze(1) * d_weight_pc.sum(dim=-1).unsqueeze(-1)
                d_weight_pc2 = d_weight_pc2 * d_loss_weight
                grad_pc2 = d_weight_pc2.sum(dim=-1)

        return grad_pc1, grad_pc2, grad_pc1_value, None


def gauss_interp(pc1, pc2, pc1_value, sigma, conservation=True):
    """
    :param pc1: (batch, 3, num_points)
    :param pc2: (batch, 3, num_nodes)
    :param pc1_value: (batch, channel, num_points)
    :param sigma: float
    :return: (batch, channel, num_nodes)
    """
    weight = torch.exp(-(pc2.unsqueeze(3) - pc1.unsqueeze(2)).norm(p=2, dim=1) ** 2 / (2*sigma**2))
    if conservation:
        return pc1_value.bmm(weight.transpose(1, 2)) / (weight.sum(dim=-1).unsqueeze(1) + 1e-6)
    else:
        return pc1_value.bmm(weight.transpose(1, 2))


def pc_grid(pc, value, grid, grid_size, sigma, dim=3, conservation=True):
    if dim == 3:
        return gauss_interp(pc, grid, value, sigma, conservation).reshape(pc.shape[0], value.shape[1], grid_size, grid_size, grid_size)
    elif dim == 2:
        return gauss_interp(pc, grid, value, sigma, conservation).reshape(pc.shape[0], value.shape[1], grid_size, grid_size)


def grid_pc(pc, value, grid, grid_size, sigma, dim=3, conservation=True):
    return gauss_interp(grid, pc, value.reshape(pc.shape[0], value.shape[1], grid_size ** dim), sigma, conservation)


def make_grid(grid_size, dim=3):
    if dim == 3:
        x = torch.linspace(-1, 1, grid_size)
        x, y, z = torch.meshgrid((x, x, x))
        return torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)
    elif dim == 2:
        x = torch.linspace(-1, 1, grid_size)
        x, y = torch.meshgrid((x, x))
        return torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1)


def gauss_interp_max(pc1, pc2, pc1_value, sigma):
    """
    :param pc1: (batch, 3, num_points)
    :param pc2: (batch, 3, num_nodes)
    :param pc1_value: (batch, channel, num_points)
    :param sigma: float
    :return: (batch, channel, num_nodes)
    """
    weight = torch.exp(-(pc2.unsqueeze(3) - pc1.unsqueeze(2)).norm(p=2, dim=1) ** 2 / (2*sigma**2))
    out = torch.zeros((pc1_value.size(0), pc1_value.size(1), pc2.size(2))).to(pc1)
    for i in range(out.size(0)):
        out[i] = (weight[i].unsqueeze(0) * pc1_value[i].unsqueeze(1)).max(dim=-1)[0]
    return out


def pc_grid_max(pc, value, grid, grid_size, sigma, dim=3):
    if dim == 3:
        return gauss_interp_max(pc, grid, value, sigma).reshape(pc.shape[0], value.shape[1], grid_size, grid_size, grid_size)
    elif dim == 2:
        return gauss_interp_max(pc, grid, value, sigma).reshape(pc.shape[0], value.shape[1], grid_size, grid_size)


def grid_pc_max(pc, value, grid, grid_size, sigma, dim=3):
    return gauss_interp_max(grid, pc, value.reshape(pc.shape[0], value.shape[1], grid_size ** dim), sigma)


if __name__ == '__main__':
    pc1 = torch.rand(2, 3, 14, dtype=torch.double, requires_grad=True)
    pc2 = make_grid(3).unsqueeze(0).expand(1, 3 ** 3, 3).transpose(1, 2).double()
    pc2.requires_grad = True
    value = torch.rand(2, 12, 14, dtype=torch.double, requires_grad=True)
    input = (pc1, pc2, value, 1)
    test = gradcheck(GaussInterp.apply, input, eps=1e-6, atol=1e-4)
    print(pc_grid(pc1, value, pc2, 3, 1).shape)
    print(test)

    pc1 = make_grid(3).unsqueeze(0).expand(1, 3 ** 3, 3).transpose(1, 2).double()
    pc1.requires_grad = True
    pc2 = torch.rand(2, 3, 14, dtype=torch.double, requires_grad=True)
    value = torch.rand(2, 12, 3**3, dtype=torch.double, requires_grad=True)
    input = (pc1, pc2, value, 1)
    test = gradcheck(GaussInterp.apply, input, eps=1e-6, atol=1e-4)
    print(grid_pc(pc2, value, pc1, 3, 1).shape)
    print(test)

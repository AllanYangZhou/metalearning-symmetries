"""Custom layers."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, init
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class WeightNorm(Module):
    append_g = "_g"
    append_v = "_v"

    def __init__(self, module, weights=["weight"]):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w / g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v * (g / torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class IdentityLinear(nn.Linear):
    def __init__(self, features, softmax=False, sigmoid=False):
        if softmax:
            raise ValueError("Softmax not supported currently.")
        self.sigmoid = sigmoid
        self.temp_warp = None
        super(IdentityLinear, self).__init__(features, features, bias=False)
        if self.sigmoid:
            self.temp_warp = nn.Parameter(torch.rand(1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.temp_warp is not None:
            init._no_grad_fill_(self.temp, 5.0)
            init._no_grad_fill_(self.weight, -1.0)
            with torch.no_grad():
                self.weight.fill_diagonal_(1.0)
        else:
            init.eye_(self.weight)

    def forward(self, x):
        weight = self.weight
        if self.sigmoid:
            weight = torch.sigmoid(self.temp * weight)
        return F.linear(x, weight, self.bias)


class ShareLinear(Module):
    def __init__(self, in_features, out_features, bias=True, sigmoid=False):
        super(ShareLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.U_warp = IdentityLinear(in_features, sigmoid=sigmoid)
        self.V_warp = IdentityLinear(out_features, sigmoid=sigmoid)

    def forward(self, x):
        x = self.U_warp(x)
        x = self.linear(x)
        x = self.V_warp(x)
        return x


class ShareLinearFull(Module):
    def __init__(self, in_features, out_features, bias=True, latent_size=3):
        super(ShareLinearFull, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.latent_params = Parameter(torch.Tensor(latent_size))
        self.warp = Parameter(torch.Tensor(in_features * out_features, latent_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def get_weight(self):
        return (self.warp @ self.latent_params).view(self.out_features, self.in_features)

    def reset_parameters(self):
        init._no_grad_normal_(self.warp, 0, 0.01)
        init._no_grad_normal_(self.latent_params, 0, 1 / self.out_features)
        if self.bias is not None:
            weight = self.get_weight()
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = self.get_weight()
        return F.linear(x, weight, self.bias)


class Identity1x1Conv(nn.Conv2d):
    def __init__(self, out_channels):
        super(Identity1x1Conv, self).__init__(out_channels, out_channels, 1, bias=False)

    def reset_parameters(self):
        with torch.no_grad():
            new_weight = torch.eye(self.out_channels, self.out_channels)[..., None, None]
            self.weight = nn.Parameter(new_weight)


class ShareConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, sigmoid=False, **kwargs):
        self.sigmoid = sigmoid
        self.A_warp, self.B_warp, self.C_warp = None, None, None
        super(ShareConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        k_size = int(np.prod(self.kernel_size))
        self.A_warp = nn.Parameter(torch.eye(self.out_channels, self.out_channels))
        self.B_warp = nn.Parameter(torch.eye(self.in_channels, self.in_channels))
        self.C_warp = nn.Parameter(torch.eye(k_size, k_size))
        if self.sigmoid:
            self.temp_warp = nn.Parameter(torch.rand(1))
        self.reset_parameters()

    def reset_parameters(self):
        super(ShareConv2d, self).reset_parameters()
        if self.A_warp is not None:
            self.reset_warp_parameters()

    def reset_warp_parameters(self):
        init._no_grad_fill_(self.A_warp, 0.0)
        init._no_grad_fill_(self.B_warp, 0.0)
        init._no_grad_fill_(self.C_warp, 0.0)
        if self.sigmoid:
            init._no_grad_fill_(self.temp_warp, 7.0)
            init._no_grad_fill_(self.A_warp, -1.0)
            init._no_grad_fill_(self.B_warp, -1.0)
            init._no_grad_fill_(self.C_warp, -1.0)
        with torch.no_grad():
            self.A_warp.fill_diagonal_(1.0)
            self.B_warp.fill_diagonal_(1.0)
            self.C_warp.fill_diagonal_(1.0)

    def forward(self, x):
        A_warp, B_warp, C_warp = self.A_warp, self.B_warp, self.C_warp
        if self.sigmoid:
            A_warp = torch.sigmoid(self.temp_warp * A_warp)
            B_warp = torch.sigmoid(self.temp_warp * B_warp)
            C_warp = torch.sigmoid(self.temp_warp * C_warp)
        orig_shape = self.weight.shape
        # (c_o, c_i, k_w * k_h)
        weight = torch.reshape(self.weight, (self.out_channels, self.in_channels, -1))
        weight = torch.einsum("ij,jkl->ikl", A_warp, weight)
        weight = torch.einsum("ik,jkl->jil", B_warp, weight)
        weight = torch.einsum("il,jkl->jki", C_warp, weight)
        weight = torch.reshape(weight, orig_shape)
        return self.conv2d_forward(x, weight)


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if 2 < tensor.dim() < 5:
            receptive_field_size = tensor[0][0].numel()
        if tensor.dim() >= 5:  # locally connected layer: kernel is stored in last dimension.
            num_input_fmaps = tensor.size(2)
            num_output_fmaps = tensor.size(1)
            receptive_field_size = tensor.size(-1)
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan = _calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class LocallyConnected1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        output_size,
        kernel_size=3,
        stride=1,
        bias=True,
        init_method="kaiming",
    ):
        super(LocallyConnected1d, self).__init__()
        self.weight = nn.Parameter(
            torch.Tensor(1, out_channels, in_channels, output_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, output_size))
        else:
            self.register_parameter("bias", None)
        self.kernel_size = kernel_size
        self.stride = stride
        self.init_method = init_method
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_method == "kaiming":
            kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
        elif self.init_method == "normal":
            init._no_grad_normal_(self.weight, 0, 1)
            if self.bias is not None:
                init._no_grad_normal_(self.bias, 0, 1)
        else:
            raise ValueError(f"Unsupported init method {self.init_method}.")

    def forward(self, x):
        _, _c, _w = x.size()
        k, d = self.kernel_size, self.stride
        x = x.unfold(2, k, d)
        x = x.contiguous().view(*x.size()[:-1], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

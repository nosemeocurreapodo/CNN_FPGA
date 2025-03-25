import torch
import torch.nn as nn
import torch.nn.functional as F

import fake_quantization as fake_quantization


# total for a pynq-z2
def getTotalData():
    total_dict = {"BRAM_18K": 280,
                  "DSP": 220,
                  "FF": 106400,
                  "LUT": 53200,
                  "URAM": 0}
    total = torch.zeros([5])
    total[0] = total_dict["BRAM_18K"]
    total[1] = total_dict["DSP"]
    total[2] = total_dict["FF"]
    total[3] = total_dict["LUT"]
    total[4] = total_dict["URAM"]
    return total


def isImplementable(M_data, x):
    return torch.sum(torch.sigmoid(10.0*(M_data - x[1:])))


def getInterpolationData():
    sample1_dict = {"period": 10,
                    "bits": 32,
                    "in_channels": 32,
                    "out_channels": 32,
                    "in_width": 32,
                    "in_height": 32,
                    "kernel_size": 1,
                    "padding": 1,
                    "latency": 100,
                    "BRAM_18K": 10,
                    "DSP": 10,
                    "FF": 10,
                    "LUT": 10,
                    "URAM": 0}
    sample2_dict = {"period": 10,
                    "bits": 32,
                    "in_channels": 32,
                    "out_channels": 32,
                    "in_width": 32,
                    "in_height": 32,
                    "kernel_size": 3,
                    "padding": 1,
                    "latency": 1000,
                    "BRAM_18K": 100,
                    "DSP": 100,
                    "FF": 100,
                    "LUT": 100,
                    "URAM": 0}
    sample3_dict = {"period": 10,
                    "bits": 32,
                    "in_channels": 128,
                    "out_channels": 128,
                    "in_width": 32,
                    "in_height": 32,
                    "kernel_size": 3,
                    "padding": 1,
                    "latency": 10000,
                    "BRAM_18K": 1000,
                    "DSP": 1000,
                    "FF": 1000,
                    "LUT": 1000,
                    "URAM": 0}
    samples = [sample1_dict, sample2_dict, sample3_dict]

    torch_X_samples = torch.zeros([len(samples), 8])
    torch_Y_samples = torch.zeros([len(samples), 6])

    for i, sample in enumerate(samples):
        torch_X_samples[i, 0] = sample["period"]
        torch_X_samples[i, 1] = sample["bits"]
        torch_X_samples[i, 2] = sample["in_channels"]
        torch_X_samples[i, 3] = sample["out_channels"]
        torch_X_samples[i, 4] = sample["in_width"]
        torch_X_samples[i, 5] = sample["in_height"]
        torch_X_samples[i, 6] = sample["kernel_size"]
        torch_X_samples[i, 7] = sample["padding"]

        torch_Y_samples[i, 0] = sample["latency"]
        torch_Y_samples[i, 1] = sample["BRAM_18K"]
        torch_Y_samples[i, 2] = sample["DSP"]
        torch_Y_samples[i, 3] = sample["FF"]
        torch_Y_samples[i, 4] = sample["LUT"]
        torch_Y_samples[i, 5] = sample["URAM"]

    return torch_Y_samples, torch_X_samples


def conv2D_3x3_estimate_params(Y_data, X_data, x):

    p = 2
    eps = 1e-8

    # 1) Compute distances from x to each sample in X_data
    diffs = X_data - x  # shape: (N, 4)
    # dist_sq = torch.sum(diffs**2, axis=1)  # shape: (N,)
    dist_sq = (diffs**2).sum(axis=1)  # shape: (N,)
    dist = torch.sqrt(dist_sq + 1e-16)     # avoid exact zeros under sqrt

    # 2) Compute weights = 1 / (distance^p + eps)
    w = 1.0 / (dist**p + eps)  # shape: (N,)

    # 3) Normalize the weights
    w_sum = torch.sum(w)
    w_normalized = w / w_sum

    # 4) Weighted average of Y_data
    y_est = torch.matmul(w_normalized, Y_data)  # shape: (D,)
    return y_est


class Conv2DHWNAS(nn.Module):
    def __init__(self, bits, use_relu,
                 in_channels, out_channels,
                 in_width, in_height,
                 kernel_size,
                 padding):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size,
                            padding=padding)
        self.bits = bits
        self.use_relu = use_relu
        Y_data, X_data = getInterpolationData()
        # self.Y_data = Y_data
        # self.X_data = X_data

        x = torch.zeros([8])
        x[0] = 10
        x[1] = bits
        x[2] = in_channels
        x[3] = out_channels
        x[4] = in_width
        x[5] = in_height
        x[6] = kernel_size
        x[7] = padding

        self.register_buffer('Y_data', Y_data)
        self.register_buffer('X_data', X_data)
        self.register_buffer('x', x)

        # self.bits = nn.Parameter(torch.zeros(16),
        #                         requires_grad=True)

    def forward(self, x):
        x = self.op(x)
        if self.use_relu:
            return F.relu(x)
        else:
            return x

        x = fake_quantization.fake_fixed_truncate(x,
                                                  self.bits,
                                                  0,
                                                  0)

        if hasattr(self.op, 'weight') and self.op.weight is not None:
            w = fake_quantization.fake_fixed_truncate(self.op.weight,
                                                      self.bits,
                                                      0,
                                                      0)
        else:
            w = None

        if hasattr(self.op, 'bias') and self.op.bias is not None:
            b = fake_quantization.fake_fixed_truncate(self.op.bias,
                                                      self.bits,
                                                      0,
                                                      0)
        else:
            b = None

        x = F.conv2d(x, w, b,
                     stride=self.op.stride,
                     padding=self.op.padding,
                     dilation=self.op.dilation,
                     groups=self.op.groups)

        if self.use_relu:
            return F.relu(x)
        else:
            return x

    def estimate_params(self):
        return conv2D_3x3_estimate_params(self.Y_data,
                                          self.X_data,
                                          self.x)


class Conv2D_1k_8b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(8, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         1, 0)


class Conv2D_1k_16b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(16, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         1, 0)


class Conv2D_1k_32b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(32, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         1, 0)


class Conv2D_3k_8b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(8, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         3, 1)


class Conv2D_3k_16b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(16, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         3, 1)


class Conv2D_3k_32b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(32, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         3, 1)


class MixedConv2D(nn.Module):
    """
    A single edge that mixes multiple candidate ops,
    weighted by architecture parameters alpha.
    """
    def __init__(self, in_channels, out_channels,
                 in_width, in_height, op_candidates):
        super().__init__()
        self._ops = nn.ModuleList()
        for op in op_candidates:
            self._ops.append(op(in_channels, out_channels,
                                in_width, in_height))

        # Architecture parameters for this edge
        # We'll treat alpha as a learnable Tensor that has one entry per op.
        self.alpha = nn.Parameter(torch.zeros(len(op_candidates)),
                                  requires_grad=True)

    def forward(self, x):
        # Softmax over alpha
        weights = F.softmax(self.alpha, dim=0)
        # Weighted sum of all candidate ops
        out = 0
        for w, op in zip(weights, self._ops):
            out = out + w * op(x)
        return out

    def estimate_params(self):
        weights = F.softmax(self.alpha, dim=0)  # shape [num_ops]

        sum = None
        for i, op in enumerate(self._ops):
            if i == 0:
                sum = weights[0] * op.estimate_params()
            else:
                sum += weights[i] * op.estimate_params()
        
        return sum
        # op_imp = []
        # for op in self._ops:
        #    op_imp.append(op.estimate_params(period).unsqueeze(dim=0))
        # op_imp = torch.cat(op_imp)

        # predicted_params = torch.sum(weights * op_imp, dim=1)
        # return predicted_params


class MergedOp(nn.Module):
    """
    A single edge that mixes multiple candidate ops,
    weighted by architecture parameters alpha.
    """
    def __init__(self, in_layers):
        super().__init__()

        # Architecture parameters for this edge
        # We'll treat alpha as a learnable Tensor that has one entry per op.
        self.alpha = nn.Parameter(torch.zeros(in_layers),
                                  requires_grad=True)

    def forward(self, xs):
        # Softmax over alpha
        weights = F.softmax(self.alpha, dim=0)
        # Weighted sum of all candidate ops
        out = 0
        for w, x in zip(weights, xs):
            out = out + w * x
        return out

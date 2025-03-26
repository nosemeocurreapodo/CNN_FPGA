import torch
import torch.nn as nn
import torch.nn.functional as F

import fake_quantization as fake_quantization


def getInterpolationData():
    samples = [{'latency': 33939, 'clock': 7.492, 'BRAM_18K': 32, 'DSP': 0, 'FF': 578, 'LUT': 2419, 'part': 'xc7z020clg400-1', 'period': '10', 'bits': 2, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
               {'latency': 679959, 'clock': 7.004, 'BRAM_18K': 2, 'DSP': 0, 'FF': 2004, 'LUT': 5335, 'part': 'xc7z020clg400-1', 'period': '10', 'bits': 2, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
               {'latency': 867209, 'clock': 7.194, 'BRAM_18K': 1, 'DSP': 0, 'FF': 503, 'LUT': 8000, 'part': 'xc7z020clg400-1', 'period': '10', 'bits': 2, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1},
               {'latency': 33939, 'clock': 6.912, 'BRAM_18K': 32, 'DSP': 0, 'FF': 567, 'LUT': 2599, 'part': 'xc7z020clg400-1', 'period': '10', 'bits': 4, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
               {'latency': 679983, 'clock': 7.28, 'BRAM_18K': 4, 'DSP': 0, 'FF': 2379, 'LUT': 7451, 'part': 'xc7z020clg400-1', 'period': '10', 'bits': 4, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
               {'latency': 873609, 'clock': 7.211, 'BRAM_18K': 2, 'DSP': 0, 'FF': 660, 'LUT': 17253, 'part': 'xc7z020clg400-1', 'period': '10', 'bits': 4, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1},
               {'latency': 33939, 'clock': 7.234, 'BRAM_18K': 32, 'DSP': 0, 'FF': 683, 'LUT': 2632, 'part': 'xc7z020clg400-1', 'period': '10', 'bits': 8, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
               {'latency': 679976, 'clock': 6.968, 'BRAM_18K': 8, 'DSP': 0, 'FF': 3496, 'LUT': 10380, 'part': 'xc7z020clg400-1', 'period': '10', 'bits': 8, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
               {'latency': 886409, 'clock': 7.194, 'BRAM_18K': 4, 'DSP': 0, 'FF': 1042, 'LUT': 22141, 'part': 'xc7z020clg400-1', 'period': '10', 'bits': 8, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1},
               {'latency': 33935, 'clock': 10.746, 'BRAM_18K': 32, 'DSP': 0, 'FF': 342, 'LUT': 2355, 'part': 'xc7z020clg400-1', 'period': '20', 'bits': 2, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
               {'latency': 679956, 'clock': 12.789, 'BRAM_18K': 2, 'DSP': 0, 'FF': 1953, 'LUT': 5280, 'part': 'xc7z020clg400-1', 'period': '20', 'bits': 2, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
               {'latency': 854407, 'clock': 14.435, 'BRAM_18K': 1, 'DSP': 0, 'FF': 387, 'LUT': 7991, 'part': 'xc7z020clg400-1', 'period': '20', 'bits': 2, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1},
               {'latency': 33935, 'clock': 13.482, 'BRAM_18K': 32, 'DSP': 0, 'FF': 360, 'LUT': 2526, 'part': 'xc7z020clg400-1', 'period': '20', 'bits': 4, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
               {'latency': 679981, 'clock': 13.404, 'BRAM_18K': 4, 'DSP': 0, 'FF': 2264, 'LUT': 7405, 'part': 'xc7z020clg400-1', 'period': '20', 'bits': 4, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
               {'latency': 860807, 'clock': 13.03, 'BRAM_18K': 2, 'DSP': 0, 'FF': 487, 'LUT': 17244, 'part': 'xc7z020clg400-1', 'period': '20', 'bits': 4, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1},
               {'latency': 33935, 'clock': 14.157, 'BRAM_18K': 32, 'DSP': 0, 'FF': 396, 'LUT': 2559, 'part': 'xc7z020clg400-1', 'period': '20', 'bits': 8, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
               {'latency': 679982, 'clock': 14.256, 'BRAM_18K': 8, 'DSP': 0, 'FF': 2863, 'LUT': 10343, 'part': 'xc7z020clg400-1', 'period': '20', 'bits': 8, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
               {'latency': 860807, 'clock': 13.452, 'BRAM_18K': 4, 'DSP': 0, 'FF': 627, 'LUT': 22100, 'part': 'xc7z020clg400-1', 'period': '20', 'bits': 8, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1}]

    torch_X_samples = torch.zeros([len(samples), 8])
    torch_Y_samples = torch.zeros([len(samples), 6])

    for i, sample in enumerate(samples):
        torch_X_samples[i, 0] = sample["clock"]
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
        torch_Y_samples[i, 5] = 0  # sample["URAM"]

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
                 padding,
                 bias=True):
        super().__init__()
        self.op = fake_quantization.QuantWrapperFixedPoint2(
                            nn.Conv2d(in_channels, out_channels,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      bias=bias), bits, False)
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

    def estimate_params(self):
        est_params = conv2D_3x3_estimate_params(self.Y_data,
                                                self.X_data,
                                                self.x)
        return est_params


class Conv2D_1k_8b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(8, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         1, 0, False)


class Conv2D_1k_16b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(16, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         1, 0, False)


class Conv2D_1k_32b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(32, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         1, 0, False)


class Conv2D_3k_2b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(2, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         3, 1, False)


class Conv2D_3k_4b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(4, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         3, 1, False)


class Conv2D_3k_8b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(8, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         3, 1, False)


class Conv2D_3k_16b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(16, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         3, 1, False)


class Conv2D_3k_32b_HWNAS(Conv2DHWNAS):
    def __init__(self, in_channels, out_channels,
                 in_width, in_height):
        super().__init__(32, True,
                         in_channels, out_channels,
                         in_width, in_height,
                         3, 1, False)


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

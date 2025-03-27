import torch
import torch.nn as nn
import torch.nn.functional as F

import fake_quantization as fake_quantization


def getInterpolationData():
    samples = [
                {'clock': 5.884, 'BRAM': 3, 'DSP': 0, 'FF': 247, 'LUT': 267, 'SLICE': 104, 'CLB': 0, 'URAM': 0, 'latency': 33939, 'part': 'xc7z020clg400-1', 'period': 10, 'bits': 2, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
                {'clock': 8.087, 'BRAM': 2, 'DSP': 0, 'FF': 1304, 'LUT': 1509, 'SLICE': 544, 'CLB': 0, 'URAM': 0, 'latency': 679959, 'part': 'xc7z020clg400-1', 'period': 10, 'bits': 2, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
                {'clock': 5.766, 'BRAM': 1, 'DSP': 0, 'FF': 311, 'LUT': 254, 'SLICE': 106, 'CLB': 0, 'URAM': 0, 'latency': 867209, 'part': 'xc7z020clg400-1', 'period': 10, 'bits': 2, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1},
                {'clock': 6.157, 'BRAM': 6, 'DSP': 0, 'FF': 302, 'LUT': 353, 'SLICE': 118, 'CLB': 0, 'URAM': 0, 'latency': 33939, 'part': 'xc7z020clg400-1', 'period': 10, 'bits': 4, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
                {'clock': 9.051, 'BRAM': 4, 'DSP': 0, 'FF': 1639, 'LUT': 1922, 'SLICE': 668, 'CLB': 0, 'URAM': 0, 'latency': 679983, 'part': 'xc7z020clg400-1', 'period': 10, 'bits': 4, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
                {'clock': 5.726, 'BRAM': 2, 'DSP': 0, 'FF': 423, 'LUT': 330, 'SLICE': 137, 'CLB': 0, 'URAM': 0, 'latency': 873609, 'part': 'xc7z020clg400-1', 'period': 10, 'bits': 4, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1},
                {'clock': 5.811, 'BRAM': 6, 'DSP': 0, 'FF': 413, 'LUT': 499, 'SLICE': 167, 'CLB': 0, 'URAM': 0, 'latency': 33939, 'part': 'xc7z020clg400-1', 'period': 10, 'bits': 8, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
                {'clock': 9.494, 'BRAM': 8, 'DSP': 0, 'FF': 2615, 'LUT': 2773, 'SLICE': 976, 'CLB': 0, 'URAM': 0, 'latency': 679976, 'part': 'xc7z020clg400-1', 'period': 10, 'bits': 8, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
                {'clock': 6.151, 'BRAM': 4, 'DSP': 0, 'FF': 647, 'LUT': 458, 'SLICE': 207, 'CLB': 0, 'URAM': 0, 'latency': 886409, 'part': 'xc7z020clg400-1', 'period': 10, 'bits': 8, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1},
                {'clock': 6.127, 'BRAM': 3, 'DSP': 0, 'FF': 183, 'LUT': 215, 'SLICE': 82, 'CLB': 0, 'URAM': 0, 'latency': 33935, 'part': 'xc7z020clg400-1', 'period': 20, 'bits': 2, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
                {'clock': 11.689, 'BRAM': 2, 'DSP': 0, 'FF': 1262, 'LUT': 1555, 'SLICE': 560, 'CLB': 0, 'URAM': 0, 'latency': 679956, 'part': 'xc7z020clg400-1', 'period': 20, 'bits': 2, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
                {'clock': 9.169, 'BRAM': 1, 'DSP': 0, 'FF': 228, 'LUT': 228, 'SLICE': 98, 'CLB': 0, 'URAM': 0, 'latency': 854407, 'part': 'xc7z020clg400-1', 'period': 20, 'bits': 2, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1},
                {'clock': 7.256, 'BRAM': 6, 'DSP': 0, 'FF': 209, 'LUT': 307, 'SLICE': 104, 'CLB': 0, 'URAM': 0, 'latency': 33935, 'part': 'xc7z020clg400-1', 'period': 20, 'bits': 4, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
                {'clock': 11.802, 'BRAM': 4, 'DSP': 0, 'FF': 1515, 'LUT': 2004, 'SLICE': 681, 'CLB': 0, 'URAM': 0, 'latency': 679981, 'part': 'xc7z020clg400-1', 'period': 20, 'bits': 4, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
                {'clock': 8.147, 'BRAM': 2, 'DSP': 0, 'FF': 321, 'LUT': 304, 'SLICE': 117, 'CLB': 0, 'URAM': 0, 'latency': 860807, 'part': 'xc7z020clg400-1', 'period': 20, 'bits': 4, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1},
                {'clock': 9.203, 'BRAM': 6, 'DSP': 0, 'FF': 261, 'LUT': 473, 'SLICE': 154, 'CLB': 0, 'URAM': 0, 'latency': 33935, 'part': 'xc7z020clg400-1', 'period': 20, 'bits': 8, 'kernel_size': 3, 'in_channels': 1, 'out_channels': 32, 'in_height': 32, 'in_width': 32, 'padding': 1},
                {'clock': 12.833, 'BRAM': 8, 'DSP': 0, 'FF': 2069, 'LUT': 2941, 'SLICE': 982, 'CLB': 0, 'URAM': 0, 'latency': 679982, 'part': 'xc7z020clg400-1', 'period': 20, 'bits': 8, 'kernel_size': 3, 'in_channels': 32, 'out_channels': 64, 'in_height': 16, 'in_width': 16, 'padding': 1},
                {'clock': 8.273, 'BRAM': 4, 'DSP': 0, 'FF': 461, 'LUT': 420, 'SLICE': 163, 'CLB': 0, 'URAM': 0, 'latency': 860807, 'part': 'xc7z020clg400-1', 'period': 20, 'bits': 8, 'kernel_size': 3, 'in_channels': 64, 'out_channels': 128, 'in_height': 8, 'in_width': 8, 'padding': 1}
               ]

    torch_X_samples = torch.zeros([len(samples), 8])
    torch_Y_samples = torch.zeros([len(samples), 8])

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
        torch_Y_samples[i, 1] = sample["BRAM"]
        torch_Y_samples[i, 2] = sample["DSP"]
        torch_Y_samples[i, 3] = sample["FF"]
        torch_Y_samples[i, 4] = sample["LUT"]
        torch_Y_samples[i, 5] = sample["SLICE"]
        torch_Y_samples[i, 6] = sample["CLB"]
        torch_Y_samples[i, 7] = sample["URAM"]

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

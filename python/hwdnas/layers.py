import torch
import torch.nn as nn
import torch.nn.functional as F

import fake_quantization as fake_quantization


def conv2D_3x3_1x32x32x32_1_estimate_params(period, bits):
    sample1_dict = {"period": 10,
                    "bits": 8,
                    "latency": 886409,
                    "BRAM_18K": 8,
                    "DSP": 9,
                    "FF": 1578,
                    "LUT": 21914,
                    "URAM": 0}
    sample2_dict = {"period": 20,
                    "bits": 32,
                    "latency": 886409,
                    "BRAM_18K": 8,
                    "DSP": 9,
                    "FF": 1578,
                    "LUT": 21914,
                    "URAM": 0}
    sample3_dict = {"period": 30,
                    "bits": 4,
                    "latency": 886409,
                    "BRAM_18K": 8,
                    "DSP": 9,
                    "FF": 1578,
                    "LUT": 21914,
                    "URAM": 0}


class ModuleHWNAS(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = None
        self.register_buffer('latency', torch.tensor(0))
        self.register_buffer('BRAM', torch.tensor(0))
        self.register_buffer('DSP', torch.tensor(0))
        self.register_buffer('FF', torch.tensor(0))
        self.register_buffer('LUT', torch.tensor(0))
        self.register_buffer('URAM', torch.tensor(0))

        # total for a pynq-z2
        self.register_buffer('t_BRAM', torch.tensor(280))
        self.register_buffer('t_DSP', torch.tensor(220))
        self.register_buffer('t_FF', torch.tensor(106400))
        self.register_buffer('t_LUT', torch.tensor(53200))
        self.register_buffer('t_URAM', torch.tensor(0))

    def forward(self, x):
        return self.op(x)

    def loadFromDict(self, resource_dict):
        self.latency = torch.tensor(resource_dict["latency"])
        self.BRAM = torch.tensor(resource_dict["BRAM"])
        self.DSP = torch.tensor(resource_dict["DSP"])
        self.FF = torch.tensor(resource_dict["FF"])
        self.LUT = torch.tensor(resource_dict["LUT"])
        self.URAM = torch.tensor(resource_dict["URAM"])

    def implementability(self):

        implementability = torch.sigmoid((self.BRAM - self.t_BRAM)*
                                         (self.DSP - self.t_DSP)*
                                         (self.FF - self.t_FF)*
                                         (self.LUT - self.t_LUT)*
                                         (self.URAM - self.t_URAM))
        return implementability

    def getLatency(self):
        return self.latency


class Conv1x1_HWNAS(ModuleHWNAS):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                            padding=0)

        self.loadFromDict({"latency": 886409,
                           "BRAM_18K": 8,
                           "DSP": 9,
                           "FF": 1578,
                           "LUT": 21914,
                           "URAM": 0})


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                            padding=0)

        self.register_buffer('latency', torch.tensor(0.4))

    def forward(self, x):
        return self.op(x)

    def getLatency(self):
        return self.latency


class Conv1x1_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                            padding=0)

        self.register_buffer('latency', torch.tensor(0.5))

    def forward(self, x):
        return F.relu(self.op(x))

    def getLatency(self):
        return self.latency


class Conv1x1_ReLU_BN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                            padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.register_buffer('latency', torch.tensor(1.1))

    def forward(self, x):
        return F.relu(self.bn(self.op(x)))

    def getLatency(self):
        return self.latency


class Conv1x1_ReLU_2_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                            padding=0)

        self.register_buffer('latency', torch.tensor(0.1))

    def forward(self, x):
        # return F.relu(self.op(x))

        x = fake_quantization.fake_fixed_truncate(x,
                                                  2,
                                                  0,
                                                  0)

        if hasattr(self.op, 'weight') and self.op.weight is not None:
            w = fake_quantization.fake_fixed_truncate(self.op.weight,
                                                      2,
                                                      0,
                                                      0)
        else:
            w = None

        if hasattr(self.op, 'bias') and self.op.bias is not None:
            b = fake_quantization.fake_fixed_truncate(self.op.bias,
                                                      2,
                                                      0,
                                                      0)
        else:
            b = None

        if isinstance(self.op, nn.Conv2d):
            x = F.conv2d(x, w, b,
                         stride=self.op.stride,
                         padding=self.op.padding,
                         dilation=self.op.dilation,
                         groups=self.op.groups)
        elif isinstance(self.op, nn.Linear):
            x = F.linear(x, w, b)

        return F.relu(x)

    def getLatency(self):
        return self.latency


class Conv1x1_ReLU_4_4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                            padding=0)

        self.register_buffer('latency', torch.tensor(0.2))

    def forward(self, x):
        # return F.relu(self.op(x))

        x = fake_quantization.fake_fixed_truncate(x,
                                                  4,
                                                  0,
                                                  0)

        if hasattr(self.op, 'weight') and self.op.weight is not None:
            w = fake_quantization.fake_fixed_truncate(self.op.weight,
                                                      4,
                                                      0,
                                                      0)
        else:
            w = None

        if hasattr(self.op, 'bias') and self.op.bias is not None:
            b = fake_quantization.fake_fixed_truncate(self.op.bias,
                                                      4,
                                                      0,
                                                      0)
        else:
            b = None

        if isinstance(self.op, nn.Conv2d):
            x = F.conv2d(x, w, b,
                         stride=self.op.stride,
                         padding=self.op.padding,
                         dilation=self.op.dilation,
                         groups=self.op.groups)
        elif isinstance(self.op, nn.Linear):
            x = F.linear(x, w, b)

        return F.relu(x)

    def getLatency(self):
        return self.latency


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                            padding=1)

        self.register_buffer('latency', torch.tensor(0.8))

    def forward(self, x):
        return self.op(x)

    def getLatency(self):
        return self.latency


class Conv3x3_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                            padding=1)

        self.register_buffer('latency', torch.tensor(0.9))

    def forward(self, x):
        return F.relu(self.op(x))

    def getLatency(self):
        return self.latency


class Conv3x3_ReLU_BN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.register_buffer('latency', torch.tensor(1.5))

    def forward(self, x):
        return F.relu(self.bn(self.op(x)))

    def getLatency(self):
        return self.latency


class Conv3x3_ReLU_2_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                            padding=1)

        self.register_buffer('latency', torch.tensor(0.2))

    def forward(self, x):
        # return F.relu(self.op(x))

        x = fake_quantization.fake_fixed_truncate(x,
                                                  2,
                                                  0,
                                                  0)

        if hasattr(self.op, 'weight') and self.op.weight is not None:
            w = fake_quantization.fake_fixed_truncate(self.op.weight,
                                                      2,
                                                      0,
                                                      0)
        else:
            w = None

        if hasattr(self.op, 'bias') and self.op.bias is not None:
            b = fake_quantization.fake_fixed_truncate(self.op.bias,
                                                      2,
                                                      0,
                                                      0)
        else:
            b = None

        if isinstance(self.op, nn.Conv2d):
            x = F.conv2d(x, w, b,
                         stride=self.op.stride,
                         padding=self.op.padding,
                         dilation=self.op.dilation,
                         groups=self.op.groups)
        elif isinstance(self.op, nn.Linear):
            x = F.linear(x, w, b)

        return F.relu(x)

    def getLatency(self):
        return self.latency


class Conv3x3_ReLU_4_4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                            padding=1)

        self.register_buffer('latency', torch.tensor(0.3))

    def forward(self, x):
        # return F.relu(self.op(x))

        x = fake_quantization.fake_fixed_truncate(x,
                                                  4,
                                                  0,
                                                  0)

        if hasattr(self.op, 'weight') and self.op.weight is not None:
            w = fake_quantization.fake_fixed_truncate(self.op.weight,
                                                      4,
                                                      0,
                                                      0)
        else:
            w = None

        if hasattr(self.op, 'bias') and self.op.bias is not None:
            b = fake_quantization.fake_fixed_truncate(self.op.bias,
                                                      4,
                                                      0,
                                                      0)
        else:
            b = None

        if isinstance(self.op, nn.Conv2d):
            x = F.conv2d(x, w, b,
                         stride=self.op.stride,
                         padding=self.op.padding,
                         dilation=self.op.dilation,
                         groups=self.op.groups)
        elif isinstance(self.op, nn.Linear):
            x = F.linear(x, w, b)

        return F.relu(x)

    def getLatency(self):
        return self.latency


class Conv5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1,
                            padding=2)

        self.register_buffer('latency', torch.tensor(1.2))

    def forward(self, x):
        return self.op(x)

    def getLatency(self):
        return self.latency


class Conv5x5_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1,
                            padding=2)

        self.register_buffer('latency', torch.tensor(1.3))

    def forward(self, x):
        return F.relu(self.op(x))

    def getLatency(self):
        return self.latency


class Conv5x5_ReLU_BN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1,
                            padding=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.register_buffer('latency', torch.tensor(1.9))

    def forward(self, x):
        return F.relu(self.bn(self.op(x)))

    def getLatency(self):
        return self.latency


class NoConv(nn.Module):
    """Skip/Identity operation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.register_buffer('latency', torch.tensor(0.1))

    def forward(self, x):
        return x

    def getLatency(self):
        return self.latency


class NoConv_ReLU(nn.Module):
    """Skip/Identity operation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.register_buffer('latency', torch.tensor(0.2))

    def forward(self, x):
        return F.relu(x)

    def getLatency(self):
        return self.latency


class NoConv_ReLU_BN(nn.Module):
    """Skip/Identity operation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.register_buffer('latency', torch.tensor(0.8))

    def forward(self, x):
        return F.relu(self.bn(x))

    def getLatency(self):
        return self.latency


class Linear(nn.Module):
    """Skip/Identity operation."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.op = nn.Linear(in_features, out_features)
        self.register_buffer('latency', torch.tensor(1.4))

    def forward(self, x):
        return self.op(x)

    def getLatency(self):
        return self.latency


class MixedOp(nn.Module):
    """
    A single edge that mixes multiple candidate ops,
    weighted by architecture parameters alpha.
    """
    def __init__(self, in_channels, out_channels, op_candidates):
        super().__init__()
        self._ops = nn.ModuleList()
        for op_class in op_candidates:
            self._ops.append(op_class(in_channels, out_channels))

        # Architecture parameters for this edge
        # We'll treat alpha as a learnable Tensor that has one entry per op.
        self.alpha = nn.Parameter(torch.zeros(len(op_candidates)),
                                  requires_grad=True)

        op_lat = []
        for op in self._ops:
            op_lat.append(op.getLatency().unsqueeze(dim=0))

        self.register_buffer('op_latencies', torch.cat(op_lat))

        op_imp = []
        for op in self._ops:
            op_imp.append(op.implementability().unsqueeze(dim=0))

        self.register_buffer('implementability', torch.cat(op_imp))

    def forward(self, x):
        # Softmax over alpha
        weights = F.softmax(self.alpha, dim=0)
        # Weighted sum of all candidate ops
        out = 0
        for w, op in zip(weights, self._ops):
            out = out + w * op(x)
        return out

    def getLatency(self):
        weights = F.softmax(self.alpha, dim=0)  # shape [num_ops]
        predicted_latency = torch.sum(weights * self.op_latencies)
        return predicted_latency

    def implementability(self):
        weights = F.softmax(self.alpha, dim=0)  # shape [num_ops]
        predicted_latency = torch.sum(weights * self.op_latencies)
        return predicted_latency

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


class TinyDARTSNetwork(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 32, 32)):
        super().__init__()

        # Define a single "cell" with MixedOp
        op_candidates = [Conv1x1, Conv1x1_ReLU, Conv3x3, Conv3x3_ReLU, Conv5x5, NoConv]
        self.mixed_op_1 = MixedOp(input_shape[0], 32, op_candidates)
        # self.mixed_op_2 = MixedOp(self.in_channels, self.out_channels, op_candidates)
        # self.mixed_op_3 = MixedOp(self.in_channels, self.out_channels, op_candidates)

        # A simple head for classification
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(128 * input_shape[1]//2 * input_shape[2]//2, 256)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.mixed_op_1(x)
        x = self.pool(x)
        # x = self.mixed_op_2(x)
        # x = self.pool(x)
        # x = self.mixed_op_3(x)
        # x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @property
    def arch_parameters(self):
        """Return a list of architecture parameters (alphas) to be optimized separately."""
        return [self.mixed_op_1.alpha]  #, self.mixed_op_2.alpha, self.mixed_op_3.alpha]


class VerySimpleDARTSNetwork(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 32, 32)):
        super().__init__()

        op_candidates_1 = [Conv1x1_ReLU, Conv3x3_ReLU,
                           Conv1x1_ReLU_BN, Conv3x3_ReLU_BN,
                           Conv1x1_ReLU_2_2, Conv3x3_ReLU_2_2,
                           Conv1x1_ReLU_4_4, Conv3x3_ReLU_4_4]
        self.mixed_op_1 = MixedOp(input_shape[0], 32, op_candidates_1)
        op_candidates_2 = [Conv1x1_ReLU, Conv3x3_ReLU,
                           Conv1x1_ReLU_BN, Conv3x3_ReLU_BN,
                           Conv1x1_ReLU_2_2, Conv3x3_ReLU_2_2,
                           Conv1x1_ReLU_4_4, Conv3x3_ReLU_4_4,
                           NoConv]
        self.mixed_op_2 = MixedOp(32, 32, op_candidates_2)

        op_candidates_3 = [Conv1x1_ReLU, Conv3x3_ReLU,
                           Conv1x1_ReLU_BN, Conv3x3_ReLU_BN,
                           Conv1x1_ReLU_2_2, Conv3x3_ReLU_2_2,
                           Conv1x1_ReLU_4_4, Conv3x3_ReLU_4_4]
        self.mixed_op_3 = MixedOp(32, 64, op_candidates_3)
        op_candidates_4 = [Conv1x1_ReLU, Conv3x3_ReLU,
                           Conv1x1_ReLU_BN, Conv3x3_ReLU_BN,
                           Conv1x1_ReLU_2_2, Conv3x3_ReLU_2_2,
                           Conv1x1_ReLU_4_4, Conv3x3_ReLU_4_4,
                           NoConv]
        self.mixed_op_4 = MixedOp(64, 64, op_candidates_4)

        op_candidates_5 = [Conv1x1_ReLU, Conv3x3_ReLU,
                           Conv1x1_ReLU_BN, Conv3x3_ReLU_BN,
                           Conv1x1_ReLU_2_2, Conv3x3_ReLU_2_2,
                           Conv1x1_ReLU_4_4, Conv3x3_ReLU_4_4]
        self.mixed_op_5 = MixedOp(64, 128, op_candidates_5)
        op_candidates_6 = [Conv1x1_ReLU, Conv3x3_ReLU,
                           Conv1x1_ReLU_BN, Conv3x3_ReLU_BN,
                           Conv1x1_ReLU_2_2, Conv3x3_ReLU_2_2,
                           Conv1x1_ReLU_4_4, Conv3x3_ReLU_4_4,
                           NoConv]
        self.mixed_op_6 = MixedOp(128, 128, op_candidates_6)

        self.fc1 = nn.Linear(128 * input_shape[1]//8 * input_shape[2]//8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.period = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        
    def forward(self, x):
        x = self.mixed_op_1(x)
        x = self.mixed_op_2(x)
        x = F.max_pool2d(x, 2)
        x = self.mixed_op_3(x)
        x = self.mixed_op_4(x)
        x = F.max_pool2d(x, 2)
        x = self.mixed_op_5(x)
        x = self.mixed_op_6(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    @property
    def getMixedOpsList(self):
        """Return a list of architecture parameters (alphas) to be optimized separately."""
        return [self.mixed_op_1, self.mixed_op_2, self.mixed_op_3,
                self.mixed_op_4, self.mixed_op_5, self.mixed_op_6]  # , self.mixed_op_3.alpha]

    def getLatency(self):
        """Return a list of architecture parameters (alphas) to be optimized separately."""
        return self.mixed_op_1.getLatency() + self.mixed_op_2.getLatency() + \
            self.mixed_op_3.getLatency() + self.mixed_op_4.getLatency() + \
            self.mixed_op_5.getLatency() + self.mixed_op_6.getLatency()


class SimpleCIFAR10Model(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 32, 32)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * input_shape[1]//8 * input_shape[2]//8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = F.dropout(x)
        x = self.fc3(x)
        return x


class VerySimpleModel(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 32, 32)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.fc1 = nn.Linear(32 * input_shape[1]//2 * input_shape[2]//2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

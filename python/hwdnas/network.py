import torch
import torch.nn as nn
import torch.nn.functional as F

import layers


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
    return torch.sum(torch.sigmoid(10.0*(M_data[:-1] - x[1:-1])/M_data[:-1]))


class VerySimpleDARTSNetwork(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 32, 32)):
        super().__init__()

        op_candidates_1 = [layers.Conv2D_3k_2b_HWNAS,
                           layers.Conv2D_3k_4b_HWNAS,
                           layers.Conv2D_3k_8b_HWNAS]
        self.mixed_op_1 = layers.MixedConv2D(input_shape[0], 32,
                                             input_shape[1], input_shape[2],
                                             op_candidates_1)

        op_candidates_2 = [layers.Conv2D_3k_2b_HWNAS,
                           layers.Conv2D_3k_4b_HWNAS,
                           layers.Conv2D_3k_8b_HWNAS]
        self.mixed_op_2 = layers.MixedConv2D(32, 32,
                                             input_shape[1], input_shape[2],
                                             op_candidates_2)

        op_candidates_3 = [layers.Conv2D_3k_2b_HWNAS,
                           layers.Conv2D_3k_4b_HWNAS,
                           layers.Conv2D_3k_8b_HWNAS]
        self.mixed_op_3 = layers.MixedConv2D(32, 64,
                                             input_shape[1]/2, input_shape[2]/2,
                                             op_candidates_3)

        op_candidates_4 = [layers.Conv2D_3k_2b_HWNAS,
                           layers.Conv2D_3k_4b_HWNAS,
                           layers.Conv2D_3k_8b_HWNAS]
        self.mixed_op_4 = layers.MixedConv2D(64, 64,
                                             input_shape[1]/2, input_shape[2]/2,
                                             op_candidates_4)

        op_candidates_5 = [layers.Conv2D_3k_2b_HWNAS,
                           layers.Conv2D_3k_4b_HWNAS,
                           layers.Conv2D_3k_8b_HWNAS]
        self.mixed_op_5 = layers.MixedConv2D(64, 128,
                                             input_shape[1]/4, input_shape[2]/4,
                                             op_candidates_5)

        op_candidates_6 = [layers.Conv2D_3k_2b_HWNAS,
                           layers.Conv2D_3k_4b_HWNAS,
                           layers.Conv2D_3k_8b_HWNAS]
        self.mixed_op_6 = layers.MixedConv2D(128, 128,
                                             input_shape[1]/4, input_shape[2]/4,
                                             op_candidates_6)

        self.fc1 = nn.Linear(128 * input_shape[1]//8 * input_shape[2]//8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        max_hardware = getTotalData()
        self.register_buffer('max_hardware', max_hardware)

        # self.period = nn.Parameter(torch.tensor(10.0), requires_grad=True)

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

    def getHardwareResults(self):
        used_hardware = self.mixed_op_1.estimate_params() + \
                        self.mixed_op_2.estimate_params() + \
                        self.mixed_op_3.estimate_params() + \
                        self.mixed_op_4.estimate_params() + \
                        self.mixed_op_5.estimate_params() + \
                        self.mixed_op_6.estimate_params()
        return used_hardware

    def getLatency(self):
        """Return a list of architecture parameters (alphas) to be optimized separately."""
        return self.getHardwareResults()[0]

    def getImplementability(self):
        used_hardware = self.getHardwareResults()
        implementability = isImplementable(self.max_hardware, used_hardware)
        return implementability


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

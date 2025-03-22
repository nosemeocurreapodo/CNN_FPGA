import torch
import torch.nn as nn
import torch.nn.functional as F

import layers


class VerySimpleDARTSNetwork(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 32, 32)):
        super().__init__()

        op_candidates_1 = [layers.Conv1x1_ReLU, layers.Conv3x3_ReLU,
                           layers.Conv1x1_ReLU_BN, layers.Conv3x3_ReLU_BN,
                           layers.Conv1x1_ReLU_2_2, layers.Conv3x3_ReLU_2_2,
                           layers.Conv1x1_ReLU_4_4, layers.Conv3x3_ReLU_4_4]
        self.mixed_op_1 = layers.MixedOp(input_shape[0], 32, op_candidates_1)
        op_candidates_2 = [layers.Conv1x1_ReLU, layers.Conv3x3_ReLU,
                           layers.Conv1x1_ReLU_BN, layers.Conv3x3_ReLU_BN,
                           layers.Conv1x1_ReLU_2_2, layers.Conv3x3_ReLU_2_2,
                           layers.Conv1x1_ReLU_4_4, layers.Conv3x3_ReLU_4_4,
                           layers.NoConv]
        self.mixed_op_2 = layers.MixedOp(32, 32, op_candidates_2)

        op_candidates_3 = [layers.Conv1x1_ReLU, layers.Conv3x3_ReLU,
                           layers.Conv1x1_ReLU_BN, layers.Conv3x3_ReLU_BN,
                           layers.Conv1x1_ReLU_2_2, layers.Conv3x3_ReLU_2_2,
                           layers.Conv1x1_ReLU_4_4, layers.Conv3x3_ReLU_4_4]
        self.mixed_op_3 = layers.MixedOp(32, 64, op_candidates_3)
        op_candidates_4 = [layers.Conv1x1_ReLU, layers.Conv3x3_ReLU,
                           layers.Conv1x1_ReLU_BN, layers.Conv3x3_ReLU_BN,
                           layers.Conv1x1_ReLU_2_2, layers.Conv3x3_ReLU_2_2,
                           layers.Conv1x1_ReLU_4_4, layers.Conv3x3_ReLU_4_4,
                           layers.NoConv]
        self.mixed_op_4 = layers.MixedOp(64, 64, op_candidates_4)

        op_candidates_5 = [layers.Conv1x1_ReLU, layers.Conv3x3_ReLU,
                           layers.Conv1x1_ReLU_BN, layers.Conv3x3_ReLU_BN,
                           layers.Conv1x1_ReLU_2_2, layers.Conv3x3_ReLU_2_2,
                           layers.Conv1x1_ReLU_4_4, layers.Conv3x3_ReLU_4_4]
        self.mixed_op_5 = layers.MixedOp(64, 128, op_candidates_5)
        op_candidates_6 = [layers.Conv1x1_ReLU, layers.Conv3x3_ReLU,
                           layers.Conv1x1_ReLU_BN, layers.Conv3x3_ReLU_BN,
                           layers.Conv1x1_ReLU_2_2, layers.Conv3x3_ReLU_2_2,
                           layers.Conv1x1_ReLU_4_4, layers.Conv3x3_ReLU_4_4,
                           layers.NoConv]
        self.mixed_op_6 = layers.MixedOp(128, 128, op_candidates_6)

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

    def implementability(self):
        """Return a list of architecture parameters (alphas) to be optimized separately."""
        return self.mixed_op_1.implementability() + self.mixed_op_2.implementability() + \
            self.mixed_op_3.implementability() + self.mixed_op_4.implementability() + \
            self.mixed_op_5.implementability() + self.mixed_op_6.implementability()


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

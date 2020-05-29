from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        zdim: int = 128,
        input_dim: int = 3,
        load_pretrained: bool = True,
        pretrained_path: Optional[str] = (
            "saves/pretrained_models/encoder/all.pt"
        ),
    ):
        super().__init__()
        self.load_pretrained = load_pretrained
        self.pretrained_path = pretrained_path

        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, zdim)

    def load_pretrained_model(self):
        if not self.load_pretrained:
            return

        weights = torch.load(self.pretrained_path)
        self.conv1.weight.data = weights["encoder.module.conv1.weight"]
        self.conv1.bias.data = weights["encoder.module.conv1.bias"]
        self.conv2.weight.data = weights["encoder.module.conv2.weight"]
        self.conv2.bias.data = weights["encoder.module.conv2.bias"]
        self.conv3.weight.data = weights["encoder.module.conv3.weight"]
        self.conv3.bias.data = weights["encoder.module.conv3.bias"]
        self.conv4.weight.data = weights["encoder.module.conv4.weight"]
        self.conv4.bias.data = weights["encoder.module.conv4.bias"]

        self.bn1.weight.data = weights["encoder.module.bn1.weight"]
        self.bn1.bias.data = weights["encoder.module.bn1.bias"]
        self.bn1.running_mean.data = weights["encoder.module.bn1.running_mean"]
        self.bn1.running_var.data = weights["encoder.module.bn1.running_var"]
        self.bn1.num_batches_tracked.data = weights[
            "encoder.module.bn1.num_batches_tracked"
        ]

        self.bn2.weight.data = weights["encoder.module.bn2.weight"]
        self.bn2.bias.data = weights["encoder.module.bn2.bias"]
        self.bn2.running_mean.data = weights["encoder.module.bn2.running_mean"]
        self.bn2.running_var.data = weights["encoder.module.bn2.running_var"]
        self.bn2.num_batches_tracked.data = weights[
            "encoder.module.bn2.num_batches_tracked"
        ]

        self.bn3.weight.data = weights["encoder.module.bn3.weight"]
        self.bn3.bias.data = weights["encoder.module.bn3.bias"]
        self.bn3.running_mean.data = weights["encoder.module.bn3.running_mean"]
        self.bn3.running_var.data = weights["encoder.module.bn3.running_var"]
        self.bn3.num_batches_tracked.data = weights[
            "encoder.module.bn3.num_batches_tracked"
        ]

        self.bn4.weight.data = weights["encoder.module.bn4.weight"]
        self.bn4.bias.data = weights["encoder.module.bn4.bias"]
        self.bn4.running_mean.data = weights["encoder.module.bn4.running_mean"]
        self.bn4.running_var.data = weights["encoder.module.bn4.running_var"]
        self.bn4.num_batches_tracked.data = weights[
            "encoder.module.bn4.num_batches_tracked"
        ]

        self.fc1.weight.data = weights["encoder.module.fc1.weight"]
        self.fc1.bias.data = weights["encoder.module.fc1.bias"]

        self.fc2.weight.data = weights["encoder.module.fc2.weight"]
        self.fc2.bias.data = weights["encoder.module.fc2.bias"]

        self.fc_bn1.weight.data = weights["encoder.module.fc_bn1.weight"]
        self.fc_bn1.bias.data = weights["encoder.module.fc_bn1.bias"]
        self.fc_bn1.running_mean.data = weights[
            "encoder.module.fc_bn1.running_mean"
        ]
        self.fc_bn1.running_var.data = weights[
            "encoder.module.fc_bn1.running_var"
        ]
        self.fc_bn1.num_batches_tracked.data = weights[
            "encoder.module.fc_bn1.num_batches_tracked"
        ]

        self.fc_bn2.weight.data = weights["encoder.module.fc_bn2.weight"]
        self.fc_bn2.bias.data = weights["encoder.module.fc_bn2.bias"]
        self.fc_bn2.running_mean.data = weights[
            "encoder.module.fc_bn2.running_mean"
        ]
        self.fc_bn2.running_var.data = weights[
            "encoder.module.fc_bn2.running_var"
        ]
        self.fc_bn2.num_batches_tracked.data = weights[
            "encoder.module.fc_bn2.num_batches_tracked"
        ]

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        ms = F.relu(self.fc_bn1(self.fc1(x)))
        ms = F.relu(self.fc_bn2(self.fc2(ms)))
        ms = self.fc3(ms)

        return ms

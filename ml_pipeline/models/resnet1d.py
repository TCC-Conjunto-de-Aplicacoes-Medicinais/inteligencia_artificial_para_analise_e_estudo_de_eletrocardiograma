import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    """
    Arquitetura Convolucional Unidimensional (1D-CNN) ResNet para séries temporais de ECG.
    Entrada: Tensor [Batch_Size, 1, 1000]
    Saída: Tensor de Logits [Batch_Size, 2] (Normal vs Patologia)
    """
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet1D, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, stride):
        block = ResNet1DBlock(self.in_planes, out_channels, stride)
        self.in_planes = out_channels
        return block

    def forward(self, x):
        # x shape: [batch, 1, 1000]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out

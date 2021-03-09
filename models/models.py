import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy

def conv3x3x3(in_channels, out_channels, stride=1):
	return nn.Conv3d(in_channels, out_channels, kernel_size=3,
					 stride=stride, padding=1, bias=False)


def conv1x3x3(in_channels, out_channels, stride=1):
	return nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
					 stride=stride, padding=(0, 1, 1), bias=False)


def conv1x3x2(in_channels, out_channels, stride=1):
	return nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 2),
					 stride=stride, padding=(0, 1, 1), bias=False)


def ConvBlock(in_channels, out_channels, k, s):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=1, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
		)

class humanFi(nn.Module):
    def __init__(self, csiChannel, numIds):
        super().__init__()
        self.h_0 = torch.nn.Parameter(torch.zeros((1,1,256)))
        self.c_0 = torch.nn.Parameter(torch.zeros((1,1,256)))
        self.inputLayer = nn.Linear(in_features=40*540, out_features=40*540, bias=False)
        self.lstm = nn.LSTM(input_size=40, hidden_size=256, num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=256*540, out_features=40, bias=False)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, s, t, c = x.shape
        x = x.reshape(b, t, c)
        x = x.permute(0, 2, 1)
        # x = x.reshape(b*s, h*w)
        # print(x.shape)
        # input layer
        # x = self.inputLayer(x)
        # LSTM + Dropout
        # if b == 1:
        #     h_0 = self.h_0[:, 0, :]
        #     c_0 = self.c_0[:, 0, :]
        # else:
        h_0 = self.h_0.expand(1, b, 256).contiguous()
        c_0 = self.c_0.expand(1, b, 256).contiguous()
        x, _ = self.lstm(x, (h_0, c_0))
        x = self.dropout(x)
        # print(h_0)
        # print(self.h_0)
        # FC
        # print(x.shape)
        x = x.reshape(b, 256*c)
        x = self.fc(x)
        # print(x.shape)
        # softmax
        x = F.softmax(x, dim=1)
        return x
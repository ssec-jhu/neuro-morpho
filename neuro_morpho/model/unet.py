# Adapted from:
# https://github.com/namdvt/skeletonization/blob/master/model/unet_att.py
# With the following license:
# MIT License

# Copyright (c) 2025 Nam Nguyen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import itertools
from typing import override

import numpy as np
import torch
import torch.nn as nn

import neuro_morpho.model.base as base


class UNet(base.BaseModel):
    def __init__(
        self,
        n_input_channels: int = 1,
        n_output_channels: int = 1,
        encoder_channels: list[int] = [64, 128, 256, 512, 1024],
        decoder_channels: list[int] = [512, 256, 128, 64],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(UNet, self).__init__()
        self.model = UNetModule(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        ).to(device)
        self.device = device

    @override
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model(torch.from_numpy(x).float()).squeeze(1).cpu().detach().numpy()


class UNetModule(nn.Module):
    def __init__(
        self,
        n_input_channels: int = 1,
        n_output_channels: int = 1,
        encoder_channels: list[int] = [64, 128, 256, 512, 1024],
        decoder_channels: list[int] = [512, 256, 128, 64],
    ):
        super(UNetModule, self).__init__()
        self.encoder = Encoder(in_channels=n_input_channels, channels=encoder_channels)
        self.decoder = Decoder(
            in_channels=encoder_channels[-1],
            channels=decoder_channels,
            out_channels=n_output_channels,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.encoder(x)
        return self.decoder(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], kernel_size: int = 3, padding: int = 1):
        super(Encoder, self).__init__()
        self._channels = channels
        channel_list = [in_channels] + channels

        for i, (in_ch, out_ch) in enumerate(itertools.pairwise(channel_list), start=1):
            setattr(self, f"conv{i}", DoubleConv2d(in_ch, out_ch, kernel_size, padding=padding))
            setattr(self, f"att{i}", AttentionGroup(out_ch))
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for i in range(1, len(self._channels) + 1):
            # apply pooling after the first set of conv/att operations
            if i > 1:
                x = self.pooling(x)
            x = getattr(self, f"conv{i}")(x)
            x = getattr(self, f"att{i}")(x)
            outs.append(x)

        return outs


class Decoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, channels: list[int], kernel_size: int = 3, padding: int = 1
    ):
        super(Decoder, self).__init__()
        self._channels = channels
        channel_list = [in_channels] + channels

        for i, (in_ch, out_ch) in enumerate(itertools.pairwise(channel_list), start=1):
            setattr(self, f"upconv{i}", UpConv2d(in_ch, out_ch, kernel_size=2, stride=2))
            setattr(self, f"conv{i}", DoubleConv2d(in_ch, out_ch, kernel_size, padding=padding))
            setattr(self, f"ca{i}", ChannelAttention(out_ch))
            setattr(self, f"sa{i}", SpatialAttention())
            setattr(
                self, f"out_conv_{i}", nn.Conv2d(out_ch, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        x, aux_inputs = x[-1], x[:-1]
        outs = []
        for i in range(1, len(self._channels) + 1):
            x = getattr(self, f"upconv{i}")(x)
            x = torch.cat([x, aux_inputs[-i]], dim=1)
            x = getattr(self, f"conv{i}")(x)
            x = getattr(self, f"ca{i}")(x) * x
            x = getattr(self, f"sa{i}")(x) * x
            outs.append(getattr(self, f"out_conv_{i}")(x))

        return outs[::-1]


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(UpConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DoubleConv2d, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGroup(nn.Module):
    def __init__(self, num_channels):
        super(AttentionGroup, self).__init__()
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2d(num_channels, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        s = torch.softmax(self.conv_1x1(x), dim=1)

        att = s[:, 0, :, :].unsqueeze(1) * x1 + s[:, 1, :, :].unsqueeze(1) * x2 + s[:, 2, :, :].unsqueeze(1) * x3

        return x + att


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class UnetAttention(nn.Module):
    def __init__(self):
        super(UnetAttention, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        out1, out2, out3, out4, x = self.encoder(x.float())
        x, aux_128, aux_64, aux_32 = self.decoder(out1, out2, out3, out4, x)

        return x.squeeze(), aux_128.squeeze(), aux_64.squeeze(), aux_32.squeeze()


if __name__ == "__main__":
    model = Encoder(in_channels=1, channels=[64, 128, 256, 512, 1024])
    x = torch.randn(1, 1, 512, 512)
    outs = model(x)
    print([y.shape for y in outs])
    model2 = Decoder(
        in_channels=outs[-1].shape[1],
        out_channels=1,
        channels=[512, 256, 128, 64],
    )
    outs2 = model2(outs)
    print([y.shape for y in outs2])

    model = UNetModule()
    x = torch.randn(1, 1, 512, 512)
    outs = model(x)
    print([y.shape for y in outs])

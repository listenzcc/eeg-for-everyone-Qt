from .Conv2dWithConstraint import Conv2dWithConstraint
from .LinearWithConstraint import LinearWithConstraint
import torch.nn.functional as F
import torch.nn as nn
import math
class EEGNetFramework(nn.Module):
    def __init__(self, freq, channel_num, time_point_num):
        super(EEGNetFramework, self).__init__()
        self.channel_num = channel_num  # C
        self.time_point_num = time_point_num  # T=128
        self.F1 = 8
        self.D = 2
        self.F2 = self.D * self.F1
        self.half_freq = freq // 2
        self.drop_out_rate = 0.5  # within subject
        self.class_num = 2

        padding_block_1 = [
            math.floor((self.half_freq - 1) / 2),
            math.ceil((self.half_freq - 1) / 2),
        ]
        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            # (1, C, T) -> (1, C, 采样率上取整+T+采样率下取整)
            nn.ZeroPad2d((padding_block_1[0], padding_block_1[1], 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=self.F1,  # num_filters
                kernel_size=(1, self.half_freq),  # filter size
                bias=False,  # TODO 确认原文关不关？
            ),  # output shape (F1, C, T)
            nn.BatchNorm2d(self.F1)  # output shape (F1, C, T)
        )

        # block 2 is implementations of Depthwise Convolution
        self.block_2 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=self.F1,  # input shape (F1, C, T)
                out_channels=self.D * self.F1,  # num_filters D * F1
                kernel_size=(self.channel_num, 1),  # filter size (C, 1)
                groups=self.F1,  # TODO: 分组卷积？需要和 F1 适配
                bias=False,
                max_norm=1,  # TODO 确认原文关不关？
            ),  # output shape (D*F1, 1, T)

            nn.BatchNorm2d(self.D * self.F1),  # output shape (D*F1, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (D*F1, 1, T//4)
            nn.Dropout(self.drop_out_rate)  # output shape (D*F1, 1, T//4)
        )

        # block 3 is implementations of Separable Convolution
        padding_block_3 = [
            math.floor((self.half_freq // 4 - 1) / 2),
            math.ceil((self.half_freq // 4 - 1) / 2),
        ]
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((padding_block_3[0], padding_block_3[1], 0, 0)),
            nn.Conv2d(
                in_channels=self.D * self.F1,  # input shape (self.D * self.F1, 1, T//4)
                out_channels=self.F2,  # num_filters
                kernel_size=(1, self.half_freq // 4),  # filter si 2Hz,这里要占到32Hz的一半
                groups=self.D * self.F1,
                bias=False
            ),  # output shape (F2, 1, T//4)
            nn.Conv2d(
                in_channels=self.F2,  # input shape (F2, 1, T//4)
                out_channels=self.F2,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(self.F2),  # output shape (F2, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (F2, 1, T//32)
            nn.Dropout(self.drop_out_rate),
            nn.Flatten(),
        )

        self.out = nn.Sequential(
            LinearWithConstraint(self.F2 * (self.time_point_num // 32), self.class_num, max_norm=0.25),
                                 )

    def forward(self, x):
        # (N, 1, T, C) -> (N, 1, C, T)
        # x = x.permute(0, 1, 3, 2)
        x = self.block_1(x)
        # print("block1", x.shape)
        x = self.block_2(x)
        # print("block2", x.shape)
        x = self.block_3(x)
        # print("block3", x.shape)
        # x = x.view(x.size(0), -1)
        logits = self.out(x)
        # 看loss
        probas = F.softmax(logits, dim=1)
        return logits, probas  # return x for visualization
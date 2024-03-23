## Adopted from https://github.com/rosinality/denoising-diffusion-pytorch with some minor changes.

import math
import random

import torch
from torch import nn
import torch.nn.functional as F


class Latent_AE_cnn_small(nn.Module):
    """定义一个名为 Latent_AE_cnn_small 的 PyTorch 模型类，继承自 nn.Module。
    __init__方法是构造函数，接收输入数据的维度 in_dim 和时间步长 time_step(默认为 1000) 作为参数。
    super().__init__() 调用基类 nn.Module 的构造函数。
    这个代码实现了一个小型的卷积自编码器模型，用于将输入数据编码为潜在表示，并从潜在表示重构出原始数据。
    模型包含四个编码层和四个解码层，每层由卷积、归一化和激活函数等操作组成。
    在编码和解码过程中，还添加了一些噪声来增强模型的鲁棒性。此外，代码还提供了 encode 和 decode 方法，
    用于分别执行编码和解码过程。
    
    """
    def __init__(
            self,
            in_dim,
            time_step=1000,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        
        # 初始化一些模型参数，包括输入维度 in_dim,
        # 下采样率 fold_rate, 卷积核大小 kernal_size,
        # 编码器各层通道数 channel_list 和解码器各层通道数 channel_list_dec。
        # 同时计算实际输入维度 real_input_dim, 以确保输入数据的长度可被 fold_rate 的 4 次方整除。
        self.in_dim = in_dim
        self.fold_rate = 3
        self.kernal_size = 3
        self.channel_list = [2, 2, 2, 2]
        self.channel_list_dec = [8, 64, 64, 2]
        self.real_input_dim = (
                int(in_dim / self.fold_rate ** 4 + 1) * self.fold_rate ** 4
        )
        ### 4 layers convolutional encoder
        # 定义编码器的四个层，每层由 nn.Sequential 模块组成，包含 InstanceNorm1d、Conv1d、LeakyReLU
        # 和最大池化操作。编码器的作用是将输入数据编码为潜在表示。
        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(1, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=0),
            nn.Tanh(),
        )
        
        
        ### decoder
        # 定义解码器的四个层，每层由 nn.Sequential 模块组成，包含 LeakyReLU、InstanceNorm1d、ConvTranspose1d
        # 和 Conv1d 操作。解码器的作用是将潜在表示解码为输出数据。
        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.ConvTranspose1d(
                self.channel_list_dec[3], self.channel_list_dec[3], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list_dec[3], self.channel_list_dec[2], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.ConvTranspose1d(
                self.channel_list_dec[2], self.channel_list_dec[2], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list_dec[2], self.channel_list_dec[1], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.ConvTranspose1d(
                self.channel_list_dec[1], self.channel_list_dec[1], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list_dec[1], self.channel_list_dec[0], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.ConvTranspose1d(
                self.channel_list_dec[0], self.channel_list_dec[0], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list_dec[0], 1, self.kernal_size, stride=1, padding=self.fold_rate),
        )

        # self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input):
        ## forward 方法定义了模型的前向传播过程。首先对输入数据添加一些小噪声，并检查输入维度是否为 2D,
        # 如果不是则将其调整为 3D。接着，将输入数据与填充的零张量连接，以匹配 real_input_dim。
        # 然后，依次通过编码器的四个层编码输入数据，得到潜在表示 emb_enc4。
        # 在 emb_enc4 上添加噪声并进行裁剪。最后，通过解码器的四个层解码 emb_enc4,
        # 得到输出数据 emb_dec4, 并将其 reshape 为原始形状返回
        input += torch.randn(input.shape).to(input.device) * 0.001
        input_shape = input.shape
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        noise_factor = 0.5
        emb_enc4 = emb_enc4 + torch.randn_like(emb_enc4) * noise_factor
        emb_enc4 = torch.clamp(emb_enc4, -1, 1)

        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :input_shape[-1]]

        return emb_dec4.reshape(input_shape)

    def encode(self, input):
    ### encode 方法用于编码输入数据，返回潜在表示 emb_enc4。预处理输入数据的步骤与 forward 方法相同。
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(input.device),
            ],
            dim=2,
        )
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        return emb_enc4

    def decode(self, emb_enc4):
    ### decode 方法用于解码潜在表示 emb_enc4, 返回输出数据。解码过程通过解码器的四个层完成。
        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :self.in_dim]

        return emb_dec4

# ae1
class Latent_AE_cnn(nn.Module):
    def __init__(
            self,
            in_dim,
            time_step=1000,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 5
        self.kernal_size = 5
        self.channel_list = [4, 4, 4, 4]
        self.channel_list_dec = [8, 256, 256, 4]
        print(self.fold_rate)
        print(self.kernal_size)
        print(self.channel_list)
        print(self.channel_list_dec)
        self.real_input_dim = (
                int(in_dim / self.fold_rate ** 4 + 1) * self.fold_rate ** 4
        )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(1, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=0),
            nn.Tanh(),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.ConvTranspose1d(
                self.channel_list_dec[3], self.channel_list_dec[3], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list_dec[3], self.channel_list_dec[2], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.ConvTranspose1d(
                self.channel_list_dec[2], self.channel_list_dec[2], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list_dec[2], self.channel_list_dec[1], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.ConvTranspose1d(
                self.channel_list_dec[1], self.channel_list_dec[1], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list_dec[1], self.channel_list_dec[0], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.ConvTranspose1d(
                self.channel_list_dec[0], self.channel_list_dec[0], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list_dec[0], 1, self.kernal_size, stride=1, padding=self.fold_rate),
        )

        # self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input):
        input_shape = input.shape
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        emb_enc4 = emb_enc4 + torch.randn_like(emb_enc4) * 0.1

        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :input_shape[-1]]

        return emb_dec4.reshape(input_shape)

    def Enc(self, input):
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(input.device),
            ],
            dim=2,
        )
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        return emb_enc4

    def Dec(self, emb_enc4):
        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :self.in_dim]

        return emb_dec4


class Latent_AE_cnn2(nn.Module):
    def __init__(
            self,
            in_dim,
            time_step=1000,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 2
        self.kernal_size = 2
        self.channel_list = [4, 4, 4, 4]
        self.channel_list_dec = [8, 64, 64, 4]
        print(self.fold_rate)
        print(self.kernal_size)
        print(self.channel_list)
        print(self.channel_list_dec)
        self.real_input_dim = (
                int(in_dim / self.fold_rate ** 4 + 1) * self.fold_rate ** 4
        )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(1, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=0),
            nn.Tanh(),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.ConvTranspose1d(
                self.channel_list_dec[3], self.channel_list_dec[3], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list_dec[3], self.channel_list_dec[2], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.ConvTranspose1d(
                self.channel_list_dec[2], self.channel_list_dec[2], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list_dec[2], self.channel_list_dec[1], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.ConvTranspose1d(
                self.channel_list_dec[1], self.channel_list_dec[1], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list_dec[1], self.channel_list_dec[0], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.ConvTranspose1d(
                self.channel_list_dec[0], self.channel_list_dec[0], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list_dec[0], 1, self.kernal_size, stride=1, padding=self.fold_rate),
        )

        # self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input):
        input_shape = input.shape
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        emb_enc4 = emb_enc4 + torch.randn_like(emb_enc4) * 0.1

        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :input_shape[-1]]

        return emb_dec4.reshape(input_shape)

    def Enc(self, input):
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(input.device),
            ],
            dim=2,
        )
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        return emb_enc4

    def Dec(self, emb_enc4):
        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :self.in_dim]

        return emb_dec4


class Latent_AE_cnn3(nn.Module):
    def __init__(
            self,
            in_dim,
            time_step=1000,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 3
        self.kernal_size = 3
        self.channel_list = [4, 4, 4, 4]
        self.channel_list_dec = [8, 256, 256, 4]
        print(self.fold_rate)
        print(self.kernal_size)
        print(self.channel_list)
        print(self.channel_list_dec)
        self.real_input_dim = (
                int(in_dim / self.fold_rate ** 4 + 1) * self.fold_rate ** 4
        )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(1, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=0),
            nn.Tanh(),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.ConvTranspose1d(
                self.channel_list_dec[3], self.channel_list_dec[3], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list_dec[3], self.channel_list_dec[2], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.ConvTranspose1d(
                self.channel_list_dec[2], self.channel_list_dec[2], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list_dec[2], self.channel_list_dec[1], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.ConvTranspose1d(
                self.channel_list_dec[1], self.channel_list_dec[1], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list_dec[1], self.channel_list_dec[0], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.ConvTranspose1d(
                self.channel_list_dec[0], self.channel_list_dec[0], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list_dec[0], 1, self.kernal_size, stride=1, padding=self.fold_rate),
        )

        # self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input):
        input_shape = input.shape
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        emb_enc4 = emb_enc4 + torch.randn_like(emb_enc4) * 0.1

        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :input_shape[-1]]

        return emb_dec4.reshape(input_shape)

    def Enc(self, input):
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(input.device),
            ],
            dim=2,
        )
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        return emb_enc4

    def Dec(self, emb_enc4):
        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :self.in_dim]

        return emb_dec4


class Latent_AE_cnn_big(nn.Module):
    def __init__(
            self,
            in_dim,
            time_step=1000,
            channel=6,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 4
        self.kernal_size = 4
        self.channel_list = [channel, channel, channel, channel]
        self.channel_list_dec = [8, 256, 256, channel]
        self.real_input_dim = (
                int(in_dim / self.fold_rate ** 4 + 1) * self.fold_rate ** 4
        )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(1, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=0),
            nn.Tanh(),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.ConvTranspose1d(
                self.channel_list_dec[3], self.channel_list_dec[3], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list_dec[3], self.channel_list_dec[2], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.ConvTranspose1d(
                self.channel_list_dec[2], self.channel_list_dec[2], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list_dec[2], self.channel_list_dec[1], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.ConvTranspose1d(
                self.channel_list_dec[1], self.channel_list_dec[1], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list_dec[1], self.channel_list_dec[0], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.ConvTranspose1d(
                self.channel_list_dec[0], self.channel_list_dec[0], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list_dec[0], 1, self.kernal_size, stride=1, padding=self.fold_rate),
        )

        # self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input):
        input_shape = input.shape
        # import pdb;pdb.set_trace()
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        emb_enc4 = emb_enc4 + torch.randn_like(emb_enc4) * 0.1

        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :input_shape[-1]]

        return emb_dec4.reshape(input_shape)

    def Enc(self, input):
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(input.device),
            ],
            dim=2,
        )
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        return emb_enc4

    def Dec(self, emb_enc4):
        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :self.in_dim]

        return emb_dec4


class Latent_AE_cnn_test(nn.Module):
    def __init__(
            self,
            in_dim,
            time_step=1000,
            channel=6,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 20
        self.kernal_size = 20
        self.channel_list = [channel, channel, channel, channel]
        self.channel_list_dec = [1, 1, 1, channel]
        self.real_input_dim = (
                int(in_dim / self.fold_rate ** 4 + 1) * self.fold_rate ** 4
        )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(1, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=0),
            nn.Tanh(),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.ConvTranspose1d(
                self.channel_list_dec[3], self.channel_list_dec[3], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list_dec[3], self.channel_list_dec[2], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.ConvTranspose1d(
                self.channel_list_dec[2], self.channel_list_dec[2], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list_dec[2], self.channel_list_dec[1], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.ConvTranspose1d(
                self.channel_list_dec[1], self.channel_list_dec[1], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list_dec[1], self.channel_list_dec[0], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.ConvTranspose1d(
                self.channel_list_dec[0], self.channel_list_dec[0], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list_dec[0], 1, self.kernal_size, stride=1, padding=self.fold_rate),
        )

        # self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input):
        input_shape = input.shape
        # import pdb;pdb.set_trace()
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        emb_enc4 = emb_enc4 + torch.randn_like(emb_enc4) * 0.1

        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :input_shape[-1]]

        return emb_dec4.reshape(input_shape)

    def Enc(self, input):
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(input.device),
            ],
            dim=2,
        )
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        return emb_enc4

    def Dec(self, emb_enc4):
        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :self.in_dim]

        return emb_dec4


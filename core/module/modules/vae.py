from typing import List, Any
from torch import nn, Tensor
from abc import abstractmethod
import torch
import torch.nn.functional as F

### 这段代码实现了一个基于 PyTorch 的 vanilla VAE (Variational Autoencoder, 变分自编码器) 模型，
# 主要包含一个基类 BaseVAE 和一个具体实现 VanillaVAE。


class BaseVAE(nn.Module):
### 定义一个名为 BaseVAE 的基类，继承自 nn.Module。这个基类定义了 VAE 模型应该具备的方法，
# 包括 encode、decode、sample、generate、forward 和 loss_function。其中 forward 和
# loss_function 被标记为抽象方法，必须在子类中实现。

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):
### 定义一个名为 VanillaVAE 的子类，继承自 BaseVAE。
# 这是一个具体的 VAE 实现。构造函数接收输入通道数 in_channels、潜在空间维度 latent_dim、
# 编码器 / 解码器隐藏层维度列表 hidden_dims 等参数。如果 hidden_dims 未提供，
# 则使用默认值 [32, 64, 128, 256, 512]。

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        ### 构建编码器网络。使用一系列 nn.Sequential 模块，每个模块包含一个 Conv1d 层 (用于特征提取)、
        # 一个 BatchNorm1d 层 (用于归一化) 和一个 LeakyReLU 激活层。
        # 编码器的输出通过两个全连接层 fc_mu 和 fc_var 分别得到潜在变量的均值 mu 和方差 var。
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []
        ### 构建解码器网络。首先使用一个全连接层 decoder_input 将潜在变量映射到一个高维空间。
        # 然后使用一系列 nn.Sequential 模块，每个模块包含一个 ConvTranspose1d 层 (用于上采样)、
        # 一个 BatchNorm1d 层和一个 LeakyReLU 激活层。最后使用一个 nn.Sequential 模块
        # final_layer 将特征映射回原始输入空间，包括一个 ConvTranspose1d 层、一个 BatchNorm1d 层、
        # 一个 LeakyReLU 激活层、一个 Conv1d 层和一个 Tanh 激活层。

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose1d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm1d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        ### encode 方法将输入数据通过编码器网络，返回潜在变量的均值 mu 和对数方差 log_var。
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        ### decode 方法将给定的潜在变量 z 通过解码器网络，返回重构的输入数据。
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        ### reparameterize 方法实现了重参数技巧 (reparameterization trick),
        # 用于从标准高斯分布 N(0,1) 中采样，得到潜在变量 z 的样本。
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        ### forward 方法定义了模型的前向传播过程。它首先通过 encode 方法获得潜在变量的均值 mu 和对数方差 log_var,
        # 然后使用 reparameterize 方法从高斯分布中采样得到 z。最后，通过 decode 方法将 z
        # 解码为重构的输入，并返回重构的输入、原始输入、mu 和 log_var。
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        ### loss_function 方法计算 VAE 的损失函数，包括重构损失 (reconstruction loss)
        # 和 KL 散度损失 (KL divergence loss)。重构损失使用均方误差损失函数 F.mse_loss
        # 计算重构输入与原始输入之间的差异。KL 散度损失根据 KL 散度公式计算，
        # 衡量编码得到的潜在变量分布与标准高斯分布之间的差异。最终损失是两者的加权和。
        # 该方法还返回了各项损失的值，方便监控和调试。
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        ### sample 方法从潜在空间中采样一些潜在变量 z, 并通过 decode 方法将它们映射回输入空间，返回生成的样本。
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        ### generate 方法将给定的输入 x 通过模型的 forward 方法，返回重构的输入。
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
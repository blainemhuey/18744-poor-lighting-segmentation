import torch
import torch.nn as nn

from mfnet.model.MFNet import ConvBnLeakyRelu2d, MiniInception


class MFNetNoIR(nn.Module):
    """
    Rewritten from scratch implementation of MFNet for more configurable hyperparameters
    """

    DEFAULT_RGB_CH_SIZE = (16, 48, 48, 96, 96)

    def __init__(self,
                 rgb_ch: tuple = DEFAULT_RGB_CH_SIZE,
                 n_class: int = 10,
                 **kwargs):
        """
        Constructor for Modified MFNet Module
        :param rgb_ch: Number of Convolutional Channels in each of the 5 RGB Encoder Layers
        :param n_class: Number of Output Classes
        :param kwargs: Additional parameters to be passed on to inner modules
        """
        super(MFNetNoIR, self).__init__()

        assert (len(rgb_ch) == 5)

        # Create Sequential for RGB (Visual) Encoder
        self.rgb_encoder = nn.ModuleList([
            nn.Sequential(
                ConvBnLeakyRelu2d(3, rgb_ch[0], **kwargs),
                nn.MaxPool2d(2, stride=2),
                ConvBnLeakyRelu2d(rgb_ch[0], rgb_ch[1], **kwargs),
                ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[1], **kwargs),
            ),
            nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[2], **kwargs),
                ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[2], **kwargs),
            ),
            nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                MiniInception(rgb_ch[2], rgb_ch[3]),
            ),
            nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                MiniInception(rgb_ch[3], rgb_ch[4]),
            )
        ])

        # Create Sequential for Combined Decoder
        self.upsampler = nn.Upsample(scale_factor=2, **kwargs)
        self.decoder = nn.ModuleList([
            ConvBnLeakyRelu2d(rgb_ch[i], rgb_ch[i - 1] if i > 0 else n_class)
            for i in reversed(range(4))
        ])

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        PyTorch Module Functor for Modified MFNet
        :param x: tensor of shape (batch, 4, height, width)
        :return: tensor of shape (batch, n_class, height, width)
        """

        # Run data through each layer of each encoder
        x_rgb = x[:, :3]
        ys = []
        for rgb_l in self.rgb_encoder:
            x_rgb = rgb_l.forward(x_rgb)
            ys.append(x_rgb)

        # Run concatenated output through decoder
        z = ys[-1]
        for i, layer in zip(reversed(range(len(ys))), self.decoder):
            z = self.upsampler(z)
            z = layer.forward(z if i == 0 else z + ys[i - 1])
        return z

import torch
from torch import Tensor
from torch import nn
from typing import Optional, Tuple, Any
import torchvision

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any
                 ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            # ch3x3red: int,
            ch3x3: int,
            # ch5x5red: int,
            ch5x5: int,
            pool_proj: int,
    ) -> None:
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.branch2 = nn.Sequential(
            # BasicConv2d(in_channels, ch3x3red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            # BasicConv2d(ch3x3red, ch3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(in_channels, ch3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            # BasicConv2d(in_channels, ch5x5red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            # BasicConv2d(ch5x5red, ch5x5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(in_channels, ch5x5, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = [branch1, branch2, branch3, branch4]

        out = torch.cat(out, 1)

        return out
    
class MSPoolAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        pools = [3,7,11]
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0]//2, count_include_pad=False)
        self.pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1]//2, count_include_pad=False)
        self.pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2]//2, count_include_pad=False)
        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        x_in = self.conv0(x)
        x_1 = self.pool1(x_in)
        x_2 = self.pool2(x_in)
        x_3 = self.pool3(x_in)
        x_out = self.sigmoid(self.conv4(x_in + x_1 + x_2 + x_3)) * u
        return x_out + u
    

class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: torch.nn.Module = torch.nn.ReLU,
        scale_activation: torch.nn.Module = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input
    
if __name__ == "__main__":
    inception = nn.Sequential(Inception(3, 64, 128, 32, 32),
                SqueezeExcitation(256, 64))
    # inception = nn.Sequential(MSPoolAttention(3, 64, 128, 32, 32),
    #             SqueezeExcitation(256, 64))
    inputs = torch.rand(2, 3, 512, 1024)
    outputs = inception(inputs)
    print(outputs.shape)

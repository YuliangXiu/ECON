""" The code is based on https://github.com/apple/ml-gsn/ with adaption. """

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.torch_utils.ops.native_ops import (
    FusedLeakyReLU,
    fused_leaky_relu,
    upfirdn2d,
)


class DiscriminatorHead(nn.Module):
    def __init__(self, in_channel, disc_stddev=False):
        super().__init__()

        self.disc_stddev = disc_stddev
        stddev_dim = 1 if disc_stddev else 0

        self.conv_stddev = ConvLayer2d(
            in_channel=in_channel + stddev_dim,
            out_channel=in_channel,
            kernel_size=3,
            activate=True
        )

        self.final_linear = nn.Sequential(
            nn.Flatten(),
            EqualLinear(in_channel=in_channel * 4 * 4, out_channel=in_channel, activate=True),
            EqualLinear(in_channel=in_channel, out_channel=1),
        )

    def cat_stddev(self, x, stddev_group=4, stddev_feat=1):
        perm = torch.randperm(len(x))
        inv_perm = torch.argsort(perm)

        batch, channel, height, width = x.shape
        x = x[perm
             ]    # shuffle inputs so that all views in a single trajectory don't get put together

        group = min(batch, stddev_group)
        stddev = x.view(group, -1, stddev_feat, channel // stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)

        stddev = stddev[inv_perm]    # reorder inputs
        x = x[inv_perm]

        out = torch.cat([x, stddev], 1)
        return out

    def forward(self, x):
        if self.disc_stddev:
            x = self.cat_stddev(x)
        x = self.conv_stddev(x)
        out = self.final_linear(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, in_res, out_res):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(out_res, 2))

        self.layers = []
        in_ch = in_channel
        for i in range(log_size_in, log_size_out):
            out_ch = in_ch // 2
            self.layers.append(
                ConvLayer2d(
                    in_channel=in_ch,
                    out_channel=out_ch,
                    kernel_size=3,
                    upsample=True,
                    bias=True,
                    activate=True
                )
            )
            in_ch = out_ch

        self.layers.append(
            ConvLayer2d(
                in_channel=in_ch, out_channel=out_channel, kernel_size=3, bias=True, activate=False
            )
        )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class StyleDiscriminator(nn.Module):
    def __init__(self, in_channel, in_res, ch_mul=64, ch_max=512, **kwargs):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(4, 2))

        self.conv_in = ConvLayer2d(in_channel=in_channel, out_channel=ch_mul, kernel_size=3)

        # each resblock will half the resolution and double the number of features (until a maximum of ch_max)
        self.layers = []
        in_channels = ch_mul
        for i in range(log_size_in, log_size_out, -1):
            out_channels = int(min(in_channels * 2, ch_max))
            self.layers.append(
                ConvResBlock2d(in_channel=in_channels, out_channel=out_channels, downsample=True)
            )
            in_channels = out_channels
        self.layers = nn.Sequential(*self.layers)

        self.disc_out = DiscriminatorHead(in_channel=in_channels, disc_stddev=True)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layers(x)
        out = self.disc_out(x)
        return out


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    """Blur layer.

    Applies a blur kernel to input image using finite impulse response filter. Blurring feature maps after
    convolutional upsampling or before convolutional downsampling helps produces models that are more robust to
    shifting inputs (https://richzhang.github.io/antialiased-cnns/). In the context of GANs, this can provide
    cleaner gradients, and therefore more stable training.

    Args:
    ----
    kernel: list, int
        A list of integers representing a blur kernel. For exmaple: [1, 3, 3, 1].
    pad: tuple, int
        A tuple of integers representing the number of rows/columns of padding to be added to the top/left and
        the bottom/right respectively.
    upsample_factor: int
        Upsample factor.

    """
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class Upsample(nn.Module):
    """Upsampling layer.

    Perform upsampling using a blur kernel.

    Args:
    ----
    kernel: list, int
        A list of integers representing a blur kernel. For exmaple: [1, 3, 3, 1].
    factor: int
        Upsampling factor.

    """
    def __init__(self, kernel=[1, 3, 3, 1], factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class Downsample(nn.Module):
    """Downsampling layer.

    Perform downsampling using a blur kernel.

    Args:
    ----
    kernel: list, int
        A list of integers representing a blur kernel. For exmaple: [1, 3, 3, 1].
    factor: int
        Downsampling factor.

    """
    def __init__(self, kernel=[1, 3, 3, 1], factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate.

    During the forward pass the weights are scaled by the inverse of the He constant (i.e. sqrt(in_dim)) to
    prevent vanishing gradients and accelerate training. This constant only works for ReLU or LeakyReLU
    activation functions.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    bias: bool
        Use bias term.
    bias_init: float
        Initial value for the bias.
    lr_mul: float
        Learning rate multiplier. By scaling weights and the bias we can proportionally scale the magnitude of
        the gradients, effectively increasing/decreasing the learning rate for this layer.
    activate: bool
        Apply leakyReLU activation.

    """
    def __init__(self, in_channel, out_channel, bias=True, bias_init=0, lr_mul=1, activate=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))
        else:
            self.bias = None

        self.activate = activate
        self.scale = (1 / math.sqrt(in_channel)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activate:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"


class EqualConv2d(nn.Module):
    """2D convolution layer with equalized learning rate.

    During the forward pass the weights are scaled by the inverse of the He constant (i.e. sqrt(in_dim)) to
    prevent vanishing gradients and accelerate training. This constant only works for ReLU or LeakyReLU
    activation functions.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    kernel_size: int
        Kernel size.
    stride: int
        Stride of convolutional kernel across the input.
    padding: int
        Amount of zero padding applied to both sides of the input.
    bias: bool
        Use bias term.

    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding
        )
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualConvTranspose2d(nn.Module):
    """2D transpose convolution layer with equalized learning rate.

    During the forward pass the weights are scaled by the inverse of the He constant (i.e. sqrt(in_dim)) to
    prevent vanishing gradients and accelerate training. This constant only works for ReLU or LeakyReLU
    activation functions.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    kernel_size: int
        Kernel size.
    stride: int
        Stride of convolutional kernel across the input.
    padding: int
        Amount of zero padding applied to both sides of the input.
    output_padding: int
        Extra padding added to input to achieve the desired output size.
    bias: bool
        Use bias term.

    """
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_channel, out_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class ConvLayer2d(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        assert not (upsample and downsample), 'Cannot upsample and downsample simultaneously'
        layers = []

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate
                )
            )
            layers.append(Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor))

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate
                )
            )

        if (not downsample) and (not upsample):
            padding = kernel_size // 2

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=1,
                    bias=bias and not activate
                )
            )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ConvResBlock2d(nn.Module):
    """2D convolutional residual block with equalized learning rate.

    Residual block composed of 3x3 convolutions and leaky ReLUs.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    upsample: bool
        Apply upsampling via strided convolution in the first conv.
    downsample: bool
        Apply downsampling via strided convolution in the second conv.

    """
    def __init__(self, in_channel, out_channel, upsample=False, downsample=False):
        super().__init__()

        assert not (upsample and downsample), 'Cannot upsample and downsample simultaneously'
        mid_ch = in_channel if downsample else out_channel

        self.conv1 = ConvLayer2d(in_channel, mid_ch, upsample=upsample, kernel_size=3)
        self.conv2 = ConvLayer2d(mid_ch, out_channel, downsample=downsample, kernel_size=3)

        if (in_channel != out_channel) or upsample or downsample:
            self.skip = ConvLayer2d(
                in_channel,
                out_channel,
                upsample=upsample,
                downsample=downsample,
                kernel_size=1,
                activate=False,
                bias=False,
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if hasattr(self, 'skip'):
            skip = self.skip(input)
            out = (out + skip) / math.sqrt(2)
        else:
            out = (out + input) / math.sqrt(2)
        return out

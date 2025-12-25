import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

from ultralytics.nn.modules.block import C2f, C3, Bottleneck
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# Wavelet Transform Conv(WTConv2d)
# class WTConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
#         super(WTConv2d, self).__init__()
#
#         assert in_channels == out_channels
#
#         self.in_channels = in_channels
#         self.wt_levels = wt_levels
#         self.stride = stride
#         self.dilation = 1
#
#         self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
#         self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
#         self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
#
#         self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
#         self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
#
#         self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
#                                    groups=in_channels, bias=bias)
#         self.base_scale = _ScaleModule([1, in_channels, 1, 1])
#
#         self.wavelet_convs = nn.ModuleList(
#             [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
#                        groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
#         )
#         self.wavelet_scale = nn.ModuleList(
#             [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
#         )
#
#         if self.stride > 1:
#             self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
#             self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
#                                                    groups=in_channels)
#         else:
#             self.do_stride = None
#
#     def forward(self, x):
#
#         x_ll_in_levels = []
#         x_h_in_levels = []
#         shapes_in_levels = []
#
#         curr_x_ll = x
#
#         for i in range(self.wt_levels):
#             curr_shape = curr_x_ll.shape
#             shapes_in_levels.append(curr_shape)
#             if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
#                 curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
#                 curr_x_ll = F.pad(curr_x_ll, curr_pads)
#
#             curr_x = self.wt_function(curr_x_ll)
#             curr_x_ll = curr_x[:, :, 0, :, :]
#
#             shape_x = curr_x.shape
#             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
#             curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
#             curr_x_tag = curr_x_tag.reshape(shape_x)
#
#             x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
#             x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
#
#         next_x_ll = 0
#
#         for i in range(self.wt_levels - 1, -1, -1):
#             curr_x_ll = x_ll_in_levels.pop()
#             curr_x_h = x_h_in_levels.pop()
#             curr_shape = shapes_in_levels.pop()
#
#             curr_x_ll = curr_x_ll + next_x_ll
#
#             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
#             next_x_ll = self.iwt_function(curr_x)
#
#             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
#
#
#         x_tag = next_x_ll
#         assert len(x_ll_in_levels) == 0
#
#         x = self.base_scale(self.base_conv(x))
#         x = x + x_tag
#
#         if self.do_stride is not None:
#             x = self.do_stride(x)
#
#         return x

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        # Wavelet transform to obtain low and high-frequency components
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]  # Low-frequency part

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])  # Low-frequency part
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])  # High-frequency parts

        next_x_ll = 0
        high_freq_fused = 0

        # Fusion of high-frequency features with upsampling
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            # Add the previous level's low-frequency part
            curr_x_ll = curr_x_ll + next_x_ll

            # Step 1: 将张量在第 2 维（大小为4）上分割成3部分
            split_x_h = torch.split(curr_x_h, 1,
                                    dim=2)  # Split along the 3rd dimension (4), each part will have shape (1, 32, 1, 8, 8)

            # Step 2: 调整每个分割后的张量的形状
            # 每个分割部分的形状是 (1, 32, 1, 8, 8)，我们需要通过 squeeze 去掉大小为1的维度
            split_x_h = [x.squeeze(2) for x in split_x_h]  # 每个张量的形状现在为 (1, 32, 8, 8)

            # Step 3: 给每一部分分配权重
            weights = [0.3333, 0.3333, 0.3334]  # 权重之和等于 1

            # Step 4: 对每个分割部分应用权重，并相加
            x_h_fused = sum(weight * x for weight, x in zip(weights, split_x_h))
            # Upsample high-frequency features to match the current low-frequency shape
            high_freq_fused_level = F.interpolate(x_h_fused, size=curr_shape[2:], mode='nearest')
            high_freq_fused += high_freq_fused_level

            # Concatenate the low-frequency part with high-frequency features
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

            # Inverse wavelet transform
            next_x_ll = self.iwt_function(curr_x)

            # Ensure the output shape matches the current shape
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # Final output by adding low-frequency part and fused high-frequency features
        x_tag = next_x_ll + high_freq_fused
        assert len(x_ll_in_levels) == 0

        # Apply base convolution
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag  # Add the final fusion of high and low-frequency parts

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x




class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class DSConvWithWT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DSConvWithWT, self).__init__()

        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)

        # 逐点卷积：使用 1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Bottleneck_WT(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = WTConv2d(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_WT(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_WT(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# 在c3k=True时，使用Bottleneck_WT特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_WT(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_WT(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

def preprocess_image(image_path):
    # 加载图片
    image = Image.open(image_path).convert('RGB')

    # 定义预处理步骤
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图片大小为模型输入所需尺寸
        transforms.ToTensor(),  # 转换为张量并归一化到[0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
    ])

    # 应用预处理
    input_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    return input_tensor


# if __name__ == '__main__':
#     # 初始化模型
#     DW = DSConvWithWT(256, 128)
#
#     # 设置模型为评估模式
#     DW.eval()
#
#     # 图像路径
#     image_path = r"C:\Users\SDS\Desktop\Scientific research\code\ultralytics-main-pri\ultralytics\data\datasets\baseline+PBANet\valid\images\fracture-5.jpg"
#
#     # 加载并预处理图像
#     image = Image.open(image_path).convert("RGB")  # 确保是RGB图像
#     preprocess = transforms.Compose([
#         transforms.Resize((64, 64)),  # 如果需要调整图像大小
#         transforms.ToTensor(),  # 转换为Tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
#     ])
#     input_tensor = preprocess(image).unsqueeze(0)  # 添加batch维度 (1, 3, 64, 64)
#
#     # 确保模型输入张量的形状是正确的
#     print(f"Input tensor shape: {input_tensor.shape}")
#
#     # 运行模型
#     with torch.no_grad():  # 不需要计算梯度
#         output_tensor = DW(input_tensor)
#
#     # 将输出张量转换回图像
#     output_image = output_tensor.squeeze(0).cpu().numpy()  # 去掉batch维度并转为NumPy数组
#     output_image = np.transpose(output_image, (1, 2, 0))  # 将通道维度移到最后
#     output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)  # 还原像素值
#
#     # 保存图像
#     output_image_pil = Image.fromarray(output_image)
#     output_image_pil.save(
#         r"C:\Users\SDS\Desktop\Scientific research\code\ultralytics-main-pri\ultralytics\data\datasets\baseline+PBANet\valid\images\fracture-55-DW.jpg")
#     print("Processed image saved.")

if __name__ == '__main__':
    DW = DSConvWithWT(256, 128)
    # 创建一个输入张量
    batch_size = 8
    input_tensor = torch.randn(batch_size, 256, 64, 64)
    # 运行模型并打印输入和输出的形状
    output_tensor = DW(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("0utput shape:", output_tensor.shape)
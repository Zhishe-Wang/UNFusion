import numpy as np
import torch
import torch.nn as nn
import fusion_strategy
import torch.nn.functional as F

Downsample = 'stride'


class BottleneckConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BottleneckConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.conv2d(x)
        out = F.relu(out, inplace=True)
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.stride_conv = nn.Conv2d(out_channels, out_channels, 3, 2)

    def forward(self, x, downsample=None):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        normal = F.relu(out, inplace=True)
        if downsample is"stride":
            out = self.reflection_pad(normal)
            down = self.stride_conv(out)
            down = F.relu(down, inplace=True)
            return normal, down
        else:
            return normal


class EncodeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(EncodeBlock, self).__init__()
        out_channels_def = int(in_channels / 2)

        self.conv1 = BottleneckConvLayer(in_channels, out_channels_def, 1, stride)
        self.conv2 = ConvLayer(out_channels_def, out_channels, kernel_size, stride)

    def forward(self, x, scales):
        normal = self.conv1(x)
        if scales == 4:
            normal = self.conv2(normal)
            return normal
        else:
            normal, out = self.conv2(normal, Downsample)
            return normal,out


class DecodeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DecodeBlock, self).__init__()
        out_channels_def = int(in_channels / 2)

        self.conv1 = ConvLayer(in_channels, out_channels_def, kernel_size, stride)
        self.conv2 = ConvLayer(out_channels_def, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class Upsample(torch.nn.Module):
    def __init__(self, Is_testing):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        if Is_testing:
            self.pad = UpsampleReshape()

    def forward(self, x1, x2, Is_testing):
        out = self.up(x2)
        if Is_testing:
            out = self.pad(x1, out)
        return out


class UpsampleReshape(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape, self).__init__()

    def forward(self, shape, x):
        shape = shape.size()
        shape_x = x.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape[3] != shape_x[3]:
            lef_right = shape[3] - shape_x[3]
            if lef_right % 2 is 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape[2] != shape_x[2]:
            top_bot = shape[2] - shape_x[2]
            if top_bot % 2 is 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x


class FusionModule(nn.Module):
    def __init__(self,Is_testing):
        super(FusionModule, self).__init__()
        rate = int(16)
        kernel_size = 3

        # encoder
        self.Conv1 = ConvLayer(1, rate, kernel_size, 1)

        self.Conv2 = ConvLayer(rate, rate*2, kernel_size, 1)
        self.ECB20 = EncodeBlock(rate*3, rate*4, kernel_size, 1)

        self.Conv3 = ConvLayer(rate*2, rate*3, kernel_size, 1)
        self.ECB30 = EncodeBlock(rate*5, rate*6, kernel_size, 1)
        self.ECB31 = EncodeBlock(rate*13, rate*16, kernel_size, 1)

        self.Conv4 = ConvLayer(rate*3, rate*4, kernel_size, 1)
        self.ECB40 = EncodeBlock(rate*7, rate*8, kernel_size, 1)
        self.ECB41 = EncodeBlock(rate*18, rate*19, kernel_size, 1)
        self.ECB42 = EncodeBlock(rate*47, rate*64, kernel_size, 1)

        # decoder
        self.DCB30 = DecodeBlock(rate*80, rate*16, kernel_size, 1)

        self.DCB20 = DecodeBlock(rate*20, rate*4, kernel_size, 1)
        self.DCB21 = DecodeBlock(rate*24, rate*4, kernel_size, 1)

        self.DCB10 = DecodeBlock(rate*5, rate, kernel_size, 1)
        self.DCB11 = DecodeBlock(rate*6, rate, kernel_size, 1)
        self.DCB12 = DecodeBlock(rate*7, rate, kernel_size, 1)

        self.UPf4 = Upsample(Is_testing)
        self.UPf3 = Upsample(Is_testing)
        self.UP30 = Upsample(Is_testing)
        self.UPf2 = Upsample(Is_testing)
        self.UP20 = Upsample(Is_testing)
        self.UP21 = Upsample(Is_testing)

        self.C1 = ConvLayer(rate, 1, 1, 1)

    def encoder(self, input):

        f_conv1, d_conv1 = self.Conv1(input, Downsample)

        f_conv2, d_conv2 = self.Conv2(d_conv1, Downsample)
        f_ECB20, d_ECB20 = self.ECB20(torch.cat([d_conv1, f_conv2], 1),2)

        f_conv3, d_conv3 = self.Conv3(d_conv2, Downsample)
        f_ECB30, d_ECB30 = self.ECB30(torch.cat([d_conv2, f_conv3], 1),3)
        f_ECB31, d_ECB31 = self.ECB31(torch.cat([d_ECB20, f_conv3, f_ECB30], 1),3)

        f_conv4 = self.Conv4(d_conv3)
        f_ECB40 = self.ECB40(torch.cat([d_conv3, f_conv4], 1),4)
        f_ECB41 = self.ECB41(torch.cat([d_ECB30, f_conv4, f_ECB40], 1),4)
        f_ECB42 = self.ECB42(torch.cat([d_ECB31, f_conv4, f_ECB40, f_ECB41], 1),4)
        return [f_conv1, f_ECB20, f_ECB31, f_ECB42]

    def fusion(self, en1, en2, p_type):
        # attention weight
        fusion_function = fusion_strategy.attention_fusion_weight

        f1_0 = fusion_function(en1[0], en2[0], p_type)
        f2_0 = fusion_function(en1[1], en2[1], p_type)
        f3_0 = fusion_function(en1[2], en2[2], p_type)
        f4_0 = fusion_function(en1[3], en2[3], p_type)

        return [f1_0, f2_0, f3_0, f4_0]

    def decoder(self, f_en, Is_testing):
        upf2 = self.UPf2(f_en[0], f_en[1], Is_testing)
        f_DCB10 = self.DCB10(torch.cat([f_en[0], upf2], 1))

        upf3 = self.UPf3(f_en[1], f_en[2], Is_testing)
        f_DCB20 = self.DCB20(torch.cat([f_en[1], upf3], 1))
        up20 = self.UP20(f_en[0], f_DCB20, Is_testing)
        f_DCB11 = self.DCB11(torch.cat([f_en[0], f_DCB10, up20], 1))

        up4 = self.UPf4(f_en[2], f_en[3], Is_testing)
        f_DCB30 = self.DCB30(torch.cat([f_en[2], up4], 1))
        up30 = self.UP30(f_en[1], f_DCB30, Is_testing)
        f_DCB21 = self.DCB21(torch.cat([f_en[1], f_DCB20, up30], 1))

        up21 = self.UP21(f_en[0], f_DCB21, Is_testing)
        f_DCB12 = self.DCB12(torch.cat([f_en[0], f_DCB10, f_DCB11, up21], 1))

        output = self.C1(f_DCB12)
        return output
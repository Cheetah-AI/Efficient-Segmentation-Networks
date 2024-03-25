##################################################################################
#Fast-SCNN: Fast Semantic Segmentation Network
#Paper-Link: https://arxiv.org/pdf/1902.04502.pdf
##################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantReLU
from brevitas.nn import QuantDropout
from brevitas.nn import TruncAdaptiveAvgPool2d
from brevitas.nn import QuantUpsample
from brevitas.quant import IntBias

from model.common import CommonIntWeightPerChannelQuant
from model.common import CommonIntWeightPerTensorQuant
from model.common import CommonUintActQuant

__all__ = ["FastSCNNq"]

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=1, 
        padding=0, 
        anbits=4, 
        wnbits=4,
        weight_quant=CommonIntWeightPerChannelQuant,
        activation_scaling_per_channel=False,
        **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(True)
            QuantConv2d(in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                        weight_quant=weight_quant,
                        weight_bit_width=wnbits),
            nn.BatchNorm2d(num_features=out_channels),
            QuantReLU(act_quant=CommonUintActQuant,
                      bit_width=anbits,
                      scaling_per_output_channel=activation_scaling_per_channel,
                      return_quant_tensor=True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, 
                 dw_channels, 
                 out_channels, 
                 stride=1,
                 anbits=4, 
                 wnbits=4,
                 weight_quant=CommonIntWeightPerChannelQuant,
                 activation_scaling_per_channel=False, 
                 **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            # nn.BatchNorm2d(dw_channels),
            # nn.ReLU(True),
            # nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(True)
            QuantConv2d(in_channels=dw_channels, 
                        out_channels=dw_channels, 
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        groups=dw_channels,
                        bias=False,
                        weight_quant=weight_quant,
                        weight_bit_width=wnbits),
            nn.BatchNorm2d(num_features=dw_channels),
            QuantReLU(act_quant=CommonUintActQuant,
                      bit_width=anbits,
                      scaling_per_output_channel=activation_scaling_per_channel,
                      return_quant_tensor=True),
            QuantConv2d(in_channels=dw_channels, 
                        out_channels=out_channels, 
                        kernel_size=1,
                        bias=False,
                        weight_quant=weight_quant,
                        weight_bit_width=wnbits),
            nn.BatchNorm2d(num_features=out_channels),
            QuantReLU(act_quant=CommonUintActQuant,
                      bit_width=anbits,
                      scaling_per_output_channel=activation_scaling_per_channel,
                      return_quant_tensor=True),
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    """Depthwise Convolutions"""
    def __init__(self, 
                 dw_channels, 
                 out_channels, 
                 stride=1, 
                 anbits=4, 
                 wnbits=4,
                 weight_quant=CommonIntWeightPerChannelQuant,
                 activation_scaling_per_channel=False, 
                 **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(True)
            QuantConv2d(in_channels=dw_channels, 
                        out_channels=out_channels, 
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        groups=dw_channels,
                        bias=False,
                        weight_quant=weight_quant,
                        weight_bit_width=wnbits),
            nn.BatchNorm2d(num_features=out_channels),
            QuantReLU(act_quant=CommonUintActQuant,
                      bit_width=anbits,
                      scaling_per_output_channel=activation_scaling_per_channel,
                      return_quant_tensor=True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.quant = QuantIdentity(act_quant=CommonUintActQuant,
                            return_quant_tensor=True,
                            bit_width=4)
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            # nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels)
            QuantConv2d(in_channels = in_channels * t, 
                           out_channels=out_channels, 
                           kernel_size=1, 
                           bias=False,
                           weight_quant=CommonIntWeightPerChannelQuant,
                           weight_bit_width=4),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        
        return self.quant(out)


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.quant = QuantIdentity(act_quant=CommonUintActQuant,
                            return_quant_tensor=True,
                            bit_width=4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        # avgpool = nn.AdaptiveAvgPool2d(size)
        avgpool = TruncAdaptiveAvgPool2d(output_size=size,
                                        float_to_int_impl_type='ROUND',
                                        bit_width=4)
        return avgpool(x)

    def upsample(self, x, size):
        # return F.interpolate(x, size, mode='bilinear', align_corners=True)
        qupsample = QuantUpsample(size, mode='bilinear', align_corners=True)
        return qupsample(x)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.quant( self.upsample(self.conv1(self.pool(x, 1)), size) )
        feat2 = self.quant( self.upsample(self.conv2(self.pool(x, 2)), size) )
        feat3 = self.quant( self.upsample(self.conv3(self.pool(x, 3)), size) )
        feat4 = self.quant( self.upsample(self.conv4(self.pool(x, 6)), size) )
        x = self.quant( x )
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            # nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels)
            QuantConv2d(in_channels = out_channels, 
                           out_channels=out_channels, 
                           kernel_size=1, 
                           bias=True,
                           bias_quant=IntBias,
                           weight_quant=CommonIntWeightPerChannelQuant,
                           weight_bit_width=4),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            # nn.Conv2d(highter_in_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels)
            QuantConv2d(in_channels = highter_in_channels, 
                           out_channels=out_channels, 
                           kernel_size=1, 
                           bias=True,
                           bias_quant=IntBias,
                           weight_quant=CommonIntWeightPerChannelQuant,
                           weight_bit_width=4),
            nn.BatchNorm2d(num_features=out_channels)
        )
        # self.relu = nn.ReLU(True)
        self.relu = QuantReLU(act_quant=CommonUintActQuant,
                      bit_width=4,
                      scaling_per_output_channel=False,
                      return_quant_tensor=True)

    def forward(self, higher_res_feature, lower_res_feature):
        _, _, h, w = higher_res_feature.size()
        lower_res_feature = F.interpolate(lower_res_feature, size=(h,w), mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, 
                 dw_channels, 
                 num_classes, 
                 stride=1,
                 wnbits=4,
                 weight_quant=CommonIntWeightPerChannelQuant,
                 **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            # nn.Dropout(0.1),
            # nn.Conv2d(dw_channels, num_classes, 1)
            QuantDropout(p=0.1, return_quant_tensor=True),
            QuantConv2d(in_channels=dw_channels,
                        out_channels=num_classes,
                        kernel_size=1,
                        bias=True,
                        bias_quant=IntBias,
                        weight_quant=weight_quant,
                        weight_bit_width=wnbits)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class FastSCNNq(nn.Module):
    def __init__(self, classes, aux=False, **kwargs):
        super(FastSCNNq, self).__init__()
        # self.aux = aux
        self.aux = False
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return x
        # return tuple(outputs)


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSCNNq(classes=19).to(device)
    summary(model,(3,512,1024))


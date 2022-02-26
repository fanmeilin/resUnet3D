import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):  # 基本模块 两个cbr+res 通道不变
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # 三个cbr+res 最后一个cbr通道数扩大4倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 对称部分
def trans_conv3x3x3(in_planes, out_planes, stride=1):
    """
    Args:
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride: 转置卷积步长

    Returns:kernel_size=3的3D转置卷积 当stride为2时就采用output_padding=1 进行还原
    """
    if stride>1:
        return nn.ConvTranspose3d(in_planes,
                              out_planes,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              output_padding=1,
                              bias=False)
    else:
        return nn.ConvTranspose3d(in_planes,
                              out_planes,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              bias=False)


def trans_conv1x1x1(in_planes, out_planes, stride=1):
    """
    Args:
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride: 转置卷积步长

    Returns:kernel_size=1的3D转置卷积,当stride为2时就采用output_padding=1 进行还原
    """
    if stride>1:
        return nn.ConvTranspose3d(in_planes,
                              out_planes,
                              kernel_size=1,
                              stride=stride,
                              output_padding=1,
                              bias=False)
    else:
        return nn.ConvTranspose3d(in_planes,
                              out_planes,
                              kernel_size=1,
                              stride=stride,
                              bias=False)


class TransBasicBlock(nn.Module):  # 基本转置模块 两个对称cbr+res 通道内部不变
    expansion = 1

    def __init__(self, out_planes, planes, stride=1, upsample=None):
        super().__init__()

        self.conv1 = trans_conv3x3x3(planes * self.expansion, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = trans_conv3x3x3(planes, out_planes)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)

        return out


# 下采样 [64, 128, 256, 512] [3, 4, 6, 3] 上采样[512, 256, 128, 64] [3, 6, 4, 3]
class TransBottleneck(nn.Module):  # 三个cbr+res 最后一个cbr通道数扩大4倍
    expansion = 4

    def __init__(self, out_planes, planes, stride=1, upsample=None):
        super().__init__()

        self.conv1 = trans_conv1x1x1(planes * self.expansion, planes)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = trans_conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = trans_conv1x1x1(planes, out_planes)  # conv1*1*1可以用于改变通道数和尺寸
        self.bn3 = nn.BatchNorm3d(out_planes)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class resUnet3D(nn.Module):

    def __init__(self,
                 block,
                 upblock,
                 layers,
                 block_inplanes,
                 n_input_channels=1,  # 3改为1
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,  # 整体放缩
                 ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]  # 每个block的通道数
        self.out_planes = []
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)

        self.up_conv1 = nn.ConvTranspose3d(self.in_planes,  # TODO
                                           n_input_channels,
                                           kernel_size=(conv1_t_size, 7, 7),
                                           stride=(conv1_t_stride, 2, 2),
                                           padding=(conv1_t_size // 2, 3, 3),
                                           output_padding=(0, 1, 1),
                                           bias=False)
        self.up_bn1 = nn.BatchNorm3d(n_input_channels)  # 待商榷 TODO
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.maxunpool = nn.MaxUnpool3d(kernel_size=3, stride=2, padding=0)
        self.upsample = nn.Upsample(scale_factor=2)
        # 34  block_inplanes ：[64, 128, 256, 512] layers：[3, 4, 6, 3]   model = ResNet(BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)
        # 下采样部分 encoder部分
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        #  上采样
        self.uplayer4 = self._trans_make_layer(upblock,
                                               block_inplanes[3],
                                               layers[3],
                                               shortcut_type,
                                               stride=2)
        self.uplayer3 = self._trans_make_layer(upblock,
                                               block_inplanes[2],
                                               layers[2],
                                               shortcut_type,
                                               stride=2)
        self.uplayer2 = self._trans_make_layer(upblock,
                                               block_inplanes[1],
                                               layers[1],
                                               shortcut_type,
                                               stride=2)
        self.uplayer1 = self._trans_make_layer(upblock, block_inplanes[0], layers[0],
                                               shortcut_type)

        # 后续预训练使用resnet
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 虚线下采样处理快连A部分
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)  # 尺寸一致
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),  # 通道数一致
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    # 下采样的分层残差块 block，planes=512，blocks = 3，B，stride=2
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    # 同时调整通道数和尺寸 W*H*in_planes->W/2*H/2*（planes * block.expansion）
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        self.out_planes.append(self.in_planes)  # 记录每一个残差块的初始输入的通道数
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    # 虚线上采样处理快连A部分
    def _upsample_basic_block(self, x, planes, stride):
        import random
        out = F.upsample_bilinear(x, scale_factor=stride)  # 尺寸一致
        # zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),  # 通道数一致
        #                         out.size(3), out.size(4))
        # if isinstance(out.data, torch.cuda.FloatTensor):
        #     zero_pads = zero_pads.cuda()
        # out = torch.cat([out.data, zero_pads], dim=1)

        result_out = out[:, random.sample(range(out.size(1)), planes), :, :, :]  # 随机取指定通道数使其一致 通道数一致
        return result_out

    # 上采样的分层残差块
    def _trans_make_layer(self, upblock, planes, blocks, shortcut_type, stride=1):
        upsample = None
        layers = []
        temp_inner_planes = planes * upblock.expansion
        temp_out_planes = self.out_planes.pop()    #每一个残差块的输入通道 【in，64，128，256】 【in，256，512，1024】
        for i in range(1, blocks):
            layers.append(upblock(temp_inner_planes, planes))

        # 定义虚线
        if stride != 1 or temp_out_planes != planes * upblock.expansion:
            if shortcut_type == 'A':
                upsample = partial(self._upsample_basic_block,
                                   planes=temp_out_planes,
                                   stride=stride)
            else:
                upsample = nn.Sequential(
                    trans_conv1x1x1(planes * upblock.expansion, temp_out_planes, stride), #使用output_padding=1
                    # 同时调整通道数和尺寸 W*H*in_planes->W*2 *H*2 *（planes * block.expansion）
                    nn.BatchNorm3d(temp_out_planes))
        # 最后一层进行上采样
        layers.append(
            upblock(out_planes=temp_out_planes,
                    planes=planes,
                    stride=stride,
                    upsample=upsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        down_feats = []
        down_feats.append(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("layer0:",x.shape)
        if not self.no_max_pool:
            x,indices = self.maxpool(x)
        down_feats.append(x)
        # print("maxpool:", x.shape)
        # 下采样
        x = self.layer1(x)
        down_feats.append(x)
        # print("下采样1:", x.shape)
        x = self.layer2(x)
        down_feats.append(x)
        # print("下采样2:", x.shape)
        x = self.layer3(x)
        down_feats.append(x)
        # print("下采样3:", x.shape)
        # x = self.layer4(x)
        # down_feats.append(x)
        # # print("下采样4:", x.shape)
        # # 上采样
        # x = self.uplayer4(x + down_feats.pop())
        # print("上采样1:", x.shape)
        x = self.uplayer3(x + down_feats.pop())
        # print("上采样2:", x.shape)
        x = self.uplayer2(x + down_feats.pop())
        # print("上采样3:", x.shape)
        x = self.uplayer1(x + down_feats.pop())
        # print("上采样4:", x.shape)

        # 还原conv1
        x = x + down_feats.pop()
        if not self.no_max_pool: #maxunpool还原尺寸有问题 暂且使用upsample
            x = self.upsample(x)
            # x = self.maxunpool(x,indices)

        # print("unmaxpooling:", x.shape)
        x = self.up_conv1(x)
        x = self.up_bn1(x)
        x = self.relu(x)
        x = x + down_feats.pop()
        # print("还原:", x.shape)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resUnet3D(BasicBlock, TransBasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = resUnet3D(BasicBlock, TransBasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = resUnet3D(BasicBlock, TransBasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = resUnet3D(Bottleneck, TransBottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = resUnet3D(Bottleneck, TransBottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = resUnet3D(Bottleneck, TransBottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = resUnet3D(Bottleneck, TransBottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


if __name__ == "__main__":
    import torch  # 命令行是逐行立即执行的
    #查看权重文件信息
    content = torch.load('r3d18_KM_200ep.pth')
    print(content.keys())  # keys()
    # 之后有其他需求比如要看 key 为 model 的内容有啥
    print("epoch",content['epoch'])
    print("arch", content['arch'])
    # print("optimizer", content['optimizer'])
    print("scheduler",content["scheduler"])
    print("state_dict", content['state_dict']['conv1.weight'].shape)
    for item in content["state_dict"]:
        print(item)


    from torchsummary import summary
    #查看模型结构
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # model = resUnet3D(BasicBlock,TransBasicBlock, [2, 2, 2, 2], get_inplanes()).to(device)
    # summary(model, input_size=(1, 96, 128, 128))
"""
几点说明：
1，完全按照对称结构还原resnet的残差基础上每一层有相加
后续对于pool的是否保留
对output_padding的在最后的部分是否会有丢失情况有待考量
另外可以再改进的地方
2，其他模块
可以考虑注意力，transfomer，传统去噪方法滤波等，蒸馏来改进
对图像增加泊松噪声的挖点
对比pooling，bn层是否需要保留
对比残差块中tranconv和conv 可验证tranconv的效果是否会比conv效果好
"""

"""
下一步：
预训练模型的导入
尺寸的确定
添加噪声
训练
"""
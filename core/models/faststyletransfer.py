import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from torchvision import transforms
from torch.autograd import Variable
from core.utils import normalize_imagenet, recover_imagenet


class ResInstanceCenterCrop(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResInstanceCenterCrop, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.instancenorm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        fx = self.conv3x3(x)
        fx = self.instancenorm(fx)
        fx = F.relu(fx)
        fx = self.conv3x3(fx)
        fx = self.instancenorm(fx)
        return fx + x[:, :, 2:-2, 2:-2]


class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvBatchRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.batchnorm(x))
        return x


class ConvInstanceRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvInstanceRelu, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, stride=stride)
        self.instancenorm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = F.relu(self.instancenorm(x))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.instancenorm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        return F.relu(x)


class FastStyleTransfer(nn.Module):
    def __init__(self):
        super(FastStyleTransfer, self).__init__()
        self.reflection_padding = nn.ReflectionPad2d(40)
        self.conv1 = ConvInstanceRelu(3, 32, kernel_size=9)
        self.conv2 = ConvInstanceRelu(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvInstanceRelu(64, 128, kernel_size=3, stride=2)
        self.resblock = ResInstanceCenterCrop(128, 128)
        self.upblock1 = UpBlock(128, 64, kernel_size=3)
        self.upblock2 = UpBlock(64, 32, kernel_size=3)
        self.reflection_pad_out = nn.ReflectionPad2d(4)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=9)

    def forward(self, x):
        x = self.reflection_padding(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.reflection_pad_out(x)
        x = self.conv_out(x)
        return F.tanh(x)


class StyleTransfer:
    def __init__(self, style):
        self.model = FastStyleTransfer()
        if style is 'starry':
            checkpoint = torch.load('core/models/state_dicts/ST_Epoch17_COCOS_Starry.pth.tar')
        else:
            checkpoint = torch.load('core/models/state_dicts/ST2_Epoch1_COCO_undie.pth.tar')
        self.model.load_state_dict(checkpoint['model_state'])

    @staticmethod
    def prepare_image(image):
        data_transforms = transforms.Compose([transforms.ToTensor(),
                                              normalize_imagenet()])
        return Variable(data_transforms(image).unsqueeze(dim=0))

    def transform(self, image):
        model = self.model.eval()
        image = self.prepare_image(image)
        hi_res_image = model(image).data.squeeze(0)
        return recover_imagenet(hi_res_image)



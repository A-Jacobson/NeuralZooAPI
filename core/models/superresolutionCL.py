import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from core.utils import normalize_imagenet, recover_imagenet


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super(ResBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_in = x
        x = self.conv3x3(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = self.batchnorm(x)
        return x + x_in


class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvBatchRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.batchnorm(x))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        return F.relu(x)


class JohnsonSR(nn.Module):
    def __init__(self):
        super(JohnsonSR, self).__init__()
        self.batch_norm = nn.BatchNorm2d(3)
        self.conv_in = ConvBatchRelu(3, 64, kernel_size=9, padding=4)
        self.resblock = ResBlock(64, 64, padding=1)
        self.upblock = UpBlock(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv_in(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.upblock(x)
        x = self.upblock(x)
        x = self.conv_out(x)
        return F.tanh(x)


class SuperResolutionCL:
    def __init__(self):
        self.model = JohnsonSR()
        checkpoint = torch.load('core/models/state_dicts/johnson_pl_bn_11.pth.tar')
        self.model.load_state_dict(checkpoint['model_state'])

    @staticmethod
    def prepare_image(image):
        data_transforms = transforms.Compose([transforms.Scale(72),
                                              transforms.ToTensor(),
                                              normalize_imagenet()])
        return Variable(data_transforms(image).unsqueeze(dim=0))

    def transform(self, image):
        model = self.model.eval()
        image = self.prepare_image(image)
        hi_res_image = model(image).data.squeeze(0)
        return recover_imagenet(hi_res_image)

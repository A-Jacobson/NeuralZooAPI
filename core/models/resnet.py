from torchvision.models import resnet18
from torchvision import transforms
from torch.autograd import Variable
from core.utils import imagenet_classes, normalize_imagenet
import torch.nn.functional as F
import torch


class ResNet18:
    def __init__(self):
        self.model = resnet18(pretrained=True)

    @staticmethod
    def prepare_image(image):
        data_transforms = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_imagenet()
        ])
        return Variable(data_transforms(image).unsqueeze(dim=0))

    def predict(self, image):
        model = self.model.eval()
        image = self.prepare_image(image)
        preds = model(image).data
        prob, idx = torch.max(F.softmax(preds), dim=1)
        return prob, imagenet_classes[idx[0]]

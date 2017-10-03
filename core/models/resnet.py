from torchvision.models import resnet18
from torchvision import transforms
from torch.autograd import Variable
from core.utils import convert_imagenet
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
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        return Variable(data_transforms(image).unsqueeze(dim=0))

    def predict(self, image):
        model = self.model.eval()
        image = self.prepare_image(image)
        preds = model(image).data
        _, idx = torch.max(preds, dim=1)
        return convert_imagenet[idx[0]]

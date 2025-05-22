
import sys
from torchvision.models import resnet50, ResNet50_Weights, vgg11, vgg16,\
    alexnet, VGG11_Weights, VGG16_Weights, AlexNet_Weights
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTConfig

class Network():
  def __init__(self, device, arch, pretrained=True):
        self.preprocess = None
        self.model = None
        self.arch = arch
        self.pretrained = pretrained
        self.device = device
  def set_model(self):
        if self.arch == "vgg11":
            if self.pretrained:
                weights = VGG11_Weights.IMAGENET1K_V1
                self.preprocess = weights.transforms()
                self.model = vgg11(weights=weights).to(self.device)
                self.model.eval()
            else:
                self.model = vgg11().to(self.device)
        elif self.arch == "vgg16":
            if self.pretrained:
                weights = VGG16_Weights.IMAGENET1K_V1
                self.preprocess = weights.transforms()
                self.model = vgg16(weights=weights).to(self.device)
                self.model.eval()
            else:
                self.model = vgg16().to(self.device)
        elif self.arch == "resnet":
            if self.pretrained:
                weights = ResNet50_Weights.DEFAULT
                self.preprocess = weights.transforms()
                self.model = resnet50(weights=weights).to(self.device)
                self.model.eval()
            else:
                self.model = resnet50().to(self.device)
        elif self.arch == "alexnet":
            if self.pretrained:
                weights = AlexNet_Weights.IMAGENET1K_V1
                self.preprocess = weights.transforms()
                self.model = alexnet(weights=weights).to(self.device)
                self.model.eval()
            else:
                self.model = alexnet().to(self.device)
        elif self.arch == "vit":
            if self.pretrained:
                self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(self.device)
                feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
                self.preprocess = feature_extractor
                self.model.eval()
            else:
                print('\n using unpretrained model')
                self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(self.device)
        else:
            sys.exit("Wrong architecture")
        return self.model

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


__all__ = ['darknet53','resnet34', 'resnet50', 'resnet18', 'resnext50_32x4d', "efficientnet_b0", "efficientnet_b4", "resnest_50", "resnet152", "resnet50_aspp"]



def darknet53(args):
    from custom_models.darknet import Darknet53, EmptyLayer
    model = Darknet53()
    model.fc = nn.Sequential(
        GeneralizedMeanPoolingP(),
        Flatten(),
        nn.Linear(1024, args.num_feature),
        nn.BatchNorm1d(args.num_feature)
    )
    return model

def resnext50_32x4d(args):
    import torchvision.models as models
    model = models.resnext50_32x4d(pretrained=args.pretrained)
    model = set_pooling(model, args, "avgpool")
    model.fc = output_layer(model.fc.in_features, args.num_feature)
    return model


def resnet50_aspp(args):
    from custom_models.resnet_v3 import resnet50
    model = resnet50(pretrained=args.pretrained)
    return model


def resnet34(args):
    import torchvision.models as models
    model = models.resnet34(pretrained=args.pretrained)
    model = set_pooling(model, args, "avgpool")
    model.fc = output_layer(model.fc.in_features, args.num_feature)
    return model


def resnet50(args):
    import torchvision.models as models
    model = models.resnet50(pretrained=args.pretrained)
    model = set_pooling(model, args, "avgpool")
    model.avgpool = GeneralizedMeanPoolingP()
    model.fc = output_layer(model.fc.in_features, args.num_feature)
    return model

def resnet50_classif(args):
    import torchvision.models as models
    model = models.resnet50(pretrained=args.pretrained, num_classes=args.num_classes)
    return model

def resnet18(args):
    import torchvision.models as models
    model = models.resnet18(pretrained=args.pretrained)
    model = set_pooling(model, args, "avgpool")
    model.fc = output_layer(model.fc.in_features, args.num_feature)
    return model

def resnet152(args):
    import torchvision.models as models
    model = models.resnet152(pretrained=args.pretrained)
    model = set_pooling(model, args, "avgpool")
    model.fc = output_layer(model.fc.in_features, args.num_feature)
    return model


def efficientnet_b0(args):
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model = set_pooling(model, args, "_avg_pooling")
    model._fc = output_layer(model._fc.in_features, args.num_feature)
    return model

def efficientnet_b4(args):
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model = set_pooling(model, args, "_avg_pooling")
    model._fc = output_layer(model._fc.in_features, args.num_feature)
    return model


def resnest_50(args):
    from resnest.torch import resnest50
    model = resnest50(pretrained=args.pretrained)
    model = set_pooling(model, args, "avgpool")
    model.fc = output_layer(model.fc.in_features, args.num_feature)
    return model


def set_pooling(model, args, name):
    if args.pooling == "gem":
        setattr(model, name, GeneralizedMeanPoolingP())
    return model



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def output_layer(last_channels, num_features):
    return nn.Sequential(
            Flatten(),
            nn.Linear(last_channels, num_features),
            nn.BatchNorm1d(num_features)
        )

class GeneralizedMeanPooling(nn.Module):
    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = Parameter(torch.ones(1) * norm)



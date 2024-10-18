import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights
import enum 
class HeadType(enum.Enum):
    PRETRAINED= "pretrained"
    UNTRAINED= "untrained"
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class CNNModelTorch(nn.Module):
    def __init__(self, 
                 num_models:int,
                 num_classes: int,
                 head_type:HeadType):
        super(CNNModelTorch, self).__init__()

        self.name = "cnn"
        self.head_type = head_type
        self.num_models = num_models
        self.num_classes = num_classes

        if self.head_type == HeadType.PRETRAINED:
            self.head = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.head.fc = Identity()
            for param in self.head.parameters():
                param.requires_grad = False
        elif self.head_type == HeadType.UNTRAINED:
            self.head = resnet18(weights=None)
            self.head.fc = Identity()
        else:
            raise RuntimeError(f"Head type {self.head_type} not recognized")

        if self.num_models == 1:
            self.model = nn.Linear(512, self.num_classes)
        else:
            self.models = nn.ModuleList([nn.Sequential(
                nn.Linear(512, self.num_classes),
            ) for _ in range(self.num_models)])

    def forward(self, x):
        assert len(
            x.shape) == 4, f"Input should have 4 dimensions: batch, channel, height, width. Got {x.shape} instead."
        # print("x shape", x.shape)
        x = self.head(x)
        # print("x shape", x.shape)
        if self.num_models == 1:
            return self.model(x)
        else:
            outputs = []
            for model in self.models:
                # print("x shape", model(x).shape)
                outputs.append(model(x))
            # shape is (batch, num_models, num_classes)
            return torch.stack(outputs, dim=1)
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from pathlib import Path


class DinoModelTorch(nn.Module):
    def __init__(self, output_size: int | tuple, use_features: bool = False,
                 dinov2_weight_path: Optional[Path] = None,
                 dinov2_model_repo_path: Optional[Path] = None,
                 num_classes: Optional[bool] = None,
                 num_models: int = 1,
                 ):
        super(DinoModelTorch, self).__init__()
        self.num_classes = num_classes
        self.use_features = use_features  # if use features, then just use the linear layer
        self.output_size = output_size
        self.num_models = num_models

        if dinov2_weight_path is not None:
            self.head = torch.hub.load(dinov2_model_repo_path, 'dinov2_vitb14', source='local', pretrained=False)
            self.head.load_state_dict(torch.load(dinov2_weight_path))
        else:
            self.head = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        for param in self.head.parameters():
            param.requires_grad = False

        if self.num_classes is not None:
            # Output shape is (num_models, num_classes, num_bins)
            self.linear_model = nn.ModuleList([nn.Sequential(
                nn.Linear(768, np.prod(output_size)),
            ) for _ in range(self.num_models)])
        else:
            # output shape is (num_models, num_bins)
            self.linear_model = nn.ModuleList([nn.Sequential(
                nn.Linear(768, output_size),
                nn.LeakyReLU(True),
            ) for _ in range(self.num_models)])

    def extract_features(self, x):
        with torch.no_grad():
            return self.head(x)

    def forward(self, x):
        if not self.use_features:
            assert len(
                x.shape) == 4, f"Input should have 4 dimensions: batch, channel, height, width. Got {x.shape} instead."
            with torch.no_grad():
                x = self.head(x)
        model_outputs = []
        for i in range(self.num_models):
            model_outputs.append(self.linear_model[i](x))
        x = torch.stack(model_outputs, dim=1)
        if self.num_classes is not None:
            x = x.view(x.shape[0], self.num_models, self.num_classes, -1)
        return x

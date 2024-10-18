import torch.nn as nn
import torch
import torch.nn.functional as F
import enum


class MemberModelName(enum.Enum):
    # jointly trained with BinEnsembleTorch
    BIN_ENSEMBLE_TORCH = "bin_ensemble_torch"
    CNN_MODEL_ABLATION_TORCH = "cnn_model_ablation_torch"
    # individually trained
    PRETRAINED_RESNET_INDIVIDUAL = "pretrained_resnet_individual"
    UNTRAINED_RESNET_INDIVIDUAL = "untrained_resnet_individual"
    LINEAR = "linear"


class BinEnsembleTorch(nn.Module):
    def __init__(self, num_models, num_classes: int):
        super().__init__()
        self.models = nn.ModuleList([nn.Sequential(
            nn.Linear(768, num_classes),
        ) for _ in range(num_models)])

    def forward(self, x):
        assert len(x.shape) == 2, "Input must be (batch, feature_size)"
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        # shape is (batch, num_models, num_classes)
        return torch.stack(outputs, dim=1)

    @staticmethod
    def calculate_uncertainty(outputs):
        """
        Calculate uncertainty as the entropy of the softmax distribution
        """
        # CALCULATES ENTROPY OF AVERAGE
        # shape is (batch, num_models, num_classes)
        softmax = F.softmax(outputs, dim=-1)
        # shape is (batch, num_models, num_classes)
        avg_softmax = softmax.mean(dim=1)
        # shape is (batch, num_classes)
        avg_log_softmax = torch.log(avg_softmax + 1e-8)
        # shape is (batch, num_classes)
        entropy = -torch.sum(avg_softmax * avg_log_softmax, dim=-1)
        # shape is (batch,)
        assert torch.min(entropy) >= -0.000001, f"Uncertainty has negative values {entropy.min()}"
        entropy = torch.clip(entropy, 0, torch.inf)  # trim off slight negatives to zero
        return entropy

    @staticmethod
    def calculate_classes(outputs):
        """
        Calculate the most likely class for each bin
        """
        # shape is (batch, num_models, num_classes)
        softmax = F.softmax(outputs, dim=-1)
        # shape is (batch,)
        return torch.argmax(softmax.mean(dim=1), dim=-1)

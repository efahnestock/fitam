import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from fitam.learning.bin_ensemble_torch import BinEnsembleTorch
from fitam.learning.load_by_name import remap_model_state_dict
from fitam.learning.bin_ensemble_torch import MemberModelName
from fitam.learning.cnn_model_ablation_torch import Identity, CNNModelTorch, HeadType


class FullEnsembleTorch(nn.Module):

    def __init__(self,
                 num_models: int,
                 num_bins: int,
                 num_classes: int,
                 use_features: bool = True,
                 dinov2_weight_path: Path = None,
                 dinov2_model_repo_path: Path = None,
                 model_name: MemberModelName = MemberModelName.BIN_ENSEMBLE_TORCH,
                 model_args: dict = None):
        super().__init__()
        self.use_features = use_features
        self.model_name = model_name
        self.model_args = model_args
        if "num_classes" in model_args:
            del model_args['num_classes']
        self.num_models = num_models
        self.num_classes = num_classes
        self.num_bins = num_bins

        # set the head
        if model_name in [MemberModelName.BIN_ENSEMBLE_TORCH, MemberModelName.LINEAR]:
            if dinov2_weight_path is not None:
                self.head = torch.hub.load(dinov2_model_repo_path, 'dinov2_vitb14', source='local', pretrained=False)
                self.head.load_state_dict(torch.load(dinov2_weight_path))
            else:
                self.head = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        elif model_name in [MemberModelName.CNN_MODEL_ABLATION_TORCH, MemberModelName.UNTRAINED_RESNET_INDIVIDUAL, MemberModelName.PRETRAINED_RESNET_INDIVIDUAL]:
            self.head = Identity()
        else:
            raise RuntimeError(f"Unrecognized model name {model_name}")

        for param in self.head.parameters():
            param.requires_grad = False

        # set tails
        if model_name == MemberModelName.BIN_ENSEMBLE_TORCH:
            self.models = nn.ModuleList([BinEnsembleTorch(num_models, num_classes=num_classes) for _ in range(num_bins)])
        elif model_name == MemberModelName.CNN_MODEL_ABLATION_TORCH:
            self.models = nn.ModuleList([CNNModelTorch(num_models, num_classes, **model_args) for _ in range(num_bins)])
        elif model_name == MemberModelName.LINEAR:
            self.models = nn.ModuleList([BinEnsembleTorch(num_models, num_classes=num_classes) for _ in range(num_bins)])
        elif model_name == MemberModelName.PRETRAINED_RESNET_INDIVIDUAL:
            self.models = nn.ModuleList([nn.ModuleList([CNNModelTorch(1, num_classes, **model_args, head_type=HeadType.PRETRAINED) for __ in range(num_models)]) for _ in range(num_bins)])
        elif model_name == MemberModelName.UNTRAINED_RESNET_INDIVIDUAL:
            self.models = nn.ModuleList([nn.ModuleList([CNNModelTorch(1, num_classes, **model_args, head_type=HeadType.UNTRAINED) for __ in range(num_models)]) for _ in range(num_bins)])
        else:
            raise RuntimeError(f"Unrecognized model name {model_name}")

    def load_from_dir(self, base_path: Path):
        """
        """
        if base_path is str:
            base_path = Path(base_path)
        if self.model_name in [MemberModelName.BIN_ENSEMBLE_TORCH,
                               MemberModelName.CNN_MODEL_ABLATION_TORCH]:
            #  File names follow the format {bin_index}_bin.ckpt
            for bin_index in range(self.num_bins):
                checkpoint = torch.load(base_path / f"{bin_index:02d}_bin.ckpt")
                model = self.models[bin_index]
                remapped_state_dict = remap_model_state_dict(checkpoint['state_dict'], model.state_dict())
                model.load_state_dict(remapped_state_dict)
        elif self.model_name == MemberModelName.LINEAR:
            #  File names follow the format 00_bin/model_00.pt
            for bin_index in range(self.num_bins):
                for model_index in range(self.num_models):
                    checkpoint = torch.load(base_path / f"{bin_index:02d}_bin" / f"model_{model_index:02d}.pt")
                    model = self.models[bin_index].models[model_index]
                    # remap net. to 0.
                    remapped_state_dict = {k.replace('net', '0'): v for k, v in checkpoint.items()}
                    model.load_state_dict(remapped_state_dict)

        elif self.model_name in [MemberModelName.PRETRAINED_RESNET_INDIVIDUAL,
                                 MemberModelName.UNTRAINED_RESNET_INDIVIDUAL]:
            #  File names follow the format 00_bin/model_00.pt
            for bin_index in range(self.num_bins):
                for model_index in range(self.num_models):
                    checkpoint = torch.load(base_path / f"{bin_index:02d}_bin" / f"model_{model_index:02d}.pt")
                    model = self.models[bin_index][model_index]
                    remapped_state_dict = checkpoint  # remap_model_state_dict(checkpoint, model.state_dict())
                    model.load_state_dict(remapped_state_dict)
        else:
            raise RuntimeError(f"Unrecognized model name {self.model_name}")

    def forward(self, x):
        if not self.use_features:
            # pass through head
            x = self.head(x)
        else:
            if self.model_name not in [MemberModelName.LINEAR,
                                       MemberModelName.BIN_ENSEMBLE_TORCH]:
                raise RuntimeError("Cannot use features with model type")
        bins = []
        for bin_index in range(self.num_bins):
            if self.model_name in [MemberModelName.LINEAR,
                                   MemberModelName.BIN_ENSEMBLE_TORCH,
                                   MemberModelName.CNN_MODEL_ABLATION_TORCH]:
                model_output = self.models[bin_index](x)  # shape is (batch, num_models, num_classes)
                bins.append(model_output.unsqueeze(1))  # shape is (batch, 1, num_models, num_classes)
            else:
                model_outputs = []
                for model_index in range(self.num_models):
                    model = self.models[bin_index][model_index]
                    model_outputs.append(model(x))
                bins.append(torch.stack(model_outputs, dim=1).unsqueeze(1))
        # return shape is (batch, num_bins, num_models, num_classes)
        return torch.cat(bins, dim=1)

    @staticmethod
    def calculate_uncertainty(outputs):
        """
        Calculate uncertainty as the entropy of the softmax distribution
        """
        assert len(outputs.shape) == 4, f"Outputs should be (batch, num_bins, num_models, num_classes). Got shape: {outputs.shape}"
        # CALCULATES ENTROPY OF AVERAGE
        softmax = F.softmax(outputs, dim=-1)
        # shape is (batch, num_bins, num_models, num_classes)
        avg_softmax = softmax.mean(dim=2)
        # shape is (batch, num_bins, num_classes)
        avg_log_softmax = torch.log(avg_softmax + 1e-8)
        # shape is (batch, num_bins, num_classes)
        entropy = -torch.sum(avg_softmax * avg_log_softmax, dim=-1)
        # shape is (batch, num_bins)
        assert torch.min(entropy) >= -0.000001, f"Uncertainty has negative values {entropy.min()}"
        entropy = torch.clip(entropy, 0, torch.inf)  # trim off slight negatives to zero
        return entropy

    @staticmethod
    def calculate_winning_confidence(outputs):
        """
        Calculate the confidence of the winning class, after reducing across ensembles
        """
        avg_softmax = F.softmax(outputs, dim=-1).mean(dim=2)
        return torch.max(avg_softmax, dim=-1)[0]

    @staticmethod
    def calculate_classes(outputs):
        """
        Calculate the most likely class for each bin
        """
        assert len(outputs.shape) == 4, f"Outputs should be (batch, num_bins, num_models, num_classes). Got shape: {outputs.shape}"
        # shape is (batch, num_bins, num_models, num_classes)
        softmax = F.softmax(outputs, dim=-1)
        # shape is (batch, num_bins, num_classes)
        return torch.argmax(softmax.mean(dim=2), dim=-1)
        # shape is (batch, num_bins)

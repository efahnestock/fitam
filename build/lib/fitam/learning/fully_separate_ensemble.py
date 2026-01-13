import pytorch_lightning as pl
from dataclasses import dataclass, field
from pathlib import Path
import torch.nn.functional as F
import pandas as pd
from fitam.learning.bin_ensemble_torch import MemberModelName
from fitam.learning.fully_separate_ensemble_torch import FullEnsembleTorch


@dataclass
class StepOutput():
    raw_outputs: list = field(default_factory=list)  # list of shape (batch, num_bins, num_classes)
    predictions: list = field(default_factory=list)  # list of shape (batch, num_bins)
    entropy: list = field(default_factory=list)  # list of shape (batch, num_bins)
    prediction_uncertainty: list = field(default_factory=list)  # list of shape (batch, num_bins)
    avg_softmax: list = field(default_factory=list)  # list of shape (batch, num_bins, num_classes)
    labels: list = field(default_factory=list)  # list of shape (batch, num_bins)
    global_dataset_index: list = field(default_factory=list)
    local_dataset_index: list = field(default_factory=list)

    def extend(self, other: 'StepOutput'):
        self.raw_outputs.extend(other.raw_outputs)
        self.predictions.extend(other.predictions)
        self.prediction_uncertainty.extend(other.prediction_uncertainty)
        self.entropy.extend(other.entropy)
        self.avg_softmax.extend(other.avg_softmax)
        self.labels.extend(other.labels)
        self.global_dataset_index.extend(other.global_dataset_index)
        self.local_dataset_index.extend(other.local_dataset_index)
        return self


class FullEnsemble(pl.LightningModule):

    def __init__(self, num_models: int, num_bins: int, num_classes: int,
                 dinov2_weight_path: Path = None,
                 dinov2_model_repo_path: Path = None,
                 model_name: MemberModelName = MemberModelName.BIN_ENSEMBLE_TORCH,
                 use_features: bool = True,
                 model_args: dict = None,
                 name='full_ensemble'):
        super().__init__()

        self.model = FullEnsembleTorch(num_models, num_bins, num_classes=num_classes, dinov2_weight_path=dinov2_weight_path, dinov2_model_repo_path=dinov2_model_repo_path,
                                       model_name=model_name, model_args=model_args, use_features=use_features)
        self.test_results = StepOutput()
        self.name = name
        self.num_models = num_models
        self.num_bins = num_bins
        self.num_classes = num_classes

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        outputs = self(batch.img)
        softmax = F.softmax(outputs, dim=-1)
        avg_softmax = softmax.mean(dim=2)

        predictions = self.calculate_classes(outputs)
        entropy = self.calculate_uncertainty(outputs)
        prediction_uncertainty = self.calculate_winning_confidence(outputs)
        step_output = StepOutput(
            raw_outputs=outputs.tolist(),
            predictions=predictions.tolist(),
            entropy=entropy.tolist(),
            prediction_uncertainty=prediction_uncertainty.tolist(),
            avg_softmax=avg_softmax.tolist(),
            labels=batch.label.tolist(),
            global_dataset_index=batch.local_index.tolist(),
            local_dataset_index=batch.local_index.tolist(),
        )
        self.test_results.extend(step_output)

    def on_test_epoch_end(self):

        output_test_df = dict(
            label_file=[str(Path(self.trainer.datamodule.test_dataloader().dataset.annotation_file)) for _ in range(len(self.test_results.global_dataset_index))],
            image_root=[str(Path(self.trainer.datamodule.test_dataloader().dataset.img_dir)) for _ in range(len(self.test_results.global_dataset_index))],
            local_dataset_index=self.test_results.local_dataset_index,
            global_dataset_index=self.test_results.global_dataset_index,
        )
        # add uncertainty and predictions
        for bin_index in range(self.model.num_bins):
            output_test_df[f"uncertainty_{bin_index}"] = [item[bin_index] for item in self.test_results.prediction_uncertainty]
            output_test_df[f"entropy_{bin_index}"] = [item[bin_index] for item in self.test_results.entropy]
            output_test_df[f"output_{bin_index}"] = [item[bin_index] for item in self.test_results.predictions]
            for class_index in range(self.model.num_classes):
                output_test_df[f"softmax_{bin_index}_{class_index}"] = [item[bin_index][class_index] for item in self.test_results.avg_softmax]
        output_test_df = pd.DataFrame(output_test_df)
        output_test_df.to_csv(str(Path(self.trainer.log_dir) / 'test_results.csv'), index=False)

        with open(str(Path(self.trainer.log_dir) / 'test_results.pkl'), 'wb') as f:
            import pickle
            pickle.dump(self.test_results, f)

        self.test_results = StepOutput()

    @staticmethod
    def calculate_uncertainty(outputs):
        return FullEnsembleTorch.calculate_uncertainty(outputs)

    @staticmethod
    def calculate_classes(outputs):
        return FullEnsembleTorch.calculate_classes(outputs)

    @staticmethod
    def calculate_winning_confidence(outputs):
        return FullEnsembleTorch.calculate_winning_confidence(outputs)

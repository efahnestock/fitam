
"""
A toy agent for the 2 test environment
It consumes the laser scan for a sector and predicts the cost map correction factor for that sector
"""
import pytorch_lightning as pl
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torch
import logging

from typing import Optional
from pathlib import Path
from fitam.learning.dino_model_torch import DinoModelTorch


class DinoModel(pl.LightningModule):
    # NOTE: input size should not include the batch dimension
    # output_size: num_bins if regression, num_bins * num_classes if classification
    def __init__(self, output_size: int, use_features: bool = False, num_classes: Optional[int] = None, num_models: int = 1,
                 dinov2_weight_path: Optional[Path] = None,
                 dinov2_model_repo_path: Optional[Path] = None,
                 name="dinov2",  # FIXED, this is the name of the model. Here so save_hyperparameters works
                 ):
        super(DinoModel, self).__init__()
        self.pass_weight_to_loss = False
        self.num_classes = num_classes
        self.use_classification = False
        self.class_weights = None
        self.num_models = num_models
        self.entropy_weight = None
        if self.num_classes is not None:
            self.use_classification = True
        self.model = DinoModelTorch(output_size, use_features, dinov2_weight_path, dinov2_model_repo_path, self.num_classes, self.num_models)
        self.name = name
        self.learning_rate = 0.001
        self.internal_logger = logging.getLogger(__name__)
        self.loss = F.mse_loss

        if not self.use_classification:
            self.mae = torchmetrics.MeanAbsoluteError()
            self.train_bin_error = nn.ModuleList([torchmetrics.MeanAbsoluteError() for _ in range(output_size)])
            self.val_bin_error = nn.ModuleList([torchmetrics.MeanAbsoluteError() for _ in range(output_size)])
            self.test_bin_error = nn.ModuleList([torchmetrics.MeanAbsoluteError() for _ in range(output_size)])

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def to(self, device):
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
        return super().to(device)

    def extract_features(self, x):
        return self.model.extract_features(x)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        if self.model.use_features:
            outputs = self(batch.feature)
        else:
            outputs = self(batch.img)

        # outputs are (batch, num_models, num_classes, num_bins) (if classification) or (batch, num_models, num_bins) (if regression)
        # log_bin_errors("train_bin_error", self, outputs, batch.label, self.train_bin_error)
        loss = 0
        for i in range(self.num_models):
            loss_args = {
                "input": outputs[:, i, ...],  # only pass in the ith model's outputs
                "target": batch.label,
                "reduction": "none"
            }
            if self.pass_weight_to_loss:
                loss_args["weight"] = batch.coverage
            if self.class_weights is not None:
                loss_args["class_weight"] = self.class_weights
            if self.entropy_weight is not None:
                loss_args["entropy_weight"] = self.entropy_weight
            loss += self.loss(**loss_args)

        self.training_step_outputs.append((loss, batch.local_index, batch.global_index, batch.label, outputs))

        loss = loss.mean()
        self.log('loss/train_loss', loss)

        return loss

    def on_train_epoch_end(self):
        # log_extreme_images("train_images", self, self.trainer.datamodule.train_dataloader(),
        #    self.training_step_outputs)
        # log_loss_histograms("train_histograms", self, self.training_step_outputs)
        # save_results_df("final_train_results", self.logger, self.trainer.datamodule.train_dataloader(), self.training_step_outputs)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        if self.model.use_features:
            outputs = self(batch.feature)
        else:
            outputs = self(batch.img)
        # log_bin_errors("val_bin_error", self, outputs, batch.label, self.val_bin_error)
        loss = 0.0
        for i in range(self.num_models):
            loss_args = {
                "input": outputs[:, i, ...],
                "target": batch.label,
                "reduction": "none"
            }
            if self.pass_weight_to_loss:
                loss_args["weight"] = batch.coverage
            if self.class_weights is not None:
                loss_args["class_weight"] = self.class_weights
            loss += self.loss(**loss_args)
        self.validation_step_outputs.append((loss, batch.local_index, batch.global_index, batch.label, outputs))
        loss = loss.mean()
        self.log('loss/val_loss', loss, batch_size=batch.img.shape[0], prog_bar=True, on_step=False, on_epoch=True, reduce_fx="mean")

    def on_validation_epoch_end(self):
        # log_extreme_images("val_images", self, self.trainer.datamodule.val_dataloader(),
        #    self.validation_step_outputs)
        # log_loss_histograms("val_histograms", self, self.validation_step_outputs)
        # save_results_df("final_val_results", self.logger, self.trainer.datamodule.val_dataloader(), self.validation_step_outputs)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        # always use images for testing
        use_features_local = self.model.use_features
        self.model.use_features = False
        outputs = self(batch.img)
        self.model.use_features = use_features_local

        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        # log_extreme_images("test_images", self, self.trainer.datamodule.test_dataloader(),
        #                     self.test_step_outputs)
        # log_loss_histograms("test_histograms", self, self.test_step_outputs)
        # save_results_df("final_test_results", self.logger, self.trainer.datamodule.test_dataloader(), self.test_step_outputs)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

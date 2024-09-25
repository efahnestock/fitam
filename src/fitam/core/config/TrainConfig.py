
from dataclasses import field
from dataclasses import dataclass
from torchvision import transforms
from typing import Optional


@dataclass
class TrainConfig:
    model_name: str = 'cnn_v1'  # cnn_v1 or dino
    loss_name: str = 'mse'  # mse or wmse
    weight_by_class: bool = False  # if true, will weight classes by their inverse frequency. Only applicable to CE/classification
    entropy_weight: Optional[float] = None  # if not None, will apply a uniform distribution to unknown bins and weight CE loss by this value.
    use_extracted_features: bool = False
    num_epochs: int = 1000

    batch_size: int = 1024
    tune_batch_size: bool = False
    learning_rate: float = 0.00229
    tune_learning_rate: bool = False

    accumulate_grad_batches: int = 4
    check_val_every_n_epoch: int = 1
    overfit_batches: float = 0  # if nonzero, will overfit to this many batches of data

    shuffle_dataset: bool = True
    num_workers: int = 20

    steps_per_printout: int = 2000

    min_value: float = 1
    max_value: float = 1 / 0.1

    # loss function?
    # optimizer?

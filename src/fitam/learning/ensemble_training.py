import tqdm
import torch
import time
from typing import Optional
import pickle
from pytorch_lightning import seed_everything
from pathlib import Path
import torch.nn as nn
from typing import NamedTuple
from torch.utils.data import random_split, DataLoader
import numpy as np
from copy import deepcopy
from fitam import DATASETS_DIR
from fitam.learning.pl_img_dataloader import PlImgDataloader
from fitam.learning.load_by_name import weighted_ce_loss
from fitam.learning.cnn_model_ablation_torch import HeadType, CNNModelTorch
from fitam.learning.bin_ensemble_torch import MemberModelName


class TrainingParams(NamedTuple):
    learning_rate: float = 1e-3
    batch_size: int = 64
    random_seed: int = 42
    training_percentage: float = 0.8
    patience: int = 10
    num_workers: int = 10
    device: str = "cuda:0"

class TrainMetrics(NamedTuple):
    training_loss: list
    training_accuracy: list
    validation_loss: list
    validation_accuracy: list

def get_class_weights(all_labels, num_classes)->list:
    total = len(all_labels)
    all_labels = np.asarray(all_labels)
    weights = np.zeros(num_classes,)
    counts = np.zeros(num_classes,)
    for i in range(num_classes):
        if (all_labels == i).sum() < 0.1:
            raise RuntimeError(f"No labels with class {i} found")
        weights[i] = total / ((all_labels == i).sum() * num_classes)
        counts[i] = (all_labels == i).sum()
    return weights, counts

def train_model(
        model: nn.Module,
        loss_fn: callable,
        optimizer,
        train_loader,
        val_loader,
        use_feature: bool,
        patience,
        device,
):
    model.to(device)
    best_val_loss = np.inf
    best_model_weights = deepcopy(model.state_dict())
    last_epoch_improved_val = 0
    current_epoch = 0

    # get class weights
    all_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(train_loader, disable=True):
            all_labels.extend(batch.label.tolist())
    class_weights, class_counts = get_class_weights(all_labels, model.num_classes)
    del all_labels
    class_weights = torch.from_numpy(class_weights).to(device).to(torch.float)

    train_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []


    model.train()
    for epoch in tqdm.tqdm(range(10000)):
        # train loop
        # print("starting epoch ", current_epoch)
        total_train_loss = 0
        all_pred = []
        all_labels = []
        times = dict(
            forward=0,
            to_device=0,
            loss_calc=0,
            backwards=0,
            stepping=0,
            validation=0,
            get_batch=0,

        )
        end_time = time.time()
        for batch in tqdm.tqdm(train_loader, disable=True):
            start_time = time.time()
            times['get_batch'] += start_time - end_time
            x = batch.feature.to(device) if use_feature else batch.img.to(device)
            y = batch.label.to(device)
            c = batch.coverage.to(device)
            t2 = time.time()
            times['to_device'] += t2 - start_time
            pred = model(x)
            t3 = time.time()
            times['forward'] += t3 - t2
            all_labels.extend(y.tolist())
            all_pred.extend(pred.argmax(dim=-1).tolist())
            # print(pred.shape, y.shape, c.shape, class_weights.shape, y.min(), y.max())
            loss = loss_fn(pred, y, c, class_weights)
            t4 = time.time()
            times['loss_calc'] += t4 - t3
            loss.backward()
            t5 = time.time()
            times['backwards'] += t5 - t4
            total_train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            end_time = time.time()
            times['stepping'] += end_time - t5
        train_accuracy.append((torch.as_tensor(all_pred) == torch.as_tensor(all_labels)).sum() / len(all_pred))
        # print(epoch, times)

        # calculate val
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            all_pred = []
            all_labels = []
            for batch in val_loader:
                x = batch.feature.to(device) if use_feature else batch.img.to(device)
                y = batch.label.to(device)
                c = batch.coverage.to(device)
                pred = model(x)
                all_pred.extend(pred.argmax(dim=-1).tolist())
                all_labels.extend(y.tolist())
                loss = loss_fn(pred, y, c, class_weights)
                total_val_loss += loss.item()
            val_accuracy.append((torch.as_tensor(all_pred) == torch.as_tensor(all_labels)).sum() / len(all_pred))
        if total_val_loss < best_val_loss:
            # print("improved validation loss from", best_val_loss, "to", total_val_loss)
            best_val_loss = total_val_loss
            best_model_weights = deepcopy(model.state_dict())
            last_epoch_improved_val = current_epoch

        train_losses.append(total_train_loss)
        val_losses.append(total_val_loss)

        current_epoch += 1
        if (current_epoch - last_epoch_improved_val > patience):
            # print(f"exiting on epoch {current_epoch}, as no improvement since epoch {last_epoch_improved_val}. Done")
            break
    model.load_state_dict(best_model_weights)

    output = TrainMetrics(
        train_losses, train_accuracy, val_losses, val_accuracy
    )
    return output



def train_ensemble(model_creation_function: callable,
                   dataloader: DataLoader,
                   num_models: int,
                   use_feature: bool,
                   loss_fn: callable,
                   training_params: TrainingParams,
                   save_dir: Optional[Path] = None) -> nn.Module:
    models = []
    training_metrics = []
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_models):
        random_seed = training_params.random_seed + i
        seed_everything(random_seed, workers=True)
        print(save_dir / f"model_{i:02d}.pt")
        if save_dir and (save_dir / f'model_{i:02d}.pt').exists():
            print("Model", i, "already exists, skipping")
            continue
        else:
            print(f"Training model {i+1}/{num_models}. Random seed: {random_seed}")
        print("Dataloader has total size", len(dataloader))
        train_loader, val_loader = random_split(dataloader, [training_params.training_percentage, 1-training_params.training_percentage], generator=torch.Generator().manual_seed(random_seed))
        train_loader = DataLoader(train_loader, training_params.batch_size, shuffle=True)  #, persistent_workers=True, num_workers=training_params.num_workers)
        val_loader = DataLoader(val_loader, training_params.batch_size, shuffle=False)  #, persistent_workers=True, num_workers=training_params.num_workers)
        model = model_creation_function()
        optimizer = torch.optim.SGD(model.parameters(), lr=training_params.learning_rate)
        metrics = train_model(
            model,
            loss_fn,
            optimizer,
            train_loader,
            val_loader,
            use_feature,
            training_params.patience,
            training_params.device
        )
        models.append(model)
        training_metrics.append(metrics)
        if save_dir:
            torch.save(model.state_dict(), save_dir / f'model_{i:02d}.pt')
            with open(save_dir / f'model_{i:02d}_metrics.pkl', 'wb') as f:
                pickle.dump(metrics, f)

    return models, training_metrics


class LinearModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.net(x)


def create_member_model(member_model_name: MemberModelName, member_model_args: dict = None):
    if member_model_name == MemberModelName.PRETRAINED_RESNET_INDIVIDUAL:
        return CNNModelTorch(**member_model_args, num_models=1, head_type=HeadType.PRETRAINED)
    elif member_model_name == MemberModelName.UNTRAINED_RESNET_INDIVIDUAL:
        return CNNModelTorch(**member_model_args, num_models=1, head_type=HeadType.UNTRAINED)
    elif member_model_name == MemberModelName.LINEAR:
        return LinearModel(**member_model_args)
    else:
        raise RuntimeError(f"Unrecognized enum member: {member_model_name}")


def train_ensemble_for_farfield(num_bins, num_models, data_module_args,
                                output_dir, member_model_name, member_model_args: dict = None,
                                training_params=TrainingParams()):
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    output_dir = Path(output_dir)
    use_feature = False  # for resnet-based pipelines
    if member_model_name == MemberModelName.LINEAR:
        use_feature = True

    for bin_index in range(num_bins):
        if (output_dir / f"{bin_index:02d}_bin").exists() and (output_dir / f"{bin_index:02d}_bin" / f"model_{num_models-1:02d}.pt").exists():
            print("SKIPPING BIN", bin_index)
            continue
        print("TRAINING BIN", bin_index)
        args = data_module_args.copy()
        args['label_index'] = bin_index
        args['seed'] = bin_index
        torch.set_float32_matmul_precision('medium')
        seed_everything(bin_index, workers=True)
        if not use_feature:
            # if using CNN head, don't return features/return images instead
            if 'rel_features_dir' in data_module_args:
                del args['rel_features_dir']
        data_module = PlImgDataloader(**args)
        data_module.setup()

        def create_model():
            return create_member_model(member_model_name, member_model_args)

        _ = train_ensemble(
            model_creation_function=create_model,
            dataloader=data_module.train_and_val,
            num_models=num_models,
            use_feature=use_feature,
            loss_fn=weighted_ce_loss,
            training_params=training_params,
            save_dir=output_dir / f"{bin_index:02d}_bin"
        )

def test():
    dataset_root = DATASETS_DIR / 'balt_standard'
    data_module_args = dict(
        batch_size=None,
        annotation_file=dataset_root / 'balanced_semantic_dataset.csv',
        img_dir=dataset_root,
        num_workers=8,
        label_clip=None,
        label_index=None,
        rel_features_dir=Path('.'),
        request_coverage=True,
        seed=0,
        use_classification=True,
        store_in_memory=True,
    )

    train_ensemble_for_farfield(
        num_bins=3,
        num_models=15,
        data_module_args=data_module_args,
        output_dir="/tmp/ensemble",
        member_model_name=MemberModelName.LINEAR,
        member_model_args=dict(num_classes=3, num_features=768),
        training_params=TrainingParams(
            device="cuda:0",
            batch_size=128
        )
    )

if __name__ == '__main__':
    test()
from pathlib import Path
import logging
import os
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from fitam.core.common import create_dir, load_json_config
from fitam.learning.fully_separate_ensemble import FullEnsemble
from fitam.learning.bin_ensemble_torch import MemberModelName
from fitam.learning.load_by_name import load_model
from fitam.learning.pl_img_dataloader import PlImgDataloader
from fitam.learning.ensemble_training import train_ensemble_for_farfield, TrainingParams

logger = logging.getLogger(__name__)

def save_model_checkpoint(model:pl.LightningModule, save_path:Path)->None:
    trainer = pl.Trainer(default_root_dir="/tmp")
    trainer.strategy.connect(model)
    trainer.save_checkpoint(str(save_path))
    return


def train_ensemble_pipeline(train_config_path: Path,
                            save_path: Path,
                            radial_map_config_path: Path,
                            num_ensemble_members: int,
                            dataset_csv_path: Path,
                            member_model_name: MemberModelName,
                            member_model_args: Optional[dict] = None,
                            num_bins: Optional[int] = None,
                            device: str = "cuda",
                            store_in_memory: bool = True,
                            ) -> None:

    train_config = load_json_config(train_config_path)
    radial_map_config = load_json_config(radial_map_config_path)

    data_module_args = dict(
        batch_size=train_config.batch_size,
        annotation_file=dataset_csv_path,
        img_dir=dataset_csv_path.parent,
        num_workers=1,
        label_clip=None,
        label_index=None,
        rel_features_dir=Path('.'),
        request_coverage=True,
        use_classification=True,
        get_semantic_classes=False,
        store_in_memory=store_in_memory,
    )
    create_dir(save_path)
    if num_bins is None:
        num_bins = radial_map_config.farfield_config.num_range_bins
    if member_model_args is None:
        member_model_args = {}
    if "num_classes" not in member_model_args:
        member_model_args['num_classes'] = radial_map_config.farfield_config.classification_config.num_classes
    train_ensemble_for_farfield(
        num_bins=num_bins,
        num_models=num_ensemble_members,
        data_module_args=data_module_args,
        output_dir=save_path,
        member_model_name=member_model_name,
        member_model_args=member_model_args,
        training_params=TrainingParams(
            batch_size=256,
            device=device
        ),
    )
    ensemble = FullEnsemble(num_ensemble_members, num_bins=num_bins,
                            num_classes=member_model_args['num_classes'],
                            model_name=member_model_name,
                            model_args=member_model_args)
    ensemble.model.load_from_dir(save_path)
    save_model_checkpoint(ensemble, save_path / f'{save_path.name}.ckpt')


def test_individual_network(train_config_path: os.PathLike,
                            model_checkpoint_path: os.PathLike,
                            save_dir: os.PathLike,
                            dataset_csv_path: os.PathLike,
                            # if specified, dataset reduces to only valid samples of this label
                            bin_index: Optional[int] = None,
                            seed: int = 42) -> None:
    """
    Test an individual network. 
    """
    seed_everything(seed, workers=True)
    create_dir(save_dir)
    # load the trial config
    train_config = load_json_config(train_config_path)

    data_module_args = dict(
        batch_size=train_config.batch_size,
        annotation_file=dataset_csv_path,
        img_dir=dataset_csv_path.parent,
        test_annotation_file=dataset_csv_path,
        test_img_dir=dataset_csv_path.parent,
        num_workers=24,
        label_clip=None,
        label_index=None,
        request_coverage=True,
        use_classification=True,
        percent_validation=0.0,
        seed=seed,
    )
    if bin_index is not None:
        data_module_args['label_index'] = bin_index
    # set up dataloader
    data_module = PlImgDataloader(**data_module_args)
    data_module.setup()
    # set up model
    model = load_model(
        checkpoint_path=model_checkpoint_path,
        get_lightning=True,
        use_features=False
    )

    trainer: pl.Trainer = pl.Trainer(accelerator='auto',
                                     max_epochs=train_config.num_epochs,
                                     log_every_n_steps=1,
                                     check_val_every_n_epoch=train_config.check_val_every_n_epoch,
                                     default_root_dir=save_dir,
                                     enable_checkpointing=True,
                                     deterministic=True,
                                     callbacks=[
                                         TQDMProgressBar(refresh_rate=1),
                                     ],
                                     )
    trainer.test(model=model, datamodule=data_module)

    # export_everything(Path(trainer.log_dir) / 'final_test_results.csv', trainer.log_dir )
    return


def test_model_cli():
    import argparse
    parser = argparse.ArgumentParser(description='Test a network')
    parser.add_argument('-m', '--model_checkpoint_path', type=str,
                        required=True, help='Path to the model checkpoint')
    parser.add_argument('-c', '--train_config_path', type=str,
                        required=True, help='Path to the training config')
    parser.add_argument('-d', '--dataset_csv_path', type=str,
                        required=True, help='Path to the dataset csv')

    args = parser.parse_args()

    test_individual_network(train_config_path=args.train_config_path,
                            model_checkpoint_path=args.model_checkpoint_path,
                            save_dir=Path('/tmp'),
                            dataset_csv_path=Path(args.dataset_csv_path)
                            )

def train_model_cli():
    import argparse, json
    parser = argparse.ArgumentParser(description='Test a network')
    parser.add_argument('-c', '--dataset_csv_path', type=str,
                        required=True)
    parser.add_argument('-s', '--save_path', type=str,
                        required=True)
    parser.add_argument('-n', '--num_ensemble_members', type=int,
                        required=True)
    parser.add_argument('-r', '--radial_map_config_path', type=str,
                        required=True)
    parser.add_argument('-t', '--train_config_path', type=str,
                        required=True)
    parser.add_argument('-m', '--member_model_name', type=str,
                        required=True)
    parser.add_argument('--num_bins', type=str,
                        required=False)
    parser.add_argument('-d', '--device', type=str,
                        required=False, default="cuda:0")
    parser.add_argument('--member_model_args', type=json.loads,
                        required=False, default=None)
    parser.add_argument('--copy_dataset', action="store_true", default=False)
    parser.add_argument('--store_in_memory', action="store_true", default=False)



    args = parser.parse_args()

    copy_path = Path("/state/partition1/user/efahnestock")
    if args.copy_dataset:
        import shutil
        copy_path.mkdir(exist_ok=True, parents=True)
        print("copying dataset to fast spot")
        shutil.copy(args.dataset_csv_path, str(copy_path))
        print("done copying dataset")

    train_ensemble_pipeline(
        dataset_csv_path=Path(args.dataset_csv_path),
        num_ensemble_members=args.num_ensemble_members,
        radial_map_config_path=Path(args.radial_map_config_path),
        save_path=Path(args.save_path),
        train_config_path=Path(args.train_config_path),
        member_model_name=MemberModelName[args.member_model_name],
        num_bins=None,
        member_model_args=args.member_model_args,
        store_in_memory=args.store_in_memory,
    )

if __name__ == "__main__":
    train_model_cli()
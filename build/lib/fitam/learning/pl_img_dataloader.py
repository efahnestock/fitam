import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torch
import pandas as pd
from torchvision import transforms
from torchvision.io import read_image
from typing import Optional
from pathlib import Path
from fitam.learning.img_sim_dataloader import ImgSimulatedDataset, load_images_from_zip, load_csv_from_zip
from fitam.learning.preprocessing_transforms import get_default_transforms


def get_default_transform_from_image_path(example_image_path, model_name='dinov2'):
    # load example image to get image size
    if issubclass(type(example_image_path), Path):
        example_image_path = str(example_image_path)
    img = read_image(example_image_path)
    img_size = img.shape[1:]
    return get_default_transform_from_image_shape(img_size, model_name)
    # add center crop to closest increments of 14 pixels


def get_default_transform_from_image_shape(img_size, model_name='dinov2'):
    return get_default_transforms(model_name, image_size=img_size)


class PlImgDataloader(pl.LightningDataModule):
    def __init__(self, batch_size: int,
                 annotation_file: str,
                 img_dir: str,
                 label_index: Optional[int] = None,
                 train_transform=None,  # NOTE: VAL USES TRAIN TRANSFORMS TOO
                 non_train_transform=None,
                 label_clip: tuple = None,
                 num_workers: int = 6,
                 percent_validation: float = 0.2,
                 test_annotation_file: str = None,
                 rel_features_dir: str = None,  # relative to img_dir
                 test_img_dir: str = None,
                 request_coverage: bool = False,
                 use_classification: bool = False,
                 get_semantic_classes: bool = False,
                 store_in_memory: bool = False,
                 seed: int = 42):
        super().__init__()
        self.batch_size = batch_size
        self.rel_features_dir = rel_features_dir
        self.percent_validation = percent_validation
        self.get_semantic_classes = get_semantic_classes
        self.num_workers = num_workers
        self.use_classification = use_classification
        self.img_dir = img_dir
        self.label_index = label_index
        self.annotation_file = annotation_file
        self.test_annotation_file = test_annotation_file
        self.test_img_dir = test_img_dir
        self.label_clip = label_clip
        self.request_coverage = request_coverage
        self.seed = seed
        self.store_in_memory = store_in_memory

        self.train_transform = train_transform
        self.non_train_transform = non_train_transform

        self.save_hyperparameters(ignore=['train_transform', 'non_train_transform'])

    def setup(self, stage: str = ""):
        self.train, self.val = None, None
        if self.annotation_file is not None:  # if no train/validation data
            import time
            # print("starting to read csv", time.time())
            if Path(self.annotation_file).suffix == ".zip":
                csv = load_csv_from_zip(self.annotation_file)
                img = load_images_from_zip(csv, self.annotation_file, None, True)
                if self.train_transform is None and self.rel_features_dir is None:  # only get train transform if we're not using extracted features
                    self.train_transform = get_default_transform_from_image_shape(img.shape[1:])
            else:
                csv = pd.read_csv(self.annotation_file)
                # print("done reading csv", time.time())
                # add center crop to closest increments of 14 pixels
                img_path = str(self.img_dir / csv.iloc[0]['img_path'])
                if self.train_transform is None and self.rel_features_dir is None:  # only get train transform if we're not using extracted features
                    self.train_transform = get_default_transform_from_image_path(img_path)

            # print("setting up datasets", time.time())
            self.train_and_val = ImgSimulatedDataset(annotation_file=self.annotation_file,
                                                     img_dir=self.img_dir,
                                                     label_index=self.label_index,
                                                     transform=transforms.Compose(self.train_transform),
                                                     label_clip=self.label_clip,
                                                     return_indices=True,
                                                     use_extracted_features=self.rel_features_dir is not None,
                                                     extracted_features_dir=self.img_dir / self.rel_features_dir if self.rel_features_dir is not None else None,
                                                     request_coverage=self.request_coverage,
                                                     use_classification=self.use_classification,
                                                     get_semantic_classes=self.get_semantic_classes,
                                                     store_in_memory=self.store_in_memory,
                                                     )
            self.train, self.val = random_split(self.train_and_val,
                                                [1-self.percent_validation, self.percent_validation],
                                                generator=torch.Generator().manual_seed(self.seed)  # fix the seed for reproducibility
                                                )
            # print("done setting up datasets", time.time())
        self.test = None
        if self.test_annotation_file is not None:
            img_path = str(self.img_dir / csv.iloc[0]['img_path'])
            if self.non_train_transform is None:
                self.non_train_transform = get_default_transform_from_image_path(img_path)
            self.test = ImgSimulatedDataset(annotation_file=self.test_annotation_file,
                                            img_dir=self.test_img_dir,
                                            label_index=self.label_index,
                                            transform=transforms.Compose(self.non_train_transform),
                                            label_clip=self.label_clip,
                                            return_indices=True,
                                            use_extracted_features=self.rel_features_dir is not None,
                                            extracted_features_dir=self.test_img_dir / self.rel_features_dir if self.rel_features_dir is not None else None,
                                            request_coverage=self.request_coverage,
                                            use_classification=self.use_classification,
                                            get_semantic_classes=self.get_semantic_classes,
                                            store_in_memory=self.store_in_memory
                                            )
        return

    def train_dataloader(self):
        if self.train is None:
            return None
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,
                          generator=torch.Generator().manual_seed(self.seed)  # fix the seed for reproducibility
                          )

    def val_dataloader(self):
        if self.val is None:
            return None
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          generator=torch.Generator().manual_seed(self.seed)  # fix the seed for reproducibility
                          )

    def test_dataloader(self):
        if self.test is None:
            return None
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def __str__(self) -> str:
        out = f"PlImgDataLoader("
        if self.train:
            out += f"train({len(self.train)}),"
        else:
            out += f"train(0),"
        if self.val:
            out += f"val({len(self.val)}),"
        else:
            out += f"val(0),"
        if self.test:
            out += f"test({len(self.test)}),"
        else:
            out += f"test(0)"
        out += ")"
        return out

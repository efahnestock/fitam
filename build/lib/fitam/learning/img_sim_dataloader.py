
from __future__ import annotations

import tqdm
import torch
import zipfile
from pathlib import Path
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from typing import Optional
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
# debugging
import matplotlib.pyplot as plt
import logging
from typing import NamedTuple


class DatasetOutput(NamedTuple):
    label: torch.Tensor
    img: Optional[torch.Tensor] = 0  # set to 0 if not requested
    local_index: int = 0  # set to 0 if not requested
    global_index: int = 0  # set to 0 if not requested
    coverage: Optional[torch.Tensor] = 0  # set to 0 if not requested
    feature: Optional[torch.Tensor] = 0  # set to 0 if not requested
    semantic_class: Optional[torch.Tensor] = 0  # set to 0 if not requested

    def __repr__(self):
        return f"DatasetOutput(img.shape={self.img.shape}, " + \
            f"label.shape={self.label.shape}, " + \
            (f"local_index={self.local_index}, " if type(self.local_index) == int else f"local_index.shape={self.local_index.shape}, ") + \
            (f"global_index={self.global_index}, " if type(self.global_index) == int else f"global_index.shape={self.global_index.shape}, ") + \
            (f"coverage={self.coverage}, " if type(self.coverage) == int else f"coverage.shape={self.coverage.shape}, ") + \
            (f"feature={self.feature})" if type(self.feature) == int else f"feature.shape={self.feature.shape})") + \
            (f"semantic_class={self.semantic_class})" if type(self.semantic_class) == int else f"semantic_class.shape={self.semantic_class.shape})")


def parse_semantic_class_df(df: pd.DataFrame, relevant_classes=None) -> tuple[dict, dict]:
    """
    Return int->name mapping,
    and
    numpy -> (df_idx, bin_idx, class_count)
    """
    if relevant_classes is None:
        relevant_classes = ['young_forest', 'forest', 'grass', 'field', 'road', 'marsh', 'beach', 'water']
    name_to_index = {}
    counter = 0
    num_bins = 0
    for row_name in df.columns:
        parts = row_name.split("_")
        bin_idx = int(parts[1])
        class_name = "_".join(parts[2:])
        if class_name not in relevant_classes:
            continue
        if class_name not in name_to_index:
            name_to_index[class_name] = counter
            counter += 1
        num_bins = max(num_bins, bin_idx)
    arr = np.zeros((len(df), num_bins+1, len(name_to_index)))
    for row_name in df.columns:
        parts = row_name.split("_")
        bin_idx = int(parts[1])
        class_name = "_".join(parts[2:])
        if class_name not in name_to_index:
            continue
        class_idx = name_to_index[class_name]

        arr[:, bin_idx, class_idx] = df[row_name].values

    index_to_name = {v: k for k, v in name_to_index.items()}
    return index_to_name, arr


def _make_key_from_args(args: dict) -> str:
    key = ""
    for k, v in args.items():
        key += f"{k}={v}|"
    return key[:-1]

def load_images_from_zip(valid_rows: pd.DataFrame,
                         zip_file_path: Path,
                         transform=None, just_one_image: bool = False) -> torch.tensor:
    with zipfile.ZipFile(zip_file_path) as zf:
        p = zipfile.Path(zf)
        dataset_root = Path(next(iter(p.iterdir())).name)
        pil_tf = transforms.PILToTensor()

        def read_img(path):
            with zf.open(path) as f:
                img = Image.open(f).convert('RGB')
            img = convert_image_dtype(pil_tf(img), torch.float32)
            return img

        image_paths = [str(dataset_root / x) for x in valid_rows['img_path'].values]
        sample_img = read_img(image_paths[0])
        if transform:
            sample_img = transform(sample_img)
        if just_one_image:
            return sample_img
        img_shape = sample_img.shape
        image_tensor = torch.zeros((len(image_paths), *img_shape), dtype=torch.float)
        for i, path in tqdm.tqdm(enumerate(image_paths), total=len(image_paths)):
            img = read_img(path)
            if transform:
                img = transform(img)
            image_tensor[i] = img
        return image_tensor


def load_features_or_images_from_dataset(valid_rows: pd.DataFrame,
                                         dataset_root_path: Path,
                                         load_images: bool = False,
                                         transform=None) -> torch.tensor:

    if load_images:
        print("loading in images")
        image_paths = [str(dataset_root_path / x) for x in valid_rows['img_path'].values]
        sample_img = convert_image_dtype(read_image(image_paths[0]), dtype=torch.float32)
        if transform:
            sample_img = transform(sample_img)
        img_shape = sample_img.shape
        image_tensor = torch.zeros((len(image_paths), *img_shape), dtype=torch.float)
        for i, path in tqdm.tqdm(enumerate(image_paths), total=len(image_paths)):
            img = convert_image_dtype(read_image(path), dtype=torch.float32)
            if transform:
                img = transform(img)
            image_tensor[i] = img
        return image_tensor

    else:
        feature_paths = [dataset_root_path / Path(x) for x in valid_rows['feature_name'].values]
        sample_feature_shape = torch.load(feature_paths[0]).shape
        feature_tensor = torch.zeros((len(feature_paths), *sample_feature_shape), dtype=torch.float)
        print("loading in features")
        for i, path in tqdm.tqdm(enumerate(feature_paths), total=len(feature_paths)):
            feature_tensor[i] = torch.load(path).type(torch.float32)
        return feature_tensor


def load_csv_from_zip(zip_path, dataset_name="balanced_semantic_dataset.csv"):
    with zipfile.ZipFile(zip_path) as zf:
        p = zipfile.Path(zf)
        dataset_root = (next(iter(p.iterdir()))).name
        with zf.open(dataset_root+"/" + dataset_name) as f:
            csv = pd.read_csv(f)
    return csv


class ImgSimulatedDataset(Dataset):

    """
    Each training pair is a tuple of (tensor[1, 3, h, w]: perspective_image, tensor[1, n_bins]: average costmap correction vector)

    NOTE: We make some sad assumptions in this code. They gather shame below
    - We assume that the resolution of the djikstra map is 1m per pixel
    """

    def decode_str(self, bytestr):
        return str(bytestr, encoding='utf-8')

    def __init__(self,
                 annotation_file: str,
                 img_dir: str,
                 label_index: Optional[int] = None,  # if specified, dataset reduces to only valid samples of this label
                 indexes: list = None,
                 transform=None,
                 label_clip: Optional[tuple] = None,
                 return_indices: bool = False,
                 request_coverage: bool = False,
                 # extracted features options
                 use_extracted_features: bool = False,
                 extracted_features_dir: str = None,
                 use_classification: bool = False,
                 get_semantic_classes: bool = False,
                 store_in_memory: bool = False):
        """
        """
        self.logger = logging.getLogger(__name__)
        self.return_indices = return_indices
        self.use_classification = use_classification
        self.request_coverage = request_coverage
        self.annotation_file = annotation_file
        self.get_semantic_classes = get_semantic_classes
        print("Setting up dataloader with annotation file path ", self.annotation_file)
        self.from_zip_file = False
        if Path(self.annotation_file).suffix == ".zip":  # if zip file extraction
            assert store_in_memory, "Need to store in memory if you're working from a zip file"
            self.from_zip_file = True
            print("loading from zip file")
            self.labels_df = load_csv_from_zip(self.annotation_file)
        else:
            self.labels_df = pd.read_csv(self.annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_clip = label_clip
        self.label_dtype = torch.float32

        if self.get_semantic_classes:
            print("loading extra.csv")
            extra_path = Path(self.annotation_file).parent / "extra.csv"
            self.semantics_df = pd.read_csv(extra_path)
            print("done loading extra.csv")

        if label_index is not None and not self.use_classification:
            raise RuntimeError("label_index can only be specified if use_classification is True")
        if self.use_classification:
            self.label_dtype = torch.long

        if indexes is not None:
            self.indexes = np.asarray(indexes, dtype=int)
        else:
            self.indexes = np.arange(len(self.labels_df))
        if len(self.indexes) == 0:
            # just keep empty dataset
            return
        assert self.indexes.min() >= 0 and self.indexes.max() < len(self.labels_df), \
            f"Invalid indexes: {self.indexes.min()}, {self.indexes.max()}, {len(self.labels_df)}"

        # print("before modifying df at all, size is ", len(self.labels_df))
        self.labels_df = self.labels_df.iloc[self.indexes]
        if self.get_semantic_classes:
            self.semantics_df = self.semantics_df.iloc[self.indexes]
        # print("after trimming to supplied indexes, size is ", len(self.labels_df))

        num_bins = 0
        for col in self.labels_df.columns:
            if col.startswith("label_"):
                num_bins += 1

        ####
        # Trim dataset to only include single label index, if specified
        ####
        if label_index is not None:
            # print("Before trimming to label index, size is ", len(self.labels_df))
            assert label_index < num_bins, f"Invalid label index {label_index}. Dataset only has {num_bins} bins"
            label_column_name = f"label_{label_index}"
            coverage_column_name = f"coverage_{label_index}"
            mask = self.labels_df[coverage_column_name] > 0
            self.labels_df = self.labels_df[mask]
            if self.get_semantic_classes:
                self.semantics_df = self.semantics_df[mask]
                self.semantics_df.reset_index(inplace=True, drop=True)
            # print("Trimmed dataset to exclude cvoerage 0 samples. Size is now ", len(self.labels_df))
            # remove other label columns
            for col in self.labels_df.columns:
                if col.startswith("label_") and col != label_column_name:
                    self.labels_df.drop(col, axis=1, inplace=True)
                if col.startswith("coverage_") and col != coverage_column_name:
                    self.labels_df.drop(col, axis=1, inplace=True)
            # reset the index
            self.labels_df.reset_index(inplace=True, drop=True)
            # update num_bins
            num_bins = 1
            self.labels_df["label"] = self.labels_df[label_column_name]
            self.labels_df["coverage"] = self.labels_df[coverage_column_name]
            # print("Trimmed dataset to only include label index ", label_index)

        else:  # create np arrays from the separated columns
            # create label column with numpy array
            if num_bins > 0:  # if label_i columns exist, create a label column with them stacked
                self.labels_df["label"] = self.labels_df.apply(lambda row: np.array(
                    [row[f"label_{i}"] for i in range(num_bins)]), axis=1)

            # if coverage is in the dataframe
            if 'coverage_0' in self.labels_df.columns:
                self.labels_df["coverage"] = self.labels_df.apply(lambda row: np.array(
                    [row[f"coverage_{i}"] for i in range(num_bins)]), axis=1)

        self.class_weights = self.get_class_weights()

        if self.get_semantic_classes:
            self.semantic_idx_to_name, self.semantic_counts = parse_semantic_class_df(self.semantics_df)
            self.semantic_counts = torch.from_numpy(self.semantic_counts).int()
            if label_index is not None:
                self.semantic_counts = self.semantic_counts[:, label_index]

        self.use_extracted_features = use_extracted_features
        self.extracted_features_dir = extracted_features_dir

        self.store_in_memory = store_in_memory
        if self.store_in_memory:
            if self.from_zip_file:
                self.loaded_objects = load_images_from_zip(self.labels_df, annotation_file, transform)
            else:
                self.loaded_objects = load_features_or_images_from_dataset(self.labels_df, annotation_file.parent, not self.use_extracted_features,
                                                                           None if self.use_extracted_features else self.transform)
            print("loaded objects shape:", self.loaded_objects.shape)
            # if not self.use_extracted_features and self.transform:  # apply tf
            #     self.loaded_objects = self.transform(self.loaded_objects)

        self.labels = torch.tensor(self.labels_df['label'], dtype=self.label_dtype)
        # clip the label
        if self.label_clip is not None:
            self.labels = torch.clip(self.labels, *self.label_clip)
        if self.request_coverage:
            self.coverage = torch.tensor(self.labels_df['coverage'], dtype=torch.float32)

    @property
    def num_semantic_classes(self):
        if not self.get_semantic_classes:
            raise RuntimeError("Semantic classes not requested in creation")
        return len(self.semantic_idx_to_name)

    def generate_dataset_statistics(self) -> None:
        """
        Generate statistics about the dataset
        """
        raise NotImplementedError

    def get_class_weights(self) -> np.ndarray:
        """
        Get the class weights for the dataset.
        Count the number of each class, divide by the total
        """
        counts = {}
        for col in self.labels_df.columns:
            if col.startswith("label_"):
                for unique_value in self.labels_df[col].unique():
                    # remove zero coverage samples
                    col_index = col.split("_")[-1]
                    value_count = len(self.labels_df[(self.labels_df[col] == unique_value) & (self.labels_df[f"coverage_{col_index}"] > 0)])
                    counts[unique_value] = counts.get(unique_value, 0) + value_count

        total = sum(counts.values())
        weights = np.zeros(len(counts))
        for i in range(len(counts)):
            try:
                weights[i] = total / (counts[i] * len(counts))
            except KeyError:
                raise RuntimeError(f"Expected classes to range from 0->C-1. Got labels {counts.keys()}")
        # 1 - weights to get the inverse, upweight rare classes
        print("in get_class_weights, counts is ", counts, "weights are ", weights)
        return weights

    def plot_histos(self, bins: int = 100) -> list[plt.Axes]:
        raise NotImplementedError

    def __len__(self):
        return len(self.labels_df)

    def get_auxilliary_data(self, idx) -> dict:
        return self.labels_df.iloc[idx].to_dict()

    def __getitem__(self, idx):
        label = self.labels[idx]

        image = None
        features = None  # either preextracted features or image
        if self.use_extracted_features:
            # load the extracted features tensor
            if self.store_in_memory:
                features = self.loaded_objects[idx]
            else:
                feature_path = os.path.join(self.extracted_features_dir, self.labels_df['feature_name'].iloc[idx])
                features = torch.load(feature_path)
                features = features.squeeze(0)  # remove batch dimension
        else:
            # load the image
            if self.store_in_memory:
                image = self.loaded_objects[idx]  # transforms are applied above
            else:
                img_path = os.path.join(self.img_dir, self.labels_df['img_path'].iloc[idx])
                image = read_image(img_path)
                image = convert_image_dtype(image, torch.float)
                if self.transform:
                    image = self.transform(image)

        output = dict(label=label, local_index=idx)
        if self.get_semantic_classes:
            output['semantic_class'] = self.semantic_counts[idx]
        if self.request_coverage:
            output['coverage'] = self.coverage[idx]
        if self.use_extracted_features:
            output['feature'] = features
        else:
            output['img'] = image
        return DatasetOutput(**output)
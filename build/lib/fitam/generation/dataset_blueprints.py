from pathlib import Path
import numpy as np
import tqdm
import pandas as pd
import cv2
from typing import NamedTuple

from dataclasses import dataclass
from fitam import MAPS_DIR, IMAGES_DIR
from fitam.core.common import load_json_config, set_all_seeds, process_df_in_parallel, create_dir, cv2_img_to_float_ndarr
from fitam.mapping.land_cover_complex_map import LandCoverComplexMap
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.costmap_swath_library import SwathLibrary
from fitam.learning.reduce_space_to_cost import reduce_single_location_to_labels
from fitam.core.config.RadialMapConfig import FarFieldConfig
from fitam.generation.dataset_generation import save_cropped_images_from_pano, get_classes_in_bins


@dataclass
class ImageItem:
    panorama_path: Path
    radial_map_config_path: Path
    complex_map_path: Path
    position: tuple[float, float]
    yaw_range: tuple[float, float]


class BalancedDatasetConfig(NamedTuple):
    max_imbalance_ratio: float  # min_samples_class / max_samples_class = max_imbalance_ratio
    dominance_percentage: float  # a bin is marked as a semantic class if at least this percentage of the cells are of that class


def get_classes_and_labels(image_df_part: pd.DataFrame, radial_map_config_path, dataset_config_path):
    radial_map_config = load_json_config(radial_map_config_path)
    swath_library = SwathLibrary.from_radial_map_config(radial_map_config, generate=True, show_progress=False)
    dataset_config = load_json_config(dataset_config_path)
    # get class counts
    current_complex_map = None
    current_occgrid = None
    current_map_path = None
    current_name_index_map = None
    current_class_map = None

    semantic_results = dict(
        map_path=[],
        location_idx=[],
        x=[], y=[], yaw_left=[], yaw_right=[],

    )
    num_labels = len(radial_map_config.farfield_config.range_bins) - 1
    for i in range(num_labels):
        semantic_results[f"label_{i}"] = []
        semantic_results[f'coverage_{i}'] = []
    for map in image_df_part.map_path.unique():
        complex_map = LandCoverComplexMap.from_map_folder(MAPS_DIR / map)
        _, classes = complex_map.create_flattened_class_map()
        for key in classes.keys():
            for i in range(num_labels):
                semantic_results[f'bin_{i}_{key}'] = []

    for row_idx, row in tqdm.tqdm(image_df_part.iterrows(), total=len(image_df_part), disable=False):
        if row.map_path != current_map_path:
            current_map_path = row.map_path
            current_complex_map = LandCoverComplexMap.from_map_folder(MAPS_DIR / row.map_path)
            current_occgrid = OccupancyGrid.from_complexmap(current_complex_map)
            current_class_map, current_name_index_map = current_complex_map.create_flattened_class_map()

        label_args = dict(
            swath_library=swath_library,
            costmap=current_occgrid,
            radial_map_config=radial_map_config,
            dataset_config=dataset_config,
            image_yaw_range=(row.yaw_left, row.yaw_right),  # left then right
        )
        labels = reduce_single_location_to_labels(
            dataset_config.label_type, (row.x, row.y),
            keep_all_points=False,
            **label_args,
        )
        pos = current_occgrid.get_index_from_point((row.x, row.y))
        for label in labels:
            (yaw_left, yaw_right), bin_labels, bin_coverage = label
            semantic_classes = get_classes_in_bins(
                center_idx=pos,
                lr_yaw_range=(yaw_left, yaw_right),
                name_index_map=current_name_index_map,
                class_map=current_class_map,
                radial_map_config=radial_map_config,
                swath_library=swath_library,
            )
            semantic_results['map_path'].append(row.map_path)
            if row_idx is float:
                raise RuntimeError("somehow, row index is a float and not an int")
            semantic_results['location_idx'].append(row_idx)
            semantic_results['x'].append(row.x)
            semantic_results['y'].append(row.y)
            semantic_results['yaw_left'].append(yaw_left)
            semantic_results['yaw_right'].append(yaw_right)
            for i in range(num_labels):
                semantic_results[f"label_{i}"].append(bin_labels[i])
                semantic_results[f'coverage_{i}'].append(bin_coverage[i])
            for key, value in semantic_classes.items():
                parts = key.split("_")
                bin = parts[0]
                sem_class = "_".join(parts[1:])
                semantic_results[f'bin_{bin}_{sem_class}'].append(value)
    df = pd.DataFrame.from_dict(semantic_results)
    df = df.astype({'location_idx': int})
    return df


def balance_classes(df: pd.DataFrame, balanced_dataset_config: BalancedDatasetConfig,
                    relevant_classes=None, seed: int = 42) -> pd.DataFrame:
    if relevant_classes is None:
        relevant_classes = ['young_forest', 'forest', 'grass', 'field', 'road', 'marsh', 'beach', 'water']
    output_df = pd.DataFrame(columns=df.columns)
    num_labels = len([col for col in df.columns if "label_" in col])
    for i in range(num_labels):  # balance for each bin
        semantic_classes = [col for col in df.columns if f"bin_{i}_" in col]
        # remove bins that don't have enough coverage
        semantics_df = df[semantic_classes]
        semantics_df = semantics_df.loc[df[f'coverage_{i}'] > balanced_dataset_config.dominance_percentage]
        # a class has to own at least dominance_percentage of the cells to be considered

        # change classes to percentages from counts
        semantics_df = semantics_df.div(semantics_df.sum(axis=1), axis=0)
        # remove irrelevant classes
        semantics_df = semantics_df[[f"bin_{i}_{x}" for x in relevant_classes if f"bin_{i}_{x}" in semantics_df]]
        semantics_df = semantics_df.mask(semantics_df < balanced_dataset_config.dominance_percentage, 0)
        semantics_df = semantics_df[semantics_df.sum(axis=1) > 0]
        if len(semantics_df) == 0:
            continue
        # get the class with the most cells
        dominant_class = semantics_df.idxmax(axis=1)
        # get class counts
        class_counts = {}
        for class_name in dominant_class.unique():
            filtered_name = class_name.replace(f"bin_{i}_", "")
            if filtered_name in relevant_classes:
                class_counts[filtered_name] = np.sum(dominant_class == class_name)
        print(class_counts.values(), class_counts)
        min_class_count = min(class_counts.values())
        max_class_count = int(1 / balanced_dataset_config.max_imbalance_ratio * min_class_count)
        print("for bin ", i, " class counts: ", class_counts, 'min/max', min_class_count, max_class_count)

        # downsample each class
        for class_name in class_counts.keys():
            class_indexes = dominant_class[dominant_class == f'bin_{i}_' + class_name]
            if class_counts[class_name] > max_class_count:
                class_indexes = class_indexes.sample(max_class_count, random_state=seed)
            # set coverage to zero for other bins
            winning_samples = df.loc[class_indexes.index].copy()
            for j in range(num_labels):
                if j != i:
                    winning_samples[f'coverage_{j}'] = 0
            output_df = pd.concat([output_df, winning_samples])
    return output_df


def add_image_paths(df: pd.DataFrame, save_img_root: Path) -> pd.DataFrame:
    img_paths = []
    for i, row in df.iterrows():
        left_yaw_str = f"{row.yaw_left:.2f}".replace(".", "p")
        img_path = Path(save_img_root) / f"{row.location_idx:08d}_ly_{left_yaw_str}.png"
        img_paths.append(img_path)
    df['img_path'] = img_paths

    return df


def create_semantically_balanced_blueprint_from_image_request(
        image_df_path: Path,
        radial_map_config_path: Path,
        dataset_config_path: Path,
        balanced_dataset_config: BalancedDatasetConfig,
        output_path: Path,
        seed: int = 42,
) -> None:
    print("starting to create semantically balanced dataset")
    set_all_seeds(seed)  # for angle sampling
    create_dir(output_path.parent)
    # iterate through image request, saving samples along the way
    output = process_df_in_parallel(image_df_path, get_classes_and_labels, radial_map_config_path=radial_map_config_path, dataset_config_path=dataset_config_path)
    output = pd.concat(output)
    output.reset_index(drop=True, inplace=True)
    # output.to_csv(output_path, index=False)

    # balance classes
    # output = pd.read_csv(output_path)
    final_dataframe = balance_classes(output, balanced_dataset_config)
    final_dataframe = add_image_paths(final_dataframe, 'images')
    final_dataframe['image_df_path'] = image_df_path.relative_to(IMAGES_DIR)
    # separate unneeded columns
    unneeded_cols = [col for col in final_dataframe.columns if col.startswith("bin_")]
    needed_cols = [col for col in final_dataframe.columns if col not in unneeded_cols]
    extra_df = final_dataframe[unneeded_cols]
    final_dataframe = final_dataframe[needed_cols]

    final_dataframe.to_csv(output_path, index=False)
    extra_df.to_csv(output_path.parent / 'extra.csv', index=False)


def crop_images_for_dataset(dataset: pd.DataFrame, dataset_csv_path: Path, farfield_config: FarFieldConfig) -> None:
    # remove duplicate image paths, don't need to crop them twice
    dataset = dataset.drop_duplicates(subset='img_path')
    location_groups = dataset.groupby('location_idx')
    for group_name, group in tqdm.tqdm(location_groups, total=len(location_groups)):
        assert len(group.image_df_path.unique()) == 1, "All images in a location group must be from the same image request"
        image_df_path = IMAGES_DIR / group.image_df_path.unique()[0]
        image_df = pd.read_csv(image_df_path)
        image_df_row = image_df.loc[group_name]

        pano = cv2.imread(str(Path(image_df_path).parent / image_df_row['img_path']))
        pano = cv2_img_to_float_ndarr(pano)
        save_cropped_images_from_pano(
            farfield_config=farfield_config,
            save_img_paths=[dataset_csv_path.parent / x for x in group.img_path],
            save_bin_yaw_ranges=group[['yaw_left', 'yaw_right']].values,
            pano=pano,
            pano_lr_yaw=image_df_row[['yaw_left', 'yaw_right']].values,
        )


def crop_images_for_dataset_parallel(dataset_csv_path: Path, farfield_config: FarFieldConfig) -> None:
    process_df_in_parallel(dataset_csv_path, crop_images_for_dataset, farfield_config=farfield_config, dataset_csv_path=dataset_csv_path)


def make_semantically_balanced_dataset_from_image_request(
        image_df_path: Path,
        radial_map_config_path: Path,
        dataset_config_path: Path,
        save_folder: Path,
        balanced_dataset_config: BalancedDatasetConfig = BalancedDatasetConfig(1.0, 0.5),
):
    create_semantically_balanced_blueprint_from_image_request(
        image_df_path,
        radial_map_config_path,
        dataset_config_path,
        balanced_dataset_config,
        save_folder / 'balanced_semantic_dataset.csv',
    )
    print("cropping images")
    crop_images_for_dataset_parallel(
        save_folder / 'balanced_semantic_dataset.csv',
        load_json_config(radial_map_config_path).farfield_config
    )


if __name__ == "__main__":
    from fitam import CONFIGS_DIR, DATASETS_DIR
    img_req_path = IMAGES_DIR / 'balt_standard' / 'points.csv'

    make_semantically_balanced_dataset_from_image_request(
        img_req_path,
        CONFIGS_DIR / 'simulated_radial_configs' / 'radial_map_config.json',
        CONFIGS_DIR / 'classification_dataset_config.json',
        DATASETS_DIR / 'balt_standard',
    )

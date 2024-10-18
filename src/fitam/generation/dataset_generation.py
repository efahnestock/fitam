"""
Contains all functions related to turning environments into datasets for the simulation evaluation pipeline
"""

from __future__ import annotations

import numpy as np
import glob
import fury
from pathlib import Path
from os import PathLike
from numpy.typing import ArrayLike
from typing import Optional
import matplotlib.pyplot as plt
import os
import tqdm
import logging
import cv2
import torch
from torchvision import transforms
from torchvision.io import read_image
import pandas as pd
from pathlib import Path
from fitam import MAPS_DIR, IMAGES_DIR, CONFIGS_DIR
from fitam.core.config.RadialMapConfig import FarFieldConfig, RadialMapConfig
from fitam.core.config.DatasetConfig import DatasetConfig
from fitam.mapping.costmap import OccupancyGrid
from fitam.core.common import create_dir, load_json_config, dump_json_config, float_tensor_to_cv2_img, cv2_img_to_float_ndarr, load_pickle_config, process_df_in_parallel
from fitam.core.product_structures import ImageGenerationRequest
from fitam.learning.reduce_space_to_cost import reduce_single_location_to_labels
from fitam.core.data_tools import crop_image_for_sector
from fitam.mapping.costmap_swath_library import load_swath_library_from_pkl, SwathLibrary
from fitam.mapping.opengl_scene_rendering import render_scene, create_scene
from fitam.mapping.land_cover_complex_map import LandCoverComplexMap
from fitam.learning.dino_model import DinoModel
from fitam.planning.planner_general import State

logger = logging.getLogger(__name__)


def create_scene_from_complexmap(costmap_path: os.PathLike, costmap: LandCoverComplexMap) -> fury.window.Scene:
    forest_floor_image_path = costmap_path / f"{costmap_path.name}.png"
    scene = create_scene(costmap, str(forest_floor_image_path))
    return scene


def sample_points_from_environments(environments: list[LandCoverComplexMap], total_points: int, dataset_config: DatasetConfig) -> list:
    """
    Sample points from environments, totaling total_points
    Assumes envs are in the same order as the biome names
    input: list of environments, total points, run config
    output: list of sampled locations 
    """

    sample_points = []
    map_count = len(environments)
    max_points_each_map = [total_points //
                           map_count for _ in range(map_count-1)]
    # last one fills the rest
    max_points_each_map.append(total_points - sum(max_points_each_map))

    # sample the points from each environment
    for i, env in enumerate(environments):
        points = sample_single_environment(
            env, max_points_each_map[i], dataset_config)
        sample_points.append(points)

    return sample_points


def sample_single_environment(env: LandCoverComplexMap, num_points: int, dataset_config: DatasetConfig) -> ArrayLike:
    kernel_width_cells = int(
        dataset_config.kernel_width_meters // env.resolution)
    dist_from_wall_cells = int(
        dataset_config.distance_from_wall_meters // env.resolution)
    min_dist_between_samples_cells = int(
        dataset_config.min_distance_between_samples_meters // env.resolution)
    env_sample_points = env.sample_freespace_dialated(kernel_width_cells, dist_from_wall_cells,
                                                      min_dist_between_samples_cells, num_points)
    return env_sample_points


def save_single_image_request(img_req_path: PathLike, sampled_points: ArrayLike, camera_angles: ArrayLike,
                              map_path: PathLike, trial_dir: PathLike = MAPS_DIR, overwrite: bool = True) -> None:
    """
    img_req_path -> directory to save the image request
    """
    if os.path.exists(img_req_path) and not overwrite:
        raise ValueError(f"Image request path {img_req_path} already exists!")

    create_dir(img_req_path.parent, overwrite=overwrite)
    img_req = ImageGenerationRequest(sampled_points.tolist(), camera_angles.tolist())
    img_req.complex_map_paths.append(map_path.absolute().relative_to(trial_dir))
    img_req.map_idx.extend([0] * sampled_points.shape[0])

    # save the image request
    dump_json_config(img_req, img_req_path)
    # render sampled locations!
    img_req.save_all_image_request_figures(img_req_path, trial_dir)

    return


def create_images_part(
    image_request_path: os.PathLike,
    start_idx: int,
    end_idx: int,
    save_root_dir: os.PathLike,
    dataset_config_path: os.PathLike,
    complex_map_path: os.PathLike | None = None,  # if none, uses image request
    # NOTE: logs the image request's complex map path as its map path. Usually this is the partially observed version of the map
    camera_name: str = 'main',  # name of the camera to add to the points.csv file
) -> None:
    """
    Calculate the images for part of a single image request! 

    @param image_request_path: path to the image request
    @param start_idx: start index of the image request
    @param end_idx: end index of the image request (exclusive)
    @param save_root_dir: root directory to save the images and labels
        Saves things with the following structure: 
        save_root_dir/
            images/ (point within the map)
                {i:07d}_{j:02d}.png
            points_{start_idx}_{end_idx}.csv
    @param dataset_config_path: path to the dataset config
    """
    # load the image request!
    image_request_path = Path(image_request_path)
    # if pickle
    if image_request_path.suffix == '.pkl':
        image_request = load_pickle_config(image_request_path)
    elif image_request_path.suffix == '.json':
        image_request = load_json_config(image_request_path)
    else:
        raise RuntimeError(f"Image request path {image_request_path} must be a pickle or json file")
    # load the map
    assert len(image_request.complex_map_paths) == 1, "Only one map per image request is supported"

    if complex_map_path is None:
        map_path = MAPS_DIR / image_request.complex_map_paths[0]
    else:
        map_path = complex_map_path
    map = LandCoverComplexMap.from_map_folder(map_path)
    scene = create_scene_from_complexmap(map_path, map)
    # load the dataset config
    dataset_config = load_json_config(dataset_config_path)
    # create output directory if it doesn't exist
    create_dir(save_root_dir)
    # create images dir
    images_dir = save_root_dir / 'images'
    create_dir(images_dir)
    df = dict(
        map_path=[],
        img_path=[],
        yaw_left=[],  # yaw of the left side of the image
        yaw_right=[],  # yaw of the right side of the image
        x=[],
        y=[],
        camera_name=[],
    )
    # iterate through points in the image request
    for point_idx in range(start_idx, end_idx):
        point = image_request.locations[point_idx]

        df['map_path'].append(image_request.complex_map_paths[0])
        if type(point) == State:
            point = (point.x, point.y)
        df['x'].append(point[0])
        df['y'].append(point[1])
        df['yaw_left'].append(0.0)
        df['yaw_right'].append(2 * np.pi)  # since its a 360 pano
        df['camera_name'].append(camera_name)
        # render image
        map_location = np.array(
            (*point, dataset_config.camera_height))
        panorama = render_scene(scene, map_location, 0.0)
        # pano index swaps it to BGR for opencv
        panorama = panorama[..., ::-1]
        df['img_path'].append(str(
            (images_dir / f'{point_idx:07d}_pano.png').relative_to(save_root_dir)))
        cv2.imwrite(str(images_dir / f'{point_idx:07d}_pano.png'), panorama)
    # save the dataframe
    df = pd.DataFrame(df)
    df.to_csv(save_root_dir / f'points_{start_idx}_{end_idx}.csv', index=False)
    return


def merge_image_csvs(root_dir: os.PathLike) -> None:
    """
    Merge all of the root_dir/points_*.csv files into a single csv
    Saves them into root_dir/points.csv
    """
    print(root_dir)
    df = pd.DataFrame()
    # do current directory
    for file in sorted(glob.glob(str(root_dir / 'points_*.csv'))):
        # adjust image paths
        # path in child csv is relative to child csv
        # path in parent csv is relative to parent dir of the parent csv
        partial_df = pd.read_csv(file)
        new_path_part = Path(file).parent.relative_to(root_dir)
        partial_df['img_path'] = partial_df['img_path'].apply(lambda x: str(new_path_part / Path(x)))

        df = pd.concat([partial_df, df], ignore_index=True)
    # do subdirectories
    for file in sorted(glob.glob(str(root_dir / "**" / 'points_*.csv'))):
        partial_df = pd.read_csv(file)
        new_path_part = Path(file).parent.relative_to(root_dir)
        partial_df['img_path'] = partial_df['img_path'].apply(lambda x: str(new_path_part / Path(x)))
        df = pd.concat([partial_df, df], ignore_index=True)
    df.to_csv(root_dir / 'points.csv', index=False)
    return


def create_labels_part(
    images_db_path: os.PathLike,
    start_idx: int,
    end_idx: int,
    save_root_dir: os.PathLike,
    radial_map_config_path: os.PathLike,
    dataset_config_path: os.PathLike,
    swath_library_path: os.PathLike,
    save_djikstra: bool = False,
    keep_all_points: bool = True,
) -> None:
    """
    Calculate the images and labels for part of a single image request! 

    @param image_request_path: path to the image request
    @param start_idx: start index of the image request
    @param end_idx: end index of the image request (exclusive)
    @param save_root_dir: root directory to save the images and labels
        Saves things with the following structure: 
        save_root_dir/
            point_{i:07d}/ (point within the map)
                djikstra.pt (optional)
                djikstra_plot.png (optional)
                labels.csv
                {j:07d}.png images
    @param radial_map_config_path: path to the radial map config
    @param dataset_config_path: path to the dataset config
    @param swath_library_path: path to the swath library
    @param save_panoramas: whether to save the panoramas
    @param save_djikstra: whether to save the djikstra maps and plots 


    """
    # load the image request!
    images_df = pd.read_csv(images_db_path)
    # load the map
    current_map_path = MAPS_DIR / Path(images_df.map_path[start_idx])
    map = LandCoverComplexMap.from_map_folder(current_map_path)
    occ_grid = OccupancyGrid.from_complexmap(map)
    # load the radial map config
    radial_map_config = load_json_config(radial_map_config_path)
    # load the dataset config
    dataset_config = load_json_config(dataset_config_path)
    # load the swath library
    swath_library = load_swath_library_from_pkl(swath_library_path)
    # make sure swath and map match
    # create output directory if it doesn't exist
    create_dir(save_root_dir)
    # iterate through points in the image request
    for point_idx in range(start_idx, end_idx):
        # map_path=[],
        # img_path=[],
        # yaw_left=[], # yaw of the left side of the image
        # yaw_right=[], # yaw of the right side of the image
        # x=[],
        # y=[],
        if MAPS_DIR / Path(images_df.map_path[point_idx]) != current_map_path:
            current_map_path = MAPS_DIR / Path(images_df.map_path[point_idx])
            map = LandCoverComplexMap.from_map_folder(current_map_path)
            occ_grid = OccupancyGrid.from_complexmap(map)

        assert swath_library.mask_resolution == occ_grid.resolution, f"Swath library and map resolution must match. Got {swath_library.mask_resolution} and {occ_grid.resolution}"
        point = (images_df.x[point_idx], images_df.y[point_idx])
        # print("processing point", point)
        camera_name = images_df.camera_name[point_idx]
        # create output directory:
        point_dir = save_root_dir / f'point_{point_idx:07d}'
        create_dir(point_dir)

        # load base image
        ref_image = cv2.imread(str(images_db_path.parent / images_df.img_path[point_idx]))
        # generate djikstra
        label_args = dict(
            swath_library=swath_library,
            costmap=occ_grid,
            radial_map_config=radial_map_config,
            dataset_config=dataset_config,
            image_yaw_range=(images_df.yaw_left[point_idx], images_df.yaw_right[point_idx]),  # left then right

        )
        # if dataset_config.label_type == 'MIN_COST':
        #     djikstra = generate_single_plan(occ_grid, radial_map_config.farfield_config.range_bins[-1],
        #                                     point, move_origin_to_center=False)
        #     if save_djikstra:
        #         torch.save(torch.from_numpy(djikstra), point_dir / 'djikstra.pt')
        #         plot_djikstra(djikstra, point_dir / 'djikstra_plot.png')
        #     label_args['djikstra'] = djikstra
        # create labels for images
        labels = reduce_single_location_to_labels(
            dataset_config.label_type, point,
            keep_all_points=keep_all_points,
            **label_args,
        )
        if len(labels) == 0:
            # save empty csv and continue
            print("saving empty csv")
            pd.DataFrame().to_csv(point_dir / 'labels.csv')
            continue  # no valid labels were found at this point

        # save images and labels
        assert radial_map_config.farfield_config.image_context_size >= radial_map_config.farfield_config.orientation_bin_size
        img_paths = [point_dir / f"{i:07d}.png" for i in range(len(labels))]
        save_cropped_images_from_pano(
            radial_map_config.farfield_config,
            img_paths,
            [(label[0][0], label[0][1]) for label in labels],
            cv2_img_to_float_ndarr(ref_image),
            (images_df.yaw_left[point_idx], images_df.yaw_right[point_idx]),
        )
        # save labels
        labels_df = label_list_to_df(images_db_path.relative_to(IMAGES_DIR),
                                     radial_map_config_path.relative_to(CONFIGS_DIR),
                                     point_idx,
                                     [x.relative_to(save_root_dir.parent) for x in img_paths],
                                     labels,
                                     camera_name)
        labels_df.to_csv(point_dir / 'labels.csv', index=False)


def save_cropped_images_from_pano(farfield_config: FarFieldConfig,
                                  save_img_paths: list,
                                  save_bin_yaw_ranges: list[tuple[float, float]],  # NOTE: padding added to these
                                  pano: np.ndarray,
                                  pano_lr_yaw: tuple[float, float]) -> None:
    image_context_padding = (farfield_config.image_context_size - farfield_config.orientation_bin_size) / 2
    for i, (save_bin_yaw_range, img_path) in enumerate(zip(save_bin_yaw_ranges, save_img_paths)):
        cropped_image = crop_image_for_sector(
            image=pano,
            image_yaws=pano_lr_yaw,
            cropped_yaws=[save_bin_yaw_range[0] + image_context_padding, save_bin_yaw_range[1] - image_context_padding],
            img_width_yaw=farfield_config.image_context_size)
        create_dir(img_path.parent)
        cv2.imwrite(str(img_path), float_tensor_to_cv2_img(cropped_image))


def extract_features_from_dataset(dataset_csv_file: os.PathLike,
                                  dataset_images_dir: os.PathLike,
                                  feature_save_dir: os.PathLike,
                                  transforms: Optional[transforms.Compose] = None,
                                  device: Optional[torch.device] = None,
                                  batch_size: int = 16

                                  ) -> None:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load dataset csv
    dataset_df = pd.read_csv(dataset_csv_file)
    feature_extractor = DinoModel(1)
    feature_extractor.to(device)
    # create output directory
    create_dir(feature_save_dir, overwrite=True)
    # iterate through images
    unique_image_paths = dataset_df.img_path.unique()
    feature_paths = []
    for i, batch_i in tqdm.tqdm(enumerate(range(0, len(unique_image_paths), batch_size)), total=len(unique_image_paths)//batch_size):
        batch = unique_image_paths[batch_i:min(batch_i+batch_size, len(dataset_df))]
        # load images
        imgs = torch.stack([read_image(str(dataset_images_dir / img_path)).to(device) for img_path in batch])
        imgs = imgs.float() / 255.0
        if transforms is not None:
            imgs = transforms(imgs)
        else:
            print("WARNING: No transforms being used for feature extractor")
        # extract features
        features = feature_extractor.extract_features(imgs).to('cpu')
        torch.save(features, feature_save_dir / f'{i:07d}.pt')
        for j in range(features.shape[0]):
            feature_paths.append((feature_save_dir / 'unbatched' / f'{j+i * batch_size:07d}.pt').relative_to(dataset_csv_file.parent))
    # create new column in dataset csv
    name_map = dict(zip(unique_image_paths, feature_paths))
    def map_func(x): return name_map[x.img_path]
    dataset_df['feature_name'] = dataset_df.apply(map_func, axis=1)

    # save dataset csv
    dataset_df.to_csv(dataset_csv_file, index=False)


def unbatch_one_file(dataframe_part: pd.DataFrame) -> None:
    for idx, row in tqdm.tqdm(dataframe_part.iterrows(), total=len(dataframe_part)):
        features = torch.load(row.feature_file)
        for i, _ in enumerate(range(features.shape[0])):
            # for some weird reason, if you don't clone it, the save file size is 1.6M instead of 3.8k
            torch.save(features[i].clone(), Path(row.feature_dir) / 'unbatched' / f'{i+row.file_idx * row.batch_size:07d}.pt')


def unbatch_features(feature_dir: os.PathLike) -> None:
    create_dir(feature_dir / 'unbatched')
    all_files = sorted(glob.glob(str(feature_dir / '*.pt')))
    batch_size = torch.load(all_files[0]).shape[0]
    file_df = dict(
        feature_file=all_files,
        file_idx=range(len(all_files)),
        batch_size=[batch_size]*len(all_files),
        feature_dir=[feature_dir]*len(all_files)
    )
    file_df = pd.DataFrame(file_df)
    file_df.to_csv('/tmp/feature_df.csv', index=False)
    process_df_in_parallel('/tmp/feature_df.csv', unbatch_one_file)


def plot_djikstra(djikstra: np.ndarray, save_path: os.PathLike) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.imshow(djikstra)
    fig.colorbar(img)
    plt.savefig(save_path)
    plt.close(fig)


def label_list_to_df(image_request_path: os.PathLike,
                     radial_map_config_path: os.PathLike,
                     location_idx: int,
                     image_paths: list[os.PathLike],
                     labels: list,
                     camera_name: str) -> pd.DataFrame:
    """
    Convert a list of labels to a dataframe 
    Labels are either a LIST of: 
        [(left yaw, right yaw), [label_0, label_1, ...]]
        [(left yaw, right yaw), class_int]
        [(left yaw, right yaw), [label_0, label_1, ...], [coverage_0, coverage_1, ...]]

    """
    df = dict(
        img_path=image_paths,
        location_idx=[location_idx]*len(image_paths),
        image_request_path=[str(image_request_path)]*len(image_paths),
        radial_map_config_path=[str(radial_map_config_path)]*len(image_paths),
        left_yaw=[label[0][0] for label in labels],
        right_yaw=[label[0][1] for label in labels],
        camera_name=[camera_name]*len(image_paths),
    )
    if type(labels[0][1]) == list:
        if len(labels[0]) == 3:  # coverage is there too
            for i in range(len(labels[0][1])):
                df[f'coverage_{i}'] = [label[2][i] for label in labels]
        for i in range(len(labels[0][1])):
            df[f'label_{i}'] = [label[1][i] for label in labels]
    else:
        df['label'] = [label[1] for label in labels]
    return pd.DataFrame(df)


def merge_individual_label_files(dataset_root_dir: os.PathLike, save_path: os.PathLike) -> None:
    """
    Collect all of the labels.csv files in the dataset_root_dir and merge them into a single file

    @param dataset_root_dir: root directory of the dataset
    @param save_path: path to save the merged labels file
    """
    labels_files = sorted(glob.glob(os.path.join(dataset_root_dir, '**', 'labels.csv'), recursive=True))
    read_csvs = [pd.read_csv(file) for file in labels_files]
    non_empty_csvs = [df for df in read_csvs if len(df) > 0]
    df = pd.concat(non_empty_csvs, ignore_index=True)
    df.to_csv(save_path, index=False)
    return


def class_cells_to_stats(class_cells, class_map):
    # get class counts
    class_counts = {}
    for class_name in class_map.keys():
        class_counts[class_name] = np.sum(class_cells == class_map[class_name])
    return class_counts


def get_classes_in_bins(center_idx: tuple[int, int],
                        lr_yaw_range: tuple[float, float],
                        name_index_map: dict,
                        class_map: dict,
                        radial_map_config: RadialMapConfig,
                        swath_library: SwathLibrary):
    output = dict()  # "bin_class" -> count
    for i in range(len(radial_map_config.farfield_config.range_bins)-1):
        swath_indices = swath_library.get_swath(center_idx, radial_map_config.farfield_config.range_bins[i], lr_yaw_range[1], class_map.shape)
        classes = class_map[swath_indices]
        class_counts = class_cells_to_stats(classes, name_index_map)
        for key in class_counts.keys():
            bin_key = f"{i}_{key}"
            output[bin_key] = class_counts[key]

    return output

import tqdm
from pathlib import Path
from typing import NamedTuple
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import binary_dilation
from fitam import MAPS_DIR, IMAGES_DIR
from fitam.mapping.land_cover_complex_map import LandCoverComplexMap
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.opengl_scene_rendering import create_scene, render_scene, IMG_HEIGHT_DEG, IMG_HEIGHT_RESOLUTION
from fitam.core.common import angle_between_lr_yaw
from fitam.core.config.RadialMapConfig import FarFieldConfig, ImagePyramidConfig


from PIL import Image
import matplotlib.pyplot as plt


def calculate_footline_distances(min_range_m, max_range_m, num_footlines: int):
    ln_A = np.log(min_range_m)
    k = (np.log((max_range_m)) - ln_A) / (num_footlines - 1)
    footline_distances = np.exp(ln_A + k * np.arange(num_footlines))
    return footline_distances


def calculate_footline_pixel_heights(image: np.ndarray, footline_distances: np.ndarray, camera_height_m: float):
    """return i coordinate of the footline in the image"""
    assert np.rad2deg(np.arctan(1 / footline_distances[0])) < IMG_HEIGHT_DEG / 2
    footline_pixel_i = IMG_HEIGHT_RESOLUTION / 2 + ((np.pi / 2 - np.arctan(footline_distances / camera_height_m)) / np.deg2rad(IMG_HEIGHT_DEG / 2) * IMG_HEIGHT_RESOLUTION / 2)
    return footline_pixel_i.astype(int)


class ImagePyramid(NamedTuple):
    image_slices: list[np.ndarray]
    footline_distances: np.ndarray


def create_image_pyramid(image, config: ImagePyramidConfig):
    footline_distances = calculate_footline_distances(config.min_range_m, config.max_range_m, config.num_footlines)
    min_height_pixels = config.image_slice_height_pixels
    tan_scale_factor = min_height_pixels / np.arctan(1 / config.max_range_m)
    image_height_pixels = (tan_scale_factor * np.arctan(1 / footline_distances)).round().astype(int)
    # make all values even
    image_height_pixels = image_height_pixels + (image_height_pixels % 2)
    footline_i_values = calculate_footline_pixel_heights(image, footline_distances, config.camera_height_m)
    # plt.figure()
    # plt.plot(footline_distances, image_height_pixels)
    # plt.plot(footline_distances, footline_i_values)
    # plt.show()
    undownsampled_image_pyramid = []

    for i, (footline_i, height) in enumerate(zip(footline_i_values, image_height_pixels)):
        # print(footline_i, height)
        image_slice = image[footline_i - height // 2: footline_i + height // 2, :]
        undownsampled_image_pyramid.append(image_slice)

    # downsample the image slices
    image_pyramid = []
    for image_slice in undownsampled_image_pyramid:
        new_width = image_slice.shape[1] * config.image_slice_height_pixels // image_slice.shape[0]
        pil_image_slice = Image.fromarray(image_slice)
        pil_image_slice = pil_image_slice.resize((new_width, config.image_slice_height_pixels), Image.Resampling.LANCZOS)
        image_pyramid.append(np.array(pil_image_slice))
        # print(np.array(pil_image_slice).shape)

    return ImagePyramid(image_pyramid, footline_distances)


def image_pyramid_to_batch(image_pyramid: ImagePyramid,
                           patch_width_pixels: int,
                           ) -> tuple[np.array, np.array]:
    batch = []
    yaw_percentage = []
    patch_range_m = []
    assert patch_width_pixels % 2 == 0
    for footline_distance, image_slice in zip(image_pyramid.footline_distances, image_pyramid.image_slices):
        h, w, _ = image_slice.shape
        if w < patch_width_pixels:
            raise RuntimeError(f'Image slice width {w} is less than patch width {patch_width_pixels}')
        delta_pixels = w - patch_width_pixels
        num_patches = delta_pixels // patch_width_pixels
        if delta_pixels % patch_width_pixels != 0:
            num_patches += 1
        for pixel_col in np.linspace(patch_width_pixels // 2, patch_width_pixels // 2 + delta_pixels, num_patches):
            pixel_col = int(pixel_col)
            patch = image_slice[:, pixel_col - patch_width_pixels // 2: pixel_col + patch_width_pixels // 2]
            batch.append(patch)
            yaw_percentage.append(pixel_col / w)
            patch_range_m.append(footline_distance)

    patches = np.array(batch)
    range_and_yaw = np.array([patch_range_m, yaw_percentage]).T
    return patches, range_and_yaw  # (num_patches, h, w, 3), (num_patches, 2)


def find_yaw_from_range_yaw_percentage(patch_range_and_yaw_percentage, lr_image_yaw_rad):
    left_yaw_rad, right_yaw_rad = lr_image_yaw_rad
    yaw_percentage = patch_range_and_yaw_percentage[:, 1]
    yaw_value = left_yaw_rad - angle_between_lr_yaw(left_yaw_rad, right_yaw_rad) * yaw_percentage
    return yaw_value


def find_xy_for_image_pyramid_batch(robot_location_m: tuple,  # (x, y)
                                    lr_image_yaw_rad: tuple,  # (left rad, right rad)
                                    patch_range_and_yaw_percentage: np.array,  # (num_patches, 2) for range_m, yaw_percentage
                                    ) -> np.array:  # (num_patches, 2) for x, y in meters
    x, y = robot_location_m
    yaw_value = find_yaw_from_range_yaw_percentage(patch_range_and_yaw_percentage, lr_image_yaw_rad)
    range_m = patch_range_and_yaw_percentage[:, 0]
    x_values = x + range_m * np.cos(yaw_value)
    y_values = y + range_m * np.sin(yaw_value)
    return np.array([x_values, y_values]).T


def label_image_pyramid_batch(
        patch_xy_values: np.ndarray,
        dialated_occ_grid: OccupancyGrid,  # all obstacles are true
        observed_occ_grid: OccupancyGrid,  # all observed cells are true
) -> np.ndarray:
    """1 is obstacle, 0 is free, -1 is no label"""
    labels = []
    for x, y in patch_xy_values:
        if not dialated_occ_grid.check_state_in((x, y)) or not observed_occ_grid.get_value((x, y)):
            labels.append(-1)
        else:
            labels.append(dialated_occ_grid.get_value((x, y)))

    labels = np.array(labels)
    return labels


class ImagePatchDataset(NamedTuple):
    patches: np.ndarray
    labels: np.ndarray
    xys: np.ndarray
    robot_locations: np.ndarray
    image_paths: np.ndarray


def create_one_batch_of_labels(
        full_image_path: Path,
        robot_position: tuple,
        panorama_yaws: tuple,
        gt_occ_grid: OccupancyGrid,
        observed_occ_grid: OccupancyGrid,
        image_pyramid_config: ImagePyramidConfig):
    img = np.array(Image.open(full_image_path))
    image_pyramid = create_image_pyramid(img, image_pyramid_config)
    patches, range_and_yaw_percentage = image_pyramid_to_batch(image_pyramid, image_pyramid_config.image_slice_width_pixels)

    xy_for_batch = find_xy_for_image_pyramid_batch(robot_position, panorama_yaws, range_and_yaw_percentage)
    labels = label_image_pyramid_batch(xy_for_batch, gt_occ_grid, observed_occ_grid)
    no_label_mask = labels == -1
    patches = patches[~no_label_mask]
    labels = labels[~no_label_mask].astype(bool)
    xy_for_batch = xy_for_batch[~no_label_mask]
    range_and_yaw_percentage = range_and_yaw_percentage[~no_label_mask]
    return patches, labels, xy_for_batch, range_and_yaw_percentage


def create_image_pyramid_labels_from_images(images_dir: Path,
                                            masked_map_path: Path,
                                            gt_map_path: Path,
                                            image_pyramid_config: ImagePyramidConfig):

    image_info = pd.read_csv(images_dir / 'points.csv')
    masked_map = LandCoverComplexMap.from_map_folder(masked_map_path)
    gt_map = LandCoverComplexMap.from_map_folder(gt_map_path)
    gt_obstacle_map = gt_map.create_occlusion_map()
    num_iterations = int(image_pyramid_config.obstacle_dialation_m / gt_map.resolution)
    dialated_obstacle_map = binary_dilation(gt_obstacle_map, iterations=num_iterations)
    gt_occ_grid = OccupancyGrid(gt_map.resolution, width=gt_obstacle_map.shape[1], height=gt_obstacle_map.shape[0], data=dialated_obstacle_map)
    observed_occ_grid = OccupancyGrid(gt_map.resolution, width=gt_obstacle_map.shape[1], height=gt_obstacle_map.shape[0], data=~masked_map.masks['unknown'])

    all_patches = []
    all_labels = []
    all_xys = []
    all_robot_locations = []
    all_image_paths = []

    for i, row in tqdm.tqdm(image_info.iterrows(), total=len(image_info)):
        patches, labels, xy_for_batch, _ = create_one_batch_of_labels(
            images_dir / row['img_path'],
            (row['x'], row['y']),
            (row['yaw_left'], row['yaw_right']),
            gt_occ_grid,
            observed_occ_grid,
            image_pyramid_config
        )

        all_patches.append(patches)
        all_labels.append(labels)
        all_xys.append(xy_for_batch)
        all_robot_locations.append(np.repeat(np.array([[row['x'], row['y']]]), len(patches), axis=0))
        all_image_paths.append(np.repeat(row['img_path'], len(patches)))

    all_patches = np.concatenate(all_patches)
    all_labels = np.concatenate(all_labels)
    all_xys = np.concatenate(all_xys)
    all_robot_locations = np.concatenate(all_robot_locations)
    all_image_paths = np.concatenate(all_image_paths)

    return ImagePatchDataset(all_patches, all_labels, all_xys, all_robot_locations, all_image_paths)


def balance_classes_in_dataset(dataset: ImagePatchDataset, generator: np.random.Generator):
    min_size = min(np.sum(dataset.labels == False), np.sum(dataset.labels == True))
    dataset_mask = np.zeros_like(dataset.labels, dtype=bool)

    for label_value in [False, True]:
        if np.sum(dataset.labels == label_value) == min_size:
            dataset_mask[dataset.labels == label_value] = True
        else:
            valid_options = np.arange(len(dataset.labels))[dataset.labels == label_value]
            lucky_picks = generator.choice(valid_options, replace=False, size=min_size)
            assert np.max(dataset_mask[lucky_picks]) == False
            dataset_mask[lucky_picks] = True

    balanced_dataset = ImagePatchDataset(
        patches=dataset.patches[dataset_mask],
        labels=dataset.labels[dataset_mask],
        xys=dataset.xys[dataset_mask],
        robot_locations=dataset.robot_locations[dataset_mask],
        image_paths=dataset.image_paths[dataset_mask],
    )
    return balanced_dataset


def save_image_patch_dataset(dataset: ImagePatchDataset, path: Path):
    with h5py.File(path, 'w') as f:
        f.create_dataset('patches', data=dataset.patches)
        f.create_dataset('labels', data=dataset.labels)
        f.create_dataset('xys', data=dataset.xys)
        f.create_dataset('robot_locations', data=dataset.robot_locations)
        f.create_dataset('image_paths', data=[s.encode('utf-8') for s in dataset.image_paths])


def visualize_dataset_sample(dataset, root_image_path, index, gt_lcm: LandCoverComplexMap, dialated_obstacle_map: OccupancyGrid, observed_occ_grid: OccupancyGrid, window_buffer_pix: int = 200):
    patch = dataset.patches[index]
    label = dataset.labels[index]
    xy = dataset.xys[index]
    patch_ij = dialated_obstacle_map.get_index_from_point(xy)
    robot_location = dataset.robot_locations[index]
    robot_ij = dialated_obstacle_map.get_index_from_point(robot_location)
    image_path = dataset.image_paths[index]
    img = np.array(Image.open(root_image_path / image_path))
    # print(xy, robot_location)
    window_buffer_pix = max([window_buffer_pix, abs(patch_ij[1] - robot_ij[1]) + 10, abs(patch_ij[0] - robot_ij[0]) + 10])

    floormap = gt_lcm.create_floormask()
    i_bounds = slice(robot_ij[0]-window_buffer_pix, robot_ij[0]+window_buffer_pix)
    j_bounds = slice(robot_ij[1]-window_buffer_pix, robot_ij[1]+window_buffer_pix)
    floormap = floormap[i_bounds, j_bounds]
    dial_obs_map = dialated_obstacle_map.data[i_bounds, j_bounds]
    obs_mask = observed_occ_grid.data[i_bounds, j_bounds]

    fig = plt.figure(figsize=(20, 12))

    # pano
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
    ax1.imshow(img)
    plt.title(f'Label: {label}, Index: {index}')
    # patch
    ax2 = plt.subplot2grid((2, 4), (1, 0))
    ax2.imshow(patch)
    # lcm floormask with annotation
    ax3 = plt.subplot2grid((2, 4), (1, 1))
    ax3.imshow(floormap)
    # ax3.scatter([100], [200], c='red')

    ax4 = plt.subplot2grid((2, 4), (1, 2))
    ax4.imshow(dial_obs_map)

    ax5 = plt.subplot2grid((2, 4), (1, 3))
    ax5.imshow(obs_mask)

    for ax in [ax3, ax4, ax5]:
        ax.scatter([window_buffer_pix], [window_buffer_pix], facecolors='none', edgecolors='blue')
        ax.scatter([window_buffer_pix + patch_ij[1] - robot_ij[1]], [window_buffer_pix + patch_ij[0] - robot_ij[0]], facecolors='none', edgecolors='red')

    return fig


def pyramid_sizes_from_image_and_config(image_shape: tuple, config: ImagePyramidConfig):
    img_pyramid = create_image_pyramid(np.zeros(image_shape, dtype=np.uint8), config)
    patches, ranges_and_yaw_percentages = image_pyramid_to_batch(img_pyramid, config.image_slice_width_pixels)
    range_values = np.sort(np.unique(ranges_and_yaw_percentages[:, 0]))
    innner_yaw_values = np.sort(ranges_and_yaw_percentages[ranges_and_yaw_percentages[:, 0] == range_values[0]])
    return range_values, innner_yaw_values


def farfield_config_from_spatial_label(image_shape: tuple, config: ImagePyramidConfig):
    range_values, inner_yaw_values = pyramid_sizes_from_image_and_config(image_shape, config)
    # shift range bins by 1m, add a final bin
    range_bin_boundaries = np.append(range_values - 1.0, range_values[-1] + 1)
    yaw_bin_size = 2 * np.pi / len(inner_yaw_values)
    return FarFieldConfig(
        yaw_bin_size,
        None,
        range_bin_boundaries,
        None
    )

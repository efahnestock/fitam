from __future__ import annotations
from fury import window
import torch.nn as nn
import numpy as np
import os
import torch
from PIL import Image
import logging
import matplotlib.pyplot as plt
import pandas as pd
from fitam.mapping.observation_types import RadialObservation
from fitam.planning.planner_general import snap_easl_state_to_planning_state, State
from fitam.core.common import chw_to_hwc_img, create_dir, dump_json_config
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.costmap_swath_library import SwathLibrary
from fitam.mapping.opengl_scene_rendering import render_scene
from fitam.core.config.EvaluationConfig import EvaluationConfig
from fitam.core.config.RadialMapConfig import RadialMapConfig, FarFieldConfig
from fitam.core.config.TrainConfig import TrainConfig
from fitam.evaluation.eval_data_structures import TrialDirectories
from fitam.evaluation.eval_utils import calculate_label_angle_pairs, prepare_image_input
from fitam.learning.reduce_space_to_cost import reduce_speed_to_class
from fitam.generation.dataset_generation import get_classes_in_bins
from fitam.mapping.belief import Belief


def observe(logger: logging.Logger, current_state: State, replan_index: int, scene: window.Scene,
            trial_directories: TrialDirectories, belief: Belief, master_costmap: OccupancyGrid, model: nn.Module,
            swath_library: SwathLibrary, device: torch.device, use_renderer: bool,
            radial_map_config: RadialMapConfig, camera_height: float, eval_config: EvaluationConfig, train_config: TrainConfig,
            num_active_bins: int, export_dataset: bool, class_map: np.ndarray, class_index_to_name_map: dict) -> None:

    # The index (in the costmap) is now [current_planning_state.y, current_planning_state.x]
    current_planning_state = snap_easl_state_to_planning_state(
        belief, current_state)
    center_idx = (current_planning_state.y, current_planning_state.x)
    heading_offset = np.random.uniform(0, 2 * np.pi)
    logger.debug(f"Observing from state {current_state} with heading offset {heading_offset}")
    if radial_map_config.farfield_config is not None:  # if farfield is enabled
        if use_renderer:
            predicted_results, cropped_imgs = predict_labels(logger=logger, current_state=current_state, replan_index=replan_index, scene=scene,
                                                             device=device, trial_dirs=trial_directories, heading_offset=heading_offset,
                                                             model=model, camera_height=camera_height, belief=belief,
                                                             radial_map_config=radial_map_config, eval_config=eval_config,
                                                             swath_library=swath_library)
            true_classes = None
            if export_dataset:
                gt_results = calculate_labels(
                    logger=logger, current_state=current_state, master_costmap=master_costmap, heading_offset=heading_offset, swath_library=swath_library,
                    radial_map_config=radial_map_config, train_config=train_config)

                extra_df = {}
                yaws, classes, uncertainties = yaws_classes_uncertainties_from_radial_observation_list(predicted_results)
                _, true_classes, _ = yaws_classes_uncertainties_from_radial_observation_list(gt_results)
                for yaw_pair in yaws:
                    counts = get_classes_in_bins(center_idx, yaw_pair, class_index_to_name_map, class_map, radial_map_config, swath_library)
                    for k, v in counts.items():
                        extra_df[f"bin_{k}"] = extra_df.get(f"bin_{k}", []) + [v]

                df = {
                    'yaw_left': [x[0] for x in yaws],
                    'yaw_right': [x[1] for x in yaws],
                    'x': [current_state.x for _ in yaws],
                    'y': [current_state.y for _ in yaws],
                    'img_path': [f"images/{replan_index:07d}_{i:02d}.png" for i in range(len(cropped_imgs))],
                }
                for label_index in range(len(classes[0])):
                    df[f'predicted_class_{label_index}'] = [x[label_index] for x in classes]
                    df[f'label_{label_index}'] = [x[label_index] for x in true_classes]
                    df[f'uncertainty_{label_index}'] = [x[label_index] for x in uncertainties]
                df = pd.DataFrame(df)
                df.to_csv(trial_directories.base_dataset_path / f"dataset_{replan_index:07d}.csv", index=False)
                extra_df = pd.DataFrame(extra_df)
                extra_df.to_csv(trial_directories.base_dataset_path / f"extra_{replan_index:07d}.csv", index=False)
                # save images
                for i, img in enumerate(cropped_imgs):
                    network_in_img = (chw_to_hwc_img(img) * 255).astype(np.uint8)
                    im = Image.fromarray(network_in_img)
                    im.save(trial_directories.base_dataset_path / f"images/{replan_index:07d}_{i:02d}.png")

            if eval_config.save_network_results:
                yaws, classes, uncertainties = yaws_classes_uncertainties_from_radial_observation_list(predicted_results)
                for i, img in enumerate(cropped_imgs):
                    network_in_img = (chw_to_hwc_img(img) * 255).astype(np.uint8)
                    fig, ax = plt.subplots()
                    ax.imshow(network_in_img)
                    title = f"Class: {[f'{x}' for x in classes[i]]}  Noise {[f'{x:.2f}' for x in uncertainties[i]]} \n Yaw range {[f'{x:.2f}' for x in yaws[i]]} "
                    if true_classes is not None:
                        title += f"True classes: {true_classes[i]}"
                    ax.set_title(title)
                    ax.axis('off')
                    fig.savefig(trial_directories.base_network_results_path / f'{replan_index:07d}_{i:02d}.png', bbox_inches='tight')
                    plt.close(fig)

        else:
            predicted_results = calculate_labels(
                logger=logger, current_state=current_state, master_costmap=master_costmap, heading_offset=heading_offset, swath_library=swath_library,
                radial_map_config=radial_map_config, train_config=train_config)

        # update radial map!
        logger.debug("Populating the unknown map")
        if num_active_bins is not None:
            assert num_active_bins > 0, "num_active_bins must be greater than 0"
            for i, obs in enumerate(predicted_results):
                obs.trim_to_num_bins(num_active_bins)

        belief.update_from_observations(
            predicted_results
        )
        logger.debug("Finished populating the unknown map")
        costmap = belief.get_full_costmap()
        if np.any(costmap < 0):
            print("negative costmap values after local update")


def construct_radial_observations_output(
        yaw_ranges: list[tuple[float, float]],
        predictions: np.ndarray,
        center_pos: tuple[float, float],
        swath_library: SwathLibrary,
        ff_config: FarFieldConfig,
        belief: Belief,
        yaw_shift=0.0) -> list[RadialObservation]:
    assert len(predictions.shape) == 4, "Predictions must be of shape (batch, num_bins, num_models, num_classes)"
    output_list = []
    center_idx = belief.get_index_from_point(center_pos)
    for i, yaw_range in enumerate(yaw_ranges):
        output_list.append(RadialObservation(
            center_idx=center_idx,
            lr_yaw=yaw_range,
            model_outputs=predictions[i],
            swath_library=swath_library,
            ff_config=ff_config,
            yaw_shift=yaw_shift,
        ))

    return output_list


def predict_labels(logger: logging.Logger, current_state: State, replan_index: int, scene: window.Scene,
                   device: torch.device, trial_dirs: TrialDirectories, heading_offset: float,
                   model: torch.nn.Module, swath_library: SwathLibrary,  belief: Belief,
                   camera_height: float, radial_map_config: RadialMapConfig,
                   eval_config: EvaluationConfig) -> list[RadialObservation]:
    logger.debug("Farfield is enabled, starting to generate panorama")
    pano = render_scene(scene, np.asarray(
        (current_state.x, current_state.y, camera_height)), heading_offset)
    pano = pano.astype(np.float32) / 255.0
    if eval_config.save_panoramas:
        pano_img = (pano * 255).astype(np.uint8)
        im = Image.fromarray(pano_img)
        im.save(os.path.join(trial_dirs.base_pano_path,
                             f'{replan_index:07d}.png'))
        logger.debug("Saved panorama to disk")
    logger.debug("Created panorama")
    logger.debug("Starting to run inference on panorama slices")
    pano = pano.transpose((2, 0, 1))
    inf_imgs, cropped_imgs, cropped_yaw_ranges = prepare_image_input(pano,
                                                                     (heading_offset, heading_offset - 2 * np.pi),
                                                                     radial_map_config.farfield_config.orientation_bin_size,
                                                                     model.name,
                                                                     device)
    logger.debug("starting forward pass")
    if eval_config.save_network_results:
        create_dir(trial_dirs.base_network_results_path / 'tensors')
    with torch.no_grad():
        predictions = model(inf_imgs)
        logger.debug("finished forward pass")
        predictions = predictions.detach()
        # to deal with numerical issues
        predictions = torch.round(predictions, decimals=3)
        if eval_config.save_network_results:
            torch.save(predictions, trial_dirs.base_network_results_path / 'tensors' / f'{replan_index:07d}.pt')
            dump_json_config(
                {
                    'yaw_ranges': cropped_yaw_ranges,
                    'replan_index': replan_index,
                    'state': current_state,
                    'radial_map_config': radial_map_config,
                }, trial_dirs.base_network_results_path / 'tensors' / f'{replan_index:07d}.json'
            )

    predictions = predictions.cpu().numpy()
    results = construct_radial_observations_output(
        cropped_yaw_ranges, predictions, (current_state.x, current_state.y),
        swath_library=swath_library,
        ff_config=radial_map_config.farfield_config,
        belief=belief,
    )

    # logger.debug(
    # f"Predicted classes {classes} with noise {noise} for yaw ranges {cropped_yaw_ranges}")
    logger.debug(
        f"Finished running ALL inference and adding observations on panorama slices.")
    return results, cropped_imgs


def calculate_labels(logger: logging.Logger, current_state: State, master_costmap: OccupancyGrid, radial_map_config: RadialMapConfig, train_config: TrainConfig,
                     heading_offset: float, swath_library: SwathLibrary, discard_invalid_labels: bool = True) -> tuple[list, list]:
    # calculate the costs from the ground truth traversability labels

    label_yaw_ranges = calculate_label_angle_pairs(
        (heading_offset, heading_offset - 2 * np.pi), radial_map_config.farfield_config.orientation_bin_size)
    labels = []
    for yaw_range in label_yaw_ranges:
        label, coverage = reduce_speed_to_class(costmap=master_costmap,
                                                location=(current_state.x, current_state.y),
                                                yaw_range=yaw_range,
                                                radial_boundaries=radial_map_config.farfield_config.range_bins,
                                                classification_config=radial_map_config.farfield_config.classification_config,
                                                swath_library=swath_library)
        labels.append(label)

    # predictions is 1-hot of shape (batch_size, num_bins, num_models, num_classes)
    predictions = np.zeros((len(label_yaw_ranges), radial_map_config.farfield_config.num_range_bins, 1, radial_map_config.farfield_config.classification_config.num_classes))

    for i, label in enumerate(labels):
        for j, speed_bin in enumerate(label):
            predictions[i, j, 0, speed_bin] = 2.5  # high confidence in label, softmax assigns prob 0.85
            # if confidence is too high, we can end up causing the planner to thrash
    results = construct_radial_observations_output(
        label_yaw_ranges, predictions, (current_state.x, current_state.y),
        swath_library=swath_library, ff_config=radial_map_config.farfield_config,
        belief=master_costmap)
    return results


def yaws_classes_uncertainties_from_radial_observation_list(obs_list: list[RadialObservation]):
    yaws, classes_out, uncertainties_out = [], [], []
    for observation in obs_list:
        yaws.append(observation.lr_yaw)
        classes_out.append(observation._calculate_classes())
        uncertainties_out.append(observation._calculate_uncertainty())
    return yaws, classes_out, uncertainties_out

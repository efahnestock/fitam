from __future__ import annotations
from fitam.mapping.spatial_label_general import create_image_pyramid, image_pyramid_to_batch, find_yaw_from_range_yaw_percentage
from fitam.learning.spatial_label_training import preprocess_image_pyramid_batch
from fury import window
import torch.nn as nn
import numpy as np
import time
import torch
import logging
from fitam.mapping.observation_types import SpatialLabelObservation
from fitam.planning.planner_general import snap_easl_state_to_planning_state, State
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.costmap_swath_library import SwathLibrary
from fitam.mapping.opengl_scene_rendering import render_scene
from fitam.core.config.EvaluationConfig import EvaluationConfig
from fitam.core.config.RadialMapConfig import RadialMapConfig
from fitam.evaluation.eval_data_structures import TrialDirectories
from fitam.mapping.belief import Belief


def spatial_label_observe(logger: logging.Logger,
                          current_state: State,
                          replan_index: int,
                          scene: window.Scene,
                          trial_directories: TrialDirectories,
                          belief: Belief,
                          model: nn.Module,
                          swath_library: SwathLibrary,
                          device: torch.device,
                          use_renderer: bool,
                          radial_map_config: RadialMapConfig,
                          eval_config: EvaluationConfig,
                          master_costmap: OccupancyGrid,
                          ) -> None:

    # The index (in the costmap) is now [current_planning_state.y, current_planning_state.x]
    current_planning_state = snap_easl_state_to_planning_state(
        belief, current_state)
    center_idx = (current_planning_state.y, current_planning_state.x)
    heading_offset = np.random.uniform(0, 2 * np.pi)
    logger.debug(f"Observing from state {current_state} with heading offset {heading_offset}")
    if radial_map_config.farfield_config is not None:  # if farfield is enabled
        if use_renderer:
            predicted_results = spatial_label_calculate_observation(
                scene=scene,
                robot_state_m=(current_state.x, current_state.y),
                left_yaw_rad=heading_offset,
                device=device,
                swath_library=swath_library,
                center_idx=center_idx,
                model=model,
                image_pyramid_config=radial_map_config.farfield_config.image_pyramid_config,
                traversable_cost_s_per_meter=radial_map_config.unknown_cost,
            )
            if eval_config.save_network_results:
                pass
                # yaws, classes, uncertainties = yaws_classes_uncertainties_from_radial_observation_list(predicted_results)
                # for i, img in enumerate(cropped_imgs):
                #     network_in_img = (chw_to_hwc_img(img) * 255).astype(np.uint8)
                #     fig, ax = plt.subplots()
                #     ax.imshow(network_in_img)
                #     title = f"Class: {[f'{x}' for x in classes[i]]}  Noise {[f'{x:.2f}' for x in uncertainties[i]]} \n Yaw range {[f'{x:.2f}' for x in yaws[i]]} "
                #     if true_classes is not None:
                #         title += f"True classes: {true_classes[i]}"
                #     ax.set_title(title)
                #     ax.axis('off')
                #     fig.savefig(trial_directories.base_network_results_path / f'{replan_index:07d}_{i:02d}.png', bbox_inches='tight')
                #     plt.close(fig)

        else:
            raise NotImplementedError("Have not implemented ground truth for spatial label")

        # update radial map!
        logger.debug("Populating the unknown map")

        belief.update_from_observations(
            predicted_results
        )
        logger.debug("Finished populating the unknown map")
        costmap = belief.get_full_costmap()
        if np.any(costmap < 0):
            print("negative costmap values after local update")


def spatial_label_calculate_observation(scene,
                                        robot_state_m: tuple[float, float],
                                        left_yaw_rad: float,
                                        device,
                                        model,
                                        swath_library: SwathLibrary,
                                        image_pyramid_config,
                                        traversable_cost_s_per_meter: float,
                                        center_idx: tuple[int, int],
                                        ) -> list[SpatialLabelObservation]:
    # render image
    # s = time.time()
    pano = render_scene(scene, (robot_state_m[0], robot_state_m[1], image_pyramid_config.camera_height_m), left_yaw_rad)
    # s1 = time.time()

    # split image into image pyramid
    image_pyramid = create_image_pyramid(pano, image_pyramid_config)
    # s2 = time.time()
    batch, range_and_yaw_percentage = image_pyramid_to_batch(image_pyramid, image_pyramid_config.image_slice_width_pixels)
    # s3 = time.time()
    model_inputs = preprocess_image_pyramid_batch(batch, device)
    # s4 = time.time()
    # run inference on all slices
    patch_classes = model(model_inputs).argmax(dim=1).cpu().numpy()
    # s5 = time.time()
    # map inference results to radial bins

    range_and_yaw_percentage[:, 1] = find_yaw_from_range_yaw_percentage(range_and_yaw_percentage, (left_yaw_rad, left_yaw_rad - 2 * np.pi))
    # s6 = time.time()
    obs = SpatialLabelObservation(center_idx, 
                                  patch_classes, 
                                  range_and_yaw_percentage, 
                                  left_yaw_rad, 
                                  swath_library, 
                                  traversable_cost_s_per_meter,
                                  image_pyramid_config.untraversable_cost_s_per_meter,
                                  left_yaw_rad)
    # s7 = time.time()
    # print("Render time: ", s1 - s)
    # print("image pyramid time: ", s2 - s1)
    # print("image pyramid to batch time: ", s3 - s2)
    # print("preprocess batch time: ", s4 - s3)
    # print("model time: ", s5 - s4)
    # print("finding yaw time: ", s6 - s5)
    # print("creating observation time: ", s7 - s6)
    return [obs]

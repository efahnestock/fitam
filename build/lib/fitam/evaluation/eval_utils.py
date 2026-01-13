from __future__ import annotations
from typing import Optional
import numpy as np
import glob
import tqdm
import time
import logging
from pathlib import Path
from os import PathLike
import os
import pickle
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from fitam.core.common import create_dir, load_json_config, dump_json_config, update_fileHandler
from fitam.core.product_structures import EvaluationRequest
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.land_cover_complex_map import LandCoverComplexMap
from fitam.planning.astar import A_Star
from fitam import MAPS_DIR
from fitam.mapping.kalman_belief_updater import KalmanBeliefUpdaterConfig, KalmanBeliefUpdater
from fitam.mapping.most_recent_belief_updater import MostRecentBeliefUpdaterConfig, MostRecentBeliefUpdater
from fitam.mapping.no_uncertaitny_average_belief_updater import NoUncertaintyBeliefUpdaterConfig, NoUncertaintyAverageBeliefUpdater
from fitam.mapping.belief import Belief
from fitam.planning.dstar_lite_interface import D_Star_Interface
from fitam.mapping.costmap_swath_library import SwathLibrary
from fitam.core.config.LoggingConfig import LoggingConfig
from fitam.core.config.RadialMapConfig import RadialMapConfig, FarFieldConfig, MapFusionType, SpatialLabelConfig
from fitam.evaluation.eval_data_structures import DebugTrialStatistics, CriticalTrialStatistics
from fitam.planning.planner_general import snap_easl_state_to_planning_state, State
from fitam.mapping.map_tools import find_map_bounds_around_point, find_close_crop_bounds_around_current_state
from fitam.core.config.RadialMapConfig import FarFieldConfig
from fitam.core.common import angle_between_lr_yaw
from fitam.core.data_tools import crop_image_for_sector
from fitam.learning.preprocessing_transforms import get_default_transforms


def create_single_eval_request_for_env(eval_config_path: os.PathLike,
                                       map_path: os.PathLike,
                                       save_path: os.PathLike) -> None:

    eval_config = load_json_config(eval_config_path)
    print(eval_config)
    cmap = LandCoverComplexMap.from_map_folder(map_path)
    occ_grid = OccupancyGrid.from_complexmap(cmap)
    planner = A_Star(occ_grid)

    create_dir(save_path.parent)

    sampled_planning_queries = occ_grid.sample_start_goal_pairs(eval_config.num_planning_queries_per_map,
                                                                eval_config.min_eval_path_length_m,
                                                                planner,
                                                                eval_config.eval_point_sampling_dialation_radius_m,
                                                                eval_config.eval_min_dist_from_map_edges_m)

    map_path_to_save = map_path
    if MAPS_DIR in map_path_to_save.parents:
        map_path_to_save = map_path_to_save.relative_to(MAPS_DIR)
    eval_request = EvaluationRequest(
        sampled_planning_queries,
        map_path_to_save
    )
    dump_json_config(eval_request, save_path)


def get_model_md5(model):
    def convert_dict(d, flat_dict):
        for k, v in d.items():
            if isinstance(v, dict):
                convert_dict(v, flat_dict)
            else:
                if isinstance(v, torch.Tensor):
                    v = v.cpu().detach().numpy().tolist()
                flat_dict[k] = v
    _dict = model.state_dict()
    flat_dict = {}
    convert_dict(_dict, flat_dict)
    data_md5 = hashlib.md5(json.dumps(flat_dict, sort_keys=True).encode('utf-8')).hexdigest()
    return data_md5


def monitor_workers(
    eval_request_paths: list[PathLike],
    save_roots: list[PathLike],
    reprint_rate: float = 1.0,  # in seconds
) -> None:
    # figure out how many points there are total
    total_points = 0
    for eval_request_path in eval_request_paths:
        eval_request = load_json_config(eval_request_path)
        total_points += len(eval_request.points)

    def count_complete_trials(save_roots: list[PathLike]) -> int:
        all_paths = []
        for save_root in save_roots:
            all_paths.extend(glob.glob(str(save_root / "**/trial_info.json")))
        # remove duplicates (if any)
        all_paths = list(set(all_paths))
        return len(all_paths)

    # start_time = time.time()
    old_finished_trials = count_complete_trials(save_roots)

    bar = tqdm.tqdm(total=total_points)
    bar.set_description("Trials Completed")
    bar.update(old_finished_trials)
    while True:
        new_finished_trials = count_complete_trials(save_roots)
        bar.update(new_finished_trials - old_finished_trials)
        time.sleep(reprint_rate)
        old_finished_trials = new_finished_trials
        # time_elapsed = time.time() - start_time
        # delta_tasks = new_finished_trials - old_finished_trials
        # rate = delta_tasks / time_elapsed
        # time_left = (total_points - new_finished_trials) / rate
        if new_finished_trials == total_points:
            break
    bar.close()
    return


def get_incomplete_queries_in_dir(job_numbers, results_dir) -> list[int]:
    """Given a list of query number IDs (job_numbers) and a results dir, return the subset of job_numbers that are incomplete.
    Relies on the fact that trial_info.json does not exist in a results directory unless the trial completed
    """
    existing_result_dirs = list(
        sorted(glob.glob(os.path.join(results_dir, '*/trial_info.json'))))
    existing_result_ids = [os.path.basename(os.path.dirname(x))
                           for x in existing_result_dirs]
    existing_result_ids = [int(x) for x in existing_result_ids if x.isdigit()]
    existing_result_ids = np.array(existing_result_ids)
    remaining_job_numbers = [
        x for x in job_numbers if x not in existing_result_ids]
    return remaining_job_numbers


def log_status_and_change_logger(num_success, num_failed: int, logger: logging.Logger, logging_config: LoggingConfig,
                                 query_i: int, global_start_time: float, results_dir: str, job_numbers: list[int], uid: str) -> None:
    """
    Printout last run stats and chagne the logger to the worker's log file
    """
    # do some semi-regular printouts
    total_queries_completed = num_success + num_failed
    remaining_queries = len(job_numbers) - total_queries_completed
    query_end_time = time.time()
    running_time = query_end_time - global_start_time
    est_completion_time = running_time * remaining_queries / total_queries_completed
    update_fileHandler(logger, os.path.join(
        results_dir, f'worker_{uid}.log'), logging_config.file_logging_level, logging_config.logging_format)
    logger.debug(f"Worker {uid}: Just finished {query_i}. Overall {total_queries_completed} queries completed in {running_time / 60 :.3f}, {remaining_queries} remaining. Estimated completion time: {est_completion_time / 60 :.3f} minutes. Failed {num_failed} queries.")


def filter_job_numbers(job_numbers: list[int], results_dir: str) -> Optional[list[int]]:
    unfinished_numbers = []
    for number in job_numbers:
        if not os.path.exists(os.path.join(results_dir, f"{number:07d}", 'trial_info.json')):
            unfinished_numbers.append(number)
    return unfinished_numbers


def change_logger_to_trial_dir(logger: logging.Logger, query_results_dir: PathLike, logging_config: LoggingConfig, query_i: int) -> None:
    # change the logger to run in the query directory
    update_fileHandler(logger, query_results_dir / f'trial-{time.strftime("%Y_%m_%d-%H_%M_%S")}.log',
                       logging_config.file_logging_level, logging_config.logging_format)
    logger.debug(f"Starting query {query_i} in directory {query_results_dir}")
    return


def setup_trial_assets(radial_map_config: RadialMapConfig,
                       master_costmap: OccupancyGrid,
                       swath_library: SwathLibrary,
                       start_state: State,
                       goal_state: State) -> tuple[Belief, D_Star_Interface]:
    costmap_fusion_type = radial_map_config.map_fusion_type
    blank_costmap = OccupancyGrid.from_costmap(
        master_costmap, copy_data=False)

    if costmap_fusion_type == MapFusionType.KALMAN:
        kbu_config = KalmanBeliefUpdaterConfig(
            process_noise=radial_map_config.process_noise,
            variance_threshold=radial_map_config.variance_threshold,
            initial_cost=radial_map_config.unknown_cost,
            unknown_space_cost=radial_map_config.unknown_cost,
            initial_variance=radial_map_config.variance_threshold,
            observation_variance_threshold=radial_map_config.observation_variance_threshold,
        )
        kbu = KalmanBeliefUpdater(kbu_config)
        belief = Belief(kbu, blank_costmap)

    elif costmap_fusion_type == MapFusionType.MIN_ENTROPY:
        raise NotImplementedError()
    elif costmap_fusion_type == MapFusionType.MOST_RECENT:
        mr_config = MostRecentBeliefUpdaterConfig(
            initial_cost=radial_map_config.unknown_cost,
            unknown_space_cost=radial_map_config.unknown_cost
        )
        mr_belief_updater = MostRecentBeliefUpdater(mr_config)
        belief = Belief(mr_belief_updater, blank_costmap)
    elif costmap_fusion_type == MapFusionType.NO_UNCERTAINTY_AVERAGE:
        avg_config = NoUncertaintyBeliefUpdaterConfig(
            initial_cost=radial_map_config.unknown_cost,
            unknown_space_cost=radial_map_config.unknown_cost
        )
        avg_belief_updater = NoUncertaintyAverageBeliefUpdater(avg_config)
        belief = Belief(avg_belief_updater, blank_costmap)
    else:
        raise ValueError(
            f"costmap_fusion_type must be one of {MapFusionType.__members__}, got {costmap_fusion_type} instead.")
    planner = D_Star_Interface(belief, start_state, goal_state,
                               check_with_astar=False, log_all_state_changes=False)
    return belief, planner


def convert_dts_to_dict(dts: DebugTrialStatistics) -> dict:
    dict_version = {x: getattr(dts, x) for x in dir(
        dts) if not x.startswith('__') and not callable(getattr(dts, x))}
    for key, value in dict_version.items():
        # check Pathlib paths
        if isinstance(value, Path):
            dict_version[key] = str(value)
    return dict_version


def convert_dict_to_dts(dict_version: dict) -> DebugTrialStatistics:
    dts = DebugTrialStatistics()
    for key, value in dict_version.items():
        setattr(dts, key, value)
    return dts


def save_results(logger: logging.Logger, cts: CriticalTrialStatistics, dts: DebugTrialStatistics, query_results_dir: PathLike) -> None:
    #           trial_info.json - small with just critical info
    #           debug_info.json - large with all info
    dump_json_config(cts, query_results_dir / 'trial_info.json')
    dict_version = convert_dts_to_dict(dts)
    with open(query_results_dir / 'debug_info.pkl', 'wb') as f:
        pickle.dump(dict_version, f)
    logger.debug(
        "Finished saving the information dump. We're done with this trial now")
    return


def plot_map_summary(
        belief: Belief,
        variance: Optional[np.ndarray] = None,
        ff_config: Optional[FarFieldConfig] = None,
        max_cost: Optional[float] = None,
        gt_costmap: Optional[OccupancyGrid] = None,
        gt_semantic_map: Optional[LandCoverComplexMap] = None,
        state_history: Optional[list[tuple[int, int]]] = None,
        current_plan: Optional[list[tuple[int, int]]] = None,
        start_state: Optional['State'] = None,
        end_state: Optional['State'] = None,) -> plt.Figure:
    """
    Plot the state, variance, and planning map with colorbars
    In the 4th subplot, in text show information about the radial map's configuration (e.g. range bins, yaw bins, etc.)
    """

    def plot_paths_on_img(img, state_history, current_plan, start, goal):
        img_w_history = gt_costmap.visualize_states(
            [state_history[-1]], states_are_idxs=True, img=img, n_pixels=5)
        if state_history is not None:
            img_w_history = gt_costmap.visualize_states(
                state_history, states_are_idxs=True, img=img_w_history, n_pixels=1, color=(0, 1, 0))
        if current_plan is not None:
            img_w_history = gt_costmap.visualize_states(
                current_plan, states_are_idxs=True, img=img_w_history, n_pixels=1, color=(1, 1, 1))
        if start is not None:
            img_w_history = gt_costmap.visualize_states(
                [start], states_are_idxs=True, img=img_w_history, n_pixels=5, color=(0.1, 0.9, 0.1))
        if goal is not None:
            img_w_history = gt_costmap.visualize_states(
                [goal], states_are_idxs=True, img=img_w_history, n_pixels=5, color=(0.9, 0.1, 0.1))
        return img_w_history
    min_idxs_close, max_idxs_close = find_close_crop_bounds_around_current_state(
        belief, state_history[-1], 150)

    def crop_local(x): return x[min_idxs_close[0]:max_idxs_close[0], min_idxs_close[1]:max_idxs_close[1]]
    min_idxs, max_idxs = [np.inf, np.inf], [-np.inf, -np.inf]
    for obs in belief.observation_history:
        center_idx = obs.center_idx
        max_obs_range = None
        if isinstance(ff_config, FarFieldConfig):
            max_obs_range = ff_config.range_bins[-1]
        elif isinstance(ff_config, SpatialLabelConfig):
            max_obs_range = ff_config.image_pyramid_config.max_range_m

        min_idxs, max_idxs = find_map_bounds_around_point(
            belief, center_idx, max_obs_range, False, min_idxs, max_idxs, True)
    for state in state_history:
        # NOTE: if keep_square is kept here, it breaks something about the keep_squre bit. Not looking into it now
        min_idxs, max_idxs = find_map_bounds_around_point(
            belief, state, 150, False, min_idxs, max_idxs, False)
    if start_state is not None:
        start_state = snap_easl_state_to_planning_state(
            belief, State(*start_state))
        start_state = (start_state.y, start_state.x)
        min_idxs, max_idxs = find_map_bounds_around_point(
            belief, start_state, 150, False, min_idxs, max_idxs, True)
    if end_state is not None:
        end_state = snap_easl_state_to_planning_state(
            belief, State(*end_state))
        end_state = (end_state.y, end_state.x)
        min_idxs, max_idxs = find_map_bounds_around_point(
            belief, end_state, 150, False, min_idxs, max_idxs, True)

    def crop_global(x): return x[min_idxs[0]:max_idxs[0], min_idxs[1]:max_idxs[1]]

    colorbar_kwargs = dict()
    if max_cost is not None:
        colorbar_kwargs['vmax'] = max_cost
        colorbar_kwargs['vmin'] = 0
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    # fig.colorbar(ax[0,0].imshow(crop_local(belief.state), **colorbar_kwargs), ax=ax[0,0])
    # ax[0,0].set_title('State')
    if variance is not None:
        fig.colorbar(ax[1, 2].imshow(crop_local(
            variance), vmin=0, vmax=0.5), ax=ax[0, 1])
        ax[1, 2].set_title('Variance')
    # apply colorbar and get image
    color_img = belief.get_full_costmap()

    def get_colors(inp, colormap=plt.cm.viridis, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))[:, :, 0:3]
    color_img = get_colors(color_img, plt.cm.viridis, **colorbar_kwargs)
    color_img = plot_paths_on_img(
        color_img, state_history, current_plan, start_state, end_state)
    ax[0, 0].imshow(crop_local(color_img))
    ax[0, 0].set_title('Planning Map')

    ax[1, 0].imshow(crop_global(color_img), **colorbar_kwargs)
    ax[1, 0].set_title('Planning map global')
    ax[1, 0].axis('off')

    ax[0, 1].axis('off')
    if gt_costmap is not None:
        base_img = None
        if gt_semantic_map is not None:
            base_img = gt_semantic_map.create_floormask()
        img_w_history = plot_paths_on_img(
            base_img, state_history, current_plan, start_state, end_state)
        ax[0, 1].imshow(crop_local(img_w_history), **colorbar_kwargs)
        ax[1, 1].imshow(crop_global(img_w_history), **colorbar_kwargs)
    ax[0, 1].set_title('Ground Truth Map')
    ax[0, 1].axis('off')
    ax[1, 1].set_title('Ground Truth Global')
    ax[1, 1].axis('off')

    fig.colorbar(ax[0, 2].imshow(crop_local(gt_costmap.data),
                 **colorbar_kwargs), ax=ax[0, 2])
    ax[0, 2].set_title('Ground Truth Cost Map')
    ax[0, 2].axis('off')

    fig.tight_layout()
    return fig


def calculate_label_angle_pairs(yaw_range, yaw_bin_size):
    # 1e-6 so if yaw_bin_size is multiple of angle_between_lr_yaw, we don't end up with
    # an extra bin
    num_bins = int((angle_between_lr_yaw(*yaw_range)-1e-6) // yaw_bin_size + 1)
    bin_left_yaws = np.linspace(yaw_range[0], yaw_range[1] + yaw_bin_size, num_bins)
    cropped_yaw_ranges = []
    for bin_left_yaw in bin_left_yaws:
        cropped_yaw_ranges.append((bin_left_yaw, bin_left_yaw - yaw_bin_size + 1e-7))
    return cropped_yaw_ranges

def crop_image_input(
        image:np.ndarray,
        image_yaws:tuple,
        yaw_bin_size:float
)->tuple[np.ndarray, list]:
    assert image.dtype == np.float32, f"image must be float32, got {image.dtype}"
    assert image.shape[0] == 3, f"image must be in CHW format, got shape {image.shape}"

    cropped_images = []
    cropped_yaw_ranges = calculate_label_angle_pairs(image_yaws, yaw_bin_size)
    for (bin_left_yaw, bin_right_yaw) in cropped_yaw_ranges:
        img = crop_image_for_sector(image, image_yaws, (bin_left_yaw, bin_right_yaw), yaw_bin_size)
        cropped_images.append(img)
    cropped_images = np.stack(cropped_images, axis=0)
    return cropped_images, cropped_yaw_ranges

def transform_image_input(cropped_images:np.ndarray,
                          model_name:str,
                          device:torch.device)->torch.tensor:
    # convert to tensor
    cropped_images = torch.from_numpy(cropped_images).to(device)
    tfs = transforms.Compose(get_default_transforms(model_name, image_size=(cropped_images.shape[2], cropped_images.shape[3])))
    inf_images = tfs(cropped_images)
    return inf_images

def prepare_image_input(image:np.ndarray, 
                        image_yaws:tuple, 
                        yaw_bin_size:float,
                        model_name:str,
                        device:torch.device)->torch.tensor:
    cropped_images, cropped_yaw_ranges = crop_image_input(image, image_yaws, yaw_bin_size) 
    inf_images = transform_image_input(cropped_images, model_name, device)
    return inf_images, cropped_images, cropped_yaw_ranges


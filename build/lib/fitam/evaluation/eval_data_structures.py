from __future__ import annotations

from typing import NamedTuple
from dataclasses import dataclass, field
import logging
from os import PathLike
from torch import nn
import torch
import numpy as np

from fury import window
from enum import Enum
from fitam.mapping.land_cover_complex_map import LandCoverComplexMap
from fitam.planning.planner_general import l2_dist_heuristic, list_of_state_to_tuples, State
from fitam.mapping.belief import Belief
from fitam.learning.load_by_name import load_model
from fitam.learning.spatial_label_training import load_spatial_label_model
from fitam.core.common import get_device, create_dir, load_json_config
from fitam.planning.astar import A_Star
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.costmap_swath_library import SwathLibrary, load_swath_library_from_pkl
from fitam.mapping.opengl_scene_rendering import create_scene
from fitam.core.product_structures import EvaluationRequest
from fitam.core.config.EvaluationConfig import EvaluationConfig
from fitam.core.config.LoggingConfig import LoggingConfig
from fitam.core.config.RadialMapConfig import RadialMapConfig, FarFieldConfig, SpatialLabelConfig
from fitam.core.config.TrainConfig import TrainConfig
from fitam.core.config.DatasetConfig import DatasetConfig
from fitam import MAPS_DIR


class ObserveFunctionType(Enum):
    FARFIELD = "farfield"
    SPATIAL_LABEL = "spatial_label"
    LOCAL = "local"


class WorkerAssets(NamedTuple):
    observe_function_type: ObserveFunctionType
    disable_farfield: bool
    use_renderer: bool
    scene: window.Scene
    master_costmap: OccupancyGrid
    land_cover_complex_map: LandCoverComplexMap
    occlusion_map: np.ndarray
    omnipresent_planner: A_Star
    observed_states: list[State]
    swath_library: SwathLibrary
    model: nn.Module  # | None
    device: torch.device
    eval_config: EvaluationConfig
    radial_map_config: RadialMapConfig
    logging_config: LoggingConfig
    eval_request: EvaluationRequest
    training_config: TrainConfig
    dataset_config: DatasetConfig


class TrialDirectories(NamedTuple):
    query_results_dir: PathLike
    base_pano_path: PathLike
    base_local_costmap_path: PathLike
    base_network_results_path: PathLike
    base_dataset_path: PathLike


@dataclass
class CriticalTrialStatistics:
    start_state: tuple = ()
    goal_state: tuple = ()
    reached_goal: bool = False
    farfield_enabled: bool = False
    ground_truth_cost: float = 0.0
    ground_truth_length: float = 0.0
    number_replans: int = 0
    total_cost: float = 0.0


@dataclass
class DebugTrialStatistics:
    ground_truth_path: list = field(default_factory=list)
    ground_truth_cost_history: list = field(default_factory=list)
    model_name: str = ""
    complex_map_path: str = ""
    run_config_path: str = ""
    config_idx: int = 0
    radial_map_observation_history: list = field(
        default_factory=list)  # the observations we've made
    # the indexes in state_history between planning queries
    state_transition_indexes: list[int] = field(default_factory=list)
    state_history: list[tuple[int, int]] = field(default_factory=list)
    # the locations we've actually moved to
    path_history: list[tuple[int, int]] = field(default_factory=list)
    # the costs of the states we've actually moved to
    cost_history: list[float] = field(default_factory=list)


def setup_worker_assets_and_config(
        eval_request_path: PathLike,
        radial_map_config_path: PathLike,
        swath_library_path: PathLike,
        logging_config_path: PathLike,
        eval_config_path: PathLike,
        training_config_path: PathLike,
        dataset_config_path: PathLike,
        model_path: str) -> WorkerAssets:

    # load configs
    eval_config = load_json_config(eval_config_path)
    radial_map_config = load_json_config(radial_map_config_path)
    logging_config = load_json_config(logging_config_path)
    training_config = None
    if not training_config_path is None:
        training_config = load_json_config(training_config_path)
    dataset_config = None
    if not dataset_config_path is None:
        dataset_config = load_json_config(dataset_config_path)

    # load the evaluation request
    eval_request = load_json_config(eval_request_path)

    complex_map_path = MAPS_DIR / eval_request.map_path
    complex_map = LandCoverComplexMap.from_map_folder(complex_map_path)
    occlusion_map = complex_map.create_occlusion_map()

    disable_farfield = radial_map_config.farfield_config is None
    use_renderer = not disable_farfield and not model_path is None
    # if we are using farfield and trained a network for it.. we need blender to render images
    scene = None

    # set up evaluation model, costmaps, planner
    # set up global and local costmap from the complex map
    master_costmap = OccupancyGrid.from_complexmap(complex_map)
    # only used to find ground truth shortest paths
    omnipresent_planner = A_Star(master_costmap)
    observed_states = np.zeros_like(master_costmap.data, dtype=bool)

    # load in the swath library
    swath_library = load_swath_library_from_pkl(swath_library_path)

    assert swath_library.mask_resolution == master_costmap.resolution, "Swath library and costmap resolution must match"

    # set up the model
    device = get_device()
    model = None
    if isinstance(radial_map_config.farfield_config, FarFieldConfig):
        observe_function_type = ObserveFunctionType.FARFIELD
        if use_renderer:
            scene = create_scene(complex_map, complex_map_path / f"{complex_map_path.name}.png")

            model = load_model(checkpoint_path=model_path, get_lightning=True)
            model.to(device)
            model.model.use_features = False  # features have not been pre-calculated
            model.eval()
            if device.type == 'cuda':
                model.cuda()
    elif isinstance(radial_map_config.farfield_config, SpatialLabelConfig):
        observe_function_type = ObserveFunctionType.SPATIAL_LABEL
        scene = create_scene(complex_map, complex_map_path / f"{complex_map_path.name}.png")
        model = load_spatial_label_model(model_path)
        model.eval()
        if device.type == 'cuda':
            model.cuda()

    else:
        observe_function_type = ObserveFunctionType.LOCAL

    return WorkerAssets(
        observe_function_type=observe_function_type,
        disable_farfield=disable_farfield,
        use_renderer=use_renderer,
        land_cover_complex_map=complex_map,
        occlusion_map=occlusion_map,
        scene=scene,
        master_costmap=master_costmap,
        omnipresent_planner=omnipresent_planner,
        observed_states=observed_states,
        swath_library=swath_library,
        model=model,
        device=device,
        eval_config=eval_config,
        radial_map_config=radial_map_config,
        logging_config=logging_config,
        training_config=training_config,
        dataset_config=dataset_config,
        eval_request=eval_request
    )


def setup_trial_directories(results_dir: PathLike, query_i: int, eval_config: EvaluationConfig) -> TrialDirectories:
    query_results_dir = results_dir / f'{query_i:07d}'
    create_dir(query_results_dir, True)
    base_pano_path = query_results_dir / 'pano'
    base_local_costmap_path = query_results_dir / 'local_costmap'
    base_network_results_path = query_results_dir / 'network_results'
    base_dataset_path = query_results_dir / 'dataset'
    if eval_config.save_panoramas:
        create_dir(base_pano_path, True)  # overwrites trial folders
    if eval_config.save_local_costmaps:
        create_dir(base_local_costmap_path)
    if eval_config.save_network_results:
        create_dir(base_network_results_path)
    if eval_config.save_dataset:
        create_dir(base_dataset_path)
        create_dir(base_dataset_path / 'images')
    return TrialDirectories(
        query_results_dir=query_results_dir,
        base_pano_path=base_pano_path,
        base_local_costmap_path=base_local_costmap_path,
        base_network_results_path=base_network_results_path,
        base_dataset_path=base_dataset_path

    )


def setup_trial_data_structures(eval_request: EvaluationRequest, query_i: int) -> tuple[CriticalTrialStatistics, DebugTrialStatistics]:
    cts = CriticalTrialStatistics()
    cts.start_state = eval_request.points[query_i][0]
    cts.goal_state = eval_request.points[query_i][1]
    dts = DebugTrialStatistics()

    return cts, dts


def compile_results(logger: logging.Logger, omnipresent_planner: A_Star, start_state: State, goal_state: State,
                    reached_goal: bool, disable_farfield: bool, cts: CriticalTrialStatistics, dts: DebugTrialStatistics, query_i: int,
                    replan_index: int, total_path_cost: float, model_path: str, complex_map_path: str,
                    local_costmap: Belief) -> None:
    # find the length and cost of the shortest path between the start and goal
    logger.debug("Compiling results!")
    logger.debug(
        f"Calling the omnipresent planner with arguments {start_state} and {goal_state}")
    path, accrued_costs = omnipresent_planner.search(start_state, goal_state)
    ground_truth_path_length = 0
    for i in range(len(path) - 1):
        ground_truth_path_length += l2_dist_heuristic(path[i], path[i+1])

    if reached_goal:
        logger.debug(f"Finished query {query_i} with {replan_index} replans")
    else:
        logger.warning(
            f"FAILED QUERY: Max planning iterations reached. Failed query {query_i} with {replan_index} replans")
    # dump critical information
    cts.reached_goal = reached_goal
    cts.farfield_enabled = not disable_farfield
    cts.ground_truth_cost = float(accrued_costs[-1])
    cts.ground_truth_length = float(ground_truth_path_length)
    cts.number_replans = replan_index
    cts.total_cost = float(total_path_cost)

    # dump debug information
    dts.ground_truth_path = list_of_state_to_tuples(path, costmap=local_costmap, convert_to_ps=True)
    dts.ground_truth_cost_history = accrued_costs
    dts.model_name = model_path
    dts.complex_map_path = complex_map_path
    dts.radial_map_observation_history = local_costmap.observation_history

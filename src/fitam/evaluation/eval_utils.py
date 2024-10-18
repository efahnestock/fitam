import os
from fitam.core.common import create_dir, load_json_config, dump_json_config
from fitam.core.product_structures import EvaluationRequest
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.land_cover_complex_map import LandCoverComplexMap
from fitam.planning.astar import A_Star
from fitam import MAPS_DIR


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

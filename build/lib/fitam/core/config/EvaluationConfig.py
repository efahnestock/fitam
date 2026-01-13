from dataclasses import dataclass


@dataclass
class EvaluationConfig:

    num_planning_queries_per_map: int = 100
    min_eval_path_length_m: float = 400  # meters
    eval_point_sampling_dialation_radius_m: float = 2  # how much to dialate the envrionment before sampling start/goal pairs
    eval_min_dist_from_map_edges_m: float = 100  # meters

    max_planning_iterations: int = 500  # max iterations before we give up on a path

    movement_per_replan_m: float = 50  # meters we move between replanning to the goal in the sim

    save_panoramas: bool = True  # save panos during eval
    resume: bool = True  # resume partially completed evaluation trials. Should not be required
    save_paths: bool = True
    save_network_results: bool = True  # save the predictions on the images from the network
    save_local_costmaps: bool = True  # save the local costmaps
    save_dataset: bool = True  # save the dataset
    overwrite_path_on_costmap: bool = True  # add annotation to the saved costmaps
    overwrite_past_states_on_costmap: bool = True  # add past states to the saved costmaps
    save_final_path: bool = True  # save the final path
    save_paths: bool = True
    save_costs: bool = True

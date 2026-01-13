#! /usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

from fitam import MAPS_DIR, SAMPLED_LOCATIONS_DIR
import fitam.mapping.land_cover_complex_map as lcm
from fitam.mapping.local_observation import local_observe_occlusions
from fitam.planning.astar import A_Star
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.kalman_belief_updater import KalmanBeliefUpdater, KalmanBeliefUpdaterConfig
from fitam.mapping.belief import Belief
from fitam.mapping.costmap_swath_library import SwathLibrary
from fitam.planning.planner_general import State
from fitam.core.product_structures import ImageGenerationRequest
from fitam.core.config.RadialMapConfig import RadialMapConfig
from fitam.core.common import dump_json_config


def get_cost_maps_and_swath_library(map_path: Path, obs_rad_m: float):
    radial_map_config = RadialMapConfig(observation_radius=obs_rad_m)
    swath_library = SwathLibrary.from_radial_map_config(
        radial_map_config, generate=True
    )
    map = lcm.LandCoverComplexMap.from_map_folder(map_path)
    gt_costmap = OccupancyGrid.from_complexmap(map)
    blank_costmap = OccupancyGrid.from_costmap(
        gt_costmap, copy_data=False)
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
    gt_occlusion_map = map.create_occlusion_map()

    return gt_costmap, belief, swath_library, gt_occlusion_map


def plot_planning_result(
    occupancy_grid: OccupancyGrid, trajectory: list[State], waypoints: list[State]
):
    plt.imshow(occupancy_grid.get_color_image())


def compute_trim_idx(trajectory: list[State], obs_rad_m: float):
    accum_dist_m = 0
    i = 1
    for i in range(1, len(trajectory)):
        curr_state = trajectory[i]
        prev_state = trajectory[i - 1]
        step_dist_m = np.linalg.norm(
            [(curr_state.x - prev_state.x), (curr_state.y - prev_state.y)]
        )
        accum_dist_m += step_dist_m
        if accum_dist_m > obs_rad_m:
            break
    return i


def step(
    start: State,
    goal: State,
    gt_costmap: OccupancyGrid,
    gt_occlusion_map: np.ndarray,
    local_observation_tracker: np.ndarray,
    swath_library: SwathLibrary,
    local_costmap: Belief,
):
    # make an observation at the start
    start_idx = local_costmap.get_index_from_point((start.x, start.y))
    observations = local_observe_occlusions(
        robot_idx=start_idx,
        gt_map=gt_costmap,
        gt_occlusion_map=gt_occlusion_map,
        local_observation_tracker=local_observation_tracker,
        swath_lib=swath_library,
    )
    local_costmap.fix_known_states(observations)

    # Make a plan towards the goal
    planner = A_Star(local_costmap)
    # print("planning", start, goal)
    path, accrued_costs = planner.search(start, goal, return_easl_coords=True, allow_failure=True)
    # print("path cost is ", accrued_costs[-1])
    # print("initial path length", len(path))
    # Trim the plan if it's longer than obs_rad_m
    threshold_idx = len(path)
    for path_idx in range(0, len(path)):
        state = path[path_idx]
        cost = local_costmap.get_value((state.x, state.y))
        idxs = local_costmap.get_index_from_point((state.x, state.y))
        if np.isinf(cost):
            raise RuntimeError("plan had an infintie cost in it??")
        if not local_observation_tracker[idxs[0], idxs[1]]:  # if cell not observed
            # trim path
            threshold_idx = path_idx
            break
    # print("threshold idx", threshold_idx)
    path = path[:threshold_idx]
    added_time_cost = accrued_costs[threshold_idx - 1]
    # Add all observations accumulated during the traversal
    for path_idx in range(0, len(path)):
        state = path[path_idx]
        idxs = local_costmap.get_index_from_point((state.x, state.y))
        cost = local_costmap.get_value((state.x, state.y))
        if np.isinf(cost):
            raise RuntimeError("plan had an infintie cost in it??")
        observations = local_observe_occlusions(
            robot_idx=idxs,
            gt_map=gt_costmap,
            gt_occlusion_map=gt_occlusion_map,
            local_observation_tracker=local_observation_tracker,
            swath_lib=swath_library,
        )
        local_costmap.fix_known_states(observations)

    reached_goal = path[-1].x == goal.x and path[-1].y == goal.y

    return reached_goal, path, local_costmap, added_time_cost


def plan_to_waypoint(
    start: State,
    goal: State,
    max_iters: int,
    gt_costmap: OccupancyGrid,
    gt_occlusion_map: np.ndarray,
    swath_lib: SwathLibrary,
    local_observation_tracker: np.ndarray,
    local_costmap: OccupancyGrid,
    max_time_s: float,
    current_accrued_time: float,
):
    trajectory = [start]
    for i in range(max_iters):
        is_at_goal, new_segment, local_costmap, added_time_cost = step(
            start=trajectory[-1],
            goal=goal,
            gt_costmap=gt_costmap,
            gt_occlusion_map=gt_occlusion_map,
            local_observation_tracker=local_observation_tracker,
            swath_library=swath_lib,
            local_costmap=local_costmap
        )
        current_accrued_time += added_time_cost
        #  The start position is part of the next segment, so we remove it
        trajectory = trajectory[:-1] + new_segment
        if is_at_goal or current_accrued_time > max_time_s:
            break
    return trajectory, local_costmap, current_accrued_time


def wander_on_map(
    max_iters_per_waypoint: int,
    max_time_s: float,
    gt_costmap: OccupancyGrid,
    gt_occlusion_map: np.ndarray,
    local_observation_tracker: np.ndarray,
    swath_library: SwathLibrary,
    local_costmap: OccupancyGrid,
    rng: np.random.Generator,
):
    num_waypoints = 1000
    # Generate a set of waypoints
    waypoint_area = 3000
    xs = rng.uniform(-waypoint_area // 2, waypoint_area // 2, size=(num_waypoints + 1,))
    ys = rng.uniform(-waypoint_area // 2, waypoint_area // 2, size=(num_waypoints + 1,))

    # make sure start position is traversable
    while True:
        start_value = gt_costmap.get_value((xs[0], ys[0]))
        if not np.isinf(start_value):
            print("found valid start position", xs[0], ys[0], "with cost", start_value)
            break
        print("Had to resample start position as it was in an obstacle")
        xs[0] = rng.uniform(-waypoint_area // 2, waypoint_area // 2)
        ys[0] = rng.uniform(-waypoint_area // 2, waypoint_area // 2)

    waypoints = [State(x=xs[i], y=ys[i]) for i in range(num_waypoints + 1)]

    trajectory = [waypoints[0]]
    used_waypoints = [waypoints[0]]
    accrued_time = 0.0
    print("Starting to plan to waypoints")
    for i, waypoint in enumerate(waypoints[1:]):
        print("setting waypoint to ", waypoint)
        new_waypoint = waypoint
        used_waypoints.append(new_waypoint)
        new_path_seg, local_costmap, accrued_time = plan_to_waypoint(
            start=trajectory[-1],
            goal=new_waypoint,
            max_iters=max_iters_per_waypoint,
            gt_costmap=gt_costmap,
            gt_occlusion_map=gt_occlusion_map,
            swath_lib=swath_library,
            local_observation_tracker=local_observation_tracker,
            local_costmap=local_costmap,
            max_time_s=max_time_s,
            current_accrued_time=accrued_time,
        )
        trajectory = trajectory[:-1] + new_path_seg
        print(f"CompAccrued time: {accrued_time} seconds after {i+1} waypoints")

        if accrued_time > max_time_s:
            break

    if accrued_time < max_time_s:
        raise RuntimeError(f"Failed to reach time limit. Only {accrued_time} seconds used out of {max_time_s} seconds.")

    return trajectory, local_costmap, used_waypoints

def generate_image_request(locations: list[State], map_path: Path, times: list[float] = None):
    return ImageGenerationRequest(
        locations=locations,
        angles=[0] * len(locations),
        map_idx=[0] * len(locations),
        complex_map_paths=[str(map_path)],
        times=times,
    )


def main(
    max_iters_per_waypoint: int,
    obs_rad_m: float,
    meters_between_keypoints: float,
    seconds_per_image_req: int,
    output_path: None | Path, # name, relative to MAPS_DIR and SAMPLED_LOCATIONS_DIR
    seed: int,
    map_path: Path,
):
    # load the map
    rng = np.random.default_rng(seed)
    gt_costmap, local_costmap, swath_library, gt_occlusion_map = get_cost_maps_and_swath_library(map_path, obs_rad_m)

    local_observation_tracker = np.zeros(gt_costmap.data.shape, dtype=bool)


    # wander on map
    trajectory, local_costmap, waypoints = wander_on_map(
        max_iters_per_waypoint=max_iters_per_waypoint,
        max_time_s=seconds_per_image_req,
        gt_costmap=gt_costmap,
        gt_occlusion_map=gt_occlusion_map,
        local_observation_tracker=local_observation_tracker,
        swath_library=swath_library,
        local_costmap=local_costmap,
        rng=rng
    )

    # Generate an image request
    time_history = [0]
    total_time = 0.0
    for i in range(len(trajectory)):
        # cost is in seconds per meter 
        seconds_per_meter = gt_costmap.get_value((trajectory[i].x, trajectory[i].y))
        if np.isinf(seconds_per_meter):
            print("INF FOUND IN TRAJECTORY. SHOULD BE IMPOSSIBLE")
            import IPython
            IPython.embed()
        total_time += seconds_per_meter * gt_costmap.resolution
        time_history.append(total_time)
    assert gt_costmap.resolution == 1, "This code assumes that the costmap resolution is 1 meter per pixel"
    request_locations = trajectory[::meters_between_keypoints]
    request_times = time_history[::meters_between_keypoints]



    if output_path:
        save_map_path = MAPS_DIR / output_path / "masked_map"
        img_req_path = SAMPLED_LOCATIONS_DIR / output_path / "image_request.json"
        if (save_map_path).exists():
            print('Warning: output path already exists. Overwriting.')
            shutil.rmtree(save_map_path)
        if (img_req_path.parent).exists():
            shutil.rmtree(img_req_path.parent)
        save_map_path.parent.mkdir(parents=True, exist_ok=True)
        img_req_path.parent.mkdir(parents=True)

        ## create limited overhead LandCoverComplexMap
        masked_map = lcm.LandCoverComplexMap.from_map_folder(map_path)
        masked_map.reduce_to_visibility_mask(local_observation_tracker)
        masked_map.save_map('masked_map', save_map_path.parent)

        print('Writing request to:',  img_req_path)
        image_gen_request = generate_image_request(request_locations, output_path / 'masked_map', request_times)
        print('created image request')
        # with open('tmp.pkl', 'wb') as f:
        #     pickle.dump(image_gen_request, f)
        # import IPython
        # IPython.embed()
        dump_json_config(image_gen_request, img_req_path)
        print("saved image request")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_dir")
    parser.add_argument("--obs_rad_m", type=int)
    parser.add_argument("--max_iters_per_waypoint", type=int)
    parser.add_argument("--num_seconds", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--meters_between_keypoints", type=int)
    parser.add_argument("--output")

    args = parser.parse_args()

    main(
        args.max_iters_per_waypoint,
        args.obs_rad_m,
        args.meters_between_keypoints,
        args.num_seconds,
        Path(args.output) if args.output else None,
        args.seed,
        Path(args.map_dir),
    )

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.observation_types import GTObservation
from fitam.mapping.costmap_swath_library import SwathLibrary, load_swath_library_from_pkl, save_swath_library_to_pkl, check_swath_against_config
from fitam.core.common import load_json_config
from fitam import CONFIGS_DIR


def local_observe_occlusions(robot_idx: tuple[int, int],
                             gt_map: OccupancyGrid,
                             gt_occlusion_map: np.ndarray,  # boolean array with true in cells that occlude views
                             local_observation_tracker: np.ndarray,  # boolean array of all cells that have or have not been locally observed
                             swath_lib: SwathLibrary):
    observations = [  # always can observe the cell you are on
        GTObservation(
            np.asarray([robot_idx[0]]),
            np.asarray([robot_idx[1]]),
            np.asarray([gt_map.data[robot_idx[0], robot_idx[1]]])
        )
    ]
    local_observation_tracker[robot_idx[0], robot_idx[1]] = True
    for angle, indexes in swath_lib.local_line_iterator(robot_idx, gt_map.data.shape):
        gt_map_vector = gt_occlusion_map[indexes[0], indexes[1]]
        occlusion_location = np.where(gt_map_vector)[0]
        if occlusion_location.size > 0:
            inf_index = min(indexes[0].shape[0], occlusion_location[0]+4)
        else:
            inf_index = indexes[0].shape[0]
        visible_indices = (indexes[0][:inf_index], indexes[1][:inf_index])
        gt_obs = GTObservation(
            i=visible_indices[0],
            j=visible_indices[1],
            costs=gt_map.data[visible_indices[0], visible_indices[1]]
        )
        observations.append(gt_obs)
        # update local observation tracker
        local_observation_tracker[visible_indices[0], visible_indices[1]] = True

    return observations


def compile_known_observations_from_gt_costmap(center_idx: tuple, gt_costmap: OccupancyGrid, swath_library: SwathLibrary) -> GTObservation:
    indexes = swath_library.get_observation_indices(center_idx, gt_costmap.data.shape)
    obs = GTObservation(
        i=indexes[0],
        j=indexes[1],
        costs=gt_costmap.data[indexes[0], indexes[1]],
    )
    return obs


def local_perfect_observations(robot_idx: tuple[int, int],
                               gt_map: OccupancyGrid,
                               local_observation_tracker: np.ndarray,  # boolean array of all cells that have or have not been locally observed
                               swath_lib: SwathLibrary):

    obs = compile_known_observations_from_gt_costmap(
        robot_idx, gt_map, swath_lib
    )
    local_observation_tracker[obs.i, obs.j] = True
    return [obs]


if __name__ == "__main__":

    radial_config_path = CONFIGS_DIR / 'simulated_radial_configs' / 'radial_map_config.json'
    radial_map_config = load_json_config(radial_config_path)
    swath_path = Path("/tmp/swath.pkl")
    # swath_path = SWATHS_DIR / 'simulated_radial_configs' / 'radial_map_config.pkl'
    swath_library = None
    if swath_path.exists():
        swath_library = load_swath_library_from_pkl(swath_path)
    if not check_swath_against_config(swath_library, radial_map_config):
        if swath_path.exists():
            swath_path.unlink()
        swath_library = SwathLibrary.from_radial_map_config(radial_map_config, generate=True, show_progress=True)
        save_swath_library_to_pkl(swath_path, swath_library)

    fake_costmap = np.random.rand(100, 100)
    fake_costmap[40, 60:70] = np.inf
    fake_costmap[45:55, 30:34] = np.inf
    occluding_vector = fake_costmap == np.inf
    map = OccupancyGrid(radial_map_config.map_resolution,
                        fake_costmap.shape[1], fake_costmap.shape[0], data=fake_costmap)

    out_map = OccupancyGrid.from_costmap(map, copy_data=False)
    observed_cells = np.zeros(out_map.data.shape, dtype=bool)
    local_observe_occlusions((50, 50), map, occluding_vector, out_map, observed_cells, swath_library)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(fake_costmap)
    ax[1].imshow(out_map.data)
    ax[2].imshow(observed_cells)
    plt.show()

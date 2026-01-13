from fitam.core.common import load_pickle_config, create_dir, dump_json_config, load_json_config
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.land_cover_complex_map import LandCoverComplexMap
from fitam.core.product_structures import ImageGenerationRequest
from fitam import MAPS_DIR
from fitam.core.config.RadialMapConfig import RadialMapConfig
from fitam.mapping.costmap_swath_library import SwathLibrary
from fitam.mapping.local_observation import local_observe_occlusions
from pathlib import Path
import numpy as np
import copy


def trim_image_request_from_time(original_image_request: ImageGenerationRequest, max_time_s: float) -> ImageGenerationRequest:
    """
    Note: this only trims the points, does not update the map
    """
    assert abs(len(original_image_request.times) - len(original_image_request.locations)) < 3, "Must have (roughly) same number of times as locations."

    new_length = np.argwhere(np.asarray(original_image_request.times) > max_time_s)[0, 0]
    new_image_request = copy.deepcopy(original_image_request)
    new_image_request.locations = new_image_request.locations[:new_length]
    new_image_request.times = new_image_request.times[:new_length]
    new_image_request.angles = new_image_request.angles[:new_length]
    new_image_request.map_idx = new_image_request.map_idx[:new_length]
    return new_image_request


if __name__ == "__main__":
    import argparse
    from fitam import MAPS_DIR, SAMPLED_LOCATIONS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_request_path", type=str, required=True, help="The path to the image request to split.")
    parser.add_argument("-m", "--map_path", type=str, required=True, help="The path to the ORIGINAL (fully observed) map used for the image request")
    parser.add_argument("-v", "--visibility", type=float, required=True, help="The visibility radius of the robot in meters.")
    parser.add_argument("-t", "--times", type=int, nargs="+", required=True, help="The times to split the image request at, in minutes")
    parser.add_argument("-o", "--output_dir_name", type=str, required=True, help="The directory to save the split image requests to. Relative to MAPS_DIR")

    args = parser.parse_args()
    args.output_dir_name = Path(args.output_dir_name)

    if args.image_request_path.endswith(".pkl"):
        original_image_request = load_pickle_config(args.image_request_path)
    elif args.image_request_path.endswith(".json"):
        original_image_request = load_json_config(args.image_request_path)
    else:
        raise RuntimeError("Unexpected extension for image request")

    rmc = RadialMapConfig(observation_radius=args.visibility)
    sl = SwathLibrary.from_radial_map_config(rmc, generate=True)

    create_dir(MAPS_DIR / args.output_dir_name, overwrite=True)
    create_dir(SAMPLED_LOCATIONS_DIR / args.output_dir_name, overwrite=True)
    for time_max in args.times:
        new_image_request = trim_image_request_from_time(original_image_request, time_max * 60)
        lcm = LandCoverComplexMap.from_map_folder(args.map_path)
        occlusions_map = lcm.create_occlusion_map()
        # update the map to the new trajectory
        occ_grid = OccupancyGrid.from_complexmap(lcm)
        observed_states = np.zeros(occ_grid.data.shape, dtype=bool)
        for location in new_image_request.locations[::5]:
            idx = occ_grid.get_index_from_point((location.x, location.y))
            _ = local_observe_occlusions(
                robot_idx=idx,
                gt_map=occ_grid,
                gt_occlusion_map=occlusions_map,
                local_observation_tracker=observed_states,
                swath_lib=sl,
            )
        lcm.reduce_to_visibility_mask(observed_states)
        lcm.save_map(f"partial_map_{time_max}m", MAPS_DIR / args.output_dir_name)
        new_image_request.complex_map_paths = [(args.output_dir_name / f"partial_map_{time_max}m")]
        dump_json_config(new_image_request, SAMPLED_LOCATIONS_DIR / args.output_dir_name / f"image_request_{time_max}m.json")

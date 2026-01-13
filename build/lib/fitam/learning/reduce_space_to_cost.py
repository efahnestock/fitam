from __future__ import annotations

from fitam.mapping.costmap import OccupancyGrid
import numpy as np
from fitam.core.config.RadialMapConfig import ClassificationConfig
from fitam.core.common import wrap_angle_2pi, get_observation_bounding_box
from fitam.core.data_tools import reduce_djikstra_sector_to_cost



def reduce_single_location_to_labels(label_type: str, location: tuple, **kwargs) -> list[tuple[float, float, float]]:
    """
    Reduce a single location to a set of labels
    Inputs: 
        labels_type: str, see DatasetConfig for options
        kwargs: dict-> other options for the label generation function 
    Outputs:
        labels: list[tuple[tuple[min_yaw, max_yaw], label]] -> label is any of int, float, or np.ndarray

    """
    if label_type == "MIN_COST":
        return find_min_cost_labels(location, **kwargs)
    elif label_type == "CLASSIFICATION":
        return calculate_classification_labels(location, **kwargs)
    elif label_type == "AVG_SPEED":
        return calculate_avg_speed_labels(location, **kwargs)
    else:
        raise ValueError(f"Invalid label type {label_type}")


def find_min_cost_labels(location: tuple, **kwargs) -> list[tuple[tuple[float, float], list]]:
    """
    """
    costmap = kwargs["costmap"]
    djikstra = kwargs["djikstra"]
    swath_library = kwargs["swath_library"]  # swath library!
    radial_map_config = kwargs["radial_map_config"]  # radial map config
    dataset_config = kwargs["dataset_config"]  # dataset config
    image_yaw_range = kwargs["image_yaw_range"]  # image yaw range
    keep_all_points = False
    if "keep_all_points" in kwargs:
        keep_all_points = kwargs["keep_all_points"]

    assert swath_library.mask_resolution == radial_map_config.map_resolution, "Swath library and radial map config must have same resolution"

    valid_yaw_pairs = find_valid_samples_at_point(location=location,
                                                  image_yaw_range=image_yaw_range,
                                                  image_context_size=radial_map_config.farfield_config.image_context_size,
                                                  yaw_bin_width=radial_map_config.farfield_config.orientation_bin_size,
                                                  dataset_yaw_sampling_density= dataset_config.sample_density,
                                                  radial_boundaries=radial_map_config.farfield_config.range_bins,
                                                  costmap=costmap,
                                                  keep_all_points=keep_all_points,)

    #assert len(valid_yaw_pairs) > 0, "No valid samples found at location"


    labels = []
    for start_yaw, end_yaw in valid_yaw_pairs:
        # calculate traversability cost for sector
        target = reduce_djikstra_sector_to_cost(djikstra=djikstra, 
                                                yaw_range=(start_yaw, end_yaw),
                                                radial_boundaries=radial_map_config.farfield_config.range_bins,
                                                map_resolution=radial_map_config.map_resolution,
                                                robot_max_speed=radial_map_config.robot_max_speed,
                                                swath_library=swath_library,
                                                )  # [num_range_bins]

        labels.append(((start_yaw, end_yaw), target.tolist()))

    return labels
def calculate_classification_labels(location: tuple, **kwargs) -> list[tuple[tuple[float, float], list]]:
    costmap = kwargs["costmap"]
    swath_library = kwargs["swath_library"]  # swath library!
    radial_map_config = kwargs["radial_map_config"]  # radial map config
    dataset_config = kwargs["dataset_config"]  # dataset config
    image_yaw_range = kwargs["image_yaw_range"]  # image yaw range
    assert swath_library.mask_resolution == radial_map_config.map_resolution, "Swath library and radial map config must have same resolution"
    assert radial_map_config.farfield_config.classification_config is not None, "Classification config must be set for classification labels"
    keep_all_points = False
    if "keep_all_points" in kwargs:
        keep_all_points = kwargs["keep_all_points"]

    valid_yaw_pairs = find_valid_samples_at_point(location=location,
                                                  image_yaw_range=image_yaw_range,
                                                  image_context_size=radial_map_config.farfield_config.image_context_size,
                                                  yaw_bin_width=radial_map_config.farfield_config.orientation_bin_size,
                                                  dataset_yaw_sampling_density= dataset_config.sample_density,
                                                  radial_boundaries=radial_map_config.farfield_config.range_bins,
                                                  costmap=costmap, 
                                                  keep_all_points=True,) # NOTE: keep all points is always true for classification labels
                                                                         # As 0 coverage is assigned to out of map labels.
                                                                         # if keep_all_points is true (in this scope), we don't discard labels with all zero coverage
    # print("found valid yaw pairs to be ", valid_yaw_pairs)
    labels = []
    for start_yaw, end_yaw in valid_yaw_pairs:
        # calculate traversability cost for sector
        # print("Starting yaws are ", start_yaw, end_yaw)
        target, coverage = reduce_speed_to_class(costmap=costmap,
                                                 location=location,
                                                 yaw_range=(start_yaw, end_yaw),
                                                 radial_boundaries=radial_map_config.farfield_config.range_bins,
                                                 classification_config=radial_map_config.farfield_config.classification_config,
                                                 swath_library=swath_library,
        ) 
        # if all bins are unknown (0 coverage), don't append
        if np.all(np.array(coverage) == 0) and not keep_all_points:
            continue
        labels.append(((start_yaw, end_yaw), target.tolist(), coverage.tolist()))
    # assert False
    return labels

def calculate_avg_speed_labels(location: tuple, **kwargs) -> list[tuple[tuple[float, float], list]]:
    """
    """
    costmap = kwargs["costmap"]
    swath_library = kwargs["swath_library"]  # swath library!
    radial_map_config = kwargs["radial_map_config"]  # radial map config
    dataset_config = kwargs["dataset_config"]  # dataset config
    image_yaw_range = kwargs["image_yaw_range"]  # image yaw range
    assert swath_library.mask_resolution == radial_map_config.map_resolution, "Swath library and radial map config must have same resolution"
    keep_all_points = False
    if "keep_all_points" in kwargs:
        keep_all_points = kwargs["keep_all_points"]

    valid_yaw_pairs = find_valid_samples_at_point(location=location,
                                                  image_yaw_range=image_yaw_range,
                                                  image_context_size=radial_map_config.farfield_config.image_context_size,
                                                  yaw_bin_width=radial_map_config.farfield_config.orientation_bin_size,
                                                  dataset_yaw_sampling_density= dataset_config.sample_density,
                                                  radial_boundaries=radial_map_config.farfield_config.range_bins,
                                                  costmap=costmap,
                                                  keep_all_points=keep_all_points,)


    labels = []
    for start_yaw, end_yaw in valid_yaw_pairs:
        # calculate traversability cost for sector
        target, coverage = reduce_speed_sector_to_costs(costmap=costmap,
                                                        location=location,
                                                        yaw_range=(start_yaw, end_yaw),
                                                        radial_boundaries=radial_map_config.farfield_config.range_bins,
                                                        swath_library=swath_library,
        )  # [num_range_bins]

        # if all bins are unknown (0 coverage), don't append
        if np.all(np.array(coverage) == 0):
            continue
        labels.append(((start_yaw, end_yaw), target.tolist(), coverage.tolist()))

    return labels

def reduce_speed_to_class(costmap: OccupancyGrid,
                          location: tuple, # in meters
                          yaw_range: tuple,
                          radial_boundaries: np.ndarray, 
                          classification_config: ClassificationConfig,
                          swath_library, 
): 
    """
    Given a speed map, get the class for each bin. Ignores unknown and untraversable cells
    yaw_range is a tuple of (left_yaw, right_yaw), both will be wrapped within 0->2pi
    costmap: torch.tensor [h, w]
        contains max speed per cell (in s/m)
        -1 indicates unknown 
        inf indicates untraversable

    CLASS STRUCTURE
    class 0 maps to class_ranges[0] <= speed < class_ranges[1]
    class 1 maps to class_ranges[1] <= speed < class_ranges[2].. etc 
    The highest speed class that has at least min_winning_percentage of total valid cells gets the label

    If less than min_valid_percentage cells are known in a bin, returns 0 class and 0 coverage
    If all cells are untraversable in a bin, returns 0 class and 1 coverage
    """
    yaw_range = [wrap_angle_2pi(yaw_range[0]), wrap_angle_2pi(yaw_range[1])]

    center_idx = costmap.get_index_from_point(location)
                 

    num_range_bins = len(radial_boundaries) - 1
    costs = np.zeros(num_range_bins, dtype=int)
    coverage = np.zeros(num_range_bins)

    for i in range(num_range_bins):
        # print("### STARTING BIN ", i)
        indices = swath_library.get_swath(
            center_idx, radial_boundaries[i], yaw_range[1], costmap.data.shape)
        values = costmap.data[indices]
        bin_num_pixels = values.shape[0]
        if bin_num_pixels == 0: # if bin is outside of the observed map
            continue
        coverage[i] = np.sum(values >= 0) / bin_num_pixels 
        if coverage[i] < classification_config.min_valid_percentage:
            coverage[i] = 0 # clear coverage, invalid label 
            continue
        values = values[values >= 0] # remove unknown values

        # convert to speeds. Inf is untraversable, so it results in a speed of zero 
        values = 1 / values
        # clip to speed range 
        values = np.clip(values, classification_config.class_ranges[0] + 1e-6, classification_config.class_ranges[-1] - 1e-6) # 1e-6 makes sure that the values are inclusive
        values = np.digitize(values, classification_config.class_ranges) - 1
        for j in range(classification_config.num_classes-1, -1, -1):
            # assign the highest class that has at least min_winning_percentage of cells
            if np.sum(values == j) / bin_num_pixels >= classification_config.min_winning_percentage:
                costs[i] = j
                break

        # print("final value is ", costs[i])


        
    return costs, coverage
def reduce_speed_sector_to_costs(costmap: OccupancyGrid,
                                 location: tuple, # in meters
                                 yaw_range: tuple,
                                 radial_boundaries: np.ndarray, 
                                 swath_library, 
                                 clip_range: tuple = None, 
): 
    """
    Given a speed map, get the average speeds and coverage in each of the bins. Ignores unknown and untraversable cells
    yaw_range is a tuple of (left_yaw, right_yaw), both will be wrapped within 0->2pi
    costmap: torch.tensor [h, w]
        contains max speed per cell (in s/m)
        -1 indicates unknown 
        inf indicates untraversable

    If all cells are unknown in a bin, returns 0 cost and 0 coverage
    If all cells are untraversable in a bin, returns 0 cost and 1 coverage
    """
    yaw_range = [wrap_angle_2pi(yaw_range[0]), wrap_angle_2pi(yaw_range[1])]

    center_idx = costmap.get_index_from_point(location)
                 

    num_range_bins = len(radial_boundaries) - 1
    costs = np.zeros(num_range_bins)
    coverage = np.zeros(num_range_bins)

    for i in range(num_range_bins):
        # grab only the values at the edge of the range bin
        indices = swath_library.get_swath(
            center_idx, radial_boundaries[i], yaw_range[1], costmap.data.shape)
        # divide by the time it would take for the robot to reach each point at max speed
        values = costmap.data[indices]
        coverage[i] = np.sum(values >= 0) / values.shape[0]
        values = values[values >= 0] # remove unknown values
        if values.shape[0] == 0: # all values in bin are untraversable or out of the map
            continue # cost is 0 and coverage is 0
        # remove untraversable values
        values = values[values != float('inf')]
        if values.shape[0] == 0: # all values in bin are untraversable
            continue # cost is 0 and coverage is <1
        costs[i] = 1 / np.mean(values) # make meters / second instead of seconds / meter
        


    if clip_range is not None:
        costs = np.clip(costs, *clip_range)
    return costs, coverage


def find_valid_samples_at_point(location: tuple,
                                image_yaw_range: tuple,
                                image_context_size: float,  # in radians
                                yaw_bin_width: float,  # in radians
                                dataset_yaw_sampling_density: float,  # in samples/bin
                                radial_boundaries: list,
                                costmap: OccupancyGrid,
                                keep_all_points: bool = False,
                                ) -> list:
    """
    Find valid (in-map) samples at a given location
    Location is in meters in the frame of the map
    """
    valid_yaw_pairs = []
    possible_yaw_pairs = []

    norm_img_yaw_range = [wrap_angle_2pi(x) for x in image_yaw_range]
    # print('norm img yaw range: ', norm_img_yaw_range)
    # left is larger than right
    img_yaw_width = norm_img_yaw_range[0] - norm_img_yaw_range[1]
    # print('img yaw width: ', img_yaw_width)
    # if the image yaw range wraps around 0
    if norm_img_yaw_range[0] < norm_img_yaw_range[1]:
        # fix the width
        img_yaw_width = norm_img_yaw_range[0] + \
            (2*np.pi - norm_img_yaw_range[1])
    # if 360 image
    if abs(norm_img_yaw_range[0] - norm_img_yaw_range[1]) < 1e-6:
        img_yaw_width = 2*np.pi

    # calculate the number of samples we need to take
    # need to remove the padding for the image context
    image_context_padding = (image_context_size / 2) - (yaw_bin_width / 2)
    img_yaw_width -= image_context_padding * 2

    num_bins = img_yaw_width / yaw_bin_width
    num_samples = int(np.ceil(num_bins * dataset_yaw_sampling_density))
    # print(f"Num samples: {num_samples}")
    # sample evenly spaced points along the image yaw range
    for i in range(num_samples):
        left_yaw = wrap_angle_2pi( # 1e-6 to prevent floating point errors
            norm_img_yaw_range[0] - image_context_padding - ((i / (num_samples-1)) * (img_yaw_width-yaw_bin_width)))
        right_yaw = wrap_angle_2pi(left_yaw - yaw_bin_width + 1e-6)
        possible_yaw_pairs.append((left_yaw, right_yaw))
    # print(f"Possible yaw pairs: {possible_yaw_pairs}")
    if keep_all_points:
        return possible_yaw_pairs
    for left_yaw, right_yaw in possible_yaw_pairs:
        # print(f"Left yaw: {left_yaw}, Right yaw: {right_yaw}")
        # sample 100 points along the arc between min and max yaw
        min_x, max_x, min_y, max_y = get_observation_bounding_box(left_yaw, right_yaw, location, radial_boundaries[-1])

        # print(f"Min x: {min_x}, Max x: {max_x}, Min y: {min_y}, Max y: {max_y}")
        # if any points reach out of bounds, we don't include this sample
        if (min_x < costmap.x_lim_meters[0]) or \
            (max_x > costmap.x_lim_meters[1]) or \
            (min_y < costmap.y_lim_meters[0]) or \
            (max_y > costmap.y_lim_meters[1]):
            continue
        # otherwise, we have a happy sample! Wrap angle and add to list
        else:
            valid_yaw_pairs.append((
                wrap_angle_2pi(left_yaw), 
                wrap_angle_2pi(right_yaw)
            ))

    return valid_yaw_pairs

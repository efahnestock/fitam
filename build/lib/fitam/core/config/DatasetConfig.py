from dataclasses import dataclass


@dataclass
class DatasetConfig:
    # NOTE: the true number of points could be less, if valid samples run out in the envs
    num_points_per_map: int = 1000  # this is total per map

    # Options:
    # MIN_COST:
    # CLASSIFICATION:
    # AVG_SPEED:
    label_type: str = "MIN_COST"

    # For going from images to samples
    sample_density: float = 1.5  # samples / bin width

    # for sampling pano points
    kernel_width_meters: float = 10.0  # the width of the kernel used to erode free space. Should be min dist to obstacles allowed
    distance_from_wall_meters: float = 50.0  # minimum distance from the wall for a valid sample point
    min_distance_between_samples_meters: float = 5.0  # minimum distance between sample points
    camera_height = 1.5  # meters off the ground plane

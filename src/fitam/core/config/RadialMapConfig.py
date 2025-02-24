
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from enum import Enum


### SPATIAL LABELING BASELINE
@dataclass
class ImagePyramidConfig:
    camera_height_m: float = 1.5  # meters
    min_range_m: float = 25.0  # meters
    max_range_m: float = 100.0  # meters
    image_slice_height_pixels: int = 10  # pixels
    image_slice_width_pixels: int = 20  # pixels
    obstacle_dialation_m: float = 1.0  # meters
    num_footlines: int = 7  # number of footlines
    untraversable_cost_s_per_meter: float = 2.0
    
@dataclass
class SpatialLabelConfig:
    image_pyramid_config: ImagePyramidConfig = field(default_factory=lambda: ImagePyramidConfig())



@dataclass
class ClassificationConfig:
    # class 0 maps to class_ranges[0] <= speed < class_ranges[1]
    class_ranges: list = field(default_factory=lambda: [0.0, 1.1, 3.5, 5.0])
    class_speeds: list = field(default_factory=lambda: [0.5, 2.25, 4.25])

    min_valid_percentage: float = 0.2  # At least this % of cells must be valid for the label to be valid
    min_winning_percentage: float = 0.2  # The highest speed class that has at least this % of cells gets the label

    @property
    def num_classes(self):
        return len(self.class_ranges) - 1


@dataclass
class FarFieldConfig:
    orientation_bin_size: float = 2 * np.pi / 32  # in radians
    image_context_size: float = 2 * np.pi / 32  # in radians, the FOV of the image, centered on the orientation bin
    # edges of range bins, in meters
    range_bins: NDArray = field(
        default_factory=lambda: np.linspace(25, 100, 4).tolist())
    classification_config: Optional[ClassificationConfig] = None

    @property
    def num_range_bins(self):
        return len(self.range_bins) - 1



class MapFusionType(Enum):
    KALMAN = "kalman"
    MIN_ENTROPY = "min_entropy"
    MOST_RECENT = "most_recent"
    NO_UNCERTAINTY_AVERAGE = "no_uncertainty_average"


@dataclass
class RadialMapConfig:
    farfield_config: Optional[FarFieldConfig | SpatialLabelConfig] = None
    robot_max_speed: float = 5.0  # meters per second
    observation_radius: float = 25  # in meters
    map_resolution: float = 1.  # in meters
    swath_angular_resolution: float = 0.001  # in radians
    unknown_value: float = -1
    unknown_cost: float = 1 / 1.0  # 1.66666 is assumed speed of unknown space
    process_noise: float = 0.001
    variance_threshold: float = 0.4
    observation_variance_threshold: float = 99999.9
    map_fusion_type: MapFusionType = MapFusionType.KALMAN
    use_occluded_local_observations: bool = True

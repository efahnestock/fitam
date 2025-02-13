from fitam.core.config.RadialMapConfig import RadialMapConfig, ClassificationConfig, FarFieldConfig, MapFusionType, SpatialLabelConfig
from fitam.core.common import dump_json_config
from pathlib import Path
import copy
import numpy as np
class_config = ClassificationConfig()
farfield_config = FarFieldConfig(classification_config=class_config)
radial_config = RadialMapConfig(farfield_config=farfield_config)

heading_options = [4, 8, 16, 32, 64]
range_options = [1, 2, 4, 8, 10]

# save default
dump_json_config(radial_config, 'radial_map_config.json', overwrite=True)

two_classes = ClassificationConfig(
    class_ranges=[0.0, 2.5, 5.0],
    class_speeds=[1.6666666, 3.333333],
)
# Set up various classification configs
slow_together = ClassificationConfig(
    class_ranges=[0.0, 1.1, 3.5, 5.0],
    class_speeds=[0.5, 2.5, 5.0]
    # [trash, young_forest, forest], [grass, field], road
)

grass_together = ClassificationConfig(
    class_ranges=[0.0, 0.3, 1.1, 3.5, 5.0],
    class_speeds=[0.15, 0.75, 2.5, 5.0]
    # trash, [young_forest, forest], [grass, field], road
)

forests_together = ClassificationConfig(
    class_ranges=[0.0, 0.3, 1.1, 2.5, 3.5, 5.0],
    class_speeds=[0.15, 0.75, 2.0, 3.0, 5.0]
    # trash, [young_forest, forest], grass, field, road
)

all_separate = ClassificationConfig(
    class_ranges=[0.0, 0.3, 0.6, 1.1, 2.5, 3.5, 5.0],
    class_speeds=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
)
names = ['2_speeds', '3_speeds', '4_speeds', '5_speeds', '6_speeds']
classification_configs = [two_classes, slow_together, grass_together, forests_together, all_separate]
Path("vary_classes").mkdir(parents=True, exist_ok=True)
for name, class_config in zip(names, classification_configs):
    config_diff_heading = copy.deepcopy(radial_config)
    config_diff_heading.farfield_config.classification_config = class_config
    dump_json_config(config_diff_heading, f'vary_classes/{name}_radial_map_config.json', overwrite=True)
# save time-varying and effective default. Larger range bins is just so standard and range can share the same model
long_range_radial_map_config = copy.deepcopy(radial_config)
long_range_radial_map_config.farfield_config.range_bins = np.linspace(25, 300, 12)
dump_json_config(long_range_radial_map_config, 'long_range_radial_map_config.json', overwrite=True)

# save baseline
baseline_config = copy.deepcopy(radial_config)
# delete farfield config
baseline_config.farfield_config = None
dump_json_config(baseline_config, 'baseline_radial_map_config.json', overwrite=True)
perfect_vision_config = copy.deepcopy(baseline_config)
perfect_vision_config.observation_radius = 100.0
perfect_vision_config.use_occluded_local_observations = False
dump_json_config(perfect_vision_config, 'perfect_vision_radial_map_config.json', overwrite=True)

Path("vary_heading").mkdir(parents=True, exist_ok=True)
# save vary heading
for num_heading_bins in heading_options:
    heading_config = copy.deepcopy(radial_config)
    heading_config.farfield_config.orientation_bin_size = 2*np.pi/num_heading_bins
    heading_config.farfield_config.image_context_size = 2*np.pi/num_heading_bins

    dump_json_config(heading_config, f'vary_heading/{num_heading_bins}h_radial_map_config.json', overwrite=True)

Path("vary_range").mkdir(parents=True, exist_ok=True)
# save vary range
for num_range_bins in range_options:
    range_config = copy.deepcopy(radial_config)
    range_config.farfield_config.range_bins = np.linspace(25, 100, num_range_bins+1).tolist()
    dump_json_config(range_config, f'vary_range/{num_range_bins}r_radial_map_config.json', overwrite=True)

# save vary range and heading
Path("vary_range_heading").mkdir(parents=True, exist_ok=True)
for num_range_bins, num_heading_bins in zip(range_options, heading_options):
    range_heading_config = copy.deepcopy(radial_config)
    range_heading_config.farfield_config.range_bins = np.linspace(25, 100, num_range_bins+1).tolist()
    range_heading_config.farfield_config.orientation_bin_size = 2*np.pi/num_heading_bins
    range_heading_config.farfield_config.image_context_size = 2*np.pi/num_heading_bins
    dump_json_config(range_heading_config, f'vary_range_heading/{num_range_bins}r{num_heading_bins}h_radial_map_config.json', overwrite=True)

# save spatial config baseline
spatial_config_baseline = copy.deepcopy(radial_config)
spatial_config_baseline.farfield_config = SpatialLabelConfig()
spatial_config_baseline.map_fusion_type = MapFusionType.MOST_RECENT
dump_json_config(spatial_config_baseline, "spatial_label_radial_map_config.json", overwrite=True)

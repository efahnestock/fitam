from fitam import SWATHS_DIR, CONFIGS_DIR
from fitam.mapping.spatial_label_general import farfield_config_from_spatial_label
from fitam.core.common import load_json_config
from fitam.mapping.costmap_swath_library import SwathLibrary, save_swath_library_to_pkl

if __name__ == "__main__":
    PANO_SIZE = (128, 4608, 3)

    radial_map_config = load_json_config(CONFIGS_DIR / "simulated_radial_configs" / "spatial_label_radial_map_config.json")
    spatial_label_ff_config = farfield_config_from_spatial_label(PANO_SIZE, radial_map_config.farfield_config.image_pyramid_config)
    sl = SwathLibrary(radial_map_config.observation_radius, radial_map_config.map_resolution, radial_map_config.swath_angular_resolution, spatial_label_ff_config, generate=True, show_progress=True)
    save_swath_library_to_pkl(SWATHS_DIR / "simulated_radial_configs" / 'spatial_label_radial_map_config.pkl', sl)
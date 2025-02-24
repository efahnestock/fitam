import numpy as np
import tqdm
import torch
import pickle
from fitam.core.data_tools import generate_range_and_bearing_tensors
from fitam.core.common import wrap_angle_2pi, mask_radius_tensor, mask_yaw_tensor
from fitam.core.config.RadialMapConfig import FarFieldConfig, RadialMapConfig, SpatialLabelConfig


def check_swath_against_config(swath: 'SwathLibrary', config: RadialMapConfig):
    if isinstance(config.farfield_config, SpatialLabelConfig):
        print("Warning: check_swath_against_config not implemented for spatial config, skipping check")
        return True
    if swath is None:
        return False
    if swath.angular_resolution != config.swath_angular_resolution:
        return False
    if swath.mask_resolution != config.map_resolution:
        return False
    if swath.observation_radius != config.observation_radius:
        return False
    if swath.ff_enabled != (isinstance(config.farfield_config, FarFieldConfig)):
        return False
    if swath.ff_enabled:
        if np.any(swath.radial_bins != config.farfield_config.range_bins):
            return False
        if swath.angular_bin_size != config.farfield_config.orientation_bin_size:
            return False
    return True


class SwathLibrary:
    """
    Stores a library of swaths for a given costmap 

    Could add reflections to cut things in 1/8ths 
    """

    LOCAL_ANG_RES = 2 * np.pi / 45.0

    def __init__(self,
                 observation_radius: float,
                 mask_resolution: float,
                 angluar_resolution: float,
                 ff_config: FarFieldConfig,
                 generate=False, show_progress=True):
        self.angular_resolution = angluar_resolution
        self.mask_resolution = mask_resolution
        self.observation_radius = observation_radius
        if isinstance(ff_config, FarFieldConfig):
            self.ff_enabled = True
            self.angular_bin_size = ff_config.orientation_bin_size
            self.radial_bins = ff_config.range_bins
        else:
            self.ff_enabled = False
            self.angular_bin_size = None
            self.radial_bins = None

        self.swath_map = {}  # save the whole radial bin
        self.arc_map = {}  # save just the end of the bin
        # save the circle around the robot that is within geometric range
        self.observation_indices = ()

        self.save_type = np.int16

        if generate:
            self._populate_masks(show_progress)

    @classmethod
    def from_radial_map_config(cls, radial_map_config, generate=False, show_progress=True):
        return cls(radial_map_config.observation_radius,
                   radial_map_config.map_resolution,
                   radial_map_config.swath_angular_resolution,
                   radial_map_config.farfield_config,
                   generate=generate, show_progress=show_progress)

    def _get_key(self, radial_bin, yaw_bin, bin_angular_size, num_decimals=5):
        # note: yaw_bin should point to the minimum of the yaw bin boundaries (right side)
        #   we shift this to the center of the bin here
        center_yaw_orig = wrap_angle_2pi(yaw_bin + bin_angular_size / 2)
        # all saved swaths are in Q2 (so that mirroring over i/j works)
        center_yaw = (center_yaw_orig % (np.pi / 2)) + np.pi / 2
        if (center_yaw_orig > np.pi and center_yaw_orig < 3 * np.pi / 2) or (center_yaw_orig < np.pi / 2):
            center_yaw = np.pi - center_yaw + np.pi / 2
        if center_yaw > 3 * np.pi / 4:  # complement if > 45 deg
            center_yaw = np.pi - center_yaw + np.pi / 2
        return (round(round(radial_bin / self.angular_resolution) * self.angular_resolution, num_decimals),
                round(round(center_yaw / self.angular_resolution) * self.angular_resolution, num_decimals))

    def _get_mask_indices(self, mask):
        return np.nonzero(mask)

    def _sort_indices_by_magnitude(self, indices):
        magnitude = np.sqrt(indices[0]**2 + indices[1]**2)
        ordering = np.argsort(magnitude)
        sorted_indices = (indices[0][ordering], indices[1][ordering])
        return sorted_indices

    def _get_range_yaw_tensors(self, max_r):
        # calculate the mask dimensions. 2 * radius_cells + 1 (center)
        mask_dim = int(np.ceil(max_r / self.mask_resolution)) * 2 + 1
        center_idx = (mask_dim//2, mask_dim//2)
        assert mask_dim % 2 == 1
        # set to 1 thread: otherwise this breaks process_df_in_parallel calls
        num_torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        range_t, yaw_t = generate_range_and_bearing_tensors(
            center_idx, mask_dim, self.mask_resolution)
        range_t = range_t.numpy()
        yaw_t = yaw_t.numpy()
        torch.set_num_threads(num_torch_threads)
        return center_idx, range_t, yaw_t

    def _populate_local_lines(self, show_progress=False):
        """
        Local line indexes are sorted by their magnitude (radius from center)
        """
        self.local_lines_map = {}
        max_r = self.observation_radius
        center_idx, range_t, yaw_t = self._get_range_yaw_tensors(max_r)
        yaw_iter = tqdm.tqdm(np.arange(np.pi / 2 - self.LOCAL_ANG_RES / 2, 3 * np.pi / 4 - self.LOCAL_ANG_RES / 2, self.angular_resolution),
                             desc='yaw bins', disable=not show_progress)
        masked_r_t = mask_radius_tensor(range_t, (0, max_r))
        for angle in yaw_iter:
            right_yaw = wrap_angle_2pi(angle)
            left_yaw = wrap_angle_2pi(angle + self.LOCAL_ANG_RES)
            masked_y_tensor = mask_yaw_tensor(yaw_t, (right_yaw - 0.05, left_yaw + 0.05))
            masked_idxs = self._get_mask_indices(masked_y_tensor & masked_r_t)
            masked_idxs = (
                masked_idxs[0] - center_idx[0], masked_idxs[1] - center_idx[1])
            sorted_idxs = self._sort_indices_by_magnitude(masked_idxs)
            self.local_lines_map[self._get_key(0, right_yaw, self.LOCAL_ANG_RES)] = sorted_idxs

    def local_line_iterator(self, center_idx, map_shape: tuple):
        for angle in np.arange(0, 2 * np.pi, self.LOCAL_ANG_RES):
            idxs = self.get_local_line(center_idx, angle, map_shape)
            yield angle, idxs

    def _populate_masks(self, show_progress=False):
        self.swath_map = {}
        self.arc_map = {}
        self.observation_indices = ()

        self._populate_local_lines()

        if not self.ff_enabled:
            max_r = self.observation_radius
        else:
            max_r = max(self.observation_radius, self.radial_bins[-1])

        center_idx, range_t, yaw_t = self._get_range_yaw_tensors(max_r)
        # generate the observation indices
        self.observation_indices = self._get_mask_indices(
            mask_radius_tensor(range_t, (0, self.observation_radius)))
        self.observation_indices = (
            self.observation_indices[0] - center_idx[0], self.observation_indices[1] - center_idx[1])
        if not self.ff_enabled:
            return

        rad_iter = tqdm.tqdm(range(len(self.radial_bins)-1),
                             desc='radial bins', disable=not show_progress)
        for radial_bin_i in rad_iter:
            masked_r_t = mask_radius_tensor(
                range_t, (self.radial_bins[radial_bin_i], self.radial_bins[radial_bin_i+1]))
            # if this is the outer-most thing, bring it in a little bit as not to catch the zeros on the edge
            if radial_bin_i == len(self.radial_bins) - 1:
                scaled_rad_boundaries = (self.radial_bins[radial_bin_i+1] - 3 * self.mask_resolution /
                                         2 + 1e-6, self.radial_bins[radial_bin_i+1] - self.mask_resolution / 2)
            else:
                scaled_rad_boundaries = (self.radial_bins[radial_bin_i+1] - self.mask_resolution /
                                         2 + 1e-6, self.radial_bins[radial_bin_i+1] + self.mask_resolution / 2)
            masked_r_t_arc = mask_radius_tensor(range_t, scaled_rad_boundaries)
            yaw_iter = tqdm.tqdm(np.arange(np.pi / 2 - self.angular_bin_size / 2, 3 * np.pi / 4 - self.angular_bin_size / 2, self.angular_resolution),
                                 desc='yaw bins', disable=not show_progress)
            for angle in yaw_iter:
                right_yaw = wrap_angle_2pi(angle)
                max_yaw = wrap_angle_2pi(angle + self.angular_bin_size)
                masked_y_t = mask_yaw_tensor(yaw_t, (right_yaw, max_yaw))
                masked_idxs = self._get_mask_indices(masked_r_t & masked_y_t)
                masked_idxs = (
                    masked_idxs[0] - center_idx[0], masked_idxs[1] - center_idx[1])
                self.swath_map[self._get_key(
                    self.radial_bins[radial_bin_i], right_yaw, self.angular_bin_size)] = masked_idxs
                # do arcs
                masked_idxs_arcs = self._get_mask_indices(
                    masked_r_t_arc & masked_y_t)
                masked_idxs_arcs = (
                    masked_idxs_arcs[0] - center_idx[0], masked_idxs_arcs[1] - center_idx[1])
                self.arc_map[self._get_key(
                    self.radial_bins[radial_bin_i], right_yaw, self.angular_bin_size)] = masked_idxs_arcs

    def _transform_indices(self, indices, right_yaw: float, bin_angular_size: float):
        # print("In transform indices")
        half_bin_yaw = wrap_angle_2pi(
            right_yaw + bin_angular_size / 2)  # [0,2pi)
        # print("half bin yaw", half_bin_yaw)
        q2_yaw = (half_bin_yaw % (np.pi / 2)) + np.pi / 2
        # print("q2 yaw", q2_yaw)
        # added 1e-12 to catch the case where half_bin_yaw is exactly pi, which before was getting flattened to pi/2 not triggering an xy flip
        if (half_bin_yaw+1e-12 > np.pi and half_bin_yaw < 3 * np.pi / 2) or (half_bin_yaw < np.pi / 2):
            q2_yaw = np.pi - q2_yaw + np.pi / 2
        # print("q2 yaw post fix", q2_yaw)
        # make valid in Q2
        if q2_yaw > 3 * np.pi / 4:
            # print("swapping xy")
            indices = self._swap_xy(indices)

        # make valid in Q1 and Q2
        if half_bin_yaw < np.pi / 2 or half_bin_yaw > 3 * np.pi / 2:
            # print("mirroring j")
            indices = self._mirror_j(indices)
        # make valid everywhere
        if half_bin_yaw > np.pi:
            # print("mirroring i")
            indices = self._mirror_i(indices)
        return indices

    def get_swath(self, center_idx: tuple, min_rad: float, right_yaw: float, map_shape: tuple):
        key = self._get_key(min_rad, right_yaw, self.angular_bin_size)
        indices = self.swath_map[key]
        tf_indices = self._transform_indices(indices, right_yaw, self.angular_bin_size)
        return self._shift_swath(tf_indices, center_idx, map_shape)

    def get_arc(self, center_idx: tuple, min_rad: float, right_yaw: float, map_shape: tuple):
        key = self._get_key(min_rad, right_yaw, self.angular_bin_size)
        indices = self.arc_map[key]
        tf_indices = self._transform_indices(indices, right_yaw, self.angular_bin_size)
        return self._shift_swath(tf_indices, center_idx, map_shape)

    def get_local_line(self, center_idx: tuple, right_yaw: float, map_shape: tuple):
        key = self._get_key(0, right_yaw, self.LOCAL_ANG_RES)
        indices = self.local_lines_map[key]
        tf_indices = self._transform_indices(indices, right_yaw, self.LOCAL_ANG_RES)
        return self._shift_swath(tf_indices, center_idx, map_shape)
    def get_observation_indices(self, center_idx: tuple, map_shape: tuple):
        return self._shift_swath(self.observation_indices, center_idx, map_shape)

    def _mirror_i(self, swath_idxs):
        return (-swath_idxs[0], swath_idxs[1])

    def _mirror_j(self, swath_idxs):
        return (swath_idxs[0], -swath_idxs[1])

    def _swap_xy(self, swath_idxs):
        return (swath_idxs[1], swath_idxs[0])

    def _shift_swath(self, swath_idxs, new_center, map_shape):
        """
        Shifts a swath to a new center
        """
        i = swath_idxs[0] + new_center[0]
        j = swath_idxs[1] + new_center[1]
        valid = (i >= 0) & (i < map_shape[0]) & (j >= 0) & (j < map_shape[1])
        return (i[valid], j[valid])


def load_swath_library_from_pkl(filepath: str) -> SwathLibrary:
    with open(filepath, 'rb') as f:
        swath_lib = pickle.load(f)
    return swath_lib


def save_swath_library_to_pkl(filepath: str, swath_lib: SwathLibrary) -> None:
    with open(filepath, 'wb') as f:
        pickle.dump(swath_lib, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the radial map config file")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output file")

    args = parser.parse_args()
    from fitam.core.common import load_json_config
    rad_map_config = load_json_config(args.config)
    slb = SwathLibrary(observation_radius=rad_map_config.observation_radius,
                       angluar_resolution=rad_map_config.swath_angular_resolution,
                       ff_config=rad_map_config.farfield_config,
                       mask_resolution=rad_map_config.map_resolution,
                       generate=True)
    save_swath_library_to_pkl(args.output, slb)

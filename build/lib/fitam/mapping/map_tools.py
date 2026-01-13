import numpy as np
from fitam.mapping.costmap import OccupancyGrid


def find_close_crop_bounds_around_current_state(
        occ_grid: OccupancyGrid,
        index_or_position: tuple[int, int] | tuple[float, float],  # x,y position in meters
        padding: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Note: may not return the same sized square if close to the edge of the map
    """
    if isinstance(index_or_position[0], int):
        i, j = index_or_position
    else:
        i, j = occ_grid.get_index_from_point(index_or_position)
    min_idxs = np.maximum([i-padding, j-padding], [0, 0])
    max_idxs = np.minimum([i+padding, j+padding], occ_grid.data.shape)
    return min_idxs, max_idxs


def find_map_bounds_around_point(occ_grid: OccupancyGrid,
                                 index_or_position: tuple[int, int] | tuple[float, float],  # x,y position in meters
                                 padding: int,
                                 convert_position_to_index: bool,
                                 existing_min_idxs: tuple[int, int] = (np.inf, np.inf),
                                 existing_max_idxs: tuple[int, int] = (-np.inf, -np.inf),
                                 keep_square: bool = False,
                                 ) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Note: may not return the same sized square if close to the edge of the map
    """
    if convert_position_to_index:
        if isinstance(index_or_position[0], int):
            print(f"WARNING: index_or_position is already an index {index_or_position}, but convert_position_to_index is True")
        i, j = occ_grid.get_index_from_point(index_or_position)
    else:
        i, j = index_or_position
    min_idxs = np.minimum(existing_min_idxs, [i-padding, j-padding])
    max_idxs = np.maximum(existing_max_idxs, [i+padding, j+padding])
    if keep_square:
        max_delta = np.max(np.diff([min_idxs, max_idxs], axis=0))
        center = np.mean([min_idxs, max_idxs], axis=0)
        min_idxs = center - max_delta / 2
        max_idxs = center + max_delta / 2
    min_idxs = np.clip(min_idxs, 0, occ_grid.data.shape).astype(int)
    max_idxs = np.clip(max_idxs, 0, occ_grid.data.shape).astype(int)
    return min_idxs, max_idxs

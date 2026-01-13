import numpy as np
from typing import List, Tuple
import pyastar2d
from fitam.mapping.costmap import OccupancyGrid
from fitam.planning.planner_general import PlanningState, snap_easl_state_to_planning_state, planning_state_to_easl_state, get_acc_costs, State


class A_Star:

    def __init__(self, costmap: OccupancyGrid):
        self.costmap = costmap
        self.resolution = self.costmap.info['resolution']
        self.start = None
        self.goal = None

    def get_costs(self, path: List[Tuple[int, int]], cost_array: np.ndarray) -> list:
        cost = []
        for item_index in range(1, len(path)):  # don't add cost for the start node
            cost.append(cost_array[path[item_index][0], path[item_index][1]])
        return np.asarray(cost, dtype=np.float32)

    def search(self, start: State, goal: State, return_easl_coords=True, allow_failure=True):
        """
        If allow_failure is true, returns path to closest point
        If allow_failure is false, returns None if no path is found
        """
        self.start = snap_easl_state_to_planning_state(self.costmap, start)
        self.goal = snap_easl_state_to_planning_state(self.costmap, goal)
        # convert start and goal states to matrix coordinates.
        # PlanningStates are in matrix coordinates -> ps.y = i, ps.x = j

        # set up the costmap
        tmp_costmap = self.costmap.get_full_costmap()
        assert np.min(tmp_costmap) >= 0, f"Costmap has negative values, {np.min(tmp_costmap)}"
        tmp_costmap = tmp_costmap.astype(np.float32)
        # make minimum cost 1 so that heuristic is admissable (l2)
        costmap_mult_shift = np.min(tmp_costmap)
        tmp_costmap /= costmap_mult_shift

        # perform the search
        success, path, cost = pyastar2d.astar_path(
            tmp_costmap, (self.start.y, self.start.x), (self.goal.y, self.goal.x), allow_diagonal=True)
        if (not success) and (not allow_failure):
            return None, None

        # get accumulated costs
        acc_costs = get_acc_costs(path, tmp_costmap, self.resolution)
        acc_costs = [x * costmap_mult_shift for x in acc_costs]
        if return_easl_coords:
            easl_path = []
            for item in path:
                easl_path.append(planning_state_to_easl_state(
                    self.costmap, PlanningState(item[1], item[0])))

            return easl_path, acc_costs
        else:
            return path, acc_costs

def format_paths_for_plot(paths, costmap: OccupancyGrid):
    ret = []
    for path in paths:
        new_path = dict(x=list(), y=list())
        for state in path:
            cmap_idx = costmap.get_index_from_point(
                (state.x * costmap.resolution, state.y * costmap.resolution))
            new_path['x'].append(cmap_idx[0])
            new_path['y'].append(cmap_idx[1])
        new_path['x'] = np.asarray(new_path['x'])
        new_path['y'] = np.asarray(new_path['y'])
        ret.append(new_path)
    return ret
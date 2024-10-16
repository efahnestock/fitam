import numpy as np
from pathlib import Path
import shutil
import dstar_lite
from fitam.mapping.costmap import OccupancyGrid
from fitam.core.common import save_compressed_pickle
from fitam.planning.astar import A_Star
from fitam.planning.planner_general import PlanningState, snap_easl_state_to_planning_state, planning_state_to_easl_state, get_acc_costs, State


class D_Star_Interface:

    def __init__(self, costmap: OccupancyGrid, start: State, goal: State,
                 max_iters: int = 9999999999, min_costmap_value: float = 1 / 5.0, check_with_astar: bool = False, log_all_state_changes: bool = False):
        self.costmap = costmap
        self.costmap_scale_factor = 1 / (min_costmap_value - 1e-6)
        self.resolution = self.costmap.info['resolution']
        self.max_iters = max_iters

        self.start = snap_easl_state_to_planning_state(self.costmap, start)
        self.goal = snap_easl_state_to_planning_state(self.costmap, goal)

        self.log_all_state_changes = log_all_state_changes
        self.check_with_astar = check_with_astar
        if self.check_with_astar:
            self.astar = A_Star(costmap)
        self.dstar = None
        self.last_costmap_state = None

    def Initialize(self):
        self.last_costmap_state = self.costmap.get_full_costmap().copy() * self.costmap_scale_factor  # make the smallest value 1
        assert np.min(self.last_costmap_state) >= 1, f"Costmap has values {self.costmap.get_full_costmap().min()} that are smaller than the min costmap value {1 / self.costmap_scale_factor}"
        self.dstar = dstar_lite.Dstar(self.last_costmap_state, self.max_iters, scale_diag_cost=True)
        self.dstar.init(self.start.y, self.start.x, self.goal.y, self.goal.x)
        initial_plan_success = self.dstar.replan()
        if not initial_plan_success:
            raise RuntimeError("Initial plan failed")

        if self.log_all_state_changes:
            self.log_dir = Path("/tmp/dstar_lite_logs")
            self.update_start_count = 0
            self.update_costmap_count = 0
            self.replan_count = 0
            if (self.log_dir).exists():
                shutil.rmtree(self.log_dir)
            (self.log_dir).mkdir(parents=True, exist_ok=False)
            with open(self.log_dir / "log.txt", "w") as f:
                f.write("Dstar Lite Log\n")

            init_state = {
                "costmap": self.last_costmap_state,
                "start": (self.start.y, self.start.x),
                "goal": (self.goal.y, self.goal.x),
                "g_values": self.dstar.getGValues(),
                "rhs_values": self.dstar.getRHSValues(),
                "keys": self.dstar.getKeys(),
            }
            save_compressed_pickle(init_state, self.log_dir / "init_state.pkl")

    def updateStart(self, start: State) -> None:
        self.start = snap_easl_state_to_planning_state(self.costmap, start)
        self.dstar.updateStart(self.start.y, self.start.x)
        if self.log_all_state_changes:
            with open(self.log_dir / "log.txt", "a") as f:
                f.write(f"Update Start idx {self.update_start_count}: {self.start.y}, {self.start.x}\n")
            self.update_start_count += 1

    def update_costmap(self) -> None:
        # find indices of cells that have changed
        latest_costmap = self.costmap.get_full_costmap() * self.costmap_scale_factor
        changed_cells = np.where(self.last_costmap_state != latest_costmap)
        indexes = np.array([changed_cells[0], changed_cells[1]]).T.astype(np.int32)
        values = latest_costmap[indexes[:, 0], indexes[:, 1]]
        if len(values) == 0:
            print("No cells changed and update_costmap was called")
            return
        assert np.min(values) >= 1, f"Costmap has values {np.min(latest_costmap / self.costmap_scale_factor)} that are smaller than the min costmap value {1 / self.costmap_scale_factor}"
        # update the dstar cells
        self.dstar.updateCells(indexes, values)
        # copy over changed values
        self.last_costmap_state[indexes[:, 0], indexes[:, 1]] = values
        if self.log_all_state_changes:
            save_compressed_pickle({"indexes": indexes, "values": values}, self.log_dir / f"costmap_update_{self.update_costmap_count}.pkl")
            with open(self.log_dir / "log.txt", "a") as f:
                f.write(f"Update Costmap idx {self.update_costmap_count}\n")
            self.update_costmap_count += 1

    def search(self, return_easl_coords=True):
        """
        """
        if self.check_with_astar:
            astar_path, astar_acc_costs = self.astar.search(planning_state_to_easl_state(self.costmap, self.start),
                                                            planning_state_to_easl_state(self.costmap, self.goal), return_easl_coords=False)
            astar_acc_costs = get_acc_costs(astar_path, self.costmap.get_full_costmap(), self.resolution)
        if self.log_all_state_changes:
            g_values = self.dstar.getGValues()
            rhs_values = self.dstar.getRHSValues()
            current_state = {
                "g_values": g_values,
                "rhs_values": rhs_values,
                "keys": self.dstar.getKeys(),
            }
            save_compressed_pickle(current_state, self.log_dir / f"pre_state_{self.replan_count}.pkl")

        dstar_success = self.dstar.replan()
        if self.log_all_state_changes:
            g_values = self.dstar.getGValues()
            rhs_values = self.dstar.getRHSValues()
            current_state = {
                "g_values": g_values,
                "rhs_values": rhs_values,
                "keys": self.dstar.getKeys(),
            }
            save_compressed_pickle(current_state, self.log_dir / f"post_state_{self.replan_count}.pkl")
            with open(self.log_dir / "log.txt", "a") as f:
                f.write(f"Replan {self.replan_count} called with success: {dstar_success}\n")
            self.replan_count += 1
        if not dstar_success:
            g_values = self.dstar.getGValues()
            rhs_values = self.dstar.getRHSValues()
            cur_map = self.dstar.getMap()
            np.save("/tmp/g_values.npy", g_values)
            np.save("/tmp/rhs_values.npy", rhs_values)
            np.save("/tmp/cur_map.npy", cur_map)
            print(f"Start: {self.start}", f"Goal: {self.goal}")
            raise RuntimeError(f"Dstar replanning failed and returned {dstar_success}")
        dstar_plan = self.dstar.getPath()

        # get accumulated costs
        acc_costs = get_acc_costs(dstar_plan, self.costmap.get_full_costmap(), self.resolution)

        if self.check_with_astar:
            if not np.isclose(acc_costs[-1], astar_acc_costs[-1]):
                if acc_costs[-1] > astar_acc_costs[-1]:
                    print("lengths", len(acc_costs), len(astar_acc_costs))
                    if len(acc_costs) == len(astar_acc_costs):
                        for i in range(len(acc_costs)):
                            if dstar_plan[i][0] != astar_path[i][0] or dstar_plan[i][1] != astar_path[i][1]:
                                print(f"Index {i} does not match: Dstar: {dstar_plan[i]}, Astar: {astar_path[i]}")
                    print("starts", dstar_plan[:10], astar_path[:10])
                    print("ends", dstar_plan[-10:], astar_path[-10:])
                    raise RuntimeError(f"DSTAR/ASTAR did not match: End of acc-costs are: \nAStar: {astar_acc_costs[-10:]}\nDstar: {acc_costs[-10:]}")

        # get the path in easl coordinates
        if return_easl_coords:
            easl_path = []
            for item in dstar_plan:
                easl_path.append(planning_state_to_easl_state(
                    self.costmap, PlanningState(item[1], item[0])))

            return easl_path, acc_costs
        else:
            return path, acc_costs

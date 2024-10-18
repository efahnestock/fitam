from __future__ import annotations
from fitam.planning.planner_general import l2_dist_heuristic, snap_easl_state_to_planning_state, list_of_state_to_tuples, State
from fitam.planning.astar import A_Star
from fitam.mapping.costmap import OccupancyGrid
from fitam.core.config.EvaluationConfig import EvaluationConfig
from fitam.evaluation.eval_data_structures import DebugTrialStatistics
import numpy as np
import logging
from pathlib import Path


def move(logger: logging.Logger, planner: A_Star, current_state: State, goal_state: State, replan_index: int, query_i: int,
         observed_states: np.ndarray, local_costmap: OccupancyGrid, dts: DebugTrialStatistics,
         eval_config: EvaluationConfig) -> tuple[bool, bool, State, float]:
    """
    Returns [done, reached_goal, traversed_cost]
    True, False means we timed out, or failed to find a path
    """
    # replan
    logger.debug("Starting to search for new plan")
    path, accrued_costs = planner.search()
    logger.debug("Finished searching for the plan")
    # note: accrued costs are the g values. They start at 0. and then sum all their parent node costs
    dts.path_history.append(list_of_state_to_tuples(path, costmap=local_costmap, convert_to_ps=True))

    if path is None:  # the planning failed
        logger.warning(
            f"Aborting query {query_i} because of a failed plan at index {replan_index}")
        return True, False, current_state, 0.0

    # take step along path until we pass our movement_per_replan_m limit
    logger.debug("Starting to take steps along the path")
    distance_traveled = 0
    last_cost = 0.0
    total_segment_cost = 0.0
    # start at 1 because the first state/cost are the start_state and 0
    for i in range(1, len(path)):
        distance_traveled += l2_dist_heuristic(current_state, path[i])
        if distance_traveled > eval_config.movement_per_replan_m:
            break
        next_state = path[i]
        next_ps = snap_easl_state_to_planning_state(
            local_costmap, next_state
        )
        if observed_states[next_ps.y, next_ps.x] == 0:
            logger.debug(f"Exiting move early (on i {i}) because we would walk into unknown space. Traveled distance {distance_traveled}")
            break

        current_state = next_state
        current_ps = next_ps

        # update the start in the planner
        planner.updateStart(current_state)

        if observed_states[current_ps.y, current_ps.x] == 0:
            # save observed states
            np.save(Path('~/debug/observed_states.npy').expanduser(), observed_states)
            with open(Path('~/debug/other_info.txt').expanduser(), 'w') as f:
                f.write(f"current_ps: {current_ps}\n")
                f.write(f"current_state: {current_state}\n")
                f.write(f"distance_traveled: {distance_traveled}\n")
                f.write(f"distance limit: {eval_config.movement_per_replan_m}\n")
                f.write(f"local costmap: {local_costmap}\n")

            logger.error(
                f"WE ARE TAKING A STEP INTO AN UNOBSERVED CELL! {current_ps}")

            raise RuntimeError("We are taking a step into an unobserved cell")

        dts.state_history.append((current_ps.y, current_ps.x))
        # the cost of this cell is its value minus the previous acrued cost
        cell_cost = accrued_costs[i] - last_cost
        last_cost = accrued_costs[i]
        dts.cost_history.append(float(cell_cost))
        total_segment_cost += cell_cost

    dts.state_transition_indexes.append(len(dts.state_history) - 1)
    # we now have a new current state! let's replan and do it all again
    logger.debug("Finished taking steps along the path")

    # if we've made it to our goal, escape!
    if current_state == goal_state:
        logger.debug("We've reached the goal!")
        return True, True, current_state, total_segment_cost

    # Note: +1 since we just finished an iteration
    if replan_index + 1 >= eval_config.max_planning_iterations:
        logger.debug(
            "WE have hit max planning iterations and will fail with our tail between our legs")
        return True, False, current_state, total_segment_cost

    return False, False, current_state, total_segment_cost  # another round!

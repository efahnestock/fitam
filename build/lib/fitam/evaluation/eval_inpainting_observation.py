from __future__ import annotations
import numpy as np
import logging
from fitam.planning.planner_general import snap_easl_state_to_planning_state, State
from fitam.mapping.costmap import OccupancyGrid
# from radial_learning.utils.easl_python import State
from fitam.mapping.belief import Belief
from fitam.planning.dstar_lite_interface import D_Star_Interface
from fitam.sim.diffusion_interface import DiffusionInterface

def observe(logger: logging.Logger,
            diffusion_interface: DiffusionInterface,
            semantic_overhead: np.ndarray,
            current_state: State,
            replan_index: int,
            belief: Belief,
            master_costmap: OccupancyGrid,
            observed_states: np.ndarray,
            planner: D_Star_Interface, 
    ) -> None:

    # The index (in the costmap) is now [current_planning_state.y, current_planning_state.x]
    current_planning_state = snap_easl_state_to_planning_state(
        belief, current_state)
    center_idx = (current_planning_state.y, current_planning_state.x)
    heading_offset = 0
    logger.debug(
        f"Observing from state {current_state} with heading offset {heading_offset}")

    if diffusion_interface is not None:
        if replan_index % diffusion_interface.config.observation_cadence == 0:
            observations = diffusion_interface.get_observation(
                center_idx=center_idx,
                occ_grid=master_costmap,
                semantic_rgb_float=semantic_overhead,
                observed_mask=observed_states,
                observation_index=replan_index,
            )

            # update radial map!
            logger.debug("Populating the unknown map")
            belief.update_from_observations(
                [observations]
            )
            logger.debug("Finished populating the unknown map")
            costmap = belief.get_full_costmap()
            if np.any(costmap < 0):
                print("negative costmap values after local update")

    logger.debug(
        f"Starting to update the dstar costmap on iter {replan_index}")
    if replan_index == 0:
        planner.Initialize()
    else:
        planner.update_costmap()
    logger.debug("Finished updating the dstar costmap")

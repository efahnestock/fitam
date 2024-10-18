from fitam.mapping.costmap import OccupancyGrid
import numpy as np


class Belief(OccupancyGrid):
    def __init__(self, belief_updater, occ_grid: OccupancyGrid):
        self.belief_updater = belief_updater
        self.state = self.belief_updater.get_initial_state(occ_grid.data.shape)
        self.observation_history = []

        super().__init__(
            resolution=occ_grid.resolution,
            width=occ_grid.info['width'],
            height=occ_grid.info['height'],
            origin=occ_grid.info['origin'],
            frame_id=occ_grid.header['frame_id'],
            data=np.zeros_like(occ_grid.data),
        )

    def update_from_observations(self, observations: list):
        self.observation_history.extend(observations)
        self.belief_updater.update_state_from_observations(
            self.state, observations)

    def get_full_costmap(self):
        return self.belief_updater.state_to_planning_map(self.state)

    def fix_known_states(self, known_observations: list):
        self.belief_updater.fix_known_states(self.state, known_observations)

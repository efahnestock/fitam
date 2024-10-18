import numpy as np
from typing import NamedTuple


class NoUncertaintyBeliefUpdaterConfig(NamedTuple):
    initial_cost: float
    unknown_space_cost: float


class NoUncertaintyAverageBeliefUpdaterState:
    def __init__(self, shape: tuple, initial_cost: float):
        self.mean = np.full(shape, initial_cost)
        self.sum = np.zeros(shape)
        self.count = np.zeros(shape, dtype=int)
        self.gt_observed = np.zeros(shape, dtype=bool)


class NoUncertaintyAverageBeliefUpdater:
    """
    We use kalman observations, just ignore the variance. We just use the last observed value for each cell.
    """
    def __init__(self, config: NoUncertaintyBeliefUpdaterConfig):
        self.config = config

    def get_initial_state(self, shape: tuple) -> NoUncertaintyAverageBeliefUpdaterState:
        return NoUncertaintyAverageBeliefUpdaterState(shape, self.config.initial_cost)

    def update_state_from_observations(self, state, observations: list) -> None:
        for observation in observations:
            kalman_obs = observation.to_kalman_observation(state.mean.shape)  # i, j, mean, variance
            assert not np.any(np.isinf(kalman_obs)), "Inf not allowed in observation. Causes issues with z_t - x_t step if both z_t and x_t are inf"

            idxs = (kalman_obs.i, kalman_obs.j)
            z_t = kalman_obs.cost

            no_local_observations = state.gt_observed[idxs] == 0

            idxs = (idxs[0][no_local_observations], idxs[1][no_local_observations])

            state.count[idxs] += 1
            state.sum[idxs] += z_t[no_local_observations]
            state.mean[idxs] = state.sum[idxs] / state.count[idxs]

            if np.any(np.isnan(state.mean[idxs])):
                print("Negative cost or NAN in Most Recent filter")
                raise RuntimeError()

    def state_to_planning_map(self, state: NoUncertaintyAverageBeliefUpdaterState) -> np.ndarray:
        """
        We want to discard uncertain high speed predictions. We should be sure about all of
        the path affecting changes we suggest to the planner.
        """
        output_map = state.mean.copy()
        return output_map

    def fix_known_states(self, state, observations: list) -> None:
        for observation in observations:
            kalman_obs = observation.to_kalman_observation(state.mean.shape)
            idxs = (kalman_obs.i, kalman_obs.j)
            state.mean[idxs] = kalman_obs.cost
            state.gt_observed[idxs] = True

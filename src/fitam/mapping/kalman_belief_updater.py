
"""
A belief representation must have:

create_state(default_cost, size)

update_state_from_observations(state, post_processed_observations)

fix_known_states(state, known_states)




Belief - holds all of the arrays
       - takes the place of radial_costmap, can produce downstream planning maps 
BeliefUpdater - updates the belief and knows how to work on an individual cell
Observation - answers questions in a spatial sort of way (gt, ff, diffusion)
            - what are the list of places I can say something about
            - can produce it in i,j format, given robot's position in the world, resolution, etc.

"""
import numpy as np
from typing import NamedTuple


class KalmanObservation(NamedTuple):
    i: np.ndarray
    j: np.ndarray
    cost: np.ndarray
    variance: np.ndarray

class KalmanBeliefUpdaterConfig(NamedTuple):
    initial_cost: float
    initial_variance: float
    variance_threshold: float
    observation_variance_threshold: float
    unknown_space_cost: float
    process_noise: float


class KalmanState:
    def __init__(self, shape: tuple, initial_cost: float, initial_variance: float):
        self.mean = np.full(shape, initial_cost)
        self.variance = np.full(shape, initial_variance)
        self.gt_observed = np.zeros(shape, dtype=bool)


class KalmanBeliefUpdater:
    # Equations:
    # K_t = p_{t-1} / (p_{t-1} + R_t) -- Kalman gain
    # State update. These are per-cell
    # x_t = x_{t-1} + K_t * (z_t - x_{t-1}) -- update state with measurement
    # p_t = (1 - K_t) * p_{t-1} -- update covariance with measurement
    # State prediction. These apply to the whole map
    # x_t = x_{t-1}
    # p_t = p_{t-1} + Q_t -- Q_t is the process noise

    # Layers:
    # Each cell needs a state and a covariance
    # Each measurement needs a value (z_t, speed) and a covariance (R_t, measurement noise)
    # Hyperparameters
    # The map needs a process noise (Q_t, process noise, how fast variance grows)
    # The map needs initial state and covariance (x_0, p_0)
    def __init__(self, config: KalmanBeliefUpdaterConfig):
        self.config = config

    def get_initial_state(self, shape: tuple) -> KalmanState:
        return KalmanState(shape, self.config.initial_cost, self.config.initial_variance)

    def update_state_from_observations(self, state, observations: list) -> None:
        for observation in observations:
            kalman_obs = observation.to_kalman_observation(state.mean.shape)  # i, j, mean, variance
            assert not np.any(np.isinf(kalman_obs)), "Inf not allowed in observation. Causes issues with z_t - x_t step if both z_t and x_t are inf"
            idxs = (kalman_obs.i, kalman_obs.j)
            mask = kalman_obs.variance < self.config.observation_variance_threshold
            idxs = (idxs[0][mask], idxs[1][mask])
            R_t = kalman_obs.variance[mask]
            p_t = state.variance[idxs]
            z_t = kalman_obs.cost[mask]

            # avoid divide by inf errors. Leave inf cells as inf
            x_t = state.mean[idxs]
            # K_t = p_{t-1} / (p_{t-1} + R_t) -- Kalman gain
            K_t = p_t / (p_t + R_t)
            # avoid divide by zero errors if zero variance in map and in measurement
            # Stops from updating values in map states with zero variance.
            K_t[p_t == 0] = 0
            # x_t = x_{t-1} + K_t * (z_t - x_{t-1}) -- update state with measurement
            inf_indexes = (idxs[0][np.isinf(state.mean[idxs])],
                           idxs[1][np.isinf(state.mean[idxs])])
            x_t[np.isinf(x_t)] = 0  # set inf to zero for now to avoid nans
            # apply update
            state.mean[idxs] = x_t + K_t * (z_t - x_t)
            # fix zeros back to inf
            state.mean[inf_indexes] = np.inf

            if np.any(np.isnan(state.mean[idxs])) or np.any(state.mean[idxs] < 0.0): # min map value is 0.0
                print("offending values value: ", state.mean[idxs][state.mean[idxs] < 0.0])
                print("Negative cost or NAN in kalman filter")
                print("x_t", x_t)
                print("K_t", K_t)
                print("R_t", R_t)
                print("p_t", p_t)
                print("z_t", z_t)
                print("x_t + K_t * (z_t - x_t)", x_t + K_t * (z_t - x_t))
                debug_output = {
                    "x_t": x_t,
                    "K_t": K_t,
                    "R_t": R_t,
                    "p_t": p_t,
                    "z_t": z_t,
                    "x_t + K_t * (z_t - x_t)": x_t + K_t * (z_t - x_t),
                    "state.mean[idxs]": state.mean[idxs],
                    "observations": observations
                }
                with open("kalman_debug.pkl", "wb") as f:
                    import pickle
                    pickle.dump(debug_output, f)
                raise RuntimeError()

            # p_t = (1 - K_t) * p_{t-1} -- update covariance with measurement
            state.variance[idxs] = (1 - K_t) * p_t

            # We don't want to update ground truth observations (variance == 0) to prevent loopy behavior
            # this doesn't prevent ground truth (local) observations from overwriting far-field observations
            state.gt_observed[idxs][R_t == 0] = True

        # insert process noise
        state.variance[state.gt_observed == False] += self.config.process_noise

    def state_to_planning_map(self, k_state: KalmanState) -> np.ndarray:
        """
        We want to discard uncertain high speed predictions. We should be sure about all of
        the path affecting changes we suggest to the planner.
        """
        output_map = k_state.mean.copy()
        output_map[k_state.variance >
                   self.config.variance_threshold] = self.config.unknown_space_cost
        return output_map

    def fix_known_states(self, state, observations: list) -> None:
        for observation in observations:
            kalman_obs = observation.to_kalman_observation(state.mean.shape)
            idxs = (kalman_obs.i, kalman_obs.j)
            state.mean[idxs] = kalman_obs.cost
            state.variance[idxs] = 0
            state.gt_observed[idxs] = True



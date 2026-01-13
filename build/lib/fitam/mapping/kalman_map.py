import numpy as np
from radial_learning.sim.costmap_swath_library import SwathLibrary

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


def entropy_to_variance(entropies: np.ndarray) -> np.ndarray:
    """
    Map the entropy of the average categorical distribution to a variance for kalman filtering.
    Maximum entropy is -ln(0.5) ~= 0.69, minimum is 0
    """
    return entropies


def kalman_map_to_planning_map(map_state_layer: np.ndarray,
                               map_covariance_layer: np.ndarray,
                               variance_threshold: float,
                               unknown_space_cost: float,
                               ) -> np.ndarray:
    """
    We want to discard uncertain high speed predictions. We should be sure about all of
    the path affecting changes we suggest to the planner.
    """
    output_map = map_state_layer.copy()
    output_map[map_covariance_layer > variance_threshold] = unknown_space_cost
    return output_map


def kalman_add_observations(observations: list,
                            map_state_layer: np.ndarray,  # x_t
                            map_covariance_layer: np.ndarray,  # p_t
                            num_obs_layer: np.ndarray,  # number of observations per cell
                            swath_library: SwathLibrary,
                            range_bins: list,
                            yaw_shift: float,
                            process_noise: float,
                            ):

    for observation in observations:
        center_idx = observation['center_idx']
        yaw_range = observation['yaw_range']
        costs = observation['costs']
        noise = observation['noise']
        for i, cost_i in enumerate(costs):
            r_min = range_bins[i]
            y_min = yaw_range[1] + yaw_shift
            idxs = swath_library.get_swath(
                center_idx, r_min, y_min, map_state_layer.shape)
            # fix the fu*king nans
            nan_indexes = (idxs[0][np.isnan(num_obs_layer[idxs])],
                           idxs[1][np.isnan(num_obs_layer[idxs])])
            num_obs_layer[nan_indexes] = 0
            # don't modify negative number of observations (local observations)
            num_obs_layer[idxs][num_obs_layer[idxs] >= 0] += 1

            R_t = entropy_to_variance(noise[i])
            p_t = map_covariance_layer[idxs]
            z_t = cost_i

            # avoid divide by inf errors. Leave inf cells as inf
            x_t = map_state_layer[idxs]
            # K_t = p_{t-1} / (p_{t-1} + R_t) -- Kalman gain
            K_t = p_t / (p_t + R_t)
            K_t[p_t == 0] = 0 # avoid divide by zero errors if zero variance in map and in measurement
            # x_t = x_{t-1} + K_t * (z_t - x_{t-1}) -- update state with measurement
            # fix the fu*king nans that result from infinite costs in x_t
            inf_indexes = (idxs[0][np.isinf(map_state_layer[idxs])],
                           idxs[1][np.isinf(map_state_layer[idxs])])
            x_t[np.isinf(x_t)] = 0  # set inf to zero for now to avoid nans
            # apply update
            map_state_layer[idxs] = x_t + K_t * (z_t - x_t)
            # fix zeros back to inf
            map_state_layer[inf_indexes] = np.inf

            if np.any(np.isnan(map_state_layer[idxs])) or np.any(map_state_layer[idxs] < 0):
                print("Negative cost or NAN in kalman filter")
                print("x_t", x_t)
                print("K_t", K_t)
                print("R_t", R_t)
                print("p_t", p_t)
                print("z_t", z_t)
                print("x_t + K_t * (z_t - x_t)", x_t + K_t * (z_t - x_t))
                raise RuntimeError()

            # p_t = (1 - K_t) * p_{t-1} -- update covariance with measurement
            map_covariance_layer[idxs] = (1 - K_t) * p_t

    # insert process noise
    map_covariance_layer[num_obs_layer >= 0] += process_noise

    return

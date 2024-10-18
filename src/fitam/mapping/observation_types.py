
import numpy as np
import torch
from fitam.mapping.costmap_swath_library import SwathLibrary
from fitam.core.config.RadialMapConfig import FarFieldConfig
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.kalman_belief_updater import KalmanObservation
from fitam.learning.fully_separate_ensemble_torch import FullEnsembleTorch


def map_classes_to_speeds_numpy(class_array, class_speeds)->np.ndarray:
    # class_tensor: (batch, num_bins)
    class_to_speed_arr = np.array(class_speeds)
    speeds = class_to_speed_arr[class_array.astype(np.int32)] 
    return speeds

class RadialObservation:
    def __init__(self,
                 center_idx: tuple[int, int],
                 lr_yaw: tuple[float, float],
                 model_outputs: np.ndarray,  # num_bins x num_models x num_classes
                 swath_library: SwathLibrary,
                 ff_config: FarFieldConfig,
                 yaw_shift: float) -> None:
        self.center_idx = center_idx
        self.lr_yaw = lr_yaw
        self.model_outputs = model_outputs
        self.swath_library = swath_library
        self.ff_config = ff_config
        self.yaw_shift = yaw_shift

    def __repr__(self) -> str:
        return f"RadialObservation(center_idx={self.center_idx}, lr_yaw={self.lr_yaw}, model_outputs={self.model_outputs}, swath_library={self.swath_library}, ff_config={self.ff_config}, yaw_shift={self.yaw_shift})"
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['swath_library'] = None
        return state


    @classmethod
    def create_random_observation(cls,
                                  center_idx: tuple[int, int],
                                  lr_yaw: tuple[float, float],
                                  swath_library: SwathLibrary,
                                  ff_config: FarFieldConfig,
                                  yaw_shift: float,
                                  num_models: int = 15) -> None:
        model_outputs = np.random.rand(
            ff_config.num_range_bins, num_models, ff_config.classification_config.num_classes)
        return cls(center_idx, lr_yaw, model_outputs, swath_library, ff_config, yaw_shift)

    def _calculate_classes(self) -> np.ndarray:
        classes = FullEnsembleTorch.calculate_classes(torch.from_numpy(
            self.model_outputs).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        return classes

    def _calculate_uncertainty(self) -> np.ndarray:
        uncertainty = FullEnsembleTorch.calculate_winning_confidence(torch.from_numpy(
            self.model_outputs).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        return 1.0 - uncertainty  # variance is more uncertain as it increases
    
    def trim_to_num_bins(self, num_bins: int)-> None:
        self.model_outputs = self.model_outputs[:num_bins]

    @staticmethod
    def _entropy_to_variance(entropies: np.ndarray) -> np.ndarray:
        """
        Map the entropy of the average categorical distribution to a variance for kalman filtering.
        Maximum entropy is -ln(0.5) ~= 0.69, minimum is 0
        """
        return entropies

    
    def to_kalman_observation(self, map_shape: tuple) -> KalmanObservation:
        center_idx = self.center_idx
        lr_yaw = self.lr_yaw
        classes = self._calculate_classes()
        speeds = map_classes_to_speeds_numpy(
            classes, self.ff_config.classification_config.class_speeds)
        uncertainty = self._calculate_uncertainty()
        costs = 1 / speeds
        noise = self._entropy_to_variance(uncertainty)
        out_i, out_j, out_cost, out_noise = [], [], [], []
        for i, cost_i in enumerate(costs):
            r_min = self.ff_config.range_bins[i]
            y_min = lr_yaw[1] + self.yaw_shift
            idxs = self.swath_library.get_swath(
                center_idx, r_min, y_min, map_shape)
            n = idxs[0].shape[0]
            out_i.append(idxs[0])
            out_j.append(idxs[1])
            out_cost.append([cost_i] * n)
            out_noise.append([noise[i]] * n)
        return KalmanObservation(np.concatenate(out_i),
                                 np.concatenate(out_j),
                                 np.concatenate(out_cost),
                                 np.concatenate(out_noise))


class GTObservation:

    def __init__(self,
                 i: np.ndarray,
                 j: np.ndarray,
                 costs: np.ndarray,
                 ) -> None:
        self.i = i
        self.j = j
        self.costs = costs

    def to_kalman_observation(self, map_shape: tuple) -> KalmanObservation:
        return KalmanObservation(self.i, self.j, self.costs, np.zeros_like(self.costs))

    def __repr__(self) -> str:
        out = "GTObservation("
        for (i,j,c) in zip(self.i, self.j, self.costs):
            out += f"({i},{j}:{c}),"
        out += ")"
        return out


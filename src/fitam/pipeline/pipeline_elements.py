
from os import PathLike
from typing import NamedTuple, Optional
from fitam.learning.bin_ensemble_torch import MemberModelName


class Swath(NamedTuple):
    name: str
    radial_map_config_path: PathLike
    save_path: PathLike


class EvaluationRequest(NamedTuple):
    name: str
    map_paths: list[PathLike]
    save_paths: list[PathLike]
    evaluation_config_path: PathLike


class MapSampling(NamedTuple):
    name: str
    env_config_path: PathLike
    map_in_paths: list[PathLike]
    map_out_paths: list[PathLike]
    make_obstacles: bool = True


class LocationSampling(NamedTuple):
    name: str
    map_paths: list[PathLike]
    save_paths: list[PathLike]
    dataset_config_path: PathLike


class TimedLocationSampling(NamedTuple):
    name: str
    seed: int
    map_path: PathLike
    save_folder_path: PathLike
    seconds_of_exploration: float
    obs_rad_m: float
    meters_between_keypoints: float = 3
    max_iters_per_waypoint: int = 10


# I think I can create one per map?
class ImageRendering(NamedTuple):
    name: str  # name of task
    dependent_task: str  # name of dependent task
    dataset_config_path: PathLike  # path to dataset config .json
    compute_config_path: PathLike  # path to compute config .json
    # paths to the image requests .json or .pkl
    location_sample_paths: list[PathLike]
    # paths to save the images from the i'th image request. Should be subdir of save_root_path
    save_paths: list[PathLike]
    save_root_path: PathLike  # The root path to do all the image saving in.
    complex_map_paths: Optional[list[PathLike]
                                ] = None  # path to the complex map


class Dataset(NamedTuple):
    name: str  # name of task
    save_root_dir: PathLike  # root of the dataset
    # dependent_task:str # name of dependent task
    radial_map_config_path: PathLike  # path to radial map config .json
    # compute_config_path:PathLike # path to compute config .json
    dataset_config_path: PathLike  # path to json dataset config
    # swath_library_path:PathLike # path to the swath library pickle file
    image_db_path: PathLike  # paths to the csv points files
    # save_djikstra:bool = False # save the djikstra maps generated (likely deprecated)
    # keep_all_points: bool = False # keep all points in the dataset, not just valid ones. Note that for classification, they're all kept anyway, with zero coverage


class Training(NamedTuple):
    name: str
    dataset_csv_path: PathLike  # path to dataset csv
    save_path: PathLike  # root save path, saved as save_path/model_i:02d
    train_config_path: PathLike
    radial_config_path: Optional[PathLike] = None
    # So models aren't trained at the same time, make each model dependent on the previous one
    training_dependency_name: Optional[str] = None
    test_dataset: Optional[PathLike] = None
    # if greater than 1, train this many ensembles and spit out a combined model at the end
    ensemble_members: int = 15
    # if None, train all the bins in the radial map config
    num_bins: Optional[int] = None
    member_model_name: MemberModelName = MemberModelName.UNTRAINED_RESNET_INDIVIDUAL
    member_model_args: Optional[dict] = None


class ModelTesting(NamedTuple):
    name: str
    model_path: PathLike
    dataset_csv_path: PathLike
    save_path: PathLike
    test_config_path: PathLike
    dependent_task: str = None


class Evaluation(NamedTuple):
    name: str
    save_root_path: PathLike
    eval_request_paths: list[PathLike]
    save_paths: list[PathLike]  # relative to the save_root_path
    model_path: PathLike  # path to the model weights
    evaluation_config_path: PathLike
    radial_map_config_path: PathLike
    dataset_config_path: PathLike
    logging_config_path: PathLike
    training_config_path: PathLike
    compute_config_path: PathLike
    swath_library_path: PathLike
    num_active_bins: int = None  # if None, use all bins
    dependent_task: str = None

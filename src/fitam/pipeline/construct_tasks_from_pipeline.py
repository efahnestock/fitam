from __future__ import annotations
from typing import Callable
from fitam.pipeline.pipeline_elements import (
    LocationSampling, MapSampling, Swath, Dataset, Training, EvaluationRequest, Evaluation, ImageRendering, ModelTesting, TimedLocationSampling
)
from fitam.core.common import load_json_config, load_pickle_config, create_dir
from fitam import (SAMPLED_LOCATIONS_DIR, MODELS_DIR, EVALUATION_REQUESTS_DIR, CONFIGS_DIR, SWATHS_DIR,
                   EVALUATIONS_DIR, FITAM_ROOT_DIR, IMAGES_DIR, MAPS_DIR)
from doit import task
import pandas as pd
from pathlib import Path


def make_tasks_from_config(config: list, headless=False) -> list[Callable]:
    resulting_tasks = []  # list of tuples (name, function)
    for c in config:
        if type(c) == Swath:
            resulting_tasks.append(make_swath_task(c))
        # elif type(c) == MapSampling:
        #     resulting_tasks.append(make_map_sampling_task(c))
        # elif type(c) == LocationSampling:
        #     resulting_tasks.append(make_location_sampling_task(c))
        # elif type(c) == TimedLocationSampling:
        #     resulting_tasks.append(make_time_based_location_sampling_task(c))
        # elif type(c) == ImageRendering:
        #     resulting_tasks.append(make_image_rendering_task(c))
        # elif type(c) == Dataset:
        #     resulting_tasks.append(make_dataset_task(c))
        #     # resulting_tasks.append(make_feature_extraction_task(c))
        # elif type(c) == Training:
        #     resulting_tasks.append(make_training_task(c))
        # elif type(c) == ModelTesting:
        #     resulting_tasks.append(make_model_testing_task(c))
        # elif type(c) == EvaluationRequest:
        #     resulting_tasks.append(make_evaluation_request_task(c))
        # elif type(c) == Evaluation:
        #     resulting_tasks.append(make_evaluation_task(c, headless))
        else:
            raise ValueError(f"Unknown config type {type(c)}")
    return resulting_tasks


def make_swath_task(c: Swath) -> tuple[str, Callable]:
    def task_create_swath():
        from fitam.mapping.costmap_swath_library import SwathLibrary, save_swath_library_to_pkl

        def create_and_save_swath_library(
                radial_map_config_path: Path,
                save_path: Path
        ):
            sl = SwathLibrary.from_radial_map_config(load_json_config(radial_map_config_path), generate=True, show_progress=False)
            create_dir(save_path.parent)
            save_swath_library_to_pkl(save_path, sl)

        yield dict(
            name=c.name,
            actions=[(create_and_save_swath_library, (), dict(radial_map_config_path=c.radial_map_config_path,
                                                              save_path=c.save_path))],
            targets=[c.save_path],
            file_dep=[c.radial_map_config_path],
        )
    return (c.name, task_create_swath)


"""
def make_map_sampling_task(c: MapSampling) -> tuple[str, Callable]:
    def task_create_maps():
        from radial_learning.pipeline.environment_sampling import setup_single_map_from_env_config
        yield {
            'basename': "map_sampling_" + c.name,
            'name': None,
        }
        all_deps = []
        all_targets = []
        for in_path, out_path in zip(c.map_in_paths, c.map_out_paths):
            new_targets = [out_path / f"{out_path.name}.pkl"]
            all_targets.extend(new_targets)
            new_deps = [in_path / f"{in_path.name}.pkl", c.env_config_path]
            all_deps.extend(new_deps)
            yield dict(
                name=f"{in_path}_{out_path}",
                actions=[(setup_single_map_from_env_config, (), dict(input_path=in_path,
                                                                     env_config_path=c.env_config_path,
                                                                     make_obstacles=c.make_obstacles,
                                                                     output_path=out_path))],
                targets=new_targets,
                file_dep=new_deps,
            )
    return ("map_sampling_" + c.name, task_create_maps)


def make_time_based_location_sampling_task(c: TimedLocationSampling) -> tuple[str, Callable]:

    def task_create_timed_sampling():
        from radial_learning.dataset_generation.limited_overhead import main

        yield {
            'name': f"timed_sampling_{c.name}",
            'actions': [(main, (), dict(
                max_iters_per_waypoint=c.max_iters_per_waypoint,
                obs_rad_m=c.obs_rad_m,
                meters_between_keypoints=c.meters_between_keypoints,
                seconds_per_image_req=c.seconds_of_exploration,
                output_path=c.save_folder_path,
                seed=c.seed,
                map_path=c.map_path,
            ))],
            'file_dep': [c.map_path / f"{c.map_path.name}.pkl"],
            'targets': [SAMPLED_LOCATIONS_DIR / c.save_folder_path, MAPS_DIR / c.save_folder_path],
        }

    return (f"timed_sampling_{c.name}", task_create_timed_sampling)


def make_location_sampling_task(c: LocationSampling) -> tuple[str, Callable]:

    def task_create_sampling():
        from radial_learning.pipeline.dataset_generation import generate_single_image_request
        yield {
            'basename': c.name,
            'name': None,
        }

        for in_path, out_path in zip(c.map_paths, c.save_paths):
            yield {
                'name': f'{in_path}_{out_path}',
                'actions': [(generate_single_image_request, (), dict(dataset_config_path=c.dataset_config_path,
                                                                     map_path=in_path,
                                                                     save_path=out_path))],
                'file_dep': [c.dataset_config_path, in_path / f"{in_path.name}.pkl"],
                'targets': [out_path],

            }
    return (c.name, task_create_sampling)


def make_image_rendering_task(c: ImageRendering) -> tuple[str, Callable]:

    def task_create_rendering():
        from radial_learning.pipeline.dataset_generation import create_images_part

        # load the compute config!
        compute_config = load_json_config(c.compute_config_path)
        all_targets = []
        all_deps = []

        yield {
            'basename': c.name,
            'name': None,
        }

        for global_idx, (img_req_path, save_root) in enumerate(zip(c.location_sample_paths, c.save_paths)):
            # now tile these!

            # load the image request!
            if img_req_path.suffix == ".json":
                img_req = load_json_config(img_req_path)
            elif img_req_path.suffix == ".pkl":
                img_req = load_pickle_config(img_req_path)
            else:
                raise RuntimeError(f"Unknown file type {img_req_path.suffix} for image request {img_req_path}")
            num_tiles = len(img_req.locations) // compute_config.num_locations_per_process
            if c.complex_map_paths is not None:
                map_path = c.complex_map_paths[global_idx]
            else:
                map_path = None

            for i in range(num_tiles):
                start_idx = i * compute_config.num_locations_per_process
                end_idx = (i+1) * compute_config.num_locations_per_process if i < num_tiles - 1 else len(img_req.locations)
                new_targets = [save_root / f'points_{start_idx}_{end_idx}.csv']
                all_targets.extend(new_targets)
                new_deps = [img_req_path, c.dataset_config_path]
                all_deps.extend(new_deps)
                yield dict(
                    name=f"{img_req_path}_{i}",
                    actions=[(create_images_part, (), dict(image_request_path=img_req_path,
                                                           start_idx=start_idx,
                                                           end_idx=end_idx,
                                                           save_root_dir=save_root,
                                                           dataset_config_path=c.dataset_config_path,
                                                           complex_map_path=map_path,
                                                           ))],
                    file_dep=new_deps,
                    targets=new_targets,
                )
        # now merge all the image info files!
        from radial_learning.pipeline.dataset_generation import merge_image_csvs
        yield dict(
            name=f"{c.name}_merge",
            actions=[(merge_image_csvs, (), dict(root_dir=c.save_root_path))],
            file_dep=all_targets,
            targets=[c.save_root_path / "points.csv"],
        )

    # task_create_rendering.doit_create_after = task.DelayedLoader(
    #     creator=task_create_rendering,
    #     executed=c.dependent_task,
    #     target_regex=None,
    #     creates=[c.name]
    # )

    return (c.name, task_create_rendering)


def make_dataset_task(c: Dataset) -> tuple[str, Callable]:
    # def task_create_dataset():
    #     from radial_learning.pipeline.dataset_generation import create_labels_part

    #     # load the compute config!
    #     compute_config = load_json_config(c.compute_config_path)
    #     all_targets = []
    #     all_deps = []

    #     yield {
    #         'basename': f"dataset_{c.name}",
    #         'name': None,
    #     }

    #     for img_df_path in c.image_db_paths:
    #         # now tile these!

    #         # load the image database!
    #         img_df = pd.read_csv(img_df_path)
    #         num_tiles = max(1, len(img_df) // compute_config.num_locations_per_process)
    #         # print(c.name, num_tiles, len(img_df))

    #         save_root = c.save_root_dir / "_".join(img_df_path.relative_to(IMAGES_DIR).parts[:-1])

    #         for i in range(num_tiles):
    #             start_idx = i * compute_config.num_locations_per_process
    #             end_idx = (i+1) * compute_config.num_locations_per_process if i < num_tiles - 1 else len(img_df)
    #             new_targets = [save_root / f"point_{i:07d}" / "labels.csv" for i in range(start_idx, end_idx)]
    #             all_targets.extend(new_targets)
    #             new_deps = [img_df_path, c.radial_map_config_path, c.dataset_config_path, c.swath_library_path]
    #             all_deps.extend(new_deps)
    #             yield dict(
    #                 name=f"{img_df_path}__{i}",
    #                 actions=[(create_labels_part, (), dict(images_db_path=img_df_path,
    #                                                          start_idx=start_idx,
    #                                                          end_idx=end_idx,
    #                                                          save_root_dir=save_root,
    #                                                          radial_map_config_path=c.radial_map_config_path,
    #                                                          dataset_config_path=c.dataset_config_path,
    #                                                          swath_library_path=c.swath_library_path,
    #                                                          save_djikstra=c.save_djikstra,
    #                                                          keep_all_points = c.keep_all_points,))],
    #                 file_dep=new_deps,
    #                 targets=new_targets,
    #                 verbosity=2,
    #             )
    #     # now merge all the labels files!
    #     from radial_learning.pipeline.dataset_generation import merge_individual_label_files
    #     yield dict(
    #         name=f"dataset_{c.name}_merge",
    #         actions=[(merge_individual_label_files, (), dict(dataset_root_dir=c.save_root_dir,
    #                                                         save_path=c.save_root_dir / "all_labels.csv"))],
    #         file_dep=all_targets,
    #         targets=[c.save_root_dir / "all_labels.csv"],
    #     )
    def task_create_balanced_dataset():
        from radial_learning.dataset_generation.dataset_blueprints import make_semantically_balanced_dataset_from_image_request
        yield dict(
            name=f"dataset_{c.name}",
            actions=[(make_semantically_balanced_dataset_from_image_request, (), dict(
                image_df_path=c.image_db_path,
                radial_map_config_path=c.radial_map_config_path,
                dataset_config_path=c.dataset_config_path,
                save_folder=c.save_root_dir,
            ))],
            verbosity=2,
            targets=[c.save_root_dir / "balanced_semantic_dataset.csv"],
            file_dep=[c.image_db_path, c.radial_map_config_path, c.dataset_config_path]
        )

    # task_create_dataset.doit_create_after = task.DelayedLoader(
    #     creator=task_create_dataset,
    #     executed=c.dependent_task,
    #     target_regex=None,
    #     creates=[f"dataset_{c.name}"]
    # )
    return (f"dataset_{c.name}", task_create_balanced_dataset)


def make_feature_extraction_task(c: Dataset) -> tuple[str, Callable]:

    from radial_learning.pipeline.dataset_generation import extract_features_from_dataset
    from radial_learning.training.pl_img_dataloader import get_default_transform_from_image_path
    from torchvision import transforms

    def task_extract_features():
        img_dir = c.save_root_dir
        csv_path = c.save_root_dir / "all_labels.csv"
        csv = pd.read_csv(csv_path)
        full_img_path = img_dir / csv['img_path'].iloc[0]
        yield dict(
            name=f"dataset_extract_{c.name}",
            actions=[(extract_features_from_dataset, (), dict(dataset_csv_file=csv_path,
                                                              dataset_images_dir=img_dir,
                                                              feature_save_dir=c.save_root_dir / "features",
                                                              transforms=transforms.Compose(get_default_transform_from_image_path(full_img_path))))],
            file_dep=[csv_path],
        )
    task_extract_features.doit_create_after = task.DelayedLoader(
        creator=task_extract_features,
        executed=f"dataset_{c.name}",
        target_regex=None,
        creates=[f"extract_{c.name}"]
    )

    return (f"dataset_extract_{c.name}", task_extract_features)


def make_training_task(c: Training):

    def task_train():
        from radial_learning.pipeline.network_training import train_ensemble_pipeline

        file_dependencies = [c.dataset_csv_path]

        yield dict(
            name="training_" + c.name,
            actions=[(train_ensemble_pipeline, (), dict(train_config_path=c.train_config_path,
                                                        num_ensemble_members=c.ensemble_members,
                                                        num_bins=c.num_bins,
                                                        radial_map_config_path=c.radial_config_path,
                                                        save_path=c.save_path,
                                                        dataset_csv_path=c.dataset_csv_path,
                                                        member_model_name=c.member_model_name,
                                                        member_model_args=c.member_model_args,
                                                        ))],
            file_dep=file_dependencies,
            targets=[c.save_path / f"{c.save_path.name}.ckpt"],
            verbosity=2,
        )

    return ("training_" + c.name, task_train)


def make_model_testing_task(c: ModelTesting):

    def task_model_test():
        from radial_learning.pipeline.network_training import test_individual_network

        yield {
            'name': 'model_testing_'+c.name,
            'actions': [(test_individual_network, (), dict(
                train_config_path=c.test_config_path,
                save_dir=c.save_path,
                dataset_csv_path=c.dataset_csv_path,
                model_checkpoint_path=c.model_path,
            ))],
            'file_dep': [c.model_path, c.dataset_csv_path],
            'targets': [c.save_path / "test_results.json"],
            'verbosity': 2,
        }

    return ('model_testing_'+c.name, task_model_test)


def make_evaluation_request_task(c: EvaluationRequest):

    def task_evaluation_request():
        from radial_learning.pipeline.approach_evaluation import create_single_eval_request_for_env

        yield {
            'name': None,
        }
        for map_path, save_path in zip(c.map_paths, c.save_paths):
            yield dict(
                name=f"{map_path}__{save_path}",
                actions=[(create_single_eval_request_for_env, (), dict(map_path=map_path,
                                                                       eval_config_path=c.evaluation_config_path,
                                                                       save_path=save_path))],
                file_dep=[map_path / f"{map_path.name}.pkl", c.evaluation_config_path],
                targets=[save_path],
            )

    return (c.name, task_evaluation_request)


def make_evaluation_task(c: Evaluation, headless: bool = False) -> tuple[str, Callable]:

    def task_evaluation():
        from radial_learning.evaluation.planner_trial_opengl import worker
        from radial_learning.evaluation.eval_save_inputs import save_all_inputs
        from radial_learning.evaluation.eval_utils import monitor_workers

        yield {
            'basename': "evaluation_" + c.name,
            'name': None,
        }

        all_file_deps = c.eval_request_paths + [c.evaluation_config_path, c.compute_config_path] + ([c.dataset_config_path] if c.dataset_config_path is not None else []) + [
            c.radial_map_config_path, c.swath_library_path, c.logging_config_path] + ([c.training_config_path] if c.training_config_path is not None else []) + ([c.model_path] if c.model_path is not None else [])
        yield dict(
            name=f"{c.name}_save_inputs",
            actions=[(save_all_inputs, (), dict(eval=c))],
            file_dep=all_file_deps,
            targets=[c.save_root_path / "inputs.json"],
        )

        # monitor task
        # yield dict(
        #     name=f"{c.name}_monitor",
        #     actions=[(monitor_workers, (), dict(
        #                                     save_roots=[c.save_root_path / x for x in c.save_paths],
        #                                     eval_request_paths=c.eval_request_paths,
        #     ))],
        #     verbosity=2,
        # )

        # load the compute config!
        compute_config = load_json_config(c.compute_config_path)

        all_targets = []
        if len(c.save_paths) != len(c.eval_request_paths):
            raise RuntimeError(f"Number of save paths ({len(c.save_paths)}) does not match number of eval requests ({len(c.eval_request_paths)})")
        for save_path, eval_req_path in zip(c.save_paths, c.eval_request_paths):
            # now tile these!
            # load the evaluation request!
            eval_req = load_json_config(eval_req_path)
            num_tiles = max(1, len(eval_req.points) // compute_config.num_planning_requests_per_process)  # make sure we have at least one tile

            for i in range(num_tiles):
                start_idx = i * compute_config.num_planning_requests_per_process
                end_idx = (i+1) * compute_config.num_planning_requests_per_process if i < num_tiles - 1 else len(eval_req.points)
                new_targets = [c.save_root_path / save_path / f"{i:07d}" / "trial_info.json" for i in range(start_idx, end_idx)]
                all_targets.extend(new_targets)
                yield dict(
                    name=f"{eval_req_path}_{i}",
                    actions=[(worker, (), dict(
                        job_numbers=list(range(start_idx, end_idx)),
                        save_root=c.save_root_path / save_path,
                        model_path=c.model_path,
                        eval_request_path=eval_req_path,
                        radial_costmap_config_path=c.radial_map_config_path,
                        training_config_path=c.training_config_path,
                        dataset_config_path=c.dataset_config_path,
                        logging_config_path=c.logging_config_path,
                        eval_config_path=c.evaluation_config_path,
                        swath_library_path=c.swath_library_path,
                        num_active_bins=c.num_active_bins,
                    ))],
                    file_dep=all_file_deps,
                    targets=new_targets,
                    verbosity=2,
                )

    # task_evaluation.doit_create_after = task.DelayedLoader(
    #     creator=task_evaluation,
    #     executed=c.dependent_task,
    #     target_regex=None,
    #     creates=[c.name]
    # )

    return ("evaluation_" + c.name, task_evaluation)
"""

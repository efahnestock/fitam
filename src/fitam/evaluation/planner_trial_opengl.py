from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import time
import pandas as pd
import argparse
import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from os import PathLike
import uuid
from pytorch_lightning import seed_everything
from fitam import MODELS_DIR, SWATHS_DIR, CONFIGS_DIR
from fitam.core.common import create_dir, setup_logger, decode_str_to_int_list, load_json_config, set_all_seeds, get_file_md5
from fitam.mapping.costmap_swath_library import check_swath_against_config
from fitam.planning.planner_general import snap_easl_state_to_planning_state, State
from fitam.mapping.local_observation import local_observe_occlusions, local_perfect_observations
from fitam.evaluation.eval_utils import setup_trial_assets, filter_job_numbers, change_logger_to_trial_dir, log_status_and_change_logger, save_results, plot_map_summary, get_model_md5
from fitam.evaluation.eval_data_structures import setup_trial_data_structures, setup_trial_directories, setup_worker_assets_and_config, compile_results, ObserveFunctionType
from fitam.evaluation.eval_observation import observe
from fitam.evaluation.spatial_label_observation import spatial_label_observe
from fitam.evaluation.eval_movement import move


def worker(
    job_numbers: list[int],
    save_root: PathLike,
    model_path: PathLike,
    eval_request_path: PathLike,
    radial_costmap_config_path: PathLike,
    training_config_path: PathLike,
    dataset_config_path: PathLike,
    logging_config_path: PathLike,
    swath_library_path: PathLike,
    eval_config_path: PathLike,
    num_active_bins: int = None,  # None use all bins
    # overwrite dumps if t/f
    dump_panos: bool = None,
    dump_network_results: bool = None,
    dump_local_costmaps: bool = None,
    dump_dataset: bool = None,
):

    # set seeds for general setup
    set_all_seeds(42)
    seed_everything(42,  workers=True)
    torch.set_num_threads(2)

    global_start_time = time.time()
    create_dir(save_root)
    uid = str(uuid.uuid4())
    logging_config = load_json_config(logging_config_path)
    logger = setup_logger(save_root / f'worker_{uid}.log', logging_config.console_logging_level,
                          logging_config.file_logging_level, logging_config.logging_format, capture_uncaught_exceptions=True)

    logger.debug(f"########Starting worker. job_numbers: {job_numbers} | save_root: {save_root} | eval_request_path: {eval_request_path}")

    eval_config = load_json_config(eval_config_path)
    # 0. filter the job numbers
    if eval_config.resume:
        job_numbers = filter_job_numbers(job_numbers, save_root)
        if len(job_numbers) == 0:
            logger.warning("No jobs to do. Exiting worker")
            return  # exit if no jobs to do
    logger.debug(f"Filtered job numbers: {job_numbers}")

    # 1. setup the assets for the worker
    worker_assets = setup_worker_assets_and_config(
        eval_request_path=eval_request_path,
        radial_map_config_path=radial_costmap_config_path,
        swath_library_path=swath_library_path,
        model_path=model_path,
        logging_config_path=logging_config_path,
        eval_config_path=eval_config_path,
        training_config_path=training_config_path,
        dataset_config_path=dataset_config_path,
    )
    # check swath validity
    if not check_swath_against_config(worker_assets.swath_library, worker_assets.radial_map_config):
        raise RuntimeError(f"Swath library {swath_library_path} is not compatible with radial map config {radial_costmap_config_path}")

    if dump_panos == True:
        worker_assets.eval_config.save_panoramas = dump_panos
    if dump_network_results == True:
        worker_assets.eval_config.save_network_results = dump_network_results
    if dump_local_costmaps == True:
        worker_assets.eval_config.save_local_costmaps = dump_local_costmaps
    if dump_dataset == True:
        worker_assets.eval_config.save_dataset = dump_dataset

    class_map = None
    class_index_to_name_map = None
    if not worker_assets.use_renderer and worker_assets.eval_config.save_dataset:
        logger.warn("Warning: Dataset export was requested but not using renderer. Disabling dataset export.")
        worker_assets.eval_config.save_dataset = False
    if worker_assets.eval_config.save_dataset:
        class_map, class_index_to_name_map = worker_assets.land_cover_complex_map.create_flattened_class_map()

    camera_height = None if worker_assets.dataset_config is None else worker_assets.dataset_config.camera_height

    num_failed = 0
    num_success = 0
    # 3. Loop through jobs
    for query_i in job_numbers:
        set_all_seeds(query_i)
        seed_everything(query_i,  workers=True)
        # 3.2 setup the job data structures and logging and directories
        trial_directories = setup_trial_directories(save_root, query_i, worker_assets.eval_config)
        change_logger_to_trial_dir(
            logger, trial_directories.query_results_dir, worker_assets.logging_config, query_i)
        cts, dts = setup_trial_data_structures(worker_assets.eval_request, query_i)
        logger.debug(f"####Starting job {query_i} with eval request path {eval_request_path} and save root {save_root}. LCM path is {worker_assets.eval_request.map_path}")
        env_vars = dict(os.environ)
        if 'LS_COLORS' in env_vars:
            del env_vars['LS_COLORS']

        logger.debug(f"os environment: {env_vars}")
        logger.debug(f"Device: {worker_assets.device}")
        # 3.1 load the assets for the job
        local_costmap, planner = setup_trial_assets(
            worker_assets.radial_map_config, worker_assets.master_costmap, worker_assets.swath_library, State(*cts.start_state), State(*cts.goal_state)
        )

        if not worker_assets.use_renderer and not worker_assets.disable_farfield:
            print("Warning: using GT labels")
            logger.warn("Warning: using GT labels")
        if worker_assets.model is not None:
            model_hash = get_file_md5(model_path)
            logger.debug(f"Model hash: {model_hash}")

        total_path_cost = 0
        # 3.3 Loop until planning exits
        current_state = State(*cts.start_state)
        replan_index = 0
        reached_goal = False
        logger.debug(
            "Finished setup before planning loop starts. Now entering inner loop")
        while True:
            # 3.3.1 get the current state
            logger.debug(
                f"Replan {replan_index}: Starting planning loop. Current state is {current_state}")
            # 3.3.2 update observations (either calculated or predicted)
            logger.debug("Starting to work on observations")
            if worker_assets.observe_function_type == ObserveFunctionType.FARFIELD \
                    or worker_assets.observe_function_type == ObserveFunctionType.LOCAL:
                observe(logger=logger, current_state=current_state, replan_index=replan_index,
                        scene=worker_assets.scene, trial_directories=trial_directories, belief=local_costmap,
                        master_costmap=worker_assets.master_costmap, model=worker_assets.model, swath_library=worker_assets.swath_library,
                        device=worker_assets.device, use_renderer=worker_assets.use_renderer,
                        radial_map_config=worker_assets.radial_map_config, eval_config=worker_assets.eval_config, train_config=worker_assets.training_config,
                        camera_height=camera_height, num_active_bins=num_active_bins, export_dataset=worker_assets.eval_config.save_dataset,
                        class_map=class_map, class_index_to_name_map=class_index_to_name_map)
                
            elif worker_assets.observe_function_type == ObserveFunctionType.SPATIAL_LABEL:
                spatial_label_observe(logger=logger, current_state=current_state, replan_index=replan_index,
                        scene=worker_assets.scene, trial_directories=trial_directories, belief=local_costmap,
                        master_costmap=worker_assets.master_costmap, model=worker_assets.model, swath_library=worker_assets.swath_library,
                        device=worker_assets.device, use_renderer=worker_assets.use_renderer,
                        radial_map_config=worker_assets.radial_map_config, eval_config=worker_assets.eval_config,
                        )

            # do local observations
            current_planning_state = snap_easl_state_to_planning_state(
                local_costmap, current_state)
            center_idx = (current_planning_state.y, current_planning_state.x)
            if worker_assets.radial_map_config.use_occluded_local_observations:
                local_observations = local_observe_occlusions(
                    robot_idx=center_idx,
                    gt_map=worker_assets.master_costmap,
                    gt_occlusion_map=worker_assets.occlusion_map,
                    local_observation_tracker=worker_assets.observed_states,
                    swath_lib=worker_assets.swath_library
                )
            else:
                local_observations = local_perfect_observations(
                    robot_idx=center_idx,
                    gt_map=worker_assets.master_costmap,
                    local_observation_tracker=worker_assets.observed_states,
                    swath_lib=worker_assets.swath_library
                )
            # add observations from within the sensor horizon to the local map
            local_costmap.fix_known_states(
                local_observations
            )

            logger.debug("Finished adding geometric observations to the costmap")

            logger.debug(f"Starting to update the dstar costmap on iter {replan_index}")
            if replan_index == 0:
                planner.Initialize()
            else:
                planner.update_costmap()
            logger.debug("Finished updating the dstar costmap")

            # 3.3.3 replan and walk along path
            done, reached_goal, current_state, cost_delta = move(logger=logger, planner=planner, current_state=current_state,
                                                                 goal_state=State(*cts.goal_state), replan_index=replan_index, query_i=query_i,
                                                                 eval_config=worker_assets.eval_config, observed_states=worker_assets.observed_states,
                                                                 local_costmap=local_costmap, dts=dts)

            if worker_assets.eval_config.save_local_costmaps:
                all_transition_indices = [0] + dts.state_transition_indexes
                path_history = dts.path_history[-1]
                state_history = dts.state_history[:max(1, all_transition_indices[replan_index])]
                var = None
                if hasattr(local_costmap.state, 'variance'):
                    var = local_costmap.state.variance
                fig = plot_map_summary(local_costmap,
                                       var,
                                       worker_assets.radial_map_config.farfield_config,
                                       10, worker_assets.master_costmap, worker_assets.land_cover_complex_map, state_history, path_history, start_state=cts.start_state, end_state=cts.goal_state)
                fig.savefig(trial_directories.base_local_costmap_path / f"{replan_index:07d}.png")
                # save_compressed_pickle(local_costmap, trial_directories.base_local_costmap_path / f"{replan_index:07d}.xz")
                plt.close(fig)
                logger.debug("Saved local costmap to disk")

            replan_index += 1
            total_path_cost += cost_delta
            if done:
                break
        # 3.4 Compile and save the results
        compile_results(logger=logger, omnipresent_planner=worker_assets.omnipresent_planner, start_state=State(*cts.start_state), goal_state=State(*cts.goal_state),
                        reached_goal=reached_goal, disable_farfield=worker_assets.disable_farfield, cts=cts, dts=dts, query_i=query_i,
                        replan_index=replan_index, total_path_cost=total_path_cost, model_path=model_path, complex_map_path=worker_assets.eval_request.map_path,
                        local_costmap=local_costmap)
        save_results(logger=logger, cts=cts, dts=dts,
                     query_results_dir=trial_directories.query_results_dir)

        if worker_assets.eval_config.save_dataset:
            # compile dataset csvs into one csv
            def compile_csvs(prefix):
                all_dataset_csvs = list(trial_directories.base_dataset_path.glob(f"{prefix}*.csv"))
                all_dataset_csvs = [pd.read_csv(csv) for csv in all_dataset_csvs]
                all_dataset_csvs = pd.concat(all_dataset_csvs)
                all_dataset_csvs.to_csv(trial_directories.base_dataset_path / f"{prefix}.csv", index=False)
            compile_csvs("dataset")
            compile_csvs("extra")

        if reached_goal:
            num_success += 1
        else:
            num_failed += 1
        log_status_and_change_logger(num_success=num_success, num_failed=num_failed, logger=logger, query_i=query_i, job_numbers=job_numbers, uid=uid,
                                     global_start_time=global_start_time, results_dir=save_root, logging_config=worker_assets.logging_config)
        # on to the next query!
    logger.debug("Worker finished!")
    return


if __name__ == "__main__":
    import argparse
    import sys

    # handle weird blender argument format
    argv = sys.argv
    full_command = " ".join(argv)
    if "--" not in argv:  # assuming things are python now, no blender launch
        argv = argv[1:]
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=False, help="Path to the input file to use. If provided other args are not required")
    parser.add_argument("--eval_request_path", type=Path, required=True, help="Path to the eval request to use.")
    parser.add_argument("--save_root", type=Path, required=True, help="Path to save results to.")
    parser.add_argument("--job_numbers", type=str, required=True, help="Comma separated list of plan numbers to run")
    #### Not required if input file is specified ####
    parser.add_argument("--swath_library_path", type=Path, required=False, help="Path of the swath library to use.")
    parser.add_argument("--model_path", type=Path, default=None, required=False, help="Path to the checkpoint to use for nn weights.")
    parser.add_argument("--radial_costmap_config_path", type=Path, required=False, help="Path to the radial costmap config to use.")
    parser.add_argument("--training_config_path", type=Path, default=None, help="Path to the training config to use.")
    parser.add_argument("--dataset_config_path", type=Path, default=None, help="Path to the dataset config to use.")
    parser.add_argument("--logging_config_path", type=Path, required=False, help="Path to the logging config to use.")
    parser.add_argument("--eval_config_path", type=Path, required=False, help="Path to the eval config to use.")
    parser.add_argument("--num_active_bins", type=int, default=None, help="Number of active bins to use")
    parser.add_argument("--dump_all_outputs", action="store_true", help="If true, will dump all outputs to disk (panoramas, network inputs, etc.)")
    parser.add_argument("-d", "--dataset", action="store_true", help="Dump dataset to disk, overwritten by dump_all_outputs")
    parser.add_argument("-n", "--network_results", action="store_true", help="Dump network results to disk, overwritten by dump_all_outputs")
    parser.add_argument("-l", "--local_costmap", action="store_true", help="Dump local_costmap to disk, overwritten by dump_all_outputs")
    parser.add_argument("-p", "--panoramas", action="store_true", help="Dump local_costmap to disk, overwritten by dump_all_outputs")

    args = parser.parse_args(argv)

    if args.input_file is not None:
        import json
        with open(args.input_file, "r") as f:
            input_dict = json.load(f)
        # overwrite args with input file
        args.dataset_config_path = CONFIGS_DIR / input_dict['dataset_config_path'] if input_dict['dataset_config_path'] is not None else None
        args.eval_config_path = CONFIGS_DIR / input_dict['evaluation_config_path']
        args.logging_config_path = CONFIGS_DIR / input_dict['logging_config_path']
        args.model_path = MODELS_DIR / input_dict['model_path'] if 'model_path' in input_dict else None
        args.num_active_bins = input_dict['num_active_bins']
        args.radial_costmap_config_path = CONFIGS_DIR / input_dict['radial_map_config_path']
        args.swath_library_path = SWATHS_DIR / input_dict['swath_library_path']
        args.training_config_path = CONFIGS_DIR / input_dict['training_config_path'] if input_dict['training_config_path'] is not None else None
    else:
        assert args.swath_library_path is not None, "swath_library_path is required"
        assert args.eval_request_path is not None, "eval_request_path is required"
        assert args.radial_costmap_config_path is not None, "radial_costmap_config_path is required"
        assert args.logging_config_path is not None, "logging_config_path is required"
        assert args.eval_config_path is not None, "eval_config_path is required"

    if args.dump_all_outputs:
        args.panoramas = True
        args.network_results = True
        args.local_costmap = True
        args.dataset = True

    job_numbers = decode_str_to_int_list(args.job_numbers)
    worker(
        job_numbers=job_numbers,
        save_root=args.save_root,
        model_path=args.model_path,
        eval_request_path=args.eval_request_path,
        radial_costmap_config_path=args.radial_costmap_config_path,
        training_config_path=args.training_config_path,
        dataset_config_path=args.dataset_config_path,
        logging_config_path=args.logging_config_path,
        eval_config_path=args.eval_config_path,
        swath_library_path=args.swath_library_path,
        num_active_bins=args.num_active_bins,
        dump_dataset=args.dataset,
        dump_local_costmaps=args.local_costmap,
        dump_network_results=args.network_results,
        dump_panos=args.panoramas,
    )

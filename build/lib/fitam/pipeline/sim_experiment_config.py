from __future__ import annotations
from fitam import (CONFIGS_DIR,
                   MAPS_DIR,
                   SAMPLED_LOCATIONS_DIR,
                   SWATHS_DIR,
                   DATASETS_DIR,
                   EVALUATION_REQUESTS_DIR,
                   EVALUATIONS_DIR,
                   MODELS_DIR,
                   IMAGES_DIR,
                   )
from fitam.pipeline.pipeline_elements import (
    Swath, EvaluationRequest, ImageRendering, Dataset, Training, Evaluation, TimedLocationSampling
)

pipeline_config = []

glob_all_radial_configs = list((CONFIGS_DIR / 'simulated_radial_configs').glob("**/*.json"))
all_radial_configs = []
# remove outdoor configs
for config_path in glob_all_radial_configs:
    rel_path = config_path.relative_to(CONFIGS_DIR)
    if 'outdoor' in rel_path.parts:
        continue
    if "spatial_label" in rel_path.parts:  # don't include the spatial baseline in the list
        continue
    all_radial_configs.append(config_path)

for config_path in all_radial_configs:
    rel_path = config_path.relative_to(CONFIGS_DIR / 'simulated_radial_configs')
    name = "_".join(rel_path.parts)[:-len(".json")]
    pipeline_config.append(
        Swath(
            name=f"swath_{name}",
            radial_map_config_path=config_path,
            save_path=SWATHS_DIR / config_path.parent.relative_to(CONFIGS_DIR) / f"{config_path.stem}.pkl",
        )
    )
train_balt_map_out_path = MAPS_DIR / 'final_experiment_train_balt'
training_county_name = 'balt_24005_lulc_2018'
# EVALUATION REQUESTS
# exclude as we use pre-generated ones

# if Path(FITAM_LAND_COVER_DIR, training_county_name, 'final_experiment_maps').exists():
#     balt_maps = Path(FITAM_LAND_COVER_DIR, training_county_name, 'final_experiment_maps')
#     balt_map_in_path = list(sorted([x for x in balt_maps.iterdir() if x.is_dir()]))[0]
#     balt_train_map_sampling = MapSampling(
#         name='balt_train',
#         env_config_path=CONFIGS_DIR / 'env_config.json',
#         map_in_paths=[balt_map_in_path],
#         map_out_paths=[train_balt_map_out_path],
#     )
#     pipeline_config.append(balt_train_map_sampling)

#     # add sampling for the all-county maps
#     for i, county_name in enumerate(Path(FITAM_LAND_COVER_DIR / 'all_county_maps').glob("*/")):
#         if training_county_name in str(county_name.name):
#             continue
#         map_in_path = county_name
#         map_out_path = MAPS_DIR / 'all_test_counties' / county_name.name
#         pipeline_config.append(MapSampling(
#             name=f'all_test_counties_{county_name.name}',
#             env_config_path=CONFIGS_DIR / 'env_config.json',
#             map_in_paths=[map_in_path],
#             map_out_paths=[map_out_path],
#         ))
# for map_out_path in Path(MAPS_DIR / 'all_test_counties').glob("*/"):
#     pipeline_config.append(EvaluationRequest(
#         name=f"eval_request_all_test_counties_{map_out_path.name}",
#         map_paths=[map_out_path],
#         save_paths=[EVALUATION_REQUESTS_DIR / 'all_test_counties' / f"{map_out_path.name}.json"],
#         evaluation_config_path=CONFIGS_DIR / 'evaluation_config.json',
#     ))
#     pipeline_config.append(EvaluationRequest(
#         name=f"eval_request_ablation_{map_out_path.name}",
#         map_paths=[map_out_path],
#         save_paths=[EVALUATION_REQUESTS_DIR / 'ablation' / f"{map_out_path.name}.json"],
#         evaluation_config_path=CONFIGS_DIR / 'evaluation_config_ablation.json',
#     ))

# Image RQs
max_time_request_seconds = 60 * 60 * 10  # 10 hours exploration max

# pipeline_config.append(
#     TimedLocationSampling(
#         name='training_map',
#         seed=0,
#         map_path=train_balt_map_out_path,
#         save_folder_path=Path('training_map'),  # saves in SAMPLED_LOCATIONS_DIR and MAPS_DIR
#         seconds_of_exploration=max_time_request_seconds,
#         obs_rad_m=25,
#         meters_between_keypoints=2,
#         max_iters_per_waypoint=50,
#     )
# )


# # splitting up image requests
# # python radial_learning/dataset_generation/split_image_request_by_time.py -i results/sampled_locations/training_map/image_request.json -m results/maps/final_experiment_train_balt -v 25 -t 10 30 60 120 360 -o split_up_image_requests
# # image rendering for balt-standard


pipeline_config.append(
    ImageRendering(
        name="image_rendering_balt_standard",
        compute_config_path=CONFIGS_DIR / 'compute_config.json',
        dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
        dependent_task=None,
        save_paths=[IMAGES_DIR / 'balt_standard'],
        location_sample_paths=[SAMPLED_LOCATIONS_DIR / 'training_map' / 'image_request.json'],
        save_root_path=IMAGES_DIR / 'balt_standard',
        complex_map_paths=[train_balt_map_out_path],
    )
)

# image rendering for balt-10m, etc (timed datasets)
times = [10, 30, 60, 120, 360]
for time in times:
    balt_timed_in_paths = [(SAMPLED_LOCATIONS_DIR / 'split_up_image_requests' / f"image_request_{time}m.json")]
    pipeline_config.append(
        ImageRendering(
            name=f"image_rendering_balt_{time}m",
            compute_config_path=CONFIGS_DIR / 'compute_config.json',
            dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
            dependent_task=None,
            save_paths=[IMAGES_DIR / f'balt_{time}m' / 'parts'],
            location_sample_paths=balt_timed_in_paths,
            save_root_path=IMAGES_DIR / f'balt_{time}m',
            complex_map_paths=[train_balt_map_out_path],
        )
    )

# Datasets and models!
# # balt-standard
pipeline_config.append(Dataset(
    name="balt_standard",
    save_root_dir=DATASETS_DIR / 'balt_standard',
    radial_map_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'radial_map_config.json',
    dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
    image_db_path=IMAGES_DIR / 'balt_standard' / 'points.csv',
))
pipeline_config.append(Training(
    name='balt_standard',
    dataset_csv_path=DATASETS_DIR / 'balt_standard' / 'balanced_semantic_dataset.csv',
    save_path=MODELS_DIR / 'balt_standard',
    train_config_path=CONFIGS_DIR / 'train_config_dino_classification.json',
    radial_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'radial_map_config.json',
    ensemble_members=15,
))

pipeline_config.append(Dataset(
    name="balt_long_range",
    save_root_dir=DATASETS_DIR / 'balt_long_range',
    radial_map_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'long_range_radial_map_config.json',
    dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
    image_db_path=IMAGES_DIR / 'balt_standard' / 'points.csv',
))
pipeline_config.append(Training(
    name='balt_long_range',
    dataset_csv_path=DATASETS_DIR / 'balt_long_range' / 'balanced_semantic_dataset.csv',
    save_path=MODELS_DIR / 'balt_long_range',
    train_config_path=CONFIGS_DIR / 'train_config_dino_classification.json',
    radial_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'long_range_radial_map_config.json',
))

# # balt-10m, etc (timed datasets)
for time in times:
    pipeline_config.append(Dataset(
        name=f"balt_{time}m",
        save_root_dir=DATASETS_DIR / f'balt_{time}m',
        radial_map_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'radial_map_config.json',
        dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
        image_db_path=IMAGES_DIR / f'balt_{time}m' / 'points.csv',
    ))
    pipeline_config.append(Training(
        name=f'balt_{time}m',
        dataset_csv_path=DATASETS_DIR / f'balt_{time}m' / 'balanced_semantic_dataset.csv',
        save_path=MODELS_DIR / f'balt_{time}m',
        train_config_path=CONFIGS_DIR / 'train_config_dino_classification.json',
        radial_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'radial_map_config.json',
    ))

for radial_config_path in all_radial_configs:
    #     # skip the non-varying ones
    if "vary_heading" not in radial_config_path.parts and \
        "vary_range" not in radial_config_path.parts and \
        "vary_range_heading" not in radial_config_path.parts and \
            "vary_classes" not in radial_config_path.parts:
        continue
    dataset_name = f"balt_{radial_config_path.stem.replace('_radial_map_config', '')}"
    pipeline_config.append(Dataset(
        name=dataset_name,
        save_root_dir=DATASETS_DIR / dataset_name,
        radial_map_config_path=CONFIGS_DIR / radial_config_path,
        dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
        image_db_path=IMAGES_DIR / 'balt_standard' / 'points.csv',
    ))
    pipeline_config.append(Training(
        name=dataset_name,
        dataset_csv_path=DATASETS_DIR / dataset_name / 'balanced_semantic_dataset.csv',
        save_path=MODELS_DIR / dataset_name,
        train_config_path=CONFIGS_DIR / 'train_config_dino_classification.json',
        radial_config_path=CONFIGS_DIR / radial_config_path,
    ))

noise_levels = [0.0001, 0.001, 0.005, 0.01, 0.1]
for noise_level in noise_levels:
    dataset_name = f"balt_noisy_odom_noise_{str(noise_level).replace('.', '_')}"
    pipeline_config.append(Dataset(
        name=dataset_name,
        save_root_dir=DATASETS_DIR / dataset_name,
        radial_map_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'radial_map_config.json',
        dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
        image_db_path=IMAGES_DIR / 'balt_standard' / f"noise_level_{str(noise_level).replace('.', '_')}.csv"
    ))
    pipeline_config.append(Training(
        name=dataset_name,
        dataset_csv_path=DATASETS_DIR / dataset_name / 'balanced_semantic_dataset.csv',
        save_path=MODELS_DIR / dataset_name,
        train_config_path=CONFIGS_DIR / 'train_config_dino_classification.json',
        radial_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'radial_map_config.json',
        ensemble_members=15,
    ))


all_test_evaluation_request_paths = sorted(list((EVALUATION_REQUESTS_DIR / "all_test_counties").glob("*.json")))
all_ablation_evaluation_request_paths = sorted(list((EVALUATION_REQUESTS_DIR / "ablation").glob("*.json")))
all_shakeout_evaluation_request_paths = sorted(list((EVALUATION_REQUESTS_DIR / "shake_out").glob("*.json")))

evaluation_configurations = {
    'baseline': dict(
        model_path=None,
        dataset_config_path=None,
        radial_map_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'baseline_radial_map_config.json',
        logging_config_path=CONFIGS_DIR / 'logging_config.json',
        training_config_path=None,
        compute_config_path=CONFIGS_DIR / 'compute_config.json',
        swath_library_path=SWATHS_DIR / 'simulated_radial_configs' / 'baseline_radial_map_config.pkl',
        dependent_task=None,
    ),
    'baseline_spatial_labels': dict(
        model_path=MODELS_DIR / 'spatial_label_propagation' / 'best_model.pt',
        dataset_config_path=None,
        radial_map_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'spatial_label_radial_map_config.json',
        logging_config_path=CONFIGS_DIR / 'logging_config.json',
        training_config_path=None,
        compute_config_path=CONFIGS_DIR / 'compute_config.json',
        swath_library_path=SWATHS_DIR / 'simulated_radial_configs' / 'spatial_label_radial_map_config.pkl',
        dependent_task=None,
    ),
    'perfect_vision': dict(
        model_path=None,
        dataset_config_path=None,
        radial_map_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'perfect_vision_radial_map_config.json',
        logging_config_path=CONFIGS_DIR / 'logging_config.json',
        training_config_path=None,
        compute_config_path=CONFIGS_DIR / 'compute_config.json',
        swath_library_path=SWATHS_DIR / 'simulated_radial_configs' / 'perfect_vision_radial_map_config.pkl',
        dependent_task=None,
    ),
    'core_farfield': dict(
        model_path=MODELS_DIR / 'balt_standard' / 'balt_standard.ckpt',
        dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
        radial_map_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'radial_map_config.json',
        logging_config_path=CONFIGS_DIR / 'logging_config.json',
        training_config_path=CONFIGS_DIR / 'train_config_dino_classification.json',
        compute_config_path=CONFIGS_DIR / 'compute_config.json',
        swath_library_path=SWATHS_DIR / 'simulated_radial_configs' / 'radial_map_config.pkl',
        dependent_task=None,
    ),
    'long_range': dict(
        model_path=MODELS_DIR / 'balt_long_range' / 'balt_long_range.ckpt',
        dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
        radial_map_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'long_range_radial_map_config.json',
        logging_config_path=CONFIGS_DIR / 'logging_config.json',
        training_config_path=CONFIGS_DIR / 'train_config_dino_classification.json',
        compute_config_path=CONFIGS_DIR / 'compute_config.json',
        swath_library_path=SWATHS_DIR / 'simulated_radial_configs' / 'long_range_radial_map_config.pkl',
        dependent_task=None,
    )
}

evaluation_configurations.update({
    'vary_heading': {},
    'vary_range': {},
    'vary_range_heading': {},
    'vary_classes': {},
    'vary_time': {},
})
for radial_config_path in all_radial_configs:
    # skip the non-varying ones
    name = None
    if "vary_heading" in radial_config_path.parts:
        name = "vary_heading"
    elif "vary_range" in radial_config_path.parts:
        name = "vary_range"
    elif "vary_range_heading" in radial_config_path.parts:
        name = "vary_range_heading"
    elif "vary_classes" in radial_config_path.parts:
        name = "vary_classes"
    else:
        continue
    swath_path = SWATHS_DIR / radial_config_path.parent.relative_to(CONFIGS_DIR) / f"{radial_config_path.stem}.pkl"
    dataset_name = f"balt_{radial_config_path.stem.replace('_radial_map_config', '')}"
    evaluation_configurations[name][radial_config_path.stem] = dict(
        dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
        radial_map_config_path=CONFIGS_DIR / radial_config_path,
        logging_config_path=CONFIGS_DIR / 'logging_config.json',
        training_config_path=CONFIGS_DIR / 'train_config_dino_classification.json',
        compute_config_path=CONFIGS_DIR / 'compute_config.json',
        model_path=MODELS_DIR / f'{dataset_name}' / f'{dataset_name}.ckpt',
        swath_library_path=swath_path,
        dependent_task=None,
    )
for time in times:
    evaluation_configurations['vary_time'][f"{time}m"] = dict(
        dataset_config_path=CONFIGS_DIR / 'classification_dataset_config.json',
        radial_map_config_path=CONFIGS_DIR / 'simulated_radial_configs' / 'radial_map_config.json',
        logging_config_path=CONFIGS_DIR / 'logging_config.json',
        training_config_path=CONFIGS_DIR / 'train_config_dino_classification.json',
        compute_config_path=CONFIGS_DIR / 'compute_config.json',
        model_path=MODELS_DIR / f'balt_{time}m' / f'balt_{time}m.ckpt',
        swath_library_path=SWATHS_DIR / 'simulated_radial_configs' / 'radial_map_config.pkl',
        dependent_task=None,
    )

eval_types = {
    'shakeout': [all_shakeout_evaluation_request_paths, 'evaluation_config.json'],
    'ablation': [all_ablation_evaluation_request_paths, 'evaluation_config_ablation.json'],
    'all_test': [all_test_evaluation_request_paths, 'evaluation_config.json'],
}


def make_evaluation(name, all_configs: dict, eval_type: str, two_levels: bool = False, custom_name=None):
    if two_levels:
        assert custom_name is None, "custom_name is only for single level evaluations"
        outputs = []
        for key, config in all_configs[name].items():
            options = config.copy()
            options['name'] = f"{eval_type}_{name}_{key}"
            options['save_root_path'] = EVALUATIONS_DIR / eval_type / name / key
            options['evaluation_config_path'] = CONFIGS_DIR / eval_types[eval_type][1]
            options['eval_request_paths'] = eval_types[eval_type][0]
            options['save_paths'] = [x.stem for x in eval_types[eval_type][0]]
            outputs.append(options)
        return outputs
    else:
        options = all_configs[name].copy()
        if custom_name is not None:
            name = custom_name
        options['name'] = f"{eval_type}_{name}"
        options['save_root_path'] = EVALUATIONS_DIR / eval_type / name
        options['evaluation_config_path'] = CONFIGS_DIR / eval_types[eval_type][1]
        options['eval_request_paths'] = eval_types[eval_type][0]
        options['save_paths'] = [x.stem for x in eval_types[eval_type][0]]
        return options


# shakeout evaluations
pipeline_config.append(Evaluation(**make_evaluation('baseline', evaluation_configurations, 'shakeout')))
pipeline_config.append(Evaluation(**make_evaluation('perfect_vision', evaluation_configurations, 'shakeout')))
pipeline_config.append(Evaluation(**make_evaluation('core_farfield', evaluation_configurations, 'shakeout')))
gt_farfield = make_evaluation('core_farfield', evaluation_configurations, 'shakeout', custom_name='gt_farfield')
gt_farfield['model_path'] = None
pipeline_config.append(Evaluation(**gt_farfield))

# ablation evaluations
pipeline_config.append(Evaluation(**make_evaluation('baseline', evaluation_configurations, 'ablation')))
pipeline_config.append(Evaluation(**make_evaluation('perfect_vision', evaluation_configurations, 'ablation')))
pipeline_config.append(Evaluation(**make_evaluation('core_farfield', evaluation_configurations, 'ablation')))
gt_farfield = make_evaluation('core_farfield', evaluation_configurations, 'ablation', custom_name='gt_farfield')
gt_farfield['model_path'] = None
pipeline_config.append(Evaluation(**gt_farfield))
pipeline_config.append(Evaluation(**make_evaluation('core_farfield', evaluation_configurations, 'ablation')))
pipeline_config.append(Evaluation(**make_evaluation('baseline_spatial_labels', evaluation_configurations, 'ablation')))


# vary models
# options = make_evaluation('core_farfield', evaluation_configurations, 'ablation', custom_name='resnet_pretrained')
# options['model_path'] = MODELS_DIR / 'balt_pretrained_resnet' / 'balt_pretrained_resnet.ckpt'
# pipeline_config.append(Evaluation(**options))
# options = make_evaluation('core_farfield', evaluation_configurations, 'ablation', custom_name='dino')
# options['model_path'] = MODELS_DIR / 'balt_dino' / 'balt_dino.ckpt'
# pipeline_config.append(Evaluation(**options))


# vary time
pipeline_config.extend([Evaluation(**x) for x in make_evaluation('vary_time', evaluation_configurations, 'ablation', two_levels=True)])
# vary heading
pipeline_config.extend([Evaluation(**x) for x in make_evaluation('vary_heading', evaluation_configurations, 'ablation', two_levels=True)])
# vary range
pipeline_config.extend([Evaluation(**x) for x in make_evaluation('vary_range', evaluation_configurations, 'ablation', two_levels=True)])
# vary range heading
pipeline_config.extend([Evaluation(**x) for x in make_evaluation('vary_range_heading', evaluation_configurations, 'ablation', two_levels=True)])
# vary classes
pipeline_config.extend([Evaluation(**x) for x in make_evaluation('vary_classes', evaluation_configurations, 'ablation', two_levels=True)])
# vary max distance
for num_bins in range(4, 13):
    name = f"vary_max_distance_{num_bins}"
    options = make_evaluation('long_range', evaluation_configurations, 'ablation', custom_name=name)
    options['num_active_bins'] = num_bins
    pipeline_config.append(Evaluation(**options))
# vary noise in odometry
for noise_level in noise_levels:
    model_name = f"balt_noisy_odom_noise_{str(noise_level).replace('.', '_')}"
    model_path = MODELS_DIR / model_name / f"{model_name}.ckpt"
    options = make_evaluation("core_farfield", evaluation_configurations, 'ablation', custom_name=model_name)
    options['model_path'] = model_path
    pipeline_config.append(Evaluation(**options))

# vary map fusion
options = make_evaluation('core_farfield', evaluation_configurations, 'ablation', custom_name='no_uncertainty')
options['radial_map_config_path'] = CONFIGS_DIR / 'simulated_radial_configs' / 'vary_map_fusion' / 'no_uncertainty_radial_map_config.json'
options['swath_library_path'] = SWATHS_DIR / 'simulated_radial_configs' / 'vary_map_fusion' / 'no_uncertainty_radial_map_config.pkl'
pipeline_config.append(Evaluation(**options))

options = make_evaluation('core_farfield', evaluation_configurations, 'ablation', custom_name='no_uncertainty_avg')
options['radial_map_config_path'] = CONFIGS_DIR / 'simulated_radial_configs' / 'vary_map_fusion' / 'avg_no_uncertainty_radial_map_config.json'
options['swath_library_path'] = SWATHS_DIR / 'simulated_radial_configs' / 'vary_map_fusion' / 'avg_no_uncertainty_radial_map_config.pkl'
pipeline_config.append(Evaluation(**options))

threshold = make_evaluation('core_farfield', evaluation_configurations, 'ablation', custom_name='core_farfield_with_confidence_threshold')
threshold['radial_map_config_path'] = CONFIGS_DIR / 'simulated_radial_configs' / 'vary_map_fusion' / 'confidence_threshold_radial_map_config.json'
threshold['swath_library_path'] = SWATHS_DIR / 'simulated_radial_configs' / 'vary_map_fusion' / 'confidence_threshold_radial_map_config.pkl'
pipeline_config.append(Evaluation(**threshold))

threshold = make_evaluation('core_farfield', evaluation_configurations, 'ablation', custom_name='core_farfield_with_confidence_threshold_04')
threshold['radial_map_config_path'] = CONFIGS_DIR / 'simulated_radial_configs' / 'vary_map_fusion' / 'confidence_threshold_radial_map_config_04.json'
threshold['swath_library_path'] = SWATHS_DIR / 'simulated_radial_configs' / 'vary_map_fusion' / 'confidence_threshold_radial_map_config_04.pkl'
pipeline_config.append(Evaluation(**threshold))

threshold = make_evaluation('core_farfield', evaluation_configurations, 'ablation', custom_name='core_farfield_with_confidence_threshold_06')
threshold['radial_map_config_path'] = CONFIGS_DIR / 'simulated_radial_configs' / 'vary_map_fusion' / 'confidence_threshold_radial_map_config_06.json'
threshold['swath_library_path'] = SWATHS_DIR / 'simulated_radial_configs' / 'vary_map_fusion' / 'confidence_threshold_radial_map_config_06.pkl'
pipeline_config.append(Evaluation(**threshold))

# full evaluations
pipeline_config.append(Evaluation(**make_evaluation('baseline', evaluation_configurations, 'all_test')))
pipeline_config.append(Evaluation(**make_evaluation('perfect_vision', evaluation_configurations, 'all_test')))
pipeline_config.append(Evaluation(**make_evaluation('core_farfield', evaluation_configurations, 'all_test')))
pipeline_config.append(Evaluation(**make_evaluation('baseline_spatial_labels', evaluation_configurations, 'all_test')))
gt_farfield = make_evaluation('core_farfield', evaluation_configurations, 'all_test', custom_name='gt_farfield')
gt_farfield['model_path'] = None
pipeline_config.append(Evaluation(**gt_farfield))

import os, pathlib

FITAM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

SWATHS_DIR = pathlib.Path(FITAM_ROOT_DIR, 'results', 'swaths')
CONFIGS_DIR = pathlib.Path(FITAM_ROOT_DIR, 'results', 'configs')
MAPS_DIR = pathlib.Path(FITAM_ROOT_DIR, 'results', 'maps')
SAMPLED_LOCATIONS_DIR = pathlib.Path(FITAM_ROOT_DIR, 'results', 'sampled_locations')
IMAGES_DIR = pathlib.Path(FITAM_ROOT_DIR, 'results', 'images')
DATASETS_DIR = pathlib.Path(FITAM_ROOT_DIR, 'results', 'datasets')
MODELS_DIR = pathlib.Path(FITAM_ROOT_DIR, 'results', 'models')
EVALUATION_REQUESTS_DIR = pathlib.Path(FITAM_ROOT_DIR, 'results', 'evaluation_requests')
EVALUATIONS_DIR = pathlib.Path(FITAM_ROOT_DIR, 'results', 'evaluations')

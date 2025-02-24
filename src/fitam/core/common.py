
from __future__ import annotations
import jsonpickle
from pathlib import Path
import random
import cv2
import pandas as pd
import multiprocessing
import pickle
from functools import partial
import numpy as np
import torch
import os
import lzma
import shutil
import logging
import sys
import matplotlib.pyplot as plt
from contextlib import contextmanager
from scipy.spatial.transform import Rotation as R
import hashlib
from fitam import FITAM_ROOT_DIR


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def quat_to_rpy(quat):
    r = R.from_quat(quat)
    rpy = r.as_euler('xyz', degrees=False)
    return rpy


def wrap_angle_pi(angle):
    """
    Wrap an angle between -pi and pi
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
# @njit


def wrap_angle_2pi(angle: float) -> float:
    """
    Wrap an angle between 0 and 2pi
    """
    return angle % (2 * np.pi)


def min_angle(angle1, angle2):
    """
    Return the minimum angle between two angles
    """
    angle1 = wrap_angle_2pi(angle1)
    angle2 = wrap_angle_2pi(angle2)
    opt1 = wrap_angle_2pi(angle1 - angle2)
    opt2 = wrap_angle_2pi(angle2 - angle1)
    return min(opt1, opt2)


def get_device():
    device = None
    logging.info("Setting device")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device {device}")
    return device


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# from https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def chw_to_hwc_img(tensor):
    if len(tensor.shape) == 4:
        # remove the batch dimension
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        else:
            raise RuntimeError(f"Tensor with batch dimension. Shape: {tensor.shape}")
    if type(tensor) == torch.Tensor:
        tensor = tensor.detach().cpu().numpy()
    return tensor.transpose((1, 2, 0))


def hwc_to_chw_img_tensor(img):
    assert len(img.shape) == 3 and img.shape[2] == 3, f"Image shape invalid for hwc to chw: {img.shape}"
    return img.transpose((2, 0, 1))


def load_image(img_path):
    img = cv2.imread(str(img_path))
    img = cv2_img_to_float_ndarr(img)
    return img


def write_image(img_path, img_np):
    assert isinstance(img_np, np.ndarray), f"Called write image with non numpy type: {type(img_np)}"
    img = float_to_cv2_img(img_np)
    cv2.imwrite(str(img_path), img)


def float_to_cv2_img(float_img):
    assert float_img.max() <= 1.0, f"Expected max of 1.0, got {float_img.max()}"
    assert float_img.min() >= 0.0, f"Expected min of 0.0, got {float_img.min()}"
    assert len(float_img.shape) == 3, f"Need 3 channel image, got shape {float_img.shape}"
    assert float_img.shape[2] == 3, f"Need rgb channel in 3rd place, got shape {float_img.shape}"
    img = (float_img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def float_tensor_to_cv2_img(float_tensor):
    if type(float_tensor) == torch.Tensor:
        float_tensor = float_tensor.detach().cpu().numpy()
    assert float_tensor.shape[0] == 3, f"Expected 3 channels in 0 index, got {float_tensor.shape[0]}"
    img = float_tensor.transpose((1, 2, 0))
    return float_to_cv2_img(img)


def cv2_img_to_float_ndarr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    return img

# @njit


def generate_range_and_bearing_tensors(center_idx, map_size_pix: int, map_resolution: float) -> tuple[torch.TensorType, torch.TensorType]:
    # Assumes that the map is square, and the origin is at the bottom left corner
    # map_resolution is in meters/pixel
    idx_values = torch.arange(0, map_size_pix, dtype=torch.float)  # assume h=w
    # print(center_idx[1], idx_values)
    # print(center_idx[1] - idx_values)
    yaw_matrix = torch.arctan2(
        (-idx_values + center_idx[0]).reshape(-1, 1), (idx_values - center_idx[1]))
    # return 0
    # [0, 2pi], there is a break between 0,2pi
    yaw_matrix = wrap_angle_2pi(yaw_matrix)

    radius_matrix = torch.sqrt(torch.pow(
        idx_values - center_idx[0], 2).reshape(-1, 1) + torch.pow(idx_values - center_idx[1], 2))
    radius_matrix = radius_matrix * map_resolution  # converts to meters
    return radius_matrix, yaw_matrix


def mask_yaw_tensor(yaw_tensor, yaw_range: tuple):
    assert yaw_range[0] >= 0 and yaw_range[0] <= 2*np.pi
    assert yaw_range[1] >= 0 and yaw_range[1] <= 2*np.pi
    if yaw_range[0] < yaw_range[1]:  # if we don't wrap around, ex [pi/4, 3pi/4]
        yaw_mask = (yaw_tensor >= yaw_range[0]) & (yaw_tensor <= yaw_range[1])
    else:  # if we do wrap around, ex [ 7pi/4, pi/4 ]
        # print('wrapping around')
        yaw_mask = (yaw_tensor >= yaw_range[0]) | (yaw_tensor <= yaw_range[1])
    return yaw_mask


def mask_radius_tensor(radius_tensor, radius_range: tuple):
    rad_mask = (radius_tensor >= radius_range[0]) & (
        radius_tensor <= radius_range[1])
    return rad_mask


def create_dir(dir_path, overwrite=False):
    if os.path.exists(dir_path):
        if overwrite:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
    else:
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            pass  # another thread might have created the dir


def setup_logger(logging_filename, console_logging_level, file_logging_level, logging_format, capture_uncaught_exceptions: bool = False) -> logging.Logger:
    # set up logging
    import sys

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(
            exc_type, exc_value, exc_traceback))

    if capture_uncaught_exceptions:
        sys.excepthook = handle_exception
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(min([console_logging_level,
                            file_logging_level]))
        if not logging_filename.exists():
            logging_filename.parent.mkdir(parents=True, exist_ok=True)
            logging_filename.touch()
        logger.debug(f"Saving results to {logging_filename}")
        # create file streamer
        fh = logging.FileHandler(logging_filename)
        fh.setLevel(file_logging_level)

        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(console_logging_level)

        # formatter
        formatter = logging.Formatter(logging_format)
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def update_fileHandler(logger, logging_filename, file_logging_level, logging_format):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
    logger.debug(f"Changing logging results to {logging_filename}")
    # create file streamer
    fh = logging.FileHandler(logging_filename)
    fh.setLevel(file_logging_level)
    # formatter
    formatter = logging.Formatter(logging_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return


def reset_path_to_system(full_path, system_base_path):
    _, maybe_code_dir = os.path.split(FITAM_ROOT_DIR)
    code_directory = (maybe_code_dir if len(maybe_code_dir) > 0
                      else os.path.basename(FITAM_ROOT_DIR[:-1]))

    full_path_parts = full_path.split('/')

    try:
        code_dir_idx = full_path_parts.index(code_directory)
        tail_path = '/'.join(full_path_parts[code_dir_idx+1:])
        return os.path.join(system_base_path, tail_path)
    except Exception as exc:
        raise ValueError(
            f'Could not find {code_directory} in {full_path}') from exc


def encode_int_list_to_str(int_list):
    return ','.join([str(x) for x in int_list])


def decode_str_to_int_list(str):
    return [int(x) for x in str.split(',')]


def load_json_config(path: os.PathLike) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        config = jsonpickle.loads(f.read())
    return config


def load_pickle_config(path: os.PathLike) -> dict:
    with open(path, 'rb') as f:
        config = pickle.load(f)
    return config


def save_compressed_pickle(object, path: os.PathLike, overwrite=False):
    if type(path) == str:
        path = Path(path)
    if path.exists() and not overwrite:
        raise ValueError(f'{path} already exists')
    with lzma.open(path, 'wb') as f:
        pickle.dump(object, f)


def load_compressed_pickle(path):
    with lzma.open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_json_config(config, path: os.PathLike, overwrite=False) -> None:
    # NOTE: will not dump attributes that don't have a type annotation? (e.g. thing: int = 5)
    if type(path) == str:
        path = Path(path)
    if path.exists() and not overwrite:
        raise ValueError(f'{path} already exists')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(jsonpickle.dumps(config, indent=4))


def list_to_str(list, precision=3):
    return ','.join([str(round(x, precision)) for x in list])


def add_label_inf_to_image(image, label, inf=None, index=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if image.max() > 1.1:
        raise RuntimeWarning(f'Image max is {image.max()}')
    image[image > 1] = 1
    ax.imshow(image)

    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    if isinstance(label, np.ndarray):
        label = label.tolist()
    if inf is not None:
        if isinstance(inf, torch.Tensor):
            inf = inf.detach().cpu().numpy()
        if isinstance(inf, np.ndarray):
            inf = inf.tolist()

    if inf is None:
        title = f'Label: {list_to_str(label)}'
        if index is not None:
            title += f'\tIndex: {index}'
    else:
        title = "(Label, Inf)"
        if index is not None:
            title += f'Index: {index}\n'
        else:
            title += '\n'
        for i in range(len(label)):
            title += f'({round(label[i], 3)}, {round(inf[i], 3)})  '

    ax.set_title(title, fontweight='bold')
    ax.set_axis_off()

    fig.tight_layout()
    return fig


def angle_between_lr_yaw(left_yaw, right_yaw):
    left_yaw_wrapped = wrap_angle_2pi(left_yaw)
    right_yaw_wrapped = wrap_angle_2pi(right_yaw)
    # if some multiple of 2pi apart, return the difference
    if abs(left_yaw_wrapped - right_yaw_wrapped) < 1e-6 and abs(left_yaw - right_yaw) > np.pi:
        return abs(left_yaw - right_yaw)
    if left_yaw_wrapped < right_yaw_wrapped:
        return 2 * np.pi - right_yaw_wrapped + left_yaw_wrapped
    else:
        return left_yaw_wrapped - right_yaw_wrapped


def rot_mtx_to_yaw(rotation_matrix: np.ndarray) -> float:
    """Converts a 3x3 rotation matrix to a yaw angle"""
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    # yaw = np.arccos(np.dot(rotation_matrix[:,0], [0,0,1]) / np.linalg.norm(rotation_matrix[:,0]))
    # if rotation_matrix[0,0] < 0:
    #     yaw = -yaw
    return yaw


def get_observation_bounding_box(left_yaw, right_yaw, location, max_radial_boundary: float, yaw_shift: float):

    left_yaw += yaw_shift
    right_yaw += yaw_shift
    left_yaw = wrap_angle_2pi(left_yaw)
    right_yaw = wrap_angle_2pi(right_yaw)

    if left_yaw < right_yaw:  # if the range wraps around 0
        sample_points = np.linspace(right_yaw, 2*np.pi + left_yaw, 100)
    else:
        sample_points = np.linspace(right_yaw, left_yaw, 100)
    x_values = location[0] + \
        np.cos(sample_points) * (max_radial_boundary)
    y_values = location[1] + \
        np.sin(sample_points) * (max_radial_boundary)
    min_x = np.min(x_values)
    max_x = np.max(x_values)
    min_y = np.min(y_values)
    max_y = np.max(y_values)
    return min_x, max_x, min_y, max_y


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.

    Parameters:
    - d: The input dictionary
    - parent_key: Used for recursion to keep track of parent keys
    - sep: Separator used between keys

    Returns:
    - A flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_and_execute_chunk(chunk_indices, large_dataframe_path, row_function, **kwargs):
    df = pd.read_csv(large_dataframe_path)
    chunk = df.iloc[chunk_indices[0]:chunk_indices[1]]
    # chunk.reset_index(inplace=True, drop=True)
    result = row_function(chunk, **kwargs)
    return result


def process_df_in_parallel(large_dataframe_path: Path, row_function, num_processes=multiprocessing.cpu_count()//2, **kwargs) -> None:
    """
    Process a large dataframe in parallel
    """
    # split the dataframe into chunks
    df = pd.read_csv(large_dataframe_path)
    df_size = len(df)
    chunk_size = int(np.ceil(df_size / num_processes))
    chunk_indexes = [(i, min(i+chunk_size, df_size)) for i in range(0, df_size, chunk_size)]
    # create a pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        # process each chunk
        results = pool.map(partial(load_and_execute_chunk, large_dataframe_path=large_dataframe_path, row_function=row_function, **kwargs), chunk_indexes)
    return results


def find_subdirectories_with_name(root_dir, target_name):
    root_path = Path(root_dir)
    subdirectories = list(root_path.rglob(target_name))

    return [str(subdir.relative_to(root_dir)) for subdir in subdirectories if subdir.is_dir()]


def get_file_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def numpy_log_softmax(x): return x - np.logaddexp.reduce(x, axis=-1, keepdims=True)


def numpy_softmax(x): return np.exp(numpy_log_softmax(x))


def floor_to_array(x: np.ndarray, floor_values: np.ndarray) -> np.ndarray:
    """Floor each element in x to closest smaller or equal value in floor_values.

    Args:
        x: Input array to be floored
        floor_values: Array of values to floor to (must be sorted ascending)

    Returns:
        Array with same shape as x, with values floored to closest values in floor_values
    """
    # Ensure floor_values is sorted
    if not np.all(np.diff(floor_values) >= 0):
        raise ValueError("floor_values must be sorted in ascending order")

    # Reshape for broadcasting
    x_flat = x.ravel()[:, None]

    # Find valid floor values (largest value in floor_values <= x)
    valid_floors = x_flat >= floor_values

    # Get indices of closest floor values
    floor_indices = (valid_floors).sum(axis=1) - 1

    # Handle case where value is smaller than all floor values
    floor_indices = np.maximum(floor_indices, 0)

    # Get floored values and reshape to original shape
    result = floor_values[floor_indices].reshape(x.shape)

    return result

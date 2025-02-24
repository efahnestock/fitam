import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from fitam.core.common import wrap_angle_2pi, min_angle, generate_range_and_bearing_tensors, chw_to_hwc_img, angle_between_lr_yaw


def check_range_within_range(outer_range: tuple, inner_range: tuple) -> bool:
    """
    Check that outer range (left, right) contains inner range (left, right)
    Returns true if valid 
    """
    assert max(outer_range) < 2*np.pi and min(outer_range) >= 0, "outer range must be in [0, 2pi]"
    assert max(inner_range) < 2*np.pi and min(inner_range) >= 0, "inner range must be in [0, 2pi]"
    # if outer range is full 360, then inner range is always within outer range
    if abs(min_angle(outer_range[0], outer_range[1])) < 1e-3:
        return True
    # check if the ranges move in opposing directions.. Sorta like <>
    # this occurs when the inner right is closer to the outer left than the inner left is to the outer left
    inner_left_to_outer_left = min_angle(inner_range[0], outer_range[0])
    inner_right_to_outer_left = min_angle(inner_range[1], outer_range[0])
    if inner_left_to_outer_left > inner_right_to_outer_left:
        return False

    def g_fp(lhs, rhs):
        return lhs - 1e-12 > rhs
    if outer_range[0] < outer_range[1]:  # if we wrap around 0
        # print("we wrap around zero")
        # check that left inner is inside outer
        if g_fp(inner_range[0], outer_range[0]) and g_fp(outer_range[1], inner_range[0]):
            return False
        # check that right inner is inside outer
        if g_fp(inner_range[1], outer_range[0]) and g_fp(outer_range[1], inner_range[1]):
            return False
    else:  # we don't wrap around 0
        # print("we don't wrap around 0")
        # print(inner_range[0], outer_range[0], inner_range[0] > outer_range[0])
        # print(inner_range[1], outer_range[1], inner_range[1] < outer_range[1])
        if g_fp(inner_range[0], outer_range[0]) or g_fp(outer_range[1], inner_range[1]):
            return False

    return True


def crop_image_for_sector(image, image_yaws: tuple, cropped_yaws: tuple, img_width_yaw: float):
    """
    Given an image, get the portion of the image that corresponds to the given cropped yaw range
    cropped_yaws and image_yaws are a tuple of (left_yaw, right_yaw), both will be wrapped within 0->2pi

    NOTE: Calculates a pixel width using img_width_yaw, so make sure to keep this constant for a dataset
          to avoid rounding errors when calculating the width

    This is remapped to [0, 2pi] counterclockwise for the yaw range 
    image: torch.tensor [3, h, w]

    Returns:
    image: torch.tensor [3, h, w_cropped]
    """
    assert image.shape[0] == 3, "image must be 3, h, w"
    image_width_pixels = image.shape[-1]
    # print(f"image_width_pixels: {image_width_pixels}", "image yaws", image_yaws, "cropped yaws", cropped_yaws)
    norm_image_yaws = [wrap_angle_2pi(image_yaws[0]), wrap_angle_2pi(image_yaws[1])]  # [0, 2pi]
    norm_cropped_yaws = [wrap_angle_2pi(cropped_yaws[0]), wrap_angle_2pi(cropped_yaws[1])]  # [0, 2pi]
    assert check_range_within_range(norm_image_yaws, norm_cropped_yaws), f"cropped_yaws must be within image_yaws. image_yaws: {norm_image_yaws}, cropped_yaws: {norm_cropped_yaws}"
    if np.isclose(image_yaws[0], image_yaws[1], atol=1e-8): # if same value, 2pi apart is assumed
        width_yaw_img = 2 * np.pi
    else:
        width_yaw_img = angle_between_lr_yaw(*image_yaws)
    output_width_pixels = int(img_width_yaw / width_yaw_img * image_width_pixels)  # width in pixels
    # print(f"output_width_pixels: {output_width_pixels}")

    def bin_yaw(yaw: float, num_pixels: int):
        # yaw is between [left_width, right_width]
        per_width = norm_image_yaws[0] - yaw if norm_image_yaws[0] > yaw else norm_image_yaws[0] + 2 * np.pi - yaw
        per_width = per_width / width_yaw_img
        return int(per_width * num_pixels)
    right_pixel_idx = bin_yaw(norm_cropped_yaws[1], image_width_pixels)
    left_pixel_idx = right_pixel_idx - output_width_pixels
    if left_pixel_idx < 0:
        if left_pixel_idx == -1:
            return image[:, :, 0:right_pixel_idx+1]
        print(image_yaws, cropped_yaws, img_width_yaw)
        print(f"left_pixel_idx: {left_pixel_idx}, right_pixel_idx: {right_pixel_idx}")
        raise ValueError("left_pixel_idx < 0, we do not support wrap around")
    return image[:, :, left_pixel_idx:right_pixel_idx]


def reduce_djikstra_sector_to_cost(djikstra: np.ndarray,  yaw_range: tuple,
                                   radial_boundaries: np.ndarray, map_resolution: float,
                                   robot_max_speed: float, swath_library, clip_range: tuple = None, printout=False):
    """
    Given a djikstra map, get the portion of the map that corresponds to the given yaw range
    yaw_range is a tuple of (min_yaw, max_yaw), both will be wrapped within 0->2pi
    djikstra: torch.tensor [h, w]

    Behavior:
      Try to mask the djikstra map just around the far radial boundary arc
        If most of these points are outside of the map (indicated by negative values)
          If any of the points in the whole radial bin [i-1] to [i] are in the map, calculate the cost using these points 
          If not, return infinite (untraversable)
        If most points are inside the map, calculate the cost by taking the median of the in-map points
    """
    yaw_range = [wrap_angle_2pi(yaw_range[0]), wrap_angle_2pi(yaw_range[1])]

    center_idx = (djikstra.shape[0] // 2,
                  djikstra.shape[1] // 2)  # 2r+1 by 2r+1

    radius_matrix, yaw_matrix = generate_range_and_bearing_tensors(
        center_idx, djikstra.shape[-1], map_resolution)

    num_range_bins = len(radial_boundaries) - 1
    result = torch.zeros(num_range_bins)

    if printout:
        indices_log = []
        values_log = []
    for i in range(num_range_bins):
        # grab only the values at the edge of the range bin
        indices = swath_library.get_arc(
            center_idx, radial_boundaries[i], yaw_range[1], djikstra.shape)
        # divide by the time it would take for the robot to reach each point at max speed
        values = djikstra[indices] / (radius_matrix[indices] / robot_max_speed)
        if printout:
            indices_log.append(indices)
            values_log.append(values)
            print("ratio values are ", values)
            print("with min ", torch.min(values))
        # remove all out-of-map influence (negative values)
        values = values[values >= 0]
        if values.shape[0] == 0:  # all values in bin are untraversable/out of map
            pass

        test_result = torch.min(values)
        # more than half the values on the arc are outside of the map (different than inf, ie untraversable) (outside bc negative)
        if test_result < 0 or values.shape[0] == 0:
            # recalculate the mask to cover the whole sector, to handle bins on the edge of the map
            indices = swath_library.get_swath(
                center_idx, radial_boundaries[i], yaw_range[0], djikstra.shape)
            if indices[0].shape[0] == 0:
                # bin compleatly outside of map, set to inf
                if printout:
                    print("bin is completely outside of map, setting to inf")
                result[i] = np.inf
            else:
                # values is unitless
                # djikstra -> seconds
                # radius_matrix -> meters
                # robot_max_speed -> meters / second
                values = djikstra[indices] / \
                    (radius_matrix[indices] / robot_max_speed)
                # remove values outside of the map
                values = values[values >= 0.0]
                test_result = torch.min(values)
                result[i] = test_result
        else:
            result[i] = test_result

        # Potential Issue if Trees take up too much of the space, it'll mark it as highest cost!! oo noo. Or maybe this is fine??
    # transform into exp space!
    if printout:
        print("result is before clipping: ", result)
    if clip_range is not None:
        result = clip_ratios(result, clip_range=clip_range)
    if printout:
        return result, indices_log, values_log
    else:
        return result


def invert_normalize(norm_tf: transforms.Normalize):
    """
    Invert the normalization of an image. 

    norm_tf:
        Either a torchvision.transforms.Normalize or..
        A list of transforms -- assumes one of the transforms is a Normalize
        A transforms.Compose object -- assumes one of the transforms is a Normalize
    """

    if type(norm_tf) == transforms.Compose:
        norm_tf = norm_tf.transforms
    if type(norm_tf) == list:
        for tf in norm_tf:
            if type(tf) == transforms.Normalize:
                norm_tf = tf
                break

    mean = torch.tensor(norm_tf.mean)
    std = torch.tensor(norm_tf.std)
    return transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


def plot_batch(images: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
    """
    Plot the images and labels in a batch
    model: pl.LightningModule model to do inference with
    images: torch.tensor [batch, 3, h, w]
    labels: torch.tensor [batch, n_bins]
    """
    num_images = images.shape[0]
    fig, ax = plt.subplots(num_images, 1)
    fig.tight_layout()
    # remove tick marks
    for i in range(num_images):
        ax[i].imshow(chw_to_hwc_img(images[i]))
        ax[i].set_title(f"labels: {labels[i].detach().cpu().numpy().tolist()}\npreds: {predictions[i].detach().cpu().numpy().tolist()}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    return fig

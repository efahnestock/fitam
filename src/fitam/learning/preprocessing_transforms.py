from torchvision import transforms


def get_default_transforms(model_name: str, **kwargs):
    """
    Returns the default transforms for the given model
    """
    if model_name in ['dinov2', 'full_ensemble']:
        return dino_preprocessing_transforms(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def dino_preprocessing_transforms(image_size):
    """
    Returns the default transforms for DINO
    normalize!
    crop to be multiple of patch size (14x14)

    @param image_size: tuple of (height, width)
    """
    tfs = [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    h, w = image_size[0] - image_size[0] % 14, image_size[1] - image_size[1] % 14
    tfs.append(transforms.CenterCrop((h, w)))
    return tfs

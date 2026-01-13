import torch.nn.functional as F
import torch
from typing import Optional


def remap_model_state_dict(state_dict_to_match: dict, model_state_dict: dict) -> dict:
    """
    Remap state_dict_to_match to match the keys in model_state_dict
    As pytorch lightning models have an extra "model." prefix, this function
    will add or remove the prefix as needed
    """
    checkpoint_key = sorted(list(state_dict_to_match.keys()))[-1]
    model_key = sorted(list(model_state_dict.keys()))[-1]
    print("ENTERED REMAP", checkpoint_key, model_key)
    def key_mod_fn(x): return x
    if checkpoint_key.startswith("model.") and not model_key.startswith("model."):
        # remove model from the checkpoint keys
        def key_mod_fn(x): return x[len("model."):]
    elif model_key.startswith("model.") and not checkpoint_key.startswith("model."):
        # add model to the checkpoint keys
        def key_mod_fn(x): return "model." + x

    state_dict_to_match = {key_mod_fn(k): v for k, v in state_dict_to_match.items()}
    parts = checkpoint_key.split(".")
    models_index = parts.index('models')
    remapping_index = models_index + 1
    parts_model_key = model_key.split(".")
    models_index_model_key = parts_model_key.index('models')
    remapping_index_model_key = models_index_model_key + 1
    print("WE ARE CHECKING FOR KEY REMAPPING", remapping_index, checkpoint_key, model_key)
    if ('_' in checkpoint_key.split(".")[remapping_index] and checkpoint_key.split(".")[remapping_index].split("_")[0].isdigit()) and model_key.split(".")[remapping_index_model_key].isdigit():
        print("REMAPPING OLD ENSEMBLE TO NEW")
        # map model.models.0_0.model.0.weight to model.models.0.models.0.0.weight

        def key_mod_fn(model_key):
            if not model_key.startswith("model"):
                return model_key
            if len(model_key.split(".")) < 3:
                return model_key
            parts = model_key.split(".")
            if 'models' not in parts:
                return model_key
            models_index = parts.index('models')
            remapping_index = models_index + 1
            key = model_key.split(".")[remapping_index].split("_")
            if "_" not in model_key.split(".")[remapping_index]:
                return model_key
            if not key[0].isdigit() or not key[1].isdigit():
                return model_key
            final_key = ".".join(model_key.split(".")[:remapping_index] + [key[0]] + [model_key.split(".")[remapping_index+1]+'s'] + [key[1]] + model_key.split(".")[remapping_index+2:])
            print("REMAPPED KEY", model_key, final_key)
            return final_key

        state_dict_to_match = {key_mod_fn(k): v for k, v in state_dict_to_match.items()}

    return state_dict_to_match


def load_model(name: str = None, checkpoint_path: str = None, get_lightning: bool = False, **kwargs):
    """
    Load a model by name.
    If checkpoint is not None, load the model from the checkpoint. Name is loaded from checkpoint if present
    If kwargs are provided, they overwrite the checkpoint hyperparameters. 

    Can load a lightning or normal model from either a lightning or normal checkpoint.
    """
    model_args = kwargs
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model_args = checkpoint["hyper_parameters"] if "hyper_parameters" in checkpoint else {}
        model_args.update(kwargs)
        name = checkpoint['hyper_parameters']["name"] if name is None else name
    if name == "full_ensemble":
        model_args = {key: model_args[key] for key in ['num_models', 'num_bins', 'use_features', 'dinov2_weight_path',
                                                       'dinov2_model_repo_path', 'num_classes', 'model_name', 'model_args'] if key in model_args}
        # load the member model
        member_model = None
        if get_lightning:
            from fitam.learning.fully_separate_ensemble import FullEnsemble
            model = FullEnsemble(**model_args)
        else:
            from fitam.learning.fully_separate_ensemble_torch import FullEnsembleTorch
            model = FullEnsembleTorch(**model_args)

    elif name == "BinEnsemble":
        from radial_learning.models.bin_ensemble import BinEnsemble
        model = BinEnsemble.load_from_checkpoint(checkpoint_path)
        return model
    else:
        raise RuntimeError(f"Model name {name} not recognized")

    if checkpoint_path is not None:
        # load in the state dict, after remapping
        modified_state_dict = checkpoint["state_dict"]
        model_state_dict = model.state_dict()
        modified_state_dict = remap_model_state_dict(modified_state_dict, model_state_dict)
        # load the state dict
        model.load_state_dict(modified_state_dict)

    return model


def weighted_mse_loss(input, target, weight, reduction='mean'):
    if reduction == 'mean':
        return (weight * (input - target) ** 2).mean()
    elif reduction == 'sum':
        return (weight * (input - target) ** 2).sum()
    elif reduction == 'none':
        return weight * (input - target) ** 2
    else:
        raise RuntimeError(f"Reduction {reduction} not recognized")


def weighted_ce_loss(input, target, weight, class_weight=None, reduction='mean', entropy_weight: Optional[float] = None):
    if entropy_weight is not None:
        assert False, "This is not implemented yet"
    if reduction == 'mean':
        return (weight * F.cross_entropy(input, target, weight=class_weight, reduction='none')).mean()
    elif reduction == 'sum':
        return (weight * F.cross_entropy(input, target, weight=class_weight, reduction='none')).sum()
    elif reduction == 'none':
        return weight * F.cross_entropy(input, target, weight=class_weight, reduction='none')
    else:
        raise RuntimeError(f"Reduction {reduction} not recognized")


def load_loss(loss_name: str):
    if loss_name == "mse":
        return F.mse_loss
    elif loss_name == "wmse":
        return weighted_mse_loss
    elif loss_name == "ce":
        return F.cross_entropy
    elif loss_name == "wce":
        return weighted_ce_loss
    else:
        raise RuntimeError(f"Loss name {loss_name} not recognized")

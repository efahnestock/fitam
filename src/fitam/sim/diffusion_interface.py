import numpy as np
import torch
import pickle
import time
import cv2
import shutil
import enum
import matplotlib.pyplot as plt
from pathlib import Path
from typing import NamedTuple, Optional
from fitam.mapping.land_cover_complex_map import LandCoverComplexMap, semantic_class_to_occ_grid_cost, lulc_to_semantic_mapping, semantic_class_to_color_map
from fitam.mapping.costmap import OccupancyGrid
from fitam.mapping.observation_types import DiffusionObservation
from learning_env_structure import utils
#import learning_env_structure.make_model as mm
#from learning_env_structure.make_model import make_conditional_model #as make_model
from learning_env_structure.make_model import make_conditional_model, make_model
#from learning_env_structure.inpainting import MCMCParams
from learning_env_structure.palettize_image import semantic_from_rgb_with_color_cube
from fitam.core.common import float_to_cv2_img, numpy_log_softmax, numpy_softmax
from fitam.core.config.RadialMapConfig import DiffusionConfig
from torchvision.transforms import ToTensor
from fitam import FITAM_ROOT_DIR
#from learning_env_structure.classifier_free_guidance import Unet, GaussianDiffusion
from torch.cuda.amp import autocast
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#mm.Unet = Unet
#mm.GaussianDiffusion = GaussianDiffusion
class ModelType(enum.Enum):
    CONDITIONAL = 1
    MCMC = 2

#mcmc_params = MCMCParams(
#    num_steps=5,
#    min_timestep=500,
#    max_timestep=1000,
#    stepsize_multiplier=20,
#)

class DiffusionInterface:

    def __init__(self, 
                 model_path: Optional[Path], 
                 model_type: ModelType,
                 config: DiffusionConfig):
        self.config = config
        with open(f"{FITAM_ROOT_DIR}/{self.config.palette_path}", 'rb') as f:
            self.palette = pickle.load(f)
        print(model_type)
        self.color_cube = np.load(f"{FITAM_ROOT_DIR}/{self.config.color_lut_path}")
        assert isinstance(config.save_root, Path), "save_root must be a Path object"
        if True in [config.save_diffusion_batch, config.save_mask_images, config.save_class_images, config.save_cost_uncertainty_images]:
            if config.save_root.exists():
                shutil.rmtree(config.save_root)
            (config.save_root / 'diffusion_observations').mkdir(parents=True, exist_ok=False)
            (config.save_root / 'diffusion_outputs').mkdir(parents=True, exist_ok=False)

        self.model_type = model_type
        if model_path is not None:
#            if model_type == ModelType.MCMC:
 #               self.model = make_model(model_path, image_size=self.config.diffusion_image_shape)
            if model_type == ModelType.CONDITIONAL:
                self.model = make_conditional_model(model_path, image_size=self.config.diffusion_image_shape)
                self.model.eval()
                self.model.cuda()

                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False
                )
                print("Diffusion model compiled:", type(self.model))
            else:
                raise ValueError("Invalid model type")
        else:
            self.model = None
        self.observation_idx = 0

    @staticmethod
    def _check_semantic_rgb(semantic_rgb, observed_mask, occ_grid):
        assert len(semantic_rgb.shape) == 3, "GT RGB must be 3D"
        assert isinstance(semantic_rgb[0, 0, 0], np.floating), "GT RGB must be Integer"
        assert semantic_rgb.shape[2] == 3, "GT RGB must have 3 channels"
        assert semantic_rgb.shape[0] == occ_grid.data.shape[0], "GT RGB must have same height as occ grid"
        assert semantic_rgb.shape[1] == occ_grid.data.shape[1], "GT RGB must have same width as occ grid"
        assert observed_mask.shape == occ_grid.data.shape, "Observed mask must have same shape as occ grid"
        assert observed_mask.dtype == bool, "Observed mask must be boolean"

    @staticmethod
    def _make_fake_rgb_batch(true_rgb: np.ndarray, batch_size: int) -> np.ndarray:
        return np.tile(true_rgb, (batch_size, 1, 1, 1))

    def get_observation(self, 
                        center_idx: int,
                        occ_grid: OccupancyGrid,
                        semantic_rgb_float: np.ndarray,
                        observed_mask: np.ndarray, 
                        observation_index: Optional[int] = None
                        ) -> DiffusionObservation:
        # input: master lcm, trajectory history, center pose
        # output: DiffusionObservation
        if observation_index is not None:
            self.observation_idx = observation_index
        self._check_semantic_rgb(semantic_rgb_float, observed_mask, occ_grid)
        cv2_semantic = float_to_cv2_img(semantic_rgb_float)

        full_cv2_img, mask, masked_cv2_img = self.get_diffusion_image_and_mask(center_idx, observed_mask, cv2_semantic, occ_grid)
        if self.model is None:
            rgb_batch = self._make_fake_rgb_batch(full_cv2_img, self.config.batch_size)
            # print("created GT batch")
        else:
            rgb_batch = self.get_diffusion_rgb_batch(mask, masked_cv2_img)
            # print("created diffusion batch")
        # print("RGB BATCH SHAPE", rgb_batch.shape)
        classes = self.get_classes_from_rgb_batch(rgb_batch, self.config.diffusion_image_shape, self.color_cube)
        # print("CLASSES SHAPE", classes.shape)
        if self.config.save_class_images:
            self._save_class_images(classes)
        costs = self.get_costs_from_class_batch(classes, self.palette, use_mode=True)
        uncertainty = self.get_uncertainty_from_class_batch(classes, self.palette)
        if self.config.save_cost_uncertainty_images:
            self._save_cost_uncertainty_image(costs, uncertainty)
        self.observation_idx += 1
        # rescale to occ grid
        costs = self._scale_from_diffusion_to_occ_grid(costs, occ_grid, self.config.diffusion_image_size_m)
        uncertainty = self._scale_from_diffusion_to_occ_grid(uncertainty, occ_grid, self.config.diffusion_image_size_m)
        return DiffusionObservation(center_idx, costs, uncertainty, occ_grid)

    @staticmethod 
    def _scale_from_diffusion_to_occ_grid(to_be_scaled: np.ndarray, 
                                          occ_grid: OccupancyGrid,
                                          diffusion_image_size_m) -> np.ndarray:
        diffusion_pixel_size = (int(diffusion_image_size_m[0] / occ_grid.resolution),
                                int(diffusion_image_size_m[1] / occ_grid.resolution))
        return cv2.resize(to_be_scaled, diffusion_pixel_size, interpolation=cv2.INTER_NEAREST)

    def get_diffusion_image_and_mask(self, 
                                     center_idx: int,
                                     observed_mask: np.ndarray, 
                                     semantic_cv2_img: np.ndarray,
                                     occ_grid: OccupancyGrid,
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        diffusion_pixel_size = (int(self.config.diffusion_image_size_m[0] / occ_grid.resolution),
                                int(self.config.diffusion_image_size_m[1] / occ_grid.resolution))
        upper_left = (center_idx[0] - diffusion_pixel_size[0] // 2, center_idx[1] - diffusion_pixel_size[1] // 2)
        lower_right = (center_idx[0] + diffusion_pixel_size[0] // 2, center_idx[1] + diffusion_pixel_size[1] // 2)


        # handle out of bounds
        shift = np.array([0, 0])
        if upper_left[0] < 0:
            shift[0] = -upper_left[0]
            upper_left = (0, upper_left[1])
        if upper_left[1] < 0:
            shift[1] = -upper_left[1]
            upper_left = (upper_left[0], 0)
        if lower_right[0] >= occ_grid.data.shape[0]:
            lower_right = (occ_grid.data.shape[0] - 1, lower_right[1])
        if lower_right[1] >= occ_grid.data.shape[1]:
            lower_right = (lower_right[0], occ_grid.data.shape[1] - 1)

        output_img = np.zeros((*diffusion_pixel_size, 3), dtype=np.uint8)
        output_mask = np.zeros(diffusion_pixel_size, dtype=bool)

        output_img[shift[0]:shift[0] + lower_right[0] - upper_left[0], 
                   shift[1]:shift[1] + lower_right[1] - upper_left[1], :] = semantic_cv2_img[upper_left[0]:lower_right[0], 
                                                                                      upper_left[1]:lower_right[1], :]
        output_mask[shift[0]:shift[0] + lower_right[0] - upper_left[0],
                    shift[1]:shift[1] + lower_right[1] - upper_left[1]] = observed_mask[upper_left[0]:lower_right[0],
                                                                                        upper_left[1]:lower_right[1]]
        output_mask_uint8 = output_mask.astype(np.uint8) * 255 
        output_img = cv2.resize(output_img, self.config.diffusion_image_shape, interpolation=cv2.INTER_NEAREST)
        output_mask_uint8 = cv2.resize(output_mask_uint8, self.config.diffusion_image_shape, interpolation=cv2.INTER_NEAREST)

        masked_rgb_img = np.zeros_like(output_img)
        masked_rgb_img[output_mask_uint8 > 0] = output_img[output_mask_uint8 > 0]

        if self.config.save_mask_images:
            cv2.imwrite(f"{self.config.save_root / 'diffusion_observations'}/{self.observation_idx:07d}_mask_{center_idx}.png", output_mask_uint8)
            cv2.imwrite(f"{self.config.save_root / 'diffusion_observations'}/{self.observation_idx:07d}_rgb_{center_idx}.png", output_img)
            cv2.imwrite(f"{self.config.save_root / 'diffusion_observations'}/{self.observation_idx:07d}_masked_rgb_{center_idx}.png", masked_rgb_img)

        output_mask = output_mask_uint8.astype(bool)

        
        return output_img, output_mask, masked_rgb_img

    def get_diffusion_rgb_batch(self, mask: np.ndarray, masked_cv2_image: np.ndarray) -> np.ndarray: # (batch, 3, height, width)
        input_img = cv2.cvtColor(masked_cv2_image, cv2.COLOR_BGR2RGB)
        input_img = ToTensor()(input_img)
        mask = torch.from_numpy(mask)
       # if self.model_type == ModelType.MCMC:
       #     with torch.no_grad():
       #         output = self.model.inpaint_ddim(
       #             input_img.cuda(), mask.cuda(),
       #             batch_size=self.config.batch_size,
       #             sampling_timesteps=200,
       #             mcmc_params=mcmc_params,
       #             is_noise_correlated=False,
       #             return_all_timesteps=False,
       #             live_viz=False,
       #         )  # (B, 3, H, W)
        if self.model_type == ModelType.CONDITIONAL:
            input_img = input_img.unsqueeze(0)
            input_img = input_img.expand((self.config.batch_size, *input_img.shape[-3:]))
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand((self.config.batch_size, 1, *mask.shape[-2:]))
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            # if not Path('/tmp/inpaint_result.pt').exists():

#           with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
#            #with torch.no_grad(), autocast("cuda", dtype=torch.float16):
#                output = self.model.sample(
#                    return_all_timesteps=False,
#                    x_obs=input_img.cuda(),
#                    x_obs_mask=mask.cuda(),
#                    batch_size=self.config.batch_size
#                )
            with torch.no_grad(), autocast(dtype=torch.float16):
             output = self.model.sample(
                return_all_timesteps=False,
                x_obs=input_img.cuda(),
                x_obs_mask=mask.cuda(),
                batch_size=self.config.batch_size
            )
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_sampling_time = end_time - start_time
            logger.debug(f"Diffusion sampling time: {total_sampling_time:.3f}s")
            print(f"Diffusion sampling time: {total_sampling_time:.3f}s")
            print("AMP enabled:", torch.is_autocast_enabled())
            # if not Path('/tmp/inpaint_result.pt').exists():
            #with torch.no_grad():
            #    output = self.model.sample(return_all_timesteps = False, x_obs = input_img.cuda(), x_obs_mask = mask.cuda(), batch_size=self.config.batch_size)
            # else:
            #     output = torch.load('/tmp/inpaint_result.pt')

            
        torch.save(output, f"{self.config.save_root / 'diffusion_outputs' / f'{self.observation_idx:07d}_diffusion_output.pt'}")
        if self.config.save_diffusion_batch:
            utils.save_gif(output.unsqueeze(1), str(self.config.save_root / 'diffusion_observations' / f"{self.observation_idx:07d}_diffusion_batch.gif"))
        output = output.cpu().numpy()
        output = (output * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        return output # (batch, height, width, 3) uint8

    @staticmethod
    def get_classes_from_rgb_batch(rgb_batch: np.ndarray, diffusion_image_shape: tuple, color_cube: np.ndarray) -> np.ndarray:
        output = np.zeros((rgb_batch.shape[0], *diffusion_image_shape), dtype=np.uint8)
        for image_idx in range(rgb_batch.shape[0]):
            rgb_img = rgb_batch[image_idx]
            output[image_idx] = semantic_from_rgb_with_color_cube(image=rgb_img, color_cube=color_cube)
        return output # [batch, height, width]
    
    def _save_class_images(self, class_batch: np.ndarray):
        output_images = self._map_palette_index_to_color(class_batch, self.palette)
        utils.save_gif(torch.from_numpy(output_images.transpose(0, 3, 1, 2)).unsqueeze(1), self.config.save_root / 'diffusion_observations' / f"{self.observation_idx:07d}_class_batch.gif")


    @staticmethod
    def get_uncertainty_from_class_batch(class_batch: np.ndarray, palette: dict, use_entropy: bool = False) -> np.ndarray:
        class_counts = DiffusionInterface._get_counts_from_batch(class_batch, palette)

        if use_entropy:
            class_distribution = class_counts / class_batch.shape[0]
            return -np.sum(class_distribution * np.log(class_distribution + 1e-9), axis=-1)
        else:
            return 1 - (np.max(class_counts, axis=-1) / class_batch.shape[0])


    @staticmethod
    def _get_counts_from_batch(class_batch: np.ndarray, palette) -> np.ndarray:
        n = len(palette)
        bin_counts = np.zeros((class_batch.shape[1], class_batch.shape[2], n))
        for class_idx in range(n):
            bin_counts[:, :, class_idx] = np.sum(class_batch == class_idx, axis=0)
        return bin_counts
    @staticmethod 
    def _map_palette_index_to_cost(class_array: np.ndarray, palette: np.ndarray)->np.ndarray:
        palette_names = [x.name for x in list(palette.keys())]
        lcm_names = [lulc_to_semantic_mapping[x] for x in palette_names]
        cost_array = np.asarray([semantic_class_to_occ_grid_cost[x] for x in lcm_names])
        return cost_array[class_array]

    @staticmethod
    def _map_palette_index_to_color(class_array: np.ndarray, palette: np.ndarray)->np.ndarray:
        palette_names = [x.name for x in list(palette.keys())]
        lcm_names = [lulc_to_semantic_mapping[x] for x in palette_names]
        color_array = np.asarray([semantic_class_to_color_map[x] for x in lcm_names])
        return color_array[class_array]

    @staticmethod
    def get_costs_from_class_batch(class_batch: np.ndarray, palette: np.ndarray, use_mode: bool = True) -> np.ndarray:
        class_counts = DiffusionInterface._get_counts_from_batch(class_batch, palette)
        if use_mode:
            batch_mode = np.argmax(class_counts, axis=-1)
            return DiffusionInterface._map_palette_index_to_cost(batch_mode, palette)
        else:
            # use the expected cost of each categorical distribution
            class_distrubtions = class_counts / class_batch.shape[0]
            class_costs = DiffusionInterface._map_palette_index_to_cost(np.arange(class_counts.shape[-1], dtype=int), palette)
            # cap costs
            class_costs[class_costs > 10.0] = 10.0
            return np.sum(class_costs * class_distrubtions, axis=-1)
    
    def _save_cost_uncertainty_image(self, costs: np.ndarray, uncertainty: np.ndarray):
        fig, ax = plt.subplots(1,2, figsize=(10, 5))
        ax[0].set_title("Costs")
        cbar = ax[0].figure.colorbar(ax[0].imshow(costs, cmap='viridis', vmax=2.0), ax=ax[0])
        cbar.set_label("Cost")
        ax[1].set_title("Uncertainty")
        cbar = ax[1].figure.colorbar(ax[1].imshow(uncertainty, cmap='viridis'), ax=ax[1])
        cbar.set_label("Uncertainty")
        plt.savefig(self.config.save_root / 'diffusion_observations' / f"{self.observation_idx:07d}_cost_uncertainty.png")
        plt.close()






if __name__ == "__main__":
    from radial_learning import MAPS_DIR, SWATHS_DIR, CONFIGS_DIR
    from radial_learning.sim.costmap_swath_library import load_swath_library_from_pkl
    import matplotlib.pyplot as plt
    import random
    from radial_learning.utils.easl_python import State
    map_path = MAPS_DIR / 'final_experiment_train_balt'
    lcm = LandCoverComplexMap.from_map_folder(map_path)
    occ_grid = OccupancyGrid.from_complexmap(lcm)
    semantic_rgb = lcm.create_floormask(include_only_visible=True)
    semantic_rgb = float_to_cv2_img(semantic_rgb)
    swath_lib_path  = SWATHS_DIR / 'simulated_radial_configs' / 'radial_map_config.pkl'
    swath_lib = load_swath_library_from_pkl(swath_lib_path)



    pallete_path = CONFIGS_DIR / 'lcm_palette.pkl'
    color_cube_path = CONFIGS_DIR / 'lcm_color_cube.npy'

    # create random trajectory
    random.seed(42)
    start = (0, 0)
    traj = [start]
    for _ in range(10):
        delta = random.choice([(0, 1)])#, (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)])
        for __ in range(100):
            new = (traj[-1][0] + delta[0], traj[-1][1] + delta[1])
            if occ_grid.check_index_in(new):
                traj.append(new)

    traj = [State(*x) for x in traj]

    obs_mask = occ_grid.get_visibility_mask(traj, swath_lib)

    config = DiffusionConfig(
        diffusion_image_shape=(128, 128),
        diffusion_image_size_m=(500,500),
        batch_size=32,
        save_mask_images=True,
        save_diffusion_batch=True,
        save_class_images=True,
        save_cost_uncertainty_images=True,
        save_root=Path("/tmp/diffusion_debug"),
    )

    model_path = '/tmp/model.ckpt'
    # run scp to get model file
    if not Path(model_path).exists():
        import os
        os.system(f"scp super:rrg_shared/learning_env_structure/models/maryland_128x128_conditional.pt {model_path}")

    diffusion_interface = DiffusionInterface(model_path, pallete_path, color_cube_path, config)

    center_idx = occ_grid.get_index_from_point((traj[-1].x, traj[-1].y))

    observation = diffusion_interface.get_observation(center_idx, occ_grid, semantic_rgb, obs_mask)

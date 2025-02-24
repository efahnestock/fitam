
from __future__ import annotations
import warnings

"""

OpenCV coordinates:
0,0 ------> +x
|
|
\/
+y

Matplotlib imshow coordinates:
0,0 ------> +y
|
|
\/
+x

Matplotlib coordinates:
+y
/\
|
|
0,0 ------> +x

"""
import os
import shutil
import pickle
import cv2
from fitam.mapping.obstacle_types import (
    MapObject, BushShapeConfig, BuildingShapeConfig,
    TreeShapeConfig, FallenTreeShapeConfig, RockShapeConfig)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.patches as mpatches
from fitam.planning.planner_general import State
from fitam.core.config.EnvConfig import EnvConfig, RealEnvConfig
from fitam import MAPS_DIR
from fitam.mapping.costmap import OccupancyGrid
from fitam.core.common import wrap_angle_2pi
from fitam.mapping.map_tools import find_map_bounds_around_point
from fitam.mapping.costmap_swath_library import SwathLibrary

lulc_to_semantic_mapping = {
    'Estuarine/Marine': 'water',
    'Lakes and Reservoirs': 'water',
    'Riverine Ponds': 'water',
    'Terrene Ponds': 'water',
    'Lotic Water (fresh)': 'water',

    'Roads': 'road',
    'Tree Canopy Over Roads': 'road',
    'Other Impervious': 'road',
    'Tree Canopy Over Other Impervious': 'road',

    'Structures': 'structure',
    'Tree Canopy Over Structures': 'structure',

    'Turf Grass': 'grass',
    'Transitional Barren': 'grass',
    'Suspended Succession Barren': 'grass',
    'Suspended Succession Herbaceous': 'grass',
    'Suspended Succession Scrub/Shrub': 'grass',
    'Tree Canopy Over Turf Grass': 'grass',

    'Forest': 'forest',
    'Other Tree Canopy': 'forest',
    'Harvested Forest Barren': 'forest',
    'Harvested Forest Herbaceous': 'forest',

    'Natural Succession Barren': 'young_forest',
    'Natural Succession Herbaceous': 'young_forest',
    'Natural Succession Scrub/Shrub': 'young_forest',

    'Cropland Barren': 'field',
    'Cropland Herbaceous': 'field',
    'Pasture/Hay Barren': 'field',
    'Pasture/Hay Herbaceous': 'field',
    'Pasture/Hay Scrub/Shrub': 'field',
    'Orchard/Vineyard Herbaceous': 'field',
    'Orchard/Vineyard Barren': 'field',
    'Orchard/Vineyard Scrub/Shrub': 'field',
    'Solar Field Herbaceous': 'field',
    'Extractive Barren': 'field',
    'Solar Field Scrub/Shrub': 'field',
    'Solar Field Barren': 'field',

    'Tidal Wetlands Barren': 'marsh',
    'Tidal Wetlands Herbaceous': 'marsh',
    'Tidal Wetlands Scrub/Shrub': 'marsh',
    'Tidal Wetlands Tree Canopy': 'marsh',
    'Tidal Wetlands Forest': 'marsh',
    'Riverine Wetlands Barren': 'marsh',
    'Riverine Wetlands Herbaceous': 'marsh',
    'Riverine Wetlands Scrub/Shrub': 'marsh',
    'Riverine Wetlands Tree Canopy': 'marsh',
    'Riverine Wetlands Forest': 'marsh',
    'Terrene Wetlands Barren': 'marsh',
    'Terrene Wetlands Herbaceous': 'marsh',
    'Terrene Wetlands Tree Canopy': 'marsh',
    'Terrene Wetlands Forest': 'marsh',
    'Terrene Wetlands Scrub/Shrub': 'marsh',

    'Bare Shore': 'beach'
}

semantic_class_to_color_map = {
    'young_forest': np.array([0.5647, 0.9333, 0.5647]),
    'forest': np.array([0.1333, 0.5451, 0.1333]),
    'grass': np.array([0.21, 0.89, 0.0]),
    'field': np.array([0.81, 0.89, 0.0]),
    'road': np.array([0., 0., 0.]),
    'water': np.array([0.0, 0.0, 0.5]),
    'structure': np.array([0.5, 0.5, 0.5]),
    'buildings': np.array([0.5, 0.5, 0.5]),
    'marsh': np.array([0.4706, 0.3922, 0.3137]),
    'beach': np.array([0.9569, 0.6431, 0.3765]),
    'trees': np.array([0.96, 0.009, 0.0]),
    'rocks': np.array([0.5, 0.5, 0.5]),
    'bushes': np.array([0.0, 1, 0.0]),
    'fallen_trees': np.array([0.96, 0.109, 0.0]),
    'unknown': np.array([1, 0, 0]),
}
occluding_classes = [
    'structure',
    'buildings',
    'trees',
    'rocks',
    'bushes',
    'fallen_trees',
]
semantic_class_to_occ_grid_cost = {
    # in the forest, roughly a meter per second
    'young_forest': 1 / 1.0,
    # dense forest, maybe half a meter per second
    'forest': 1 / 0.5,
    # forest, maybe can get up to 3 meters per second
    'grass': 1 / 2.0,
    'field': 1 / 3.0,
    # road, max speed of 5 meters per second
    'road': 1 / 5.0,
    # wetland, slow and muddy
    'marsh': 1 / 0.25,
    # beach, slow and sandy
    'beach': 1 / 0.1,
    # building, no go,
    'structure': np.inf,  # untraversable
    'buildings': np.inf,  # untraversable
    # lake, no go
    'water': np.inf,  # untraversable
    # trees, no go
    'trees': np.inf,  # untraversable
    # rocks, no go
    'rocks': np.inf,  # untraversable
    # bushes we can cross at 0.1 meters per second
    'bushes': 1 / 0.1,
    # fallen trees we can cross at 0.1 meters per second
    'fallen_trees': 1 / 0.1,
    # unknown regions? We don't know
    'unknown': -1,  # negative cost
}


class LandCoverComplexMap:
    resolution: float  # resolution, in meters per pixel

    # dict of semantic masks
    masks: dict[str, np.ndarray]

    # costs of different semantic classes
    occgrid_costs: dict[str, float]

    def __init__(self, semantic_image: np.ndarray, resolution: float, class_map: dict[str, int]):
        """
        semantic_image: np.ndarray with values corresponding to the class_map
        resolution: resolution of the map, in meters per pixel
        class_map: dict mapping class names to class indices
        """
        super().__init__()
        self.semantic_image = semantic_image
        self.resolution = resolution
        self.class_map = class_map
        self.__post_init__()

    def __repr__(self):
        return f"LandCoverComplexMap(res:{self.resolution}, pix_x,y:({self.width_pixels},{self.height_pixels}), size_x,y:({self.width},{self.height})"

    @property
    def width(self):
        return self.semantic_image.shape[1] * self.resolution

    @property
    def height(self):
        return self.semantic_image.shape[0] * self.resolution

    @property
    def width_pixels(self):
        return self.semantic_image.shape[1]

    @property
    def height_pixels(self):
        return self.semantic_image.shape[0]

    @classmethod
    def create_from_land_cover_raster(cls, land_cover_raster: np.ndarray, resolution: float, pixel_types: dict):
        # pixel types maps [semantic_idx] ->
        # Class descriptions are at: https://cicwebresources.blob.core.windows.net/docs/LU_Classification_Methods_2017_2018.pdf
        """
        ##Remap to -> water
        Water|Estuarine/Marine
        Water|Lakes and Reservoirs
        Water|Riverine Ponds
        Water|Terrene Ponds
        Water|Lotic Water (fresh)
        ####Remap to -> road
        ## Note: Impervious other are objects below 2 meters that are made of materials impenetrable by water
        Impervious Roads|Roads
        Tree Canopy over Impervious|Tree Canopy Over Roads
        Impervious, Other|Other Impervious
        Tree Canopy over Impervious|Tree Canopy Over Other Impervious
        ####Remap to -> structure
        Impervious Structures|Structures
        Tree Canopy over Impervious|Tree Canopy Over Structures
        ####Remap to -> grass
        Turf Grass|Turf Grass
        Pervious Developed, Other|Transitional Barren
        Pervious Developed, Other|Suspended Succession Barren
        Pervious Developed, Other|Suspended Succession Herbaceous
        Pervious Developed, Other|Suspended Succession Scrub/Shrub
        Tree Canopy over Turf Grass|Tree Canopy Over Turf Grass
        ### Remap to -> forest
        Forest|Forest
        Tree Canopy, Other|Other Tree Canopy
        Harvested Forest|Harvested Forest Barren
        Harvested Forest|Harvested Forest Herbaceous

        ### Remap to -> young_forest
        Natural Succession|Natural Succession Barren
        Natural Succession|Natural Succession Herbaceous
        Natural Succession|Natural Succession Scrub/Shrub

        ### remap to -> field
        Cropland|Cropland Barren
        Cropland|Cropland Herbaceous
        Pasture/Hay|Pasture/Hay Barren
        Pasture/Hay|Pasture/Hay Herbaceous
        Pasture/Hay|Pasture/Hay Scrub/Shrub
        Cropland|Orchard/Vineyard Herbaceous
        Pervious Developed, Other|Solar Field Herbaceous
        Extractive|Extractive Barren
        ### remap to marsh
        Wetlands, Tidal Non-forested|Tidal Wetlands Barren
        Wetlands, Tidal Non-forested|Tidal Wetlands Herbaceous
        Wetlands, Tidal Non-forested|Tidal Wetlands Scrub/Shrub
        Tree Canopy, Other|Tidal Wetlands Tree Canopy
        Forest|Tidal Wetlands Forest
        Wetlands, Riverine Non-forested|Riverine Wetlands Barren
        Wetlands, Riverine Non-forested|Riverine Wetlands Herbaceous
        Wetlands, Riverine Non-forested|Riverine Wetlands Scrub/Shrub
        Tree Canopy, Other|Riverine Wetlands Tree Canopy
        Forest|Riverine Wetlands Forest
        Wetlands, Terrene Non-forested|Terrene Wetlands Barren
        Wetlands, Terrene Non-forested|Terrene Wetlands Herbaceous
        Tree Canopy, Other|Terrene Wetlands Tree Canopy
        Forest|Terrene Wetlands Forest
        ### remap to -> beach
        Natural Succession|Bare Shore 
        """
        real_class_mapping = lulc_to_semantic_mapping
        remapped_raster = np.zeros_like(land_cover_raster)
        new_mapping = dict()
        class_idx = 1
        for key in pixel_types:
            pixel_info = pixel_types[key]
            cmap_class = real_class_mapping[pixel_info['LandUse']]
            if cmap_class not in new_mapping:
                new_mapping[cmap_class] = class_idx
                class_idx += 1
            remapped_raster[land_cover_raster == key] = new_mapping[cmap_class]

        return cls(remapped_raster, resolution, new_mapping)

    def __post_init__(self):
        self.masks = {}
        for key in self.class_map:
            class_index = self.class_map[key]
            if key not in self.masks:
                self.masks[key] = np.zeros_like(
                    self.semantic_image, dtype=bool)
            self.masks[key][self.semantic_image == class_index] = True
        # costs of different semantic classes
        # costs are seconds to traverse each meter

        self.obstacles = dict()

    @classmethod
    def from_complexmap(cls, complexmap: 'LandCoverComplexMap'):
        new_map = cls(complexmap.semantic_image,
                      complexmap.resolution, complexmap.class_map)
        return new_map

    @classmethod
    def from_map_folder(cls, map_folder_path: str):
        """Loads a ComplexMap class from a pickle file. NOTE: re-init's the class too """
        map_folder_path = str(map_folder_path)
        if map_folder_path.endswith("/"):
            map_folder_path = map_folder_path[:-1]
        name = os.path.basename(map_folder_path)
        with open(os.path.join(map_folder_path, f'{name}.pkl'), 'rb') as f:
            loaded_map = pickle.load(f)

        reinit_map = cls.from_complexmap(loaded_map)
        reinit_map.masks = loaded_map.masks
        if hasattr(loaded_map, 'obstacles'):
            reinit_map.obstacles = loaded_map.obstacles

        return reinit_map

    def create_flattened_class_map(self):
        """
        Create a flatened class map from the semantic layers
        """
        output = np.zeros_like(self.semantic_image)
        class_index_map = {}
        for i, class_name in enumerate(self.masks.keys()):
            output[self.masks[class_name]] = i
            class_index_map[class_name] = i
        return output, class_index_map
    

    def expand(self, new_shape: tuple[int, int]):
        flat, names = self.create_flattened_class_map()
        assert new_shape[0] % 2 == 0 and new_shape[1] % 2 == 0 and flat.shape[0] % 2 == 0 and flat.shape[1] % 2 == 0
        unknown_index = names["unknown"]
        i_initial = new_shape[0] // 2 - flat.shape[0] // 2
        j_initial = new_shape[1] // 2 - flat.shape[1] // 2
        new_flat = np.ones(new_shape, dtype=int) * unknown_index
        new_flat[i_initial:i_initial+flat.shape[0], j_initial:j_initial+flat.shape[1]] = flat
        self.update_from_flattened_class_map(new_flat, names)

    def update_from_flattened_class_map(self, flattened, class_index_map):
        for name, i in class_index_map.items():
            self.masks[name] = flattened == i
        self.semantic_image = flattened
        self.class_map = class_index_map
        return

    def create_stacked_class_map(self):
        num_layers = len(self.masks)
        ex_mask = list(self.masks.values())[0]
        out = np.zeros((num_layers, *ex_mask.shape), dtype=bool)
        layer_names = sorted(list(self.masks.keys()))
        for i, l in enumerate(layer_names):
            out[i] = self.masks[l]
        return out, layer_names
    
    def update_from_stacked_class_map(self, layers, layer_names):
        for i, name in enumerate(layer_names):
            self.masks[name] = layers[i]

    def create_occlusion_map(self):
        output = np.zeros(self.semantic_image.shape, dtype=bool)
        for name in occluding_classes:
            if name in self.masks:
                output = output + self.masks[name]
        return output

    def crop(self, i_bounds: tuple, j_bounds: tuple) -> None:
        """
        Crop the map to the given bounds
        """

        for name, mask in self.masks.items():
            self.masks[name] = mask[i_bounds[0]:i_bounds[1], j_bounds[0]:j_bounds[1]]
        self.semantic_image = self.semantic_image[i_bounds[0]:i_bounds[1], j_bounds[0]:j_bounds[1]]

    def create_floormask(self, include_only_visible: bool = False):
        img = np.zeros((self.height_pixels, self.width_pixels, 3))
        visible = ['forest', 'young_forest', 'grass', 'field',
                   'road', 'water', 'beach', 'structure', 'marsh', 'unknown']
        for name, mask in self.masks.items():
            if include_only_visible and name not in visible:
                continue
            img[mask > 0] = semantic_class_to_color_map[name]
        return img

    def create_occgrid(self):
        img = np.zeros((self.height_pixels, self.width_pixels))
        for name, mask in self.masks.items():
            img[mask > 0] = semantic_class_to_occ_grid_cost[name]

        # make sure buildings is last, as the contour fitting can cause it to overlap with other classes
        if 'buildings' in self.masks:
            img[self.masks['buildings'] > 0] = semantic_class_to_occ_grid_cost['buildings']
        return img

    def reduce_to_trajectory_visiblity(self, trajectory: list[State], swath_library: SwathLibrary):
        """Reduces the map to only the visible area from the trajectory"""
        occ_grid = OccupancyGrid.from_complexmap(self)
        visibility_mask = occ_grid.get_visibility_mask(trajectory, swath_library)

        for name, mask in self.masks.items():
            self.masks[name] = mask * visibility_mask
        if "unknown" not in self.masks:
            self.masks["unknown"] = np.zeros((self.height_pixels, self.width_pixels), dtype=bool)
        self.masks["unknown"] = np.logical_not(visibility_mask)

    def reduce_to_visibility_mask(self, observed_states: np.ndarray):
        """Reduces the map to only the visible area from the trajectory"""

        for name, mask in self.masks.items():
            self.masks[name] = mask * observed_states
        if "unknown" not in self.masks:
            self.masks["unknown"] = np.zeros((self.height_pixels, self.width_pixels), dtype=bool)
        self.masks["unknown"] = np.logical_not(observed_states)

    def map_with_legend(self):
        floormask = self.create_floormask()
        plt.figure()
        plt.imshow(floormask)
        for name, color in semantic_class_to_color_map.items():
            plt.plot([0], [0], lw=5, label=name, color=color)
        plt.legend()
        plt.show()

    def reset_map(self):
        """ Reset the masks, and clears obstacles """
        # Note that the buildings mask will cause some issues, as with the fact
        # that most of the masks have been edited in generation
        raise NotImplementedError

    def reset(self):
        """ Clear all obstacles and reset everything """
        self.reset_map()
        self.__post_init__()

    def get_mask_area(self, mask: np.ndarray):
        """Return area of a mask in meters squared"""
        return np.sum(mask * self.resolution ** 2)

    def sample_unif_range(self, generator: np.random.Generator, range, num_samples: int = 1):
        values = generator.uniform(range[0], range[1], num_samples)
        if num_samples == 1:
            return values[0]
        return values

    def create_fallen_tree(self, generator: np.random.Generator, mask: np.ndarray, location: np.ndarray, sample_config: dict) -> tuple[np.ndarray, MapObject]:
        fallen_tree_height = self.sample_unif_range(
            generator, sample_config['height_range'])
        fallen_tree_diam = self.sample_unif_range(
            generator, sample_config['diameter_range'])
        fallen_tree_angle = self.sample_unif_range(generator, (0, 2*np.pi))

        h_pixels = max(1, int(fallen_tree_height/self.resolution))
        w_pixels = max(1, int(fallen_tree_diam/self.resolution))
        box = cv2.boxPoints(((int(location[1]), int(
            location[0])), (h_pixels, w_pixels), fallen_tree_angle * -180 / np.pi))
        box = np.int0(box)
        fallen_tree_mask = cv2.drawContours(mask, [box], 0, 1, -1)
        location_m = location * self.resolution
        location_m = [location_m[1], self.width - location_m[0], 0]
        loc_obj = FallenTreeShapeConfig(
            fallen_tree_diam, fallen_tree_height, fallen_tree_angle)
        map_obj = MapObject("fallen_tree", location_m, loc_obj)
        return fallen_tree_mask, map_obj

    def create_bush(self, generator: np.random.Generator, mask: np.ndarray, location: np.ndarray, sample_config: dict) -> tuple[np.ndarray, MapObject]:
        bush_diam = self.sample_unif_range(
            generator, sample_config['diameter_range'])
        bush_height = self.sample_unif_range(
            generator, sample_config['height_range'])
        rad_pixels = min(1, int(bush_diam/self.resolution))
        bush_mask = cv2.circle(
            mask, (location[1], location[0]), rad_pixels, 1, -1)
        location_m = location * self.resolution
        location_m = [location_m[1], self.width - location_m[0], 0]
        shape_cfg = BushShapeConfig(bush_diam, bush_height)
        bush = MapObject("bush", position=location_m, shape_config=shape_cfg)
        return bush_mask, bush

    def create_tree(self, generator: np.random.Generator, mask: np.ndarray, location: np.ndarray, sample_config: dict) -> tuple[np.ndarray, MapObject]:
        tree_diam = self.sample_unif_range(
            generator, sample_config['diameter_range'])
        tree_height = self.sample_unif_range(
            generator, sample_config['height_range'])
        tree_leaf_diam = self.sample_unif_range(
            generator, sample_config['leaf_diameter_range'])

        rad_pixels = min(1, int(tree_diam/self.resolution))
        tree_mask = cv2.circle(
            mask, (location[1], location[0]), rad_pixels, 1, -1)
        location_m = location * self.resolution
        location_m = [location_m[1], self.width - location_m[0], 0]
        shape_cfg = TreeShapeConfig(tree_diam, tree_height, tree_leaf_diam)
        tree = MapObject("tree", position=location_m, shape_config=shape_cfg)
        return tree_mask, tree

    def create_rock(self, generator: np.random.Generator, mask: np.ndarray, location: np.ndarray, sample_config: dict) -> tuple[np.ndarray, MapObject]:
        rock_diam = self.sample_unif_range(
            generator, sample_config['diameter_range'])

        rad_pixels = min(1, int(rock_diam/self.resolution))
        rock_mask = cv2.circle(
            mask, (location[1], location[0]), rad_pixels, 1, -1)
        location_m = location * self.resolution
        location_m = [location_m[1], self.width - location_m[0], 0]
        shape_cfg = RockShapeConfig(rock_diam)
        rock = MapObject("rock", position=location_m, shape_config=shape_cfg)
        return rock_mask, rock

    def generate_uniform_obstacle(self, generator: np.random.Generator, mask_name, density, obs_name: str, obs_sampling_properties: dict, create_obstacle_fn):

        parent_mask = self.masks[mask_name]

        if obs_name not in self.masks:
            child_mask = np.zeros_like(parent_mask).astype(np.uint8)
        else:
            child_mask = self.masks[obs_name].astype(np.uint8)
        if obs_name not in self.obstacles:
            self.obstacles[obs_name] = []

        obstacle_locations = self.sample_uniform_overlapping_mask(
            parent_mask, density)
        for i in range(obstacle_locations.shape[0]):
            child_mask, map_obj = create_obstacle_fn(
                generator, child_mask, obstacle_locations[i], obs_sampling_properties)
            self.obstacles[obs_name].append(map_obj)

        child_mask = child_mask.astype(bool)
        self.masks[obs_name] = child_mask
        self.masks[mask_name][child_mask] = False

    def generate_buildings(self, generator: np.random.Generator, mask_name: str, height_range: tuple):
        # reduce buildings to rectangles
        building_mask = 255 * self.masks[mask_name].astype(np.uint8)
        ret, thresh = cv2.threshold(building_mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        heights = self.sample_unif_range(
            generator, height_range, num_samples=len(contours))

        if "buildings" not in self.obstacles:
            self.obstacles['buildings'] = []

        for i, contour in enumerate(contours):
            rect = cv2.minAreaRect(contour)
            location = rect[0]
            location_m = np.asarray(location) * self.resolution
            location_m = [location_m[0], self.width - location_m[1], 0]
            h, w = np.asarray(rect[1]) * self.resolution

            angle = -rect[2] * np.pi / 180
            if h < w:
                tmp = h
                h = w
                w = tmp
                angle += np.pi / 2
            height = heights[i]
            shape_config = BuildingShapeConfig(w, h, height, angle)
            building = MapObject(
                "building", position=location_m, shape_config=shape_config)
            self.obstacles['buildings'].append(building)

            # fill in building footprint to make it rectangular in the mask
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            building_mask = cv2.drawContours(
                building_mask, [box], 0, 1, -1)
        self.masks[mask_name] = building_mask.astype(bool)
        # remove buildings from other masks
        for name in self.masks.keys():
            if name == mask_name:
                continue
            self.masks[name][building_mask > 0] = False

        # NOTE: does not create a new layer for buildings, just adds them to the obstacle dict

    def generate_obstacles(self, config: RealEnvConfig, generator: np.random.Generator):
        # NOTE: modifies the masks (buildings). Should only be called once on a map
        # clear the obs dict
        self.obstacles = dict()
        # do buildings first, as they modify other masks
        if 'structure' in config.obstacle_configs:
            if len(config.obstacle_configs['structure']) > 1:
                raise RuntimeError(
                    "Only one building type is supported at this time")
            self.generate_buildings(generator, "structure", config.obstacle_configs['structure'][0].other_properties["height_range"])

        for mask_name, obs_list in config.obstacle_configs.items():
            if mask_name not in self.masks:
                warnings.warn(f"Warning: {mask_name} not in masks. Current masks are {self.masks.keys()}. Skipping")
                continue
            for obstacle in obs_list:
                if obstacle.name == "trees":
                    self.generate_uniform_obstacle(generator, mask_name, obstacle.density, "trees", obstacle.other_properties,
                                                   self.create_tree)
                elif obstacle.name == "bushes":
                    self.generate_uniform_obstacle(generator, mask_name, obstacle.density, "bushes", obstacle.other_properties,
                                                   self.create_bush)
                elif obstacle.name == "fallen_trees":
                    self.generate_uniform_obstacle(generator, mask_name, obstacle.density, "fallen_trees", obstacle.other_properties,
                                                   self.create_fallen_tree)
                elif obstacle.name == "rocks":
                    self.generate_uniform_obstacle(generator, mask_name, obstacle.density, "rocks", obstacle.other_properties,
                                                   self.create_rock)
                elif obstacle.name == "buildings":
                    continue
                else:
                    raise RuntimeError(
                        f"Unidentified obstacle name: {obstacle.name}")

    def generate_map(self, params: EnvConfig):
        # self.reset_map()
        raise NotImplementedError

    def plot_masks(self):
        for name, mask in self.masks.items():
            plt.figure()
            plt.imshow(mask)
            plt.title(name)
            plt.show()
        return

    def save_map(self, name: str, path: str = MAPS_DIR, overwrite=False):
        """Saves the ComplexMap class to a pickle file, and saves the floormask"""
        if os.path.exists(os.path.join(path, name)):
            if not overwrite:
                raise RuntimeError(
                    "Map with name {} already exists. Not writing the map".format(name))
            else:
                shutil.rmtree(os.path.join(path, name))

        os.mkdir(os.path.join(path, name))
        with open(os.path.join(path, name, name + '.pkl'), 'wb') as f:
            pickle.dump(self, f)

        # save floormask
        color_img = self.create_floormask(True)
        matplotlib.image.imsave(os.path.join(
            path, name, name+'.png'), color_img)
        # save informational plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(color_img, interpolation='none')
        patches = []
        for terrain, color in semantic_class_to_color_map.items():
            patches.append(mpatches.Patch(color=color, label=terrain))
        ax[0].legend(handles=patches, bbox_to_anchor=(0.5, -0.1),
                     loc='upper center', borderaxespad=0., ncol=4)
        ax[0].set_title("Color floormap")
        plot_full_range = False
        max_cost = -np.inf
        if plot_full_range:
            for terrain, cost in semantic_class_to_occ_grid_cost.items():
                if np.isinf(cost):
                    continue
                max_cost = max(max_cost, cost)
        else:
            occ_grid_no_inf = self.create_occgrid()
            occ_grid_no_inf[occ_grid_no_inf == np.inf] = 0
            max_cost = np.max(occ_grid_no_inf)
        # occ_grid = self.create_occgrid()
        img_plot = ax[1].imshow(self.create_occgrid(),
                                interpolation='none', vmin=0, vmax=max_cost)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        plt.colorbar(img_plot, cax=cbar_ax, label="seconds / meter")

        ax[1].set_title("Costmap")
        plt.savefig(os.path.join(path, name, name +
                    '_plot.png'), bbox_inches='tight', dpi=600)
        plt.close(fig)

    def get_mask_map(self):
        free = np.zeros_like(self.masks['forest'])
        not_free = np.zeros_like(self.masks['forest'])
        for mask_name in self.masks:
            if np.isinf(semantic_class_to_occ_grid_cost[mask_name]):
                not_free += self.masks[mask_name]
            else:
                free += self.masks[mask_name]
        free[not_free] = 0
        return free

    def sample_uniform_overlapping_mask(self, mask_in: np.ndarray, density: float) -> np.ndarray:
        """Samples a uniform distribution over a mask. Samples can overlap"""
        num_samples = int(self.get_mask_area(mask_in) * density)

        mask = np.copy(mask_in)
        mask = mask.astype(bool)
        idxs = np.argwhere(mask)
        if len(idxs) == 0:
            return np.array([])
        idx = idxs[np.random.randint(
            0, idxs.shape[0], size=min(idxs.shape[0], num_samples))]
        return idx
        # return np.array([idx[1], idx[0]]) * self.resolution

    def sample_rejection_mask_dialated(self, mask_in: np.ndarray, dialation_kernel_width: int, side_buffer: int, min_distance_between_samples: int, max_samples: int):
        """
        dialation_kernel_width: width of dialation kernel to use when sampling in index space 
        side_buffer: number of indexes to ignore on each side of the map 
        min_distance_between_samples: minimum distance between samples in index space 
        max_samples: maximum number of samples to sample
        Returns Nx2 array of samples in meters. 0,0 IS THE BOTTOM LEFT CORNER OF THE MAP
        """
        mask = np.copy(mask_in)
        min_idx = side_buffer
        max_idx = mask.shape[0] - side_buffer

        mask = cv2.erode(mask.astype(np.uint8), np.ones(
            (dialation_kernel_width, dialation_kernel_width))).astype(bool)
        mask = mask[min_idx:max_idx, min_idx:max_idx]

        samples = np.ones((max_samples, 2), dtype=int) * -1
        for i in range(max_samples):
            if True not in mask:
                raise RuntimeWarning(
                    "no more space for samples, mask all occupied with {} samples".format(i))
            idx = np.unravel_index(
                np.random.choice(
                    np.where(
                        mask.flatten() == True
                    )[0]), mask.shape)
            # carve out part of mask to stop identical samples
            bounds_x = [idx[0] - min_distance_between_samples,
                        idx[0] + min_distance_between_samples]
            bounds_x = np.clip(bounds_x, 0, mask.shape[0])
            bounds_y = [idx[1] - min_distance_between_samples,
                        idx[1] + min_distance_between_samples]
            bounds_y = np.clip(bounds_y, 0, mask.shape[1])
            mask[bounds_x[0]:bounds_x[1], bounds_y[0]:bounds_y[1]] = False
            idx = list(idx)
            idx[0] += min_idx
            idx[1] += min_idx
            samples[i, :] = idx
        samples = samples[samples[:, 0] != -1, :]

        # convert samples to locations in meters
        samples = samples * self.resolution + self.resolution / \
            2  # add half a resolution to center on pixel
        # flip y axis, as indexes are in image coordinate frame
        samples[:, 0] = self.width - samples[:, 0]
        # swap the axis i,j to x,y
        samples = samples[:, [1, 0]]
        return samples

    def sample_freespace_dialated(self, dialation_kernel_width: int, side_buffer: int, min_distance_between_samples: int, max_samples: int) -> np.ndarray:
        mask = self.get_mask_map()
        min_idx = side_buffer
        max_idx = mask.shape[0] - side_buffer

        mask = cv2.erode(mask.astype(np.uint8), np.ones(
            (dialation_kernel_width, dialation_kernel_width))).astype(bool)
        mask = mask[min_idx:max_idx, min_idx:max_idx]

        samples = np.ones((max_samples, 2), dtype=int) * -1
        for i in range(max_samples):
            if True not in mask:
                raise RuntimeWarning(
                    "no more space for samples, mask all occupied with {} samples".format(i))
            idx = np.unravel_index(
                np.random.choice(
                    np.where(
                        mask.flatten() == True
                    )[0]), mask.shape)
            # carve out part of mask to stop identical samples
            bounds_x = [idx[0] - min_distance_between_samples,
                        idx[0] + min_distance_between_samples]
            bounds_x = np.clip(bounds_x, 0, mask.shape[0])
            bounds_y = [idx[1] - min_distance_between_samples,
                        idx[1] + min_distance_between_samples]
            bounds_y = np.clip(bounds_y, 0, mask.shape[1])
            mask[bounds_x[0]:bounds_x[1], bounds_y[0]:bounds_y[1]] = False
            idx = list(idx)
            idx[0] += min_idx
            idx[1] += min_idx
            samples[i, :] = idx
        samples = samples[samples[:, 0] != -1, :]

        # convert samples to locations in meters
        samples = samples * self.resolution + self.resolution / \
            2  # add half a resolution to center on pixel
        # flip y axis, as indexes are in image coordinate frame
        samples[:, 0] = self.width - samples[:, 0]
        # swap the axis i,j to x,y
        samples = samples[:, [1, 0]]
        return samples

    def overlay_poses(self,
                      positions: list[tuple],
                      yaw_ranges: list[tuple],
                      radial_boundaries: list[float],
                      img=None,
                      crop_local: bool = False,
                      keep_square: bool = False,
                      return_bounds: bool = False) -> np.ndarray:
        """
        positions: list of (x,y) tuples in meters 
        yaw_ranges: list of (left, right) tuples
        radial_boundaries: list of floats for each radial bin
        """
        occ_grid = OccupancyGrid.from_complexmap(self)
        if img is None:
            img = self.create_floormask()
            img = (255 * img).astype(np.uint8)
        min_idxs = [np.inf, np.inf]
        max_idxs = [-np.inf, -np.inf]

        for position, yaw_range in zip(positions, yaw_ranges):
            if crop_local:
                img, min_idx, max_idx = self.overlay_pose(occ_grid, img, position, yaw_range, radial_boundaries, crop_local=False, return_idxs=True, keep_square=keep_square)
                min_idxs = np.minimum(min_idxs, min_idx)
                max_idxs = np.maximum(max_idxs, max_idx)
            else:
                img = self.overlay_pose(occ_grid, img, position, yaw_range, radial_boundaries)
        if crop_local:
            bounds = [
                [int(np.clip(min_idxs[0], 0, img.shape[0]-1)), int(np.clip(max_idxs[0], 0, img.shape[0]-1))],
                [int(np.clip(min_idxs[1], 0, img.shape[1]-1)), int(np.clip(max_idxs[1], 0, img.shape[1]-1))]
            ]
            if keep_square:
                larger_delta = np.max(np.abs(np.array(max_idxs) - np.array(min_idxs)))
                center = np.mean([min_idxs, max_idxs], axis=0)
                bounds = [
                    [int(np.clip(center[0] - larger_delta/2, 0, img.shape[0]-1)), int(np.clip(center[0] + larger_delta/2, 0, img.shape[0]-1))],
                    [int(np.clip(center[1] - larger_delta/2, 0, img.shape[1]-1)), int(np.clip(center[1] + larger_delta/2, 0, img.shape[1]-1))]
                ]
            img = img[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]]
        if return_bounds and crop_local:
            return img, bounds
        else:
            return img

    @staticmethod
    def overlay_pose(occ_grid,
                     in_img: np.ndarray,
                     position: tuple,
                     yaw_range: tuple,
                     radial_boundaries: list[float],
                     bin_color: tuple = (0, 255, 0),
                     bin_thickness: int = 2,
                     crop_local: bool = False,
                     return_idxs: bool = False,
                     keep_square: bool = False,
                     ) -> np.ndarray:
        """
        position: (x,y) tuple in meters 
        yaw_range: (left, right) tuple
        """
        img = np.copy(in_img)
        # draw robot position/orientation as an arrow
        min_idxs = [np.inf, np.inf]
        max_idxs = [-np.inf, -np.inf]
        arrow_scale = 20
        arrow_color = (0, 0, 255)
        arrow_thickness = 5
        max_radius_pixels = int(radial_boundaries[-1] / occ_grid.resolution)
        padding = max_radius_pixels + 10
        angles = [wrap_angle_2pi(x) for x in yaw_range]
        delta_angle = np.abs(np.diff(yaw_range)[0])
        center_yaw = angles[1] + (angles[0] - angles[1]) / 2
        if angles[0] < angles[1]:  # if left is larger than right, we pass through 0
            delta_angle = (2*np.pi - angles[1]) + angles[0]
            center_yaw = wrap_angle_2pi(angles[1] + delta_angle / 2)

        i, j = occ_grid.get_index_from_point(position)
        start_point = [int(i + np.sin(center_yaw) * arrow_scale / 2), int(j - np.cos(center_yaw) * arrow_scale / 2)]
        end_point = [int(i - np.sin(center_yaw) * arrow_scale / 2), int(j + np.cos(center_yaw) * arrow_scale / 2)]
        img = cv2.arrowedLine(img, tuple(reversed(start_point)), tuple(reversed(end_point)), arrow_color, arrow_thickness, tipLength=0.5)

        if return_idxs or crop_local:
            min_idxs, max_idxs = find_map_bounds_around_point(occ_grid, (i, j), padding, convert_position_to_index=False, keep_square=keep_square)
            min_idxs, max_idxs = find_map_bounds_around_point(occ_grid, start_point, padding, convert_position_to_index=False,
                                                              existing_min_idxs=min_idxs, existing_max_idxs=max_idxs, keep_square=keep_square)
            # min_idxs, max_idxs = find_map_bounds_around_point(occ_grid, end_point, padding, convert_position_to_index=False, existing_min_idxs=min_idxs, existing_max_idxs=max_idxs, keep_square=keep_square)
            # print(3, min_idxs, max_idxs)

        # draw arcs for radial_boundaries
        for r in radial_boundaries:
            radius_pixels = int(r / occ_grid.resolution)
            img = cv2.ellipse(img, (j, i), (radius_pixels, radius_pixels),
                              -(angles[1] * 180 / np.pi),
                              0,
                              -(delta_angle * 180 / np.pi),
                              bin_color,
                              bin_thickness)

        # draw lines for radial_boundaries
        left_line_end = [int(i - np.sin(yaw_range[0]) * max_radius_pixels), int(j + np.cos(yaw_range[0]) * max_radius_pixels)]
        img = cv2.line(img, (j, i), tuple(reversed(left_line_end)), bin_color, bin_thickness)
        right_line_end = [int(i - np.sin(yaw_range[1]) * max_radius_pixels), int(j + np.cos(yaw_range[1]) * max_radius_pixels)]
        img = cv2.line(img, (j, i), tuple(reversed(right_line_end)), bin_color, bin_thickness)

        if return_idxs or crop_local:
            min_idxs, max_idxs = find_map_bounds_around_point(occ_grid, left_line_end, padding, convert_position_to_index=False,
                                                              existing_min_idxs=min_idxs, existing_max_idxs=max_idxs, keep_square=keep_square)
            min_idxs, max_idxs = find_map_bounds_around_point(occ_grid, right_line_end, padding, convert_position_to_index=False,
                                                              existing_min_idxs=min_idxs, existing_max_idxs=max_idxs, keep_square=keep_square)
            if crop_local:
                img = img[
                    min_idxs[0]:max_idxs[0],
                    min_idxs[1]:max_idxs[1]
                ]
            if return_idxs:
                return img, min_idxs, max_idxs

        return img

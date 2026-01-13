import numpy as np
from dataclasses import dataclass
import enum


@dataclass
class TreeShapeConfig:
    diameter: float  # diameter of the tree
    height: float = 0.0  # height of the tree (only for 3d)
    leaf_diameter: float = 0.0  # diameter of the leaves (only for 3d)


@dataclass
class RockShapeConfig:
    diameter: float  # diam in meters


@dataclass
class BushShapeConfig:
    diameter: float = 0.0
    height: float = 0.0  # only for 3d


@dataclass
class FallenTreeShapeConfig:
    diameter: float
    height: float
    angle: float  # in radians


@dataclass
class BuildingShapeConfig:
    width: float
    height: float
    depth: float
    angle: float


class ShapeConfigs(enum.Enum):
    tree = TreeShapeConfig
    rock = RockShapeConfig
    bush = BushShapeConfig
    fallen_tree = FallenTreeShapeConfig
    building = BuildingShapeConfig


@dataclass
class MapObject:
    name: str
    position: np.ndarray  # (x,y) specifies centroid, (z) specifies height of the bottom of the object
    shape_config: TreeShapeConfig

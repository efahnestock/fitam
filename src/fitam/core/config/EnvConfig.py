from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvConfig:
    name: str = "unnamed_biome"


@dataclass
class ObstacleConfig:
    name: str  # name of the obstacle
    density: float  # number of obstacles per square meter of terrain
    self_overlap: bool  # if true, can overlap with itself
    other_properties: dict  # other properties of the obstacle
    # for tree-> {"height_range", "diameter_range", "leaf_diameter_range"}
    # for bush-> {"diameter_range", "height_range"}
    # for rock-> {"diameter_range"}
    # for fallen tree-> {"diameter_range", "height_range"}
    # for building-> {"height_range"}


@dataclass
class RealEnvConfig(EnvConfig):

    obstacle_configs: dict[str, list[ObstacleConfig]] = None  # obstacles in this biome

    def __post_init__(self):
        # define a default obstacle config if none is provided
        if self.obstacle_configs is None:
            self.obstacle_configs = {
                'young_forest': [
                    ObstacleConfig("trees", 0.001, False, {"height_range": (5, 15), "diameter_range": (0.7, 1.0), "leaf_diameter_range": (4, 8)}),
                    ObstacleConfig("bushes", 0.0001, False, {"diameter_range": (0.9, 2.5), "height_range": (1, 2)}),
                    ObstacleConfig("fallen_trees", 0.0005, True, {"diameter_range": (1.0, 1.0), "height_range": (10, 10)}),
                ],
                'forest': [
                    ObstacleConfig("trees", 0.005, False, {"height_range": (10, 25), "diameter_range": (0.7, 1.0), "leaf_diameter_range": (4, 8)}),
                    ObstacleConfig("bushes", 0.0005, False, {"diameter_range": (0.9, 2.5), "height_range": (1, 2)}),
                    ObstacleConfig("fallen_trees", 0.001, True, {"diameter_range": (1.0, 1.0), "height_range": (10, 10)}),  # all fixed so can render
                ],
                'grass': [
                    ObstacleConfig("rocks", 0.0005, False, {"diameter_range": (0.5, 1.5)}),
                ],
                'field': [
                    ObstacleConfig("rocks", 0.0005, False, {"diameter_range": (0.5, 1.5)}),
                ],
                'structure': [
                    ObstacleConfig("buildings", 1, False, {"height_range": (5, 15)}),
                ],
                'marsh': [
                    ObstacleConfig("fallen_trees", 0.0005, True, {"diameter_range": (1, 1), "height_range": (10, 10)}),
                ],

            }

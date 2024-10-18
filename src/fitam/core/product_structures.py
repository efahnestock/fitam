from __future__ import annotations
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import os
from fitam.mapping.land_cover_complex_map import LandCoverComplexMap


@dataclass
class ImageGenerationRequest():
    ###
    # Reference frame: 0,0 is center of the map. Positive x is right, positive y is up.
    ###
    locations: list  # list of (x,y) tuples in meters
    angles: list  # list of angles in radians to sample images at, for each location

    # allows tracking of complex map to
    map_idx: list[int] = field(default_factory=list)  # for each location, index of the complex map that it came from

    complex_map_paths: list[str] = field(default_factory=list)
    times: list[float] = field(default_factory=list)  # list of times in seconds to sample images at, for each location

    def save_all_image_request_figures(self, img_req_path: os.PathLike, trial_dir: str, point_width: int = 3) -> None:
        """
        Generate figures displaying the sample locations in each map
        """
        for i in range(len(self.complex_map_paths)):
            ax = self.plot_one_map(i, trial_dir, point_width)
            plt.savefig(img_req_path.parent / f'{img_req_path.name}_map_{i}.png')
            plt.close()
        return

    @classmethod
    def from_old_img_request(cls, img_request: "ImageGenerationRequest"):
        locations = [img_request.locations[i] for i in range(len(img_request.locations))]
        angles = [img_request.angles[i] for i in range(len(img_request.angles))]
        map_idx = [img_request.map_idx[i] for i in range(len(img_request.map_idx))]
        return cls(locations, angles, map_idx, img_request.complex_map_paths)

    def plot_one_map(self, map_idx: int, trial_dir: str, point_width: int = 3) -> plt.Axes:
        cmap = LandCoverComplexMap.from_map_folder(os.path.join(trial_dir, self.complex_map_paths[map_idx]))
        background_img = cmap.create_floormask()

        fig, ax = plt.subplots()
        points_in_map = [self.locations[i] for i in range(len(self.locations)) if self.map_idx[i] == map_idx]

        # for each point, plot a circle
        # point is (x,y) in meters
        # i is y axis, with 0 at the top
        # j is x axis, with 0 at the left
        # plt.Circle takes x,y in pixels, with 0,0 at the top left
        for point in points_in_map:
            i = int(cmap.width_pixels - point[1] / cmap.resolution)
            j = int(point[0] / cmap.resolution)
            ax.add_patch(plt.Circle((j, i), point_width, color='r'))

        ax.imshow(background_img)

        return ax


@dataclass
class EvaluationRequest():
    points: list  # list of ((x_start,y_start), (x_goal, y_goal)) tuples in meters, 0,0 is center of the map
    map_path: os.PathLike  # path to the complex map used to generate this request. If relative, relative to MAPS_DIR

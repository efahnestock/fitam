from math import floor
from typing import Tuple
import xml.etree.ElementTree as ET
from fitam.planning.planner_general import State
import numpy as np
import cv2


class OccupancyGrid:
    header: dict  # {seq: 0, stamp: {secs: 0, nsecs: 0}, frame_id: 'map'}
    # {map_load_time: {secs, nsecs}, resolution, width, height, origin: {position: {x, y, z}, orientation: {x, y, z, w}}
    info: dict
    data: np.ndarray  # float

    """
    Origin is usually set to be the center of the map (0,0), y up, x right

    NOTE: data is of shape (height, width)
    """

    def __init__(self, resolution: float, width: int, height: int, origin: dict = None, frame_id: str = 'map', data: np.ndarray = None):
        if data is None:
            self.data = np.zeros((height, width), dtype=float)
        else:
            self.data = data

        self.header = dict(
            seq=0,
            stamp=dict(secs=0, nsecs=0),
            frame_id=frame_id
        )

        if origin is None:
            origin = dict(
                position=dict(x=-resolution * width / 2, y=-
                              resolution * height / 2, z=0),
                orientation=dict(x=0, y=0, z=0, w=1)
            )
        self.info = dict(
            map_load_time=dict(secs=0, nsecs=0),
            resolution=resolution,
            width=width,
            height=height,
            origin=origin
        )

    @classmethod
    def from_costmap(cls, other: 'OccupancyGrid', copy_data: bool = True):
        """
        Create a new OccupancyGrid from an existing costmap
        args:
          other: OccupancyGrid to copy from
        returns:
          new OccupancyGrid
        """
        data = other.data.copy() if copy_data else None
        return cls(
            resolution=other.info['resolution'],
            width=other.info['width'],
            height=other.info['height'],
            origin=other.info['origin'],
            frame_id=other.header['frame_id'],
            data=data
        )

    @classmethod
    def from_msg_parsed_yml(cls, msg_parsed_yaml: dict):
        # NOTE: we don't transfer over the header timestamps
        header = msg_parsed_yaml['header']
        info = msg_parsed_yaml['info']

        data = np.array(msg_parsed_yaml['data'], dtype=float)
        data = data.reshape(info['height'], info['width'])
        # data is from bottom left corner to top right corner
        data = np.flipud(data)

        return cls(info['resolution'], info['width'], info['height'], info['origin'], header['frame_id'], data)

    # NOTE: averages the channels if it is a color image

    @classmethod
    def from_png(cls, image_path, resolution):
        img = cv2.imread(image_path)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)

        img = (255 - img) / 255.0  # invert the image, white is freespace
        img = img.astype(float)

        return cls(resolution, img.shape[1], img.shape[0], data=img)

    @classmethod
    def from_np_array(cls, resolution: float, arr: np.ndarray):
        """
        Create a new OccupancyGrid from a numpy array
        args:
          resolution: float of the resolution of the grid
          arr: np.ndarray of shape (height, width)
        returns:
          new OccupancyGrid
        """
        return cls(resolution, arr.shape[1], arr.shape[0], data=arr)

    @classmethod
    def from_complexmap(cls, other: 'ComplexMap'):
        """
        Create a new OccupancyGrid from an existing ComplexMap
        args:
          other: ComplexMap to copy from
        returns:
          new OccupancyGrid
        """
        data = other.create_occgrid()
        return cls(resolution=other.resolution, width=data.shape[1], height=data.shape[0], data=data)

    @property
    def width_m(self):
        return self.info['width'] * self.info['resolution']

    @property
    def height_m(self):
        return self.info['height'] * self.info['resolution']

    @property
    def x_lim_meters(self):
        return (self.info['origin']['position']['x'], self.info['origin']['position']['x'] + self.width_m)

    @property
    def y_lim_meters(self):
        return (self.info['origin']['position']['y'], self.info['origin']['position']['y'] + self.height_m)

    @property
    def resolution(self):
        """Meters per pixel"""
        return self.info['resolution']

    def __str__(self):
        return f'OccupancyGrid(x:{self.x_lim_meters}, y:{self.y_lim_meters},resolution:{self.info["resolution"]})'

    def get(self, i, j):
        return self.data[i, j]

    def get_full_costmap(self):
        """
        Get the full costmap
        returns:
          costmap: np.ndarray of shape (height, width)
        """
        return np.copy(self.data)

    def get_color_image(self, max_value: float = None):
        """
        Get a color image of the costmap.
        The color image is in RGB format.

        returns:
          color_image: np.ndarray of shape (height, width, 3)
        """
        if max_value is None:
            max_value = np.max(self.data)
            if abs(max_value) < 1e-6:
                max_value = 1
            if np.isinf(max_value) or np.isnan(max_value):
                max_value = 1
        color_image = 1.0 - \
            (np.stack([self.data, self.data, self.data], axis=-1) / max_value)
        color_image = np.clip(color_image, 0, 1)
        return color_image.astype(float)

    def get_index_from_point(self, point: tuple):
        """
        Get the index of the cell in the costmap that corresponds to the given point.
        Remember that the costmap is stored in row major order, and thus is indexed by (i (y), j (x))

        REMEMBER: the cell i,j spans from x_lim_meters[0] + (j-0.5) * resolution TO x_lim_meters[0] + (j+0.5) * resolution
                                          y_lim_meters[1] - (i-0.5) * resolution TO y_lim_meters[1] - (i+0.5) * resolution

        args:
          point: (x,y) tuple of the point in the frame of self.header['frame_id']

        returns:
          (i,j) tuple of the index of the cell in the costmap that corresponds to the given point
        """
        if (point[0] < self.x_lim_meters[0] or point[0] > self.x_lim_meters[1]) \
                or (point[1] < self.y_lim_meters[0] or point[1] > self.y_lim_meters[1]):
            raise ValueError(f'Point {point} is not in the costmap with x limits: {self.x_lim_meters} and y limits: {self.y_lim_meters}')

        j_index = int(
            floor(
                (point[0] - self.x_lim_meters[0]) / self.width_m * self.info['width']
            + 0.5)
        )
        i_index = int(
            floor(
                (self.y_lim_meters[1] - point[1]) / self.height_m * self.info['height'] 
            + 0.5)
        )
        assert j_index >= 0 and j_index < self.info['width']
        assert i_index >= 0 and i_index < self.info['height']
        return (i_index, j_index)

    def get_value(self, point: Tuple[float, float]) -> float:
        """
        Get the value of the costmap at a given point (x,y, in meters)
        args:
          point: Tuple[float,float] of the point in the frame of self.header['frame_id']
        returns:
          float value of the costmap at the given point
        Throws:
          ValueError if the point is not in the costmap
        """
        index = self.get_index_from_point(point)
        return self.data[index[0], index[1]]

    def get_point_from_index(self, index: tuple):
        """
        Get the point (x,y) in meters from an index into the costmap

        args:
          index: (i,j) tuple of the index into the costmap

        returns:
          (x,y) tuple of the point in the frame of self.header['frame_id']
        """
        x = float(self.x_lim_meters[0] + index[1] * self.info['resolution'])
        y = float(self.y_lim_meters[1] - index[0] * self.info['resolution'])

        return (x, y)

    def set_unobserved_as_freespace(self):
        self.data[self.data == -1] = 0

    def check_index_in(self, idx: Tuple[int, int]) -> bool:
        """
        Check if an index is in the costmap 
        """
        return 0 <= idx[0] < self.info['height'] and 0 <= idx[1] < self.info['width']

    def check_state_in(self, state: Tuple[float, float]) -> bool:
        """
        Check if the given state is in the costmap
        args:
          state: (x,y) tuple of the state in the frame of self.header['frame_id']
        returns:
          True if the state is in the costmap, False otherwise
        """
        return self.x_lim_meters[0] <= state[0] <= self.x_lim_meters[1] and self.y_lim_meters[0] <= state[1] <= self.y_lim_meters[1]

    def sample_point(self):
        """
        Sample a point that lies within the costmap

        returns:
          (x,y) tuple of the sampled point, where x,y are in meters in the frame of self.header['frame_id']
        """
        point = np.random.rand(2)
        point[0] *= self.width_m
        point[0] += self.info['origin']['position']['x']
        point[1] *= self.height_m
        point[1] += self.info['origin']['position']['y']
        return point

    def add_observation(self, state: State, angles: np.ndarray, ranges: np.ndarray):
        """
        Add an observation to the costmap
        args:
          state: (x,y) tuple of the state in the frame of self.header['frame_id']
          angles: np.ndarray of angles in radians
          ranges: np.ndarray of ranges in meters
        """
        raise NotImplementedError()

    def get_visibility_mask(self, trajectory: list[State], swath_library:"SwathLibrary")->np.ndarray:

        vis_mask = np.zeros_like(self.data, dtype=bool)
        for state in trajectory:
            idx = self.get_index_from_point((state.x, state.y))
            swath = swath_library.get_observation_indices(idx, self.data.shape)
            vis_mask[swath] = True
        return vis_mask


    def get_freespace(self):
        """
        Get a binary mask where 1 indicates freespace and 0 indicates occupied
        Currently we consider all non inf values to be freespace
        """
        freespace = np.zeros(self.data.shape, dtype=bool)
        freespace[~np.isinf(self.data)] = True
        return freespace

    def sample_free_space_idx(self):
        """
        Sample an index that lies in free space (i.e. is not occupied)

        returns:
          (i,j) tuple of the sampled index, where i,j are the indices into the costmap
        """
        freespace = self.get_freespace()
        return np.unravel_index(np.random.choice(np.where(freespace.flatten())[0]), self.data.shape)

    def sample_region_dialated(self, dialation_radius: float, i_slice: slice, j_slice: slice):
        """
        Sample an index that lies in free space (i.e. is not occupied). Only samples in a region of the map

        @arg dialation_radius: float, radius of the dialation in meters
        @arg i_slice: slice, slice of the indices to sample in the i dimension
        @arg j_slice: slice, slice of the indices to sample in the j dimension

        returns:
          (i,j) tuple of the sampled index, where i,j are the indices into the costmap
        """
        kernel = np.ones((int(dialation_radius / self.resolution),
                         int(dialation_radius / self.resolution)), dtype=float)
        freespace = self.get_freespace()
        freespace = freespace.astype(float)
        idx = np.unravel_index(
            np.random.choice(
                np.where(
                    cv2.erode(freespace[i_slice, j_slice].astype(
                        float), kernel, iterations=1).flatten() == 1
                )[0]), self.data[i_slice, j_slice].shape)

        # add the offset for the part of the map we ignored due to the distance from wall
        return (idx[0] + i_slice.start, idx[1] + j_slice.start)

    def sample_free_space_dialated(self, dialation_radius: float = 1.0, distance_from_wall: float = 0.0):
        """
        Sample an index that lies in free space (i.e. is not occupied). 

        @arg dialation_radius: float, radius of the dialation in meters

        returns:
          (i,j) tuple of the sampled index, where i,j are the indices into the costmap
        """
        if int(distance_from_wall / self.resolution) >= 1:
            smaller_slice = slice(int(
                distance_from_wall / self.resolution), -int((distance_from_wall) / self.resolution))
        else:
            smaller_slice = slice(0, -1)

        return self.sample_region_dialated(dialation_radius, smaller_slice, smaller_slice)

    def sample_start_goal_pairs(self, num_pairs: int, min_path_length_m: float, planner, dialated_radius_m: float = 1.0, distance_from_wall_m: float = 0.0, max_num_tries=1e6):
        """
        Sample start goal pairs from the costmap.
        Use rejection sampling to make sure the following conditions are met
          - the start and end points are in free space 
          - the start and end points are not too close to each other
          - a valid path exists between the start and end points

        args:
          num_pairs: number of points to sample
          min_path_length: minimum distance in meters between start and end points
        """

        resulting_pairs = []
        num_tries = 0
        while True:
            start_idx = self.sample_free_space_dialated(
                dialated_radius_m, distance_from_wall_m)
            start_pt = self.get_point_from_index(start_idx)
            start_state = State(start_pt[0], start_pt[1])

            end_idx = self.sample_free_space_dialated(
                dialated_radius_m, distance_from_wall_m)
            end_pt = self.get_point_from_index(end_idx)
            end_state = State(end_pt[0], end_pt[1])
            if np.linalg.norm(np.asarray(start_pt) - np.asarray(end_pt)) > min_path_length_m:
                # check that a valid path exists
                # print(f"Checking path from {start_state} to {end_state}")
                path, costs = planner.search(
                    start_state, end_state, allow_failure=False)
                if path is not None:  # if a valid path exists
                    resulting_pairs.append((start_pt, end_pt))
                    # print(f"Found valid start goal pair {len(resulting_pairs)}/{num_pairs}")
                    num_tries = 0
                else:
                    pass
                    # print(f"Failed to find valid path from {start_state} to {end_state}")
                    # print("Dumping arguments to search")
                    # with open('/tmp/args.pkl', 'wb') as f:
                    #   pickle.dump((self, planner, start_state,end_state, path), f)

            if len(resulting_pairs) >= num_pairs:
                break
            num_tries += 1
            if num_tries > max_num_tries:
                raise RuntimeError(
                    f"Could not sample enough valid start goal pairs. Found {len(resulting_pairs)} before failure")

        return resulting_pairs

    def visualize_states(self, states: list, color: tuple = (1.0, 0, 0), n_pixels=2, img:np.ndarray=None, states_are_idxs:bool = False) -> np.ndarray:
        """
        Visualize the given states in the costmap
        args:
          states: list of states to visualize
          color: color to use for visualization
        """
        if img is None:
            color_img = self.get_color_image()
        else:
            color_img = img.copy()
        for state in states:
            if states_are_idxs:
                idx = (state[0], state[1])
            else:
                idx = self.get_index_from_point((state.x, state.y))
            start_idx = np.clip(
                [idx[0] - n_pixels, idx[0] + n_pixels], 0, self.data.shape[0])
            end_idx = np.clip([idx[1] - n_pixels, idx[1] +
                              n_pixels], 0, self.data.shape[1])
            color_img[start_idx[0]:start_idx[1], end_idx[0]:end_idx[1]] = color

        return color_img

    def expand(self, num_cells: int) -> None:
        """
        Expand the costmap by adding num_cells cells of freespace to all sides of the costmap

        """
        new_data = np.zeros(
            (self.data.shape[0] + 2 * num_cells, self.data.shape[1] + 2 * num_cells))
        new_data[num_cells:-num_cells, num_cells:-num_cells] = self.data
        self.data = new_data
        self.info['width'] += 2 * num_cells
        self.info['height'] += 2 * num_cells
        self.info['origin']['position']['x'] -= num_cells * \
            self.info['resolution']
        self.info['origin']['position']['y'] -= num_cells * \
            self.info['resolution']

    def to_easl_costmap_str(self):
        """
        returns: the xml string readable by easl
        """
        xml = self.to_easl_costmap_xml()
        return ET.tostring(xml, encoding='unicode')

    def to_easl_costmap_xml(self):
        """
        Dump the current costmap to an XML file readable by EASL

        returns: the xml string 
        """
        true_root = ET.Element('root')
        return self.to_easl_costmap_xml(true_root)

    def to_easl_costmap_xml(self, true_root: ET.Element):
        root = ET.SubElement(true_root, 'cost-map')
        root.set('min-x', str(self.x_lim_meters[0]))
        root.set('min-y', str(self.y_lim_meters[0]))
        root.set('resolution', str(self.info['resolution']))
        root.set('width', str(self.info['width']))
        root.set('height', str(self.info['height']))

        cells = ET.SubElement(root, 'cells')
        occ_grid_data_format = np.flipud(self.data)
        occ_grid_data_format *= 255
        occ_grid_data_format = occ_grid_data_format.astype(np.uint8)
        for r in range(occ_grid_data_format.shape[0]):
            for c in range(occ_grid_data_format.shape[1]):
                if occ_grid_data_format[r, c] == 0:
                    continue
                cell = ET.SubElement(cells, 'cell')
                cell.set('c', str(c))
                cell.set('r', str(r))
                cell.set('value', str(occ_grid_data_format[r, c]))

        return true_root

    @classmethod
    def from_xml_file(cls, xml_file: str):
        """
        Load a costmap from an xml file

        args:
          xml_file: str, path to the xml file
        """
        root = ET.parse(xml_file).getroot()
        return cls.from_xml_root(root)

    @classmethod
    def from_xml_root(cls, costmap_root: ET.Element):
        """
        Load a costmap from an xml root

        args:
          costmap_root: ET.Element, root of the xml tree
        """
        min_x = float(costmap_root.get('min-x'))
        min_y = float(costmap_root.get('min-y'))
        resolution = float(costmap_root.get('resolution'))
        width = int(costmap_root.get('width'))
        height = int(costmap_root.get('height'))
        origin = dict(
            position=dict(
                x=min_x,
                y=min_y,
                z=0.0,
            ),
            orientation=dict(
                x=0.0, y=0.0, z=0.0, w=1.0
            )
        )
        costmap = cls(resolution, width, height, origin=origin)
        cells = costmap_root.find('cells')
        for cell in cells:
            c = int(cell.get('c'))
            r = int(cell.get('r'))
            value = int(cell.get('value'))
            costmap.data[r, c] = value
        return costmap


if __name__ == "__main__":
    r = 0.2
    size = 1000
    c = OccupancyGrid(r, size, size)
    print(f"created occupancy grid {c}")

    disturbances = np.linspace(-r / 2 + 1e-8, r / 2 - 1e-8, 10)

    for d in disturbances:
        for d2 in disturbances:
            i, j = c.get_index_from_point((d, d2))
            assert i == 500 and j == 500
            # check going back the other way
            x, y = c.get_point_from_index((i, j))
            assert np.abs(x - 0) < 1e-8 and np.abs(y - 0) < 1e-8

    # check index stuff

    max_x = c.x_lim_meters[1]
    try:
        point = c.get_index_from_point((max_x+0.1, 0))
        print()
        print("ERROR: should have thrown an error")
    except ValueError:
        pass
    point = c.get_index_from_point((max_x - 0.1, 0))
    print("max_x point produces index of ",
          point[1], " which should not exceed ", c.data.shape[1]-1)
    assert point[1] < c.data.shape[1]
    min_x = c.x_lim_meters[0]
    point = c.get_index_from_point((min_x+0.1, 0))
    print("min_x point produces index of ",
          point[1], " which should not be negative")
    assert point[1] >= 0

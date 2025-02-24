from __future__ import annotations
import vtkmodules.vtkRenderingOpenGL2 as roglvtk
import vtkmodules.vtkRenderingCore as rcvtk
from fury import actor, window, utils, io
import numpy as np
from fitam.mapping.land_cover_complex_map import LandCoverComplexMap, semantic_class_to_color_map
from fitam.mapping.obstacle_types import MapObject
import os
from fury.lib import RenderWindow, numpy_support, WindowToImageFilter

AMBIENT = 0.2
DIFFUSE = 0.3
SPECULAR = 0.0
# img_dims = (4 * 4096, 2 * 4096)
# img_dims = (4 * 2048, 2 * 2048)
# img_dims = (4 * 1024, 2 * 1024)
# img_dims = (4 * 512, 2 * 512)
IMG_HEIGHT_DEG = 10
IMG_HEIGHT_RESOLUTION = 128

# at 10 deg, 256 height, 1536 side res, about 500mb gpu memory per process, around 0.01 second per image
# roughly, doubling height doubles gpu memory

img_dims = (int(IMG_HEIGHT_RESOLUTION * 360 / IMG_HEIGHT_DEG), IMG_HEIGHT_RESOLUTION)

class NonLeakingScene(window.Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_to_image_filter = None
        self.render_window = None
    def __del__(self):
        del self.window_to_image_filter
        del self.render_window

def set_lighting_props(obj, amb, dif, spec):
    obj.GetProperty().SetAmbient(amb)
    obj.GetProperty().SetDiffuse(dif)
    obj.GetProperty().SetSpecular(spec)

def create_scene(map: LandCoverComplexMap, map_floormask_filepath) -> window.Scene:
    """
    Create a scene for the given map
    """
    scene = NonLeakingScene()
    attach_render_window(scene, img_dims)
    enable_pano_pass(scene)
    scene.SetAutomaticLightCreation(False)
    scene.SetNearClippingPlaneTolerance(0.0001)
    scene.RemoveAllLights()
    for x in np.linspace(-map.width/2, map.width/2, 10):
        for y in np.linspace(-map.width/2, map.width/2, 10):
            light = rcvtk.vtkLight()
            light.SetPosition(x, y, 200)
            light.SetFocalPoint(x + (np.random.rand()*2 -1) * 100, y+ (np.random.rand()*2 -1) * 100, 0)
            light.SetIntensity(0.1)
            light.SetConeAngle(180)
            light.SetLightTypeToSceneLight()
            scene.AddLight(light)
    scene.background((0.529, 0.808, 0.922))
    ground = create_ground(map_floormask_filepath)
    set_lighting_props(ground, 0.3, 0.1, SPECULAR)
    scene.add(ground)
    for key, value in map.obstacles.items():
        if len(value) == 0: # dont try to generate empty obstacles
            continue
        if key == "trees":
            for obj in create_trees(map, value):
                set_lighting_props(obj, AMBIENT, DIFFUSE, SPECULAR)
                scene.add(obj)
        elif key == "rocks":
            obj = create_spheres(
                map, value, semantic_class_to_color_map['rocks'], in_ground=True)
            set_lighting_props(obj, 0.4, 0.05, SPECULAR)
            scene.add(obj)
        elif key == "bushes":
            obj = create_spheres(map, value, semantic_class_to_color_map['bushes'])
            set_lighting_props(obj, AMBIENT, DIFFUSE, SPECULAR)
            scene.add(obj)
        elif key == "fallen_trees":
            obj = create_fallen_trees(map, value)
            set_lighting_props(obj, AMBIENT, DIFFUSE, SPECULAR)
            scene.add(obj)
        elif key == "buildings":
            obj = create_buildings(map, value, semantic_class_to_color_map['structure'])
            set_lighting_props(obj, AMBIENT, DIFFUSE, SPECULAR)
            scene.add(obj)
        else:
            raise ValueError(f"Unknown obstacle type {key}")
    return scene

def enable_pano_pass(scene: window.Scene):

    panorama_mapper = roglvtk.vtkPanoramicProjectionPass()
    panorama_mapper.SetAngle(360.0)
    panorama_mapper.SetvFOV(IMG_HEIGHT_DEG)
    panorama_mapper.SetCubeResolution(1536)
    # panorama_mapper.SetGlobalWarningDisplay(1)
    pass_sequence = roglvtk.vtkSequencePass()
    pass_sequence_passes = roglvtk.vtkRenderPassCollection()
    light_pass = roglvtk.vtkLightsPass()
    default_pass = roglvtk.vtkDefaultPass()
    pass_sequence_passes.AddItem(light_pass)
    pass_sequence_passes.AddItem(default_pass)
    pass_sequence.SetPasses(pass_sequence_passes)

    camera_pass = roglvtk.vtkCameraPass()
    camera_pass.SetDelegatePass(pass_sequence)

    panorama_mapper.SetDelegatePass(camera_pass)
    scene.UseDepthPeelingOff()
    scene.GetCullers().RemoveAllItems()
    scene.SetPass(panorama_mapper)



def create_ground(image_filename: str):
    image = io.load_image(str(image_filename))
    image = np.flipud(image)
    # loads image, 1 unit per pixel, center is center of the image
    ground_plane_actor = actor.texture(image)
    ground_plane_actor.GetProperty().SetDiffuse(1)
    return ground_plane_actor


def create_trees(map: LandCoverComplexMap, tree_objs: list[MapObject]):
    locations = np.array([obj.position for obj in tree_objs])
    locations -= np.array([map.width / 2, map.width / 2, 0])
    locations -= np.array([-0.5, 0.5, 0.0]) # shift to deal with slight shift in ground texture
    n = locations.shape[0]
    dirs = np.repeat(np.array([0, 0, 1]).reshape(1, -1), n, axis=0)
    colors = np.repeat(np.array([0.1, 0.8, 0.1]).reshape(1, -1), n, axis=0)
    heights = np.array([obj.shape_config.height / 2.5 for obj in tree_objs])
    locations[:, 2] += heights / 2 + np.random.rand(n) * 1 + 1
    leaves = actor.cone(locations, dirs, colors, heights)
    leaves.SetScale(1, 1, 2.5)
    leaves.SetPosition(0, 0, 0)

    trunk_locations = np.stack(
        [locations[:, 0], locations[:, 2], -locations[:, 1]], axis=1)
    locations -= np.array([-0.5, 0.0, -0.5]) # shift to deal with slight shift in ground texture
    trunk_locations[:, 1] = 2.5
    trunk_dirs = np.repeat(np.array([1, 0, 0]).reshape(1, -1), n, axis=0)
    trunk_colors = np.repeat(
        np.array([0.8, 0.2, 0.2]).reshape(1, -1), n, axis=0)
    trunks = actor.cylinder(trunk_locations, trunk_dirs,
                            trunk_colors, radius=0.1, heights=5)
    trunks.SetOrientation(90, 0, 0)
    return [leaves, trunks]


def create_spheres(map: LandCoverComplexMap, map_objs: list[MapObject], color, in_ground=False) -> actor.sphere:
    locations = np.array([obj.position for obj in map_objs])
    locations -= np.array([map.width / 2, map.width / 2, 0])
    locations -= np.array([-0.5, 0.5, 0.0]) # shift to deal with slight shift in ground texture
    n = locations.shape[0]
    radii = np.array(
        [obj.shape_config.diameter for obj in map_objs]).reshape(n) / 2
    if not in_ground:
        locations[:, 2] += radii
    colors = np.repeat(color.reshape(1, -1), n, axis=0)
    return actor.sphere(locations, colors, radii)


def create_buildings(map: LandCoverComplexMap, building_objs: list[MapObject], color: tuple):
    locations = np.array([obj.position for obj in building_objs])
    locations -= np.array([map.width / 2, map.width / 2, 0])
    locations -= np.array([-0.5, 0.5, 0.0]) # shift to deal with slight shift in ground texture
    n = locations.shape[0]
    scales = np.array([[max(1,obj.shape_config.height-1.5), max(obj.shape_config.width-1.5, 1),
                      obj.shape_config.depth] for obj in building_objs])
    locations[:, 2] += scales[:, 2] / 2
    orientations = np.array([[np.cos(obj.shape_config.angle/2),
                            np.sin(obj.shape_config.angle/2), 0] for obj in building_objs])
    colors = np.repeat(color.reshape(1, -1), n, axis=0)
    return actor.box(locations, orientations, colors, scales)

def create_fallen_trees(map: LandCoverComplexMap, fallen_tree_objs: list[MapObject]):
    locations = np.array([obj.position for obj in fallen_tree_objs])
    locations -= np.array([map.width / 2, map.width / 2, 0])
    locations -= np.array([-0.5, 0.5, 0.0]) # shift to deal with slight shift in ground texture
    radius = 0.1
    locations[:,2] = radius - 0.07
    n = locations.shape[0]
    dirs = np.random.rand(n, 3)
    thetas = [obj.shape_config.angle + np.pi / 2 for obj in fallen_tree_objs] # +pi/2 as angle specifies angle from x axis to long axis of log
    dirs[:, 0] = np.cos(thetas)
    dirs[:, 1] = np.sin(thetas)
    dirs[:, 2] = 0
    trunk_colors = np.repeat(
        np.array([0.8, 0.2, 0.2]).reshape(1, -1), n, axis=0)
    trunks = actor.cylinder(locations, dirs,
                            trunk_colors, radius=radius, heights=9, capped=True)
    return trunks


def render_single_image(scene: window.Scene, location: np.ndarray, focal_point: np.ndarray) -> np.ndarray:
    img_dims = (512, 512)
    scene.set_camera(position=location,
                     focal_point=focal_point, view_up=(0, 0, 1))

    render_window = RenderWindow()
    render_window.AddRenderer(scene)
    render_window.SetSize(img_dims[0], img_dims[1])
    render_window.SetOffScreenRendering(1)
    scene.GetActiveCamera().UseHorizontalViewAngleOn()
    scene.GetActiveCamera().SetViewAngle(45.0)
    scene.GetActiveCamera().OrthogonalizeViewUp()
    scene.GetActiveCamera().UseExplicitAspectRatioOn()
    scene.GetActiveCamera().SetExplicitAspectRatio(1.0)
    scene.GetActiveCamera().SetClippingRange(0.001, 5000)
    # img = window.snapshot(scene, fname=None, size=img_dims, offscreen=True)
    render_window.Render()
    window_to_image_filter = WindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()
    h, w, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    img = numpy_support.vtk_to_numpy(vtk_array).reshape(w, h, components)

    return np.flipud(img).copy()

def render_scene_individual_image_stitch(scene: window.Scene, location: np.ndarray, left_yaw_rad: float) -> np.ndarray:
    # location: (x,y,z) in Fury coordinates (0,0 is center of map)
    #   print(scene, location)
    if scene.GetPass() is not None:
        raise RuntimeError("Scene has pass set, cannot use individual image rendering")
    left_yaw = np.rad2deg(left_yaw_rad)
    num_imgs = 5
    img_dims = (256, 256)
    pano = np.zeros((img_dims[0], img_dims[1]*num_imgs, 3), dtype=np.uint8)
    focal_point = (location[0] + np.cos(np.deg2rad(0)),
                   location[1] + np.sin(np.deg2rad(0)), location[2])

    render_window = RenderWindow()
    scene.set_camera(position=location,
                     focal_point=focal_point, view_up=(0, 0, 1))
    render_window.AddRenderer(scene)
    render_window.SetSize(img_dims[0], img_dims[1])
    scene.SetNearClippingPlaneTolerance(0.00001)
    scene.reset_clipping_range()
    render_window.SetOffScreenRendering(1)

    for i in range(num_imgs):
        angle = i * 360 / num_imgs + left_yaw - (360 / num_imgs / 2)
        scene.GetActiveCamera().Yaw(angle)
        scene.GetActiveCamera().UseHorizontalViewAngleOn()
        scene.GetActiveCamera().SetViewAngle(360.0 / num_imgs)
        scene.GetActiveCamera().OrthogonalizeViewUp()
        scene.GetActiveCamera().UseExplicitAspectRatioOn()
        scene.GetActiveCamera().SetExplicitAspectRatio(1.0)
        scene.GetActiveCamera().SetClippingRange(0.001, 5000)
        tf = utils.Transform().RotateY(-angle)
        scene.SetUserLightTransform(tf)
        render_window.Render()
        window_to_image_filter = WindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.Update()

        vtk_image = window_to_image_filter.GetOutput()
        h, w, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        img = numpy_support.vtk_to_numpy(vtk_array).reshape(w, h, components)
        scene.GetActiveCamera().Yaw(-angle)
        if i != 0:
            g = num_imgs - i
        else:  # first image (pointing +x) goes on the left of the panorama
            g = 0  # the panorama then rotates clockwise from left->right
        pano[:, g*img_dims[1]:(g+1)*img_dims[1], :] = np.flipud(img)
    return pano



def render_scene(scene: window.Scene, location: np.ndarray, left_yaw_rad: float) -> np.ndarray:
    # location: (x,y,z) in Fury coordinates (0,0 is center of map)
    #   print(scene, location)

    center_yaw_deg = np.rad2deg(left_yaw_rad - np.pi) # specifies center camera yaw
    # start with 4 x 2 aspect ration (360 to 180 deg) -> crop to 4 x 1 (90 deg vfov)
    focal_point = (location[0] + np.cos(np.deg2rad(center_yaw_deg)),
                   location[1] + np.sin(np.deg2rad(center_yaw_deg)), location[2])

    scene.set_camera(position=location,
                     focal_point=focal_point, view_up=(0, 0, 1))
    # these two seem required to get an updated image
    scene.window_to_image_filter = WindowToImageFilter()
    scene.window_to_image_filter.SetInput(scene.render_window)
    # scene.render_window.Render()
    # with suppress_stdout():
    scene.window_to_image_filter.Update()
    vtk_image = scene.window_to_image_filter.GetOutput()
    h, w, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    img = numpy_support.vtk_to_numpy(vtk_array).reshape(w, h, components)
    return np.flipud(img)

def attach_render_window(scene: window.Scene, img_dims:tuple):
    scene.render_window = RenderWindow()
    scene.render_window.AddRenderer(scene)
    scene.render_window.SetSize(img_dims[0], img_dims[1])
    scene.render_window.SetOffScreenRendering(1)
    # scene.window_to_image_filter = WindowToImageFilter()
    # scene.window_to_image_filter.SetInput(scene.render_window)

def get_random_location():
    loc = np.random.rand(3) * 2000 - 1000
    loc[2] = 1.5
    return loc
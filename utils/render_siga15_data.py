# -*- coding: utf-8 -*-
# author: yulia
# modified: pinakinathc

import os
import glob
import itertools
import argparse
import time
import numpy as np
import warnings
from multiprocessing import Process, Pool

import bpy
from render_freestyle_svg import register
from mathutils import Vector

register()
warnings.filterwarnings('ignore')

def look_at(obj_camera, point):
    direction = point - obj_camera.location
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def spherical_to_euclidian(elev, azimuth, r):
    x_pos = r * np.cos(elev/180.0*np.pi) * np.cos(azimuth/180.0*np.pi)
    y_pos = r * np.cos(elev/180.0*np.pi) * np.sin(azimuth/180.0*np.pi)
    z_pos = r * np.sin(elev/180.0*np.pi)
    return x_pos, y_pos, z_pos


def iterateTillInsideBounds(val, bounds, origin, mean, std, random_state):
    while val < bounds[0] or val > bounds[1] or (val > bounds[2] and val < bounds[3]):
        val = round(origin + mean + std*random_state.randn())
    return val


def find_longest_diagonal_old(imported):
    local_bbox_center = 0.125 * sum((Vector(b) for b in imported.bound_box), Vector())
    ld = 0.0
    for v in imported.bound_box:
        lv = Vector(local_bbox_center) - Vector(v)
        ld = max(ld, lv.length)
    return ld


def find_longest_diagonal(imported):
    points = np.array([Vector(b) for b in imported.bound_box])
    points = points.max(axis=0) - points.min(axis=0)
    return points.max()


def compute_longest_diagonal(mesh_path):
    try:
        bpy.ops.import_scene.obj(filepath=mesh_path, axis_forward='-X')
        obj_object = bpy.context.selected_objects[0]
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        ld = find_longest_diagonal(obj_object)
        center = obj_object.location
        return ld, list(center)
    except:
        return 1.0


def fill_in_camera_positions():
    num_base_viewpoints = 360 # TODO: change to 360
    num_add_viewpoints = 0

    random_state = np.random.RandomState()


    mean_azi = 0
    std_azi = 7
    mean_elev = 0
    std_elev = 7
    mean_r = 0
    std_r = 7

    delta_azi_max = 15
    delta_elev_max = 15
    delta_azi_min = 5
    delta_elev_min = 5
    delta_r = 0.1

    # azi_origins = np.linspace(0, 359, num_base_viewpoints)
    azi_origins = np.linspace(0, 350, 36)
    elev_origin = 10
    r_origin = 1.5

    bound_azi = [(azi - delta_azi_max, azi + delta_azi_max, azi - delta_azi_min, azi + delta_azi_min) for azi in azi_origins]
    bound_elev = (elev_origin - delta_elev_max, elev_origin + delta_elev_max, elev_origin - delta_elev_min, elev_origin + delta_elev_min)
    bound_r = (r_origin - delta_r, r_origin + delta_r)

    azis = []
    elevs = []
    for azi in azi_origins:
        azis.append(azi)
        elevs.append(elev_origin)

    x_pos = []
    y_pos = []
    z_pos = []
    for azi, elev in zip(azis, elevs):
        x_pos_, y_pos_, z_pos_ = spherical_to_euclidian(elev, azi, r_origin)
        x_pos.append(x_pos_)
        y_pos.append(y_pos_)
        z_pos.append(z_pos_)

    return azis, elevs, x_pos, y_pos, z_pos


def render(folder_name, list_files):
    global opt
    global max_ld

    obj_path = os.path.join(opt.output_dir, 'GEO', 'OBJ', folder_name)
    render_path = os.path.join(opt.output_dir, 'RENDER', folder_name)

    if os.path.exists(os.path.join(render_path, '350_0_00.png')):
        print ('skipping...', render_path)
        return

    os.makedirs(obj_path, exist_ok=True)
    os.makedirs(render_path, exist_ok=True)

    # clean the default blender scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    ###### Import Obj #########
    for idx, filepath in enumerate(list_files):
        bpy.ops.import_scene.obj(filepath=filepath, axis_forward='-X')
        print ('loading...', bpy.data.objects[-1].location)

    # Join objects
    if len(list_files) > 1:
        c = {}
        obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        pseudo_center = obs[0].location
        print ('pseudo_center: ', pseudo_center)
        c["object"] = c["active_object"] = bpy.data.objects[0]
        c["selected_objects"] = c["selected_editable_objects"] = obs
        print (c, obs)
        bpy.ops.object.join(c)
        obj_object = bpy.data.objects[0]
        obj_object = -1 * pseudo_center
        # for ob in bpy.context.scene.objects:
        #     if ob.type == 'MESH':
        #         ob.select = True
        #         bpy.context.scene.objects.active = ob
        #     else:
        #         ob.select = False
        # bpy.ops.object.join()

    obj_object = bpy.data.objects[0]
    print (obj_object)
    maxDimension = 1.0
    scaleFactor = maxDimension / max_ld
    obj_object.scale = (scaleFactor, scaleFactor, scaleFactor)
    center = Vector((0.0, 0.0, -0.7))
    obj_object.location = center

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

    ###### Add a camera ########
    bpy.ops.object.camera_add()
    obj_camera = bpy.data.objects['Camera']

    obj_camera.data.sensor_height = 32
    obj_camera.data.sensor_width = 32
    obj_camera.data.lens = 35
    bpy.context.scene.camera = bpy.context.object

    # Camera parameters:
    azimuths, elevations, x_pos, y_pos, z_pos = fill_in_camera_positions()

    # Set the canvas:
    bpy.data.scenes['Scene'].render.resolution_x = 540
    bpy.data.scenes['Scene'].render.resolution_y = 540
    bpy.data.scenes['Scene'].render.resolution_percentage = 100

    # Render preferences:
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    cycles_preferences = bpy.context.preferences.addons['cycles'].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()
    cycles_preferences.compute_device_type = opt.device
    cuda_devices, opencl_devices = cycles_preferences.get_devices()
    for device in cuda_devices:
        device.use = True
    
    bpy.context.scene.cycles.samples = 4
    bpy.context.scene.cycles.preview_samples = 4
    bpy.context.scene.render.tile_x = 540
    bpy.context.scene.render.tile_y = 540
    bpy.context.scene.render.threads = 1024
    bpy.context.scene.render.threads_mode = 'AUTO'

    ########## Render Sketches ############
    bpy.context.scene.view_layers['View Layer'].use_sky = False
    bpy.context.scene.view_layers['View Layer'].use_ao = False
    bpy.context.scene.view_layers['View Layer'].use_solid = False
    bpy.context.scene.view_layers['View Layer'].use_strand = False
    bpy.context.scene.view_layers['View Layer'].use_volumes = False
    bpy.context.scene.render.film_transparent = True
    bpy.data.scenes['Scene'].render.use_freestyle = True

    # Parameters
    freestyle_settings = bpy.context.scene.view_layers['View Layer'].freestyle_settings
    freestyle_settings.use_culling = True
    freestyle_settings.use_smoothness = True
    freestyle_settings.use_suggestive_contours = True
    # freestyle_settings.crease_angle = np.random.uniform(np.pi/180.0*90, np.pi/180.0*134)
    freestyle_settings.crease_angle = np.pi/180.0*90
    freestyle_settings.use_advanced_options = True
    freestyle_settings.sphere_radius = 0.01
    freestyle_settings.kr_derivative_epsilon = 0

    bpy.data.linestyles['LineStyle'].color = (0, 0, 0)
    bpy.data.linestyles['LineStyle'].geometry_modifiers["Sampling"].sampling = 1.0
    bpy.data.linestyles['LineStyle'].use_length_max = True
    bpy.data.linestyles['LineStyle'].use_length_min = True
    # bpy.data.linestyles['LineStyle'].length_min = 3
    bpy.data.linestyles['LineStyle'].use_chain_count = False
    bpy.data.linestyles['LineStyle'].use_nodes = False

    freestyle_settings.linesets['LineSet'].select_border = True
    freestyle_settings.linesets['LineSet'].select_crease = True
    freestyle_settings.linesets['LineSet'].select_contour = True
    freestyle_settings.linesets['LineSet'].select_suggestive_contour = False
    freestyle_settings.linesets['LineSet'].edge_type_combination = 'OR'
    freestyle_settings.linesets['LineSet'].select_silhouette = True
    freestyle_settings.linesets['LineSet'].select_external_contour = True
    freestyle_settings.linesets['LineSet'].select_edge_mark = True
    # freestyle_settings.linesets['LineSet'].linestyle.thickness = np.random.uniform(1, 1.5)
    freestyle_settings.linesets['LineSet'].linestyle.thickness = 3

    bpy.context.scene.render.image_settings.file_format = 'PNG'

    center = Vector((0.0, 0.0, 0.0))

    for azi, elev, xx, yy, zz in zip(azimuths, elevations, x_pos, y_pos, z_pos):
        obj_camera.location = (xx, yy, zz)
        look_at(obj_camera, center)

        # Render PNG
        bpy.context.scene.render.filepath = os.path.join(render_path, str(int(azi))+'_0_00')
        bpy.context.scene.svg_export.use_svg_export = True
        bpy.ops.render.render(write_still = True)

    # Export resized OBJ
    bpy.ops.export_scene.obj(filepath=os.path.join(obj_path, '%s.obj'%folder_name),
        filter_glob='*.obj', axis_forward='-X')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create realistic 2D render sketch from 3D')
    parser.add_argument('--input_dir', type=str, default='/vol/research/sketchcaption/adobe/adobe-dataset/duygu_dataset/sigasia15/',
        help='Enter input dir to SigA15 dataset')
    parser.add_argument('--output_dir', type=str, default='./../training_data/', help='Enter output dir')
    parser.add_argument('--device', type=str, default='CUDA', help='Use CPU or GPU')
    parser.add_argument('--num_process', type=int, default=None, help='Number of Parallel Processes')
    opt = parser.parse_args()

    print ('Options:\n', opt)

    obj_shirt_list = sorted(glob.glob(os.path.join(opt.input_dir, '*', '*.obj')))

    with Pool(processes=opt.num_process) as pool:
        return_data = pool.map(compute_longest_diagonal, obj_shirt_list)
    longest_diagonal = [item[0] for item in return_data]
    max_ld = max(longest_diagonal)

    global_center = np.array([item[1] for item in return_data])
    global_center = np.mean(global_center, axis=0)
    
    obj_shirt_list = os.listdir(os.path.join(opt.input_dir))
    
    # TODO: remove the next line
    # obj_shirt_list = ['6']

    for folder_name in obj_shirt_list:
        
        # Full OBJ
        full_obj = glob.glob(os.path.join(opt.input_dir, folder_name, '%s.obj'%folder_name))
        render(folder_name, full_obj)

        # Components OBJ
        body_obj = glob.glob(os.path.join(opt.input_dir, folder_name, 'component_obj', 'body_up.obj'))
        if len(body_obj) != 1:
            continue
        components_list = glob.glob(os.path.join(opt.input_dir, folder_name, 'component_obj', '*.obj'))
        components_list = [item for item in components_list if os.path.split(item)[-1] != 'body_up.obj']

        for L in range(0, len(components_list)+1):
            for component in itertools.combinations(components_list, L):
                obj_list = body_obj + list(component)
                part_name = [os.path.split(item)[-1][:-4] for item in component]
                part_name = 'body-'+'-'.join(part_name)
                render('%s-%s'%(folder_name, part_name), obj_list)

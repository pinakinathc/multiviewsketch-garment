# -*- coding: utf-8 -*-
# author: ygryadit
# modified: pinakinathc

import os
import glob
import argparse
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


def find_longest_diagonal(imported):
    local_bbox_center = 0.125 * sum((Vector(b) for b in imported.bound_box), Vector())
    ld = 0.0
    for v in imported.bound_box:
        lv = Vector(local_bbox_center) - Vector(v)
        ld = max(ld, lv.length)
    return ld


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

    # azi_origins = [180, 300, 60]
    azi_origins = [180, 0]
    # azi_origins = [180]
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


def render(filepath, output_dir, filename=None):
    global opt

    ####### Init Setup #########
    if filename is None:
        filename = os.path.split(filepath)[-1]

    print (output_dir, filename)
    render_path = os.path.join(output_dir, filename+'_NPR')

    # clean the default blender scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    ###### Import Obj #########
    print (filepath)
    bpy.ops.import_scene.obj(filepath=filepath, axis_forward='-X')
    # bpy.ops.import_scene.obj(filepath=os.path.join(
    #     '/vol/research/NOBACKUP/CVSSP/scratch_4weeks/pinakiR/tmp_dataset/siga15/GEO/OBJ',
    #     filename, filepath+'.obj'
    #     ), axis_forward='-X')
    obj_object = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    maxDimension = 0.56
    ld = find_longest_diagonal(obj_object)
    scaleFactor = maxDimension / ld
    obj_object.scale = (scaleFactor, scaleFactor, scaleFactor)

    center = Vector((0.0, 0.0, 0.0))
    obj_object.location = center

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
    cycles_preferences.compute_device_type = 'CUDA'
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
    bpy.data.scenes['Scene'].render.line_thickness = np.random.uniform(1, 1.5)
    bpy.data.linestyles['LineStyle'].color = (0, 0, 0)

    obj_object.data.use_auto_smooth = True
    obj_object.data.auto_smooth_angle = np.pi/180.0 * 90
    obj_object.cycles.shadow_terminator_offset = 0.9
    obj_object.modifiers.new(name='Smooth', type='SMOOTH')
    # obj_object.modifiers['Smooth'].iterations = 5

    freestyle_settings = bpy.context.scene.view_layers['View Layer'].freestyle_settings
    freestyle_settings.use_culling = True
    freestyle_settings.use_smoothness = True
    freestyle_settings.use_suggestive_contours = True
    freestyle_settings.crease_angle = np.random.uniform(np.pi/180.0*120, np.pi/180.0*134)
    freestyle_settings.use_advanced_options = True
    freestyle_settings.sphere_radius = 0.0
    freestyle_settings.kr_derivative_epsilon = 0.001
    
    bpy.data.linestyles['LineStyle'].geometry_modifiers["Sampling"].sampling = 1.0
    bpy.data.linestyles['LineStyle'].use_length_max = True
    bpy.data.linestyles['LineStyle'].use_length_min = True
    bpy.data.linestyles['LineStyle'].length_min = 3 # 17 # 11.1
    bpy.data.linestyles['LineStyle'].use_chain_count = False
    bpy.data.linestyles['LineStyle'].use_nodes = False
    bpy.ops.scene.freestyle_geometry_modifier_add(type='BACKBONE_STRETCHER')
    bpy.data.linestyles['LineStyle'].geometry_modifiers["Backbone Stretcher"].backbone_length = 3.0
    bpy.data.linestyles['LineStyle'].geometry_modifiers["Sampling"].sampling = 1.0
    bpy.ops.scene.freestyle_geometry_modifier_add(type='BEZIER_CURVE')
    bpy.data.linestyles['LineStyle'].geometry_modifiers["Bezier Curve"].error = 10.0
    bpy.ops.scene.freestyle_geometry_modifier_add(type='SIMPLIFICATION')
    bpy.data.linestyles['LineStyle'].geometry_modifiers["Simplification"].tolerance = 0.5


    freestyle_settings.linesets['LineSet'].select_border = True
    freestyle_settings.linesets['LineSet'].select_silhouette = True
    freestyle_settings.linesets['LineSet'].select_crease = True
    freestyle_settings.linesets['LineSet'].select_contour = True
    freestyle_settings.linesets['LineSet'].select_external_contour = True
    freestyle_settings.linesets['LineSet'].select_suggestive_contour = True
    freestyle_settings.linesets['LineSet'].edge_type_combination = 'OR'
    freestyle_settings.linesets['LineSet'].select_edge_mark = False
    freestyle_settings.linesets['LineSet'].linestyle.thickness =  np.random.uniform(1, 1.5)

    bpy.context.scene.render.image_settings.file_format = 'PNG'

    center = Vector((0.0, 0.0, 0.0))
    for azi, elev, xx, yy, zz in zip(azimuths, elevations, x_pos, y_pos, z_pos):
        obj_camera.location = (xx, yy, zz)
        look_at(obj_camera, center)

        # Render PNG
        # bpy.context.scene.render.filepath = os.path.join(render_path, str(int(azi))+'_0_00')
        bpy.context.scene.render.filepath = render_path+str(int(azi))+''#+'_0_00'
        bpy.context.scene.svg_export.use_svg_export = False
        bpy.ops.render.render(write_still = True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create realistic 2D render NPR from 3D')
    parser.add_argument('--input_dir', type=str, default='output/*.obj', help='Enter input dir to raw dataset')
    parser.add_argument('--output_dir', type=str, default='output/', help='Enter output dir')
    parser.add_argument('--device', type=str, default='CUDA', help='Use CPU or GPU')
    parser.add_argument('--num_process', type=int, default=None, help='Number of Parallel Processes')
    opt = parser.parse_args()

    print ('Options:\n', opt)

    # obj_shirt_list = np.loadtxt(opt.input_dir, dtype=str)
    obj_shirt_list = glob.glob(opt.input_dir)

    # with Pool(processes=opt.num_process) as pool:
    #     pool.map(render, obj_shirt_list)
    for obj_shirt in obj_shirt_list:
        render(obj_shirt, opt.output_dir)

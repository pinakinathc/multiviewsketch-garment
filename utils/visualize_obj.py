# -*- coding: utf-8 -*-
# author: pinakinathc

import sys,os
import bpy
from render_freestyle_svg import register
import warnings
from mathutils import Vector
import numpy as np
import argparse
import glob
import imageio

register()
warnings.filterwarnings("ignore")

def look_at(obj_camera, point):
    
    direction = point - obj_camera.location
    print(direction)
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()
    print(obj_camera.rotation_euler)

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
    # num_base_viewpoints = 6
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
    azi_origins = np.linspace(0, 360, 10)[:-1]
    elev_origin = 10
    r_origin = 1.5
    
    bound_azi = [(azi - delta_azi_max, azi + delta_azi_max, azi - delta_azi_min, azi + delta_azi_min) for azi in azi_origins]
    bound_elev = (elev_origin - delta_elev_max, elev_origin + delta_elev_max, elev_origin - delta_elev_min, elev_origin + delta_elev_min)
    bound_r = (r_origin - delta_r, r_origin + delta_r)
    
    # 8 defalut viewpoints:
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
        
    # Additional viewpoints:
    # for n_azi in range(num_base_viewpoints):
    #     for _ in range(num_add_viewpoints):
    #         azi = round(azi_origins[n_azi] + mean_azi + std_azi*random_state.randn())
    #         iterateTillInsideBounds(azi, bound_azi[n_azi], azi_origins[n_azi], mean_azi, std_azi, random_state)
            
    #         elev = round(elev_origin + mean_elev + std_elev*random_state.randn())
    #         iterateTillInsideBounds(elev, bound_elev, elev_origin, mean_elev, std_elev, random_state)

    #         while (azi, elev) in list(zip(azis, elevs)):   # control (azi, elev) not repeated
    #             azi = round(azi_origins[n_azi] + mean_azi + std_azi*random_state.randn())
    #             iterateTillInsideBounds(azi, bound_azi[n_azi], azi_origins[n_azi], mean_azi, std_azi, random_state)
            
    #             elev = round(elev_origin + mean_elev + std_elev*random_state.randn())
    #             iterateTillInsideBounds(elev, bound_elev, elev_origin, mean_elev, std_elev, random_state)

    #         r = r_origin + mean_r + std_r * random_state.randn()
    #         while r < bound_r[0] or r > bound_r[1]:       # control bound for r
    #             r = r_origin + mean_r + std_r * random_state.randn()

    #         azis.append(azi)
    #         elevs.append(elev)
    #         x_pos_, y_pos_, z_pos_ = spherical_to_euclidian(elev, azi, r)
    #         x_pos.append(x_pos_)
    #         y_pos.append(y_pos_)
    #         z_pos.append(z_pos_)
        
    return azis, elevs, x_pos, y_pos, z_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise obj mesh")
    parser.add_argument('--obj_path', type=str, default='./', help='enter path to obj mesh')
    parser.add_argument('--output_path', type=str, default='./', help='enter path to rendered images')
    opt = parser.parse_args()

    # Enter obj_path that needs visualization
    obj_path = opt.obj_path
    output_path = opt.output_path # Enter output path

    # Clean the default blender scene:
    bpy.ops.object.select_all(action="SELECT")  
    bpy.ops.object.delete(use_global=False)

    # Import Obj
    filepath = obj_path
    print ("filepath:", filepath)

    bpy.ops.import_scene.obj(filepath=filepath, axis_forward='X')
    obj_object = bpy.context.selected_objects[0];
    
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    maxDimension = 0.7 #0.56
    ld = find_longest_diagonal(obj_object)
    #scaleFactor = maxDimension / max(imported.dimensions)
    scaleFactor = maxDimension / ld
    obj_object.scale = (scaleFactor,scaleFactor,scaleFactor)
    
    local_bbox_center = 0.125 * sum((Vector(b) for b in obj_object.bound_box), Vector())
    center = Vector((0.0, 0.0, 0.0));
    obj_object.location = center

    print('Imported name: ', obj_object.name)
    
    ################
    #Add a camera
    bpy.ops.object.camera_add()
    obj_camera = bpy.data.objects["Camera"]
    
    obj_camera.data.sensor_height = 45
    obj_camera.data.sensor_width = 45
    obj_camera.data.lens = 35
    
    bpy.context.scene.camera = bpy.context.object
    
    ################
    #Camera parameters:    
    azimuths, elevations, x_pos, y_pos, z_pos = fill_in_camera_positions()
   
    #Set the canvas
    bpy.data.scenes['Scene'].render.resolution_x = 540
    bpy.data.scenes['Scene'].render.resolution_y = 540
    bpy.data.scenes['Scene'].render.resolution_percentage = 100
    
    #Set the canvas
    bpy.data.scenes['Scene'].render.resolution_x = 540
    bpy.data.scenes['Scene'].render.resolution_y = 540
    bpy.data.scenes['Scene'].render.resolution_percentage = 100

    #Render preferences:
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 4
    bpy.context.scene.cycles.preview_samples = 4
    # bpy.context.scene.view_layers["View Layer"].use_sky = False

    #################
    # Add a Light
    bpy.ops.object.light_add(type='POINT', location=(3,0,0))
    bpy.context.object.data.energy = 100

    bpy.ops.object.light_add(type='POINT', location=(-3, 0, 0))
    bpy.context.object.data.energy = 100

    # bpy.ops.object.light_add(type='POINT', location=(-1.5,1.5,0))
    # bpy.context.object.data.energy = 100

    # bpy.ops.object.light_add(type='POINT', location=(1.5,-1.5,0))
    # bpy.context.object.data.energy = 100

    # bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    # 
    # bpy.ops.object.light_add(type='POINT', location=(0,0,3))
    # bpy.context.object.data.energy = 70    

    i = 0    
    for azi, elev, xx, yy, zz in zip(azimuths, elevations, x_pos, y_pos, z_pos):
        i = i+1
        obj_camera.location = (xx,yy,zz)        
        look_at(obj_camera, center)

        # azi = 359-azi
        bpy.context.scene.render.filepath = os.path.join(output_path, 'azi_{}_elev_{}_'.format(int(azi), int(elev)))
        bpy.context.scene.svg_export.use_svg_export = False #True
        bpy.ops.render.render(write_still = True)

    images = []
    for filename in glob.glob(os.path.join(output_path, '*.png')):
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(output_path, 'result.gif'), images, duration=3.0)

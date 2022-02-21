import os
import glob

from numpy import random
import tqdm
import argparse
import numpy as np
import trimesh
# from igl import signed_distance
from igl import winding_number
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Precomputes saved points from Meshs')
parser.add_argument('--data_dir', type=str, help='Path to all Meshes')
opts = parser.parse_args()


def compute_scale (mesh_path):
    try:
        mesh = trimesh.load(mesh_path)
        mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
        scene = trimesh.Scene(mesh)
        return scene.scale
    except:
        return 0


if __name__ == '__main__':
    output_dir = os.path.join(opts.data_dir, 'all_mesh_points')
    os.makedirs(output_dir, exist_ok=True)
    local_state = np.random.RandomState()

    all_obj_shirt_list = glob.glob(os.path.join(opts.data_dir, 'GEO', 'OBJ', '*', '*.obj'))

    # global max_ld # Make LD global since it would be constant for entire dataset
    with Pool(processes=None) as pool:
        all_scale = pool.map(compute_scale, all_obj_shirt_list)
    max_scale = max(all_scale)

    # Get all garments path
    for mesh_path in tqdm.tqdm(all_obj_shirt_list):
        mesh_name = os.path.split(mesh_path)[0]
        mesh_name = os.path.split(mesh_name)[-1]
        if os.path.exists(os.path.join(output_dir, '%s.npy'%mesh_name)):
            continue
        mesh = trimesh.load(mesh_path)
        
        # Resize mesh
        try:
            mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
            scene = trimesh.Scene(mesh)
            scene = scene.scaled(1.5/max_scale)
            mesh = scene.geometry[list(scene.geometry.keys())[0]]
        except:
            continue

        # Sample points
        surface_points, _ = trimesh.sample.sample_surface(mesh, 250000)

        # Outside close to mesh
        near_points_out = surface_points + local_state.normal(
            0.01, 0.03, surface_points.shape)
        # sdf_out = signed_distance(near_points_out, mesh.vertices, mesh.faces)[0][:, np.newaxis]
        sdf_out = winding_number(mesh.vertices, mesh.faces, near_points_out)[:, np.newaxis]

        # Inside close to mesh
        near_points_in = surface_points + local_state.normal(
            -0.01, 0.03, surface_points.shape)
        # sdf_in = signed_distance(near_points_in, mesh.vertices, mesh.faces)[0][:, np.newaxis]
        sdf_in = winding_number(mesh.vertices, mesh.faces, near_points_in)[:, np.newaxis]

        # Random points
        B_MAX = surface_points.max(0)
        B_MIN = surface_points.min(0)
        length = (B_MAX - B_MIN)
        random_points = local_state.rand(25000, 3) * length + B_MIN
        # sdf_rand = signed_distance(random_points, mesh.vertices, mesh.faces)[0][:, np.newaxis]
        sdf_rand = winding_number(mesh.vertices, mesh.faces, random_points)[:, np.newaxis]

        to_save = {
            'outside': np.concatenate([near_points_out, sdf_out], axis=-1),
            'inside': np.concatenate([near_points_in, sdf_in], axis=-1),
            'random': np.concatenate([random_points, sdf_rand], axis=-1)
        }
        
        np.save(os.path.join(output_dir, '%s.npy'%mesh_name), to_save)

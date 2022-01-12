import os
import glob
import tqdm
import argparse
import numpy as np
import trimesh
# from igl import signed_distance
from igl import winding_number
from multiprocessing import Pool
from utils import save_vertices_ply

parser = argparse.ArgumentParser(description='Precomputes saved points from Meshs')
parser.add_argument('--data_dir', type=str, help='Path to all Meshes')
opts = parser.parse_args()


def resize_ply(mesh, scale, centroid):
    mesh = trimesh.Trimesh(mesh.vertices - centroid)
    scene = trimesh.Scene(mesh)
    scene = scene.scaled(1.5/scale)
    mesh = scene.geometry[list(scene.geometry.keys())[0]]
    return mesh.vertices


def render_partial_mesh(azi):
    save_path = os.path.join(opts.data_dir, 'partial_mesh_points', garment, '%d.npy'%azi)
    if os.path.exists(save_path):
        return

    ray_direction = np.array(
        [[-1*np.sin(np.radians(azi)), 0, -1*np.cos(np.radians(azi))]]).T
    dot_product = np.dot(normals, ray_direction)[:, 0]
    idx = np.where(dot_product >= 0.5)

    surface_points = mesh_verts[idx]
    # save_vertices_ply(
    #     os.path.join('output', '%d.ply'%count),
    #     surface_points
    # )
    # count += 1
    # continue
    # surface_points = resize_ply(partial_mesh, scale, centroid)

    surface_points = surface_points[local_state.choice(
        surface_points.shape[0], 250000)]

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
    near_points = np.concatenate([near_points_in, near_points_out], axis=0)
    B_MAX = near_points.max(0)
    B_MIN = near_points.min(0)
    length = (B_MAX - B_MIN)
    random_points = local_state.rand(25000, 3) * length + B_MIN
    # sdf_rand = signed_distance(random_points, mesh.vertices, mesh.faces)[0][:, np.newaxis]
    sdf_rand = winding_number(mesh.vertices, mesh.faces, random_points)[:, np.newaxis]

    to_save = {
        'outside': np.concatenate([near_points_out, sdf_out], axis=-1),
        'inside': np.concatenate([near_points_in, sdf_in], axis=-1),
        'random': np.concatenate([random_points, sdf_rand], axis=-1)
    }            
    np.save(save_path, to_save)


def main(garment):
    if len(glob.glob(os.path.join(
        opts.data_dir, 'partial_mesh_points', garment, '*.npy')))==36:
        print ('skipping %s'%garment)
        return

    os.makedirs(os.path.join(
        opts.data_dir, 'partial_mesh_points', garment), exist_ok=True)
    
    # Loads original mesh
    mesh_path = glob.glob(os.path.join(
        opts.data_dir, 'GEO', 'OBJ', garment, '*.obj'))[0]
    global mesh
    mesh = trimesh.load(mesh_path, skip_materials=False)
    
    # Resize mesh
    mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
    scene = trimesh.Scene(mesh)
    scene = scene.scaled(1.5/scene.scale)
    mesh = scene.geometry[list(scene.geometry.keys())[0]]
    global normals
    normals = trimesh.geometry.mean_vertex_normals(
        mesh.vertices.shape[0], mesh.faces, mesh.face_normals)
    global mesh_verts
    mesh_verts = mesh.vertices

    # count = 0
    # for azi in tqdm.tqdm(range(0, 360, 10)):
    with Pool(processes=None) as pool:
        pool.map(render_partial_mesh, list(range(0, 360, 10)))


if __name__ == '__main__':
    os.makedirs(os.path.join(opts.data_dir, 'partial_mesh_points'), exist_ok=True)
    local_state = np.random.RandomState()
    all_garments = os.listdir(os.path.join(
        opts.data_dir, 'GEO', 'OBJ'))
    # all_garments = np.loadtxt(os.path.join(
    #     opts.data_dir, 'val.txt'
    # ), dtype=str)[:21]

    # # Get all garments path

    print (all_garments)
    for garment in tqdm.tqdm(all_garments):
        main(garment)
    
    # with Pool(processes=8) as pool:
    #     pool.map(main, all_garments)

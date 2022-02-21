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


def resize_ply(mesh, scale, centroid):
    mesh = trimesh.Trimesh(mesh.vertices - centroid)
    scene = trimesh.Scene(mesh)
    scene = scene.scaled(1.5/scale)
    mesh = scene.geometry[list(scene.geometry.keys())[0]]
    return mesh.vertices

def render_partial_mesh(azi):
    save_path = os.path.join(opt.output_dir, 'partial_mesh_points', garment, '%d.npy'%azi)
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


def main(mesh_path, scale):

    global garment
    garment = os.path.split(mesh_path)[0]
    if os.path.split(garment)[-1] == 'component_obj':
        garment = os.path.split(garment)[0]
    garment = os.path.split(garment)[-1]

    if len(glob.glob(os.path.join(
        opt.output_dir, 'partial_mesh_points', garment, '*.npy')))==36:
        print ('skipping %s'%garment)
        return

    os.makedirs(os.path.join(
        opt.output_dir, 'partial_mesh_points', garment), exist_ok=True)
    
    # Loads original mesh
    global mesh
    mesh = trimesh.load(mesh_path, skip_materials=False)
    
    # Resize mesh
    try:
        mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
        scene = trimesh.Scene(mesh)
        scene = scene.scaled(1.5/scale)
        mesh = scene.geometry[list(scene.geometry.keys())[0]]
        global normals
        normals = trimesh.geometry.mean_vertex_normals(
            mesh.vertices.shape[0], mesh.faces, mesh.face_normals)
        global mesh_verts
        mesh_verts = mesh.vertices
    except:
        return

    # count = 0
    # for azi in tqdm.tqdm(range(0, 360, 10)):
    with Pool(processes=None) as pool:
        pool.map(render_partial_mesh, list(range(0, 360, 10)))


def compute_scale (mesh_path):
    try:
        mesh = trimesh.load(mesh_path)
        mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
        scene = trimesh.Scene(mesh)
        return scene.scale
    except:
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create realistic 2D render sketch from 3D')
    parser.add_argument('--input_dir', type=str, default='garment_dataset/shirt_dataset_rest/*/shirt_mesh_r.obj', help='Enter input dir to raw dataset')
    parser.add_argument('--output_dir', type=str, default='training_data/', help='Enter output dir')
    parser.add_argument('--device', type=str, default='CUDA', help='Use CPU or GPU')
    parser.add_argument('--num_process', type=int, default=None, help='Number of Parallel Processes')
    opt = parser.parse_args()

    print ('Options:\n', opt)

    local_state = np.random.RandomState()
    obj_shirt_list = sorted(glob.glob(opt.input_dir))

    # global max_ld # Make LD global since it would be constant for entire dataset
    with Pool(processes=opt.num_process) as pool:
        all_scale = pool.map(compute_scale, obj_shirt_list)
    max_scale = max(all_scale)

    for mesh_path in tqdm.tqdm(obj_shirt_list):
        main(mesh_path, max_scale)

    


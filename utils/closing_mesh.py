import trimesh
from skimage import measure
from igl import signed_distance
import numpy as np
import os
import tqdm
from multiprocessing import Pool

reso = 256
grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T
root_dir = '/vol/research/NOBACKUP/CVSSP/scratch_4weeks/pinakiR/tmp_dataset/training_data/wang18'

list_foldernames = os.listdir(root_dir)

def compute_scale (mesh_path):
    mesh = trimesh.load(mesh_path)
    mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
    scene = trimesh.Scene(mesh)
    return scene.scale

# global max_ld # Make LD global since it would be constant for entire dataset
with Pool(processes=None) as pool:
    all_scale = pool.map(compute_scale, list_foldernames)
    max_scale = max(all_scale)

for foldername in tqdm.tqdm(list_foldernames):
    if os.path.exists(os.path.join('data/obj_meshes', 'watertight_%s.obj'%foldername)):
        continue
    
    mesh_path = '%s/%s/%s.obj'%(root_dir, foldername, foldername)
    if not os.path.exists(mesh_path):
        print ('skipping: %s'%mesh_path)
        continue
    mesh = trimesh.load(mesh_path)
    scene = trimesh.Scene(mesh)
    scene = scene.scaled(1.5/max_scale)
    mesh = scene.geometry[list(scene.geometry.keys())[0]]
    mesh.rezero()
    mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)

    gt_sdf = signed_distance(grid, mesh.vertices, mesh.faces)[0][:, np.newaxis]
    gt_sdf = gt_sdf.reshape((reso, reso, reso))
    verts, faces, normals, values = measure.marching_cubes(gt_sdf, 0.0)

    with open(os.path.join('data/obj_meshes', 'watertight_%s.obj'%foldername), 'w') as fp:

        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces+1: # faces are 1-based, not 0-based in obj
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

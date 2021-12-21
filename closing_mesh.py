import trimesh
from skimage import measure
from igl import signed_distance
import numpy as np
import os
import tqdm

reso = 256
grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T

list_foldernames = os.listdir('../adobe-dataset/duygu_dataset/sigasia15/')
for foldername in tqdm.tqdm(list_foldernames):
        if os.path.exists(os.path.join('data/obj_meshes', 'watertight_%s.obj'%foldername)):
            continue
        # mesh_path = '../adobe-dataset/duygu_dataset/sigasia15/%s/component_obj/body_up.obj'%(foldername)
        mesh_path ='../adobe-dataset/duygu_dataset/sigasia15/%s/%s.obj'%(foldername, foldername)
        if not os.path.exists(mesh_path):
            print ('skipping: %s'%mesh_path)
            continue
        mesh = trimesh.load(mesh_path)
        scene = trimesh.Scene(mesh)
        scene = scene.scaled(1.5/scene.scale)
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

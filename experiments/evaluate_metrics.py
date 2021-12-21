import argparse
import os
import trimesh
import numpy as np
import glob
import tqdm
import torch
from pytorch3d.loss.chamfer import chamfer_distance
from igl import signed_distance
from skimage import measure
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser(description='Evaluate Multiview Garment Modeling')

parser.add_argument('--root_dir', type=str, default='output',
        help='Enter directory where all OBJs are kept')

opt = parser.parse_args()

reso = 256
grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T


def mesh_normals(mesh):
        # resize mesh
        scene = trimesh.Scene(mesh)
        scene = scene.scaled(1.5/scene.scale)
        mesh = scene.geometry[list(scene.geometry.keys())[0]]
        mesh.rezero()
        mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)

        sdf = signed_distance(grid, mesh.vertices, mesh.faces)[0][:, np.newaxis]
        sdf = sdf.reshape((reso, reso, reso))
        verts, faces, normals, values = measure.marching_cubes(sdf, 0.0)
        return verts, normals


def compute_view_normals(verts, normals, azi):
        ray_direction = np.array(
                                [[np.sin(np.radians(azi)), 0, np.cos(np.radians(azi))]]).T

        dot_product = np.dot(normals, ray_direction)[:, 0]
        idx = np.where(dot_product >= 0.0)

        subset_verts = verts[idx]
        subset_norms = dot_product[idx]
        [x_med, y_med, z_med] = np.median(verts, axis=0)

        canvas = np.zeros((reso, reso))
        for ((x, y, z), norm) in zip(subset_verts, subset_norms):
                x1 = (x-x_med)*np.cos(np.radians(azi)) - (z-z_med)*np.sin(np.radians(azi))
                x1 = x1 + x_med
                canvas[reso - int(y), reso - int(x1)] = norm
        return canvas


if __name__ == '__main__':
        filename_list = ['12', '14', '100', '117', '141', '143', '146']
        filename_list = ['141', '143', '146']
        # filename_list = ['146', '143']

        # list_shirts = np.loadtxt(os.path.join('..', 'data', 'body_up', 'val.txt'), dtype=str)
        # filename_list = [os.path.split(path)[-1][11:-4] for path in list_shirts]

        # all_azi = [180, 240, 60] # normal
        # all_azi = [180, 180, 180] # overlap
        all_azi = [180, 0] # non-overlap

        for filename in (filename_list):
                gt_filename = 'watertight_%s_gt.obj'%filename
                for vid in range(len(all_azi)):
                        pred_filename = 'watertight_%s_pred_view_%d_*.obj'%(filename, vid)

                        gt_obj = trimesh.load(os.path.join(opt.root_dir, gt_filename))
                        pred_obj = trimesh.load(glob.glob(os.path.join(opt.root_dir, pred_filename))[0])

                        chamfer_dist = chamfer_distance(
                                torch.tensor(gt_obj.vertices).unsqueeze(0).type(torch.float32),
                                torch.tensor(pred_obj.vertices).unsqueeze(0).type(torch.float32))[0].item()

                        print ('Filename: %s, view: %d, Chamfer Distance: %f'%(
                                filename, vid, chamfer_dist))

                        # Calculate SSIM of Normal maps
                        gt_verts, gt_normals = mesh_normals(gt_obj)
                        pred_verts, pred_normals = mesh_normals(pred_obj)

                        for azi in all_azi:
                                gt_normal_maps = compute_view_normals(gt_verts, gt_normals, azi)
                                pred_normal_maps = compute_view_normals(pred_verts, pred_normals, azi)

                                ssim_val = ssim(gt_normal_maps, pred_normal_maps)

                                print ('Filename: %s, view: %d, Normal Maps: %f'%(
                                filename, azi, ssim_val))

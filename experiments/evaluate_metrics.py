import argparse
from ctypes import resize
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
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Evaluate Multiview Garment Modeling')

parser.add_argument('--data_dir', type=str, default='/vol/research/NOBACKUP/CVSSP/scratch_4weeks/pinakiR/tmp_dataset/adobe_training_data/siga15/',
        help='Enter directory where all OBJs are kept')
parser.add_argument('--output_dir', type=str, default='output',
        help='Enter directory where all OBJs are kept')

opt = parser.parse_args()

reso = 128
grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T


def compute_scale (mesh_path):
    try:
        mesh = trimesh.load(mesh_path)
        mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
        scene = trimesh.Scene(mesh)
        return scene.scale
    except:
        return 0


def resize_mesh(mesh, max_scale):
        mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
        scene = trimesh.Scene(mesh)
        scene = scene.scaled(1.5/max_scale)
        mesh = scene.geometry[list(scene.geometry.keys())[0]]
        return mesh


def mesh_normals(mesh, scale):
        # resize mesh
        scene = trimesh.Scene(mesh)
        scene = scene.scaled(1.5/scene.scale)
        mesh = scene.geometry[list(scene.geometry.keys())[0]]
        mesh.rezero()
        mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)

        sdf = signed_distance(grid, mesh.vertices, mesh.faces)[0][:, np.newaxis]
        sdf = sdf.reshape((reso, reso, reso))
        verts, faces, normals, values = measure.marching_cubes(sdf, 0.0)
        # verts = mesh.vertices
        # verts = (verts - verts.min()) / (verts.max() - verts.min()) * 255
        # normals = trimesh.geometry.mean_vertex_normals(len(mesh.vertices), mesh.faces, mesh.face_normals)
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
        all_obj_shirt_list = glob.glob(os.path.join(opt.data_dir, 'GEO', 'OBJ', '*', '*.obj'))
        # global max_ld # Make LD global since it would be constant for entire dataset
        with Pool(processes=None) as pool:
                all_scale = pool.map(compute_scale, all_obj_shirt_list)
        max_scale = max(all_scale)

        pred_obj_list = glob.glob(os.path.join(opt.output_dir, '*.obj'))
        with Pool(processes=None) as pool:
                all_scale = pool.map(compute_scale, pred_obj_list)
        max_pred_scale = max(all_scale)

        val_list = np.loadtxt(os.path.join(opt.data_dir, 'val.txt'), dtype=str)[:5]
        filename_list = val_list

        # all_azi = [180, 240, 60] # normal
        # all_azi = [180, 180, 180] # overlap
        all_azi = [180, 0] # non-overlap

        all_chamfer_dist = []

        for filename in filename_list:
                all_gt_obj, all_gt_verts, all_gt_normals, all_pred_obj, all_pred_verts, all_pred_normals = [], [], [], [], [], []
                all_tmp_filename = []
                closest_mesh = np.loadtxt(os.path.join(opt.data_dir, 'closest_mesh', '%s.txt'%filename), dtype=str)
                closest_mesh = [filename]
                for tmp_filename in closest_mesh:
                        gt_filename = os.path.join(opt.data_dir, 'GEO', 'OBJ', tmp_filename, '%s.obj'%tmp_filename)
                        gt_obj = trimesh.load(gt_filename)
                        gt_obj = resize_mesh(gt_obj, max_scale)
                        # gt_verts, gt_normals = mesh_normals(gt_obj)
                        all_gt_obj.append(gt_obj)
                        # all_gt_verts.append(gt_verts)
                        # all_gt_normals.append(gt_normals)
                        all_tmp_filename.append(tmp_filename)
                
                for vid in range(len(all_azi[:1])):
                        pred_filename = glob.glob(os.path.join(opt.output_dir, '%s_pred_view_1_*.obj'%(filename)))[0]
                        pred_obj = trimesh.load(pred_filename)
                        pred_obj = resize_mesh(pred_obj, max_pred_scale)
                        # pred_verts, pred_normals = mesh_normals(pred_obj)
                        all_pred_obj.append(pred_obj)
                        # all_pred_verts.append(pred_verts)
                        # all_pred_normals.append(pred_normals)

                for gt_idx in range(len(all_gt_obj)):
                        tmp_filename = all_tmp_filename[gt_idx]
                        for vid in range(len(all_pred_obj)):
                                try:
                                        gt_obj = all_gt_obj[gt_idx]
                                        pred_obj = all_pred_obj[vid]

                                        chamfer_dist = chamfer_distance(
                                                torch.tensor(gt_obj.vertices).unsqueeze(0).type(torch.float32),
                                                torch.tensor(pred_obj.vertices).unsqueeze(0).type(torch.float32))[0].item()

                                        all_chamfer_dist.append(chamfer_dist)

                                        # print ('Filename: %s, view: %d, Chamfer Distance: %f'%(
                                        #         filename, vid, chamfer_dist))

                                        # # Calculate SSIM of Normal maps
                                        # gt_verts, gt_normals = all_gt_verts[gt_idx], all_gt_normals[gt_idx]
                                        # pred_verts, pred_normals = all_pred_verts[vid], all_pred_normals[vid]

                                        # for azi in all_azi:
                                        #         gt_normal_maps = compute_view_normals(gt_verts, gt_normals, azi)
                                        #         pred_normal_maps = compute_view_normals(pred_verts, pred_normals, azi)

                                        #         ssim_val = ssim(gt_normal_maps, pred_normal_maps)

                                        #         print ('Filename: %s, GT: %s viewID: %d, azi: %d, SSIM: %f'%(
                                        #         filename, tmp_filename, vid, azi, ssim_val))
                                except:
                                        pass

        print ("Chamfer distance: mean={}, std={}".format(np.mean(all_chamfer_dist), np.std(all_chamfer_dist)))
      

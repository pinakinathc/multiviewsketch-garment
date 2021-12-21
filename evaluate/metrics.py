import argparse
import os
import time
import trimesh
import numpy as np
import glob
import tqdm
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss.chamfer import chamfer_distance
from igl import signed_distance
from skimage import measure
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser(description='Evaluate Multiview Garment Modeling')

parser.add_argument('--val_txt', type=str, help='enter val txt path')
parser.add_argument('--gt_obj', type=str, help='enter GT obj')
parser.add_argument('--pred_obj', type=str, help='enter pred obj')
parser.add_argument('--azi', type=int, nargs='+', default=[180, 300, 60], help='enter azis')

opt = parser.parse_args()

reso = 256
grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T

def resize_mesh(mesh):
# resize mesh
        scene = trimesh.Scene(mesh)
        scene = scene.scaled(1.5/scene.scale)
        mesh = scene.geometry[list(scene.geometry.keys())[0]]
        mesh.rezero()
        mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
        return mesh


def mesh_normals(mesh):
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

    gt_filename = opt.gt_obj
    pred_filename = opt.pred_obj

    list_filenames = np.loadtxt(opt.val_txt, dtype=str)
    list_filenames = [item[:-4] for item in list_filenames]

    metrics = np.zeros((len(opt.azi), len(opt.azi)))
    metrics  = {
        'chamfer': [[] for i in range(len(opt.azi))],
        'ssim': [[] for i in range(len(opt.azi))]
    }

    for filenames in tqdm.tqdm(list_filenames[:]):
        gt_obj = trimesh.load(os.path.join(
            opt.gt_obj, filenames, '%s.obj'%filenames
        ))
        prev_chamfer = None
        prev_ssim = None
        for idx, azii in enumerate(opt.azi):
            try:
                pred_obj = trimesh.load(os.path.join(
                    opt.pred_obj, '%s_pred_view_%d_%d.obj'%(filenames, idx, azii)
                ))
            except:
                break

            # chamfer_dist = chamfer_distance(
            #     torch.tensor(gt_obj.vertices).unsqueeze(0).type(torch.float32),
            #     torch.tensor(pred_obj.vertices).unsqueeze(0).type(torch.float32))[0].item()
            chamfer_dist = 0.0

            if prev_chamfer is None:
                metrics['chamfer'][0].append(chamfer_dist)
                prev_chamfer = chamfer_dist
            else:
                metrics['chamfer'][idx].append(chamfer_dist-prev_chamfer)

            gt_mesh = Meshes(
                    verts=[torch.FloatTensor(gt_obj.vertices)],
                    faces=[torch.LongTensor(gt_obj.faces)])

            pred_mesh = Meshes(
                    verts=[torch.FloatTensor(pred_obj.vertices)],
                    faces=[torch.LongTensor(pred_obj.faces)])

            gt_verts = gt_obj.vertices
            gt_normals = gt_mesh.verts_normals_list()[0].numpy()
            pred_verts = pred_obj.vertices
            pred_normals = pred_mesh.verts_normals_list()[0].numpy()

            ssim_curr = np.zeros((len(opt.azi)))
            for jdx, azi in enumerate(opt.azi):
                gt_normal_maps = compute_view_normals(gt_verts, gt_normals, azi)
                pred_normal_maps = compute_view_normals(pred_verts, pred_normals, azi)

                ssim_val = ssim(gt_normal_maps, pred_normal_maps)
                ssim_curr[jdx] = ssim_val

                # print ('Azi: %d, Normal Maps: %f'%(azi, ssim_val))
            
            if prev_ssim is None:
                metrics['ssim'][0].append(ssim_curr)
                prev_ssim = ssim_curr
            else:
                metrics['ssim'][idx].append(ssim_curr-prev_ssim)
    
    metrics['chamfer'] = np.array(metrics['chamfer'])
    metrics['ssim'] = np.array(metrics['ssim'])

    for i in range(len(opt.azi)):
        print ('Metrics: Iteration: %d \nChamfer Distance: %.5f +- %.5f'%(
            i, np.mean(metrics['chamfer'][i]), np.std(metrics['chamfer'][i])
        ))
        for j in range(len(opt.azi)):
            print ('Iteration: %d, view: %d, SSIM: %.5f +- %.5f'%(
                i, j, np.mean(metrics['ssim'][i, :, j]), np.std(metrics['ssim'][i, :, j])
            ))

    np.save('metrics.npy', metrics)

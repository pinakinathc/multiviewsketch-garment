# -*- coding: utf-8 -*-

import os
import glob
import tqdm
import argparse
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from skimage import measure
from igl import signed_distance
from PIL import Image, ImageOps
from utils.error_surface import error_surface


parser = argparse.ArgumentParser(description='Evaluate Multiview Garment Modeling')
parser.add_argument('--model_name', type=str, default='model_A')
parser.add_argument('--data_dir', type=str, default='/vol/research/NOBACKUP/CVSSP/scratch_4weeks/pinakiR/tmp_dataset/adobe_training_data/siga15/', help='enter data dir')
parser.add_argument('--output_dir', type=str, help='enter output mesh path')
parser.add_argument('--ckpt', type=str, default='saved_models_full/new_model_A_siga15_full.ckpt', help='enter model path')
opt = parser.parse_args()

if opt.model_name == 'model_A':
    from src.model_A import GarmentModel
elif opt.model_name == 'model_AA':
    from src.model_AA import GarmentModel
elif opt.model_name == 'model_B':
    from src.model_B import GarmentModel
elif opt.model_name == 'model_BB':
    from src.model_BB import GarmentModel
elif opt.model_name == 'model_C':
    from src.model_C import GarmentModel
elif opt.model_name == 'model_D':
    from src.model_D import GarmentModel
elif opt.model_name == 'model_E':
    from src.model_E import GarmentModel
elif opt.model_name == 'model_F':
    from src.model_F import GarmentModel
elif opt.model_name == 'model_G':
    from src.model_G import GarmentModel
elif opt.model_name == 'model_H':
    from src.model_H import GarmentModel
elif opt.model_name == 'model_I':
    from src.model_I import GarmentModel
else:
    raise ValueError('opts.model_name option wrong: %s'%opt.model_name)

os.makedirs(os.path.join(opt.output_dir), exist_ok=False)

# Image transforms
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

if __name__ == '__main__':
    output_dir = opt.output_dir
    model = GarmentModel.load_from_checkpoint(opt.ckpt)
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    reso = 256
    num_samples = 2048
    grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T

    view_ids = [180, 300, 60]
    # view_ids = [180, 180, 180]
    # view_ids = [180, 0]

    list_garments = np.loadtxt(os.path.join(opt.data_dir, 'val.txt'), dtype=str)
    list_garments = list_garments[:5]

    for garment in list_garments:
        list_input_imgs = []
        for idx, view_id in enumerate(view_ids):
            
            # # For disentanglement experiment
            # closest_garment = np.loadtxt(os.path.join(opt.data_dir, 'closest_mesh', '%s.txt'%garment), dtype=str)
            # tmp_garment = closest_garment[0::10][idx]
            # tmp_garment = list_garments[idx]
            # list_input_imgs.append(os.path.join(
            #     opt.data_dir, 'RENDER', tmp_garment, '%d_0_00.png'%view_id))

            # For consistent setup
            list_input_imgs.append(os.path.join(
                opt.data_dir, 'RENDER', garment, '%d_0_00.png'%view_id))
        img_tensor = []
        pos_emb_feat = []
        
        for idx, view_id in enumerate(view_ids):
            img_path = list_input_imgs[idx]
            image = Image.open(img_path).convert('RGBA').split()[-1].convert('RGB')
            ImageOps.invert(image).save(os.path.join(opt.output_dir, '%s_pred_view_%d_%d.jpg'%(garment, idx, view_id)))
            t_img = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0))

            pos_emb = []
            for p in [1, 2, 4, 8, 16]:
                    pos_emb.append(np.sin(np.radians(view_id*p)))
                    pos_emb.append(np.cos(np.radians(view_id*p)))
            
            img_tensor.append(t_img)
            pos_emb_feat.append(pos_emb)
        
        num_views = len(view_ids)
        predicted_sdf = np.zeros((num_views, grid.shape[0], 1))

        img_tensor = torch.cat(img_tensor, dim=0)
        img_tensor = Variable(torch.FloatTensor(img_tensor).unsqueeze(0)).to(device)
        pos_emb_feat = Variable(torch.FloatTensor(pos_emb_feat).unsqueeze(0)).to(device)

        """ Evaluate mesh """
        for idx in tqdm.tqdm(range(0, grid.shape[0], num_samples)):
            lst_sdf = np.arange(idx, idx+num_samples)
            xyz = grid[lst_sdf, 0:3]
            xyz = Variable(torch.FloatTensor(xyz)).to(device)
            
            # Shape: 1 x num_views x num_samples x 3
            xyz = xyz.unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1, 1)
            
            # shape of all_pred_sdf: B x num_views x num_points x 1
            all_pred_sdf, _, all_alpha = model(img_tensor, pos_emb_feat, xyz)[:3]
            
            for i, view_id in enumerate(range(num_views)):
                # predicted_sdf[i, lst_sdf, :] = all_pred_sdf[i][i*2048:(i+1)*2048].cpu().data.numpy()
                predicted_sdf[i, lst_sdf, :] = all_pred_sdf[i].cpu().data.numpy()

                # Visualise alpha
                if idx == 0:
                    alpha = all_alpha[:, i, :].cpu().data.numpy().reshape(-1)
                    plt.plot(np.arange(alpha.shape[0]), alpha)
                    plt.savefig(os.path.join(opt.output_dir, '%s_pred_alpha_%d_%d.jpg'%(garment, i, view_id)))
                    plt.clf()

        for vid, view_id in enumerate(view_ids):
            """ Performing Marching Cubes on Predicted """
            winding_num = predicted_sdf[vid].reshape((reso, reso, reso))
            winding_num = np.gradient(winding_num)
            winding_num = np.stack(winding_num)
            winding_num = (winding_num**2).sum(axis=0)**0.5

            """ Mesh Paths """
            gt_mesh_path = glob.glob(os.path.join(opt.data_dir, 'GEO', 'OBJ', '%s/*.obj'%(garment)))[0]
            pred_mesh_path =  os.path.join(opt.output_dir, '%s_pred_view_%d_%d.obj'%(garment, vid, view_id))
            error_mesh_path = os.path.join(opt.output_dir, '%s_error_view_%d_%d.obj'%(garment, vid, view_id))

            """ Saving Mesh """
            try:
                verts, faces, normals, values = measure.marching_cubes(-winding_num, -0.3)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=10)
                verts, faces = mesh.vertices, mesh.faces
                with open(pred_mesh_path, 'w') as fp:
                    for v in verts:
                        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                    for f in faces+1: # faces are 1-based, not 0-based in obj
                        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
            except:
                    print ('Failed to perform marching cubes on %s View %d'%(garment, view_id))
                    continue
    
            """ Error Mesh """
            # error_surface(gt_mesh_path, pred_mesh_path, error_mesh_path)
        

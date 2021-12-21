# -*- coding: utf-8 -*-

import os
import glob
import tqdm
import argparse
import numpy as np
import trimesh
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from skimage import measure
from igl import signed_distance
from PIL import Image
from utils.extract_image_feature import get_vector
from src.model_new import GarmentModel


parser = argparse.ArgumentParser(description='Evaluate Multiview Garment Modeling')
parser.add_argument('--output_dir', type=str, default='evaluate/1/', help='enter val txt path')
opt = parser.parse_args()

# Image transforms
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

if __name__ == '__main__':
    output_dir = 'evaluate/1/'
    output_dir = opt.output_dir
    # model = GarmentModel(output_dir='/vol/research/sketchcaption/adobe/multiview_modeling/output/sigasia15-body-up')
    model = GarmentModel.load_from_checkpoint('saved_models_lab/siggraph15_data_new_updater-val_loss=0.1.ckpt')
    # model = GarmentModel()
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    reso = 256
    num_samples = 2048
    grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T

    view_ids = [180, 300, 60]
    # view_ids = [180, 180, 180]
    # view_ids = [180, 0]

    # list_garments = np.loadtxt('/vol/research/sketchcaption/adobe/multiview_modeling/data/body_up/val.txt', dtype=str)
    # list_garments = [item[:-4] for item in list_garments]
    list_garments = ['watertight_313']
    # list_garments = ['W8EXCGDKFZST']
    
    for garment in list_garments:
        list_input_imgs = []
        for idx, view_id in enumerate(view_ids):
            list_input_imgs.append(os.path.join(
                output_dir, '%s_NPR%d.png'%(garment, view_id)))
        img_tensor = []
        pos_emb_feat = []
        
        for idx, view_id in enumerate(view_ids):
            img_path = list_input_imgs[idx]
            image = Image.open(img_path).convert('RGBA').split()[-1].convert('RGB')
            feat = get_vector(image)
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
            all_pred_sdf = model(img_tensor, pos_emb_feat, xyz)[0]
            
            # print ('shape of all_pred_sdf: ', [x.shape for x in all_pred_sdf])
            for i in range(num_views):
                predicted_sdf[i, lst_sdf, :] = all_pred_sdf[i][i*2048:(i+1)*2048].cpu().data.numpy()
            # predicted_sdf[:, lst_sdf, :] = all_pred_sdf.cpu().data.numpy()

        """ Performing Marching Cubes on Predicted """
        # for view_id in range(num_views):
        for vid, view_id in enumerate(view_ids):
            sdf = predicted_sdf[vid].reshape((reso, reso, reso))
            try:
                verts, faces, normals, values = measure.marching_cubes(sdf, 0.0)
                with open(os.path.join(
                    output_dir, '%s_pred_view_%d_%d.obj'%(garment, vid, view_id)), 'w') as fp:

                    for v in verts:
                        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                    for f in faces+1: # faces are 1-based, not 0-based in obj
                        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
            except:
                    print ('Failed to perform marching cubes on %s View %d'%(shirtname, view_id))
    
        
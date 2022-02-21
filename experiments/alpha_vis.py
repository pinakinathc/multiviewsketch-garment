# -*- coding: utf-8 -*-

import os
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from src.model_A import GarmentModel
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description='Evaluate Multiview Garment Modeling')
parser.add_argument('--data_dir', type=str, default='/vol/research/NOBACKUP/CVSSP/scratch_4weeks/pinakiR/tmp_dataset/adobe_training_data/siga15/', help='enter data dir')
parser.add_argument('--input_dir', type=str, default='output/siga15/1/', help='enter evaluation image path')
parser.add_argument('--output_dir', type=str, help='enter output mesh path')
parser.add_argument('--ckpt', type=str, default='saved_models_jade/new_model_A_siga15_full.ckpt', help='enter model path')
opt = parser.parse_args()

# Image transforms
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

if __name__ == '__main__':
    model = GarmentModel.load_from_checkpoint(opt.ckpt)
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    all_alphas = [] # Store all alpha vertices for tSNE plot
    color = [] # Color coding each garment

    reso = 10
    num_samples = 2048
    grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T

    views = [180, 300, 60]
    views = np.arange(0, 360, 36)
    list_garments = np.loadtxt(os.path.join(opt.data_dir, 'val.txt'), dtype=str)
    # list_garments = list_garments[:21]

    for g_idx, garment in enumerate(list_garments):
        list_inp_imgs = []
        for idx, view_id in enumerate(views):
            list_inp_imgs.append(
                os.path.join(opt.data_dir, 'RENDER', garment, '%d_0_00.png'%view_id)
            )

        img_tensor = []
        pos_emb_feat = []

        for idx, view_id in enumerate(views):
            img_path = list_inp_imgs[idx]
            image = Image.open(img_path).convert('RGBA').split()[-1].convert('RGB')
            t_img = Variable(normalize(to_tensor(scaler(image)))).unsqueeze(0)

            pos_emb = []
            for p in [1, 2, 4, 8, 16]:
                pos_emb.append(np.sin(np.radians(view_id*p)))
                pos_emb.append(np.cos(np.radians(view_id*p)))
            
            img_tensor.append(t_img)
            pos_emb_feat.append(pos_emb)

        num_views = len(views)
        pred = np.zeros((num_views, grid.shape[0], 1))

        img_tensor = torch.cat(img_tensor, dim=0)
        img_tensor = Variable(torch.FloatTensor(img_tensor).unsqueeze(0)).to(device)
        pos_emb_feat = Variable(torch.FloatTensor(pos_emb_feat).unsqueeze(0)).to(device)

        """ Evaluate mesh """
        xyz = grid[:10, 0:3]
        xyz = Variable(torch.FloatTensor(xyz)).to(device)
        
        # Shape: 1 x num_views x num_samples x 3
        xyz = xyz.unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1, 1)
        
        # shape of alphas: B x num_views x 512
        _, _, alphas = model(img_tensor, pos_emb_feat, xyz)[:3]
        alphas = alphas.reshape(-1, 512).cpu().detach().numpy()
        all_alphas.append(alphas)
        color.append(np.ones(alphas.shape[0])*g_idx)

    all_alphas = np.concatenate(all_alphas, axis=0) # N x 512
    color = np.concatenate(color, axis=0) # N
    color = (color - color.min()) / (color.max() - color.min()) * 255.0

    X_embedded = TSNE(
        n_components=2,
        learning_rate='auto',
        init='random').fit_transform(all_alphas)

    xcoord = X_embedded[:, 0]
    ycoord = X_embedded[:, 1]

    plt.scatter(xcoord[0::3], ycoord[0::3], c=color[0::3], marker='x')
    plt.scatter(xcoord[1::3], ycoord[1::3], c=color[1::3], marker='^')
    plt.scatter(xcoord[2::3], ycoord[2::3], c=color[2::3], marker='.')
    plt.axis('off')
    plt.savefig(
        os.path.join(opt.output_dir, 'tSNE.png'),
        bbox_inches='tight',
        dpi=512)
    

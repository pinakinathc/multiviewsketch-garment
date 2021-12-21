# -*- coding: utf-8 -*-

import os
import tqdm
import numpy as np
import trimesh
import torch
from torch.autograd import Variable
from skimage import measure
from igl import signed_distance
from PIL import Image
from utils.extract_image_feature import get_vector
from options import opt
from model import GarmentModel

if __name__ == '__main__':
        output_dir = 'output/sigasia15-body-full-inconsistent'
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

        model = GarmentModel(output_dir=output_dir, use_partial=False)
        if len(opt.device) and torch.cuda.is_available():
                device = torch.device('cuda')
        else:
                device = torch.device('cpu')
        model = model.to(device)
        model.load_model(output_dir)
        model.eval()

        reso = 256
        num_samples = 2048
        grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T
        yaw_list = list(range(0, 360, 1))

        """ Getting Data for Evaluation """
        list_shirts = os.listdir(os.path.join(opt.root_dir, "GEO", "OBJ"))
        # list_shirts = ['3', '11', '10']
        list_shirts = np.loadtxt(os.path.join('data', 'body_up', 'val.txt'), dtype=str)
        # list_shirts = ['watertight_146.obj', 'watertight_143.obj']
        for shirtname in list_shirts[:]:
                shirtname = shirtname[:-4]
                mesh = trimesh.load(os.path.join(
                        opt.root_dir, "GEO", "OBJ", shirtname, "%s.obj"%shirtname))

                """ Resize mesh and center it """
                scene = trimesh.Scene(mesh)
                scene = scene.scaled(1.5/scene.scale)
                mesh = scene.geometry[list(scene.geometry.keys())[0]]
                mesh.rezero()
                mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)

                gt_sdf = signed_distance(grid, mesh.vertices, mesh.faces)[0][:, np.newaxis]

                # view_ids = [180, 300, 60]
                # view_ids = [180, 240, 60] # normal 
                # view_ids = [180, 180, 180] # overlap
                view_ids = [180, 0] # non-overlap
                img_feat = []
                pos_emb_feat = []
                all_images = []

                """ Get image features """
                for idx, view_id in enumerate(view_ids):
                        img_path = os.path.join(opt.root_dir, 'RENDER', shirtname, '%d_0_00.png'%(view_id))

                        """ inconsistent experiment """
                        # if view_id == 180:
                        #     img_path = os.path.join(root_dir, 'RENDER', 'L9UVRMFHQWVK', '%d_0_00.png'%(view_id))
                        # if view_id == 300:
                        #     img_path = os.path.join(root_dir, 'RENDER', 'G5YBBARPCUQB', '%d_0_00.png'%(view_id))
                        # if view_id == 60:
                        #     img_path = os.path.join(root_dir, 'RENDER', 'B4USJQDASCMA', '%d_0_00.png'%(view_id))

                        image = Image.open(img_path).convert('RGBA').split()[-1].convert('RGB')
                        all_images.append(image)
                        feat = get_vector(image)

                        pos_emb = []
                        for p in [1, 2, 4, 8, 16]:
                                pos_emb.append(np.sin(np.radians(view_id*p)))
                                pos_emb.append(np.cos(np.radians(view_id*p)))
                        
                        img_feat.append(feat)
                        pos_emb_feat.append(pos_emb)

                """ Save raw input sketch """
                widths, heights = zip(*(i.size for i in all_images))
                total_width = sum(widths)
                max_height = max(heights)
                new_im = Image.new('RGB', (total_width, max_height))
                x_offset = 0
                for im in all_images:
                  new_im.paste(im, (x_offset,0))
                  x_offset += im.size[0]
                new_im.save(os.path.join(
                        output_dir, 'results', '%s_sketch.jpg'%(shirtname)))

                num_views = len(view_ids)
                # Store the resting predicted sdf
                predicted_sdf = np.zeros((num_views, grid.shape[0], 1))

                # Shape: 1 x num_views x dim
                img_feat = Variable(torch.FloatTensor(img_feat).unsqueeze(0)).to(device)
                pos_emb_feat = Variable(torch.FloatTensor(pos_emb_feat).unsqueeze(0)).to(device)

                """ Evaluate mesh """
                print ('Evaluating mesh %s'%shirtname)
                for idx in tqdm.tqdm(range(0, grid.shape[0], num_samples)):
                        lst_sdf = np.arange(idx, idx+num_samples)
                        xyz = grid[lst_sdf, 0:3]
                        xyz = Variable(torch.FloatTensor(xyz)).to(device)
                        # Shape: 1 x num_views x num_samples x 3
                        xyz = xyz.unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1, 1)
                        # shape of all_pred_sdf: B x num_views x num_points x 1
                        all_pred_sdf = model(img_feat, pos_emb_feat, xyz)[0]
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
                                        output_dir, 'results', '%s_pred_view_%d_%d.obj'%(shirtname, vid, view_id)), 'w') as fp:

                                        for v in verts:
                                                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                                        for f in faces+1: # faces are 1-based, not 0-based in obj
                                                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
                        except:
                                print ('Failed to perform marching cubes on %s View %d'%(shirtname, view_id))

                """ Perform Marching Vubes on GT SDF for comparison """
                gt_sdf = gt_sdf.reshape((reso, reso, reso))
                verts, faces, normals, values = measure.marching_cubes(gt_sdf, 0.0)
                with open(os.path.join(output_dir, 'results', '%s_gt.obj'%shirtname), 'w') as fp:

                        for v in verts:
                                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                        for f in faces+1: # faces are 1-based, not 0-based in obj
                                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

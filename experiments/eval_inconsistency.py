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
	output_dir = 'experiments'

	os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

	model = GarmentModel(output_dir='output/inconsistent', use_partial=False)
	if len(opt.device) and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	model = model.to(device)
	model.load_model()
	model.eval()

	reso = 256
	num_samples = 2048
	grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T
	yaw_list = list(range(0, 360, 1))

	""" Getting Data for Evaluation """
	list_shirts = np.loadtxt(os.path.join('data', 'val.txt'), dtype=str)[:2]

	for shirtname in list_shirts:
		related_shirts = np.loadtxt(os.path.join(
			'data', 'closest_mesh', '%s.txt'%shirtname), dtype=str)

		# for topk in [2, 10, 20]:
		for topk in [2]:	
			all_shirtname = [shirtname, related_shirts[topk]]
			# all_shirtname = [shirtname, shirtname, shirtname]
			print ('processing: ', all_shirtname)
			
			all_mesh = []
			all_gt_sdf = []

			for tmp_shirtname in all_shirtname:
				mesh = trimesh.load(os.path.join(
					opt.root_dir, 'GEO', 'OBJ', tmp_shirtname, 'shirt_mesh_r_tmp_watertight.obj'))
				""" Resize mesh and center it """
				scene = trimesh.Scene(mesh)
				scene = scene.scaled(1.5/scene.scale)
				mesh = scene.geometry['shirt_mesh_r_tmp_watertight.obj']
				mesh.rezero()
				mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
				gt_sdf = signed_distance(grid, mesh.vertices, mesh.faces)[0][:, np.newaxis]
				all_mesh.append(mesh)
				all_gt_sdf.append(gt_sdf)

			view_ids = [180, 0]
			# view_ids = [180, 180, 180]
			
			img_feat = []
			pos_emb_feat = []
			all_images = []

			""" Get image features """
			for idx, view_id in enumerate(view_ids):
				img_path = os.path.join(opt.root_dir, 'RENDER', all_shirtname[idx], '%d_0_00.png'%view_id)
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
				output_dir, 'results', '%s_input.jpg'%('-'.join(all_shirtname))))

			num_views = len(view_ids)
			# Store the resting predicted sdf
			predicted_sdf = np.zeros((num_views, grid.shape[0], 1))

			# Shape: 1 x num_views x dim
			img_feat = Variable(torch.FloatTensor(img_feat).unsqueeze(0)).to(device)
			pos_emb_feat = Variable(torch.FloatTensor(pos_emb_feat).unsqueeze(0)).to(device)

			""" Evaluate mesh """
			print ('Evaluating mesh %s'%'-'.join(all_shirtname))
			for idx in tqdm.tqdm(range(0, grid.shape[0], num_samples)):
				lst_sdf = np.arange(idx, idx+num_samples)
				xyz = grid[lst_sdf, 0:3]
				xyz = Variable(torch.FloatTensor(xyz)).to(device)
				# Shape: 1 x num_views x num_samples x 3
				xyz = xyz.unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1, 1)
				# shape of all_pred_sdf: B x num_views x num_points x 1
				all_pred_sdf = model(img_feat, pos_emb_feat, xyz)[0]
				for vid in range(num_views):
					predicted_sdf[vid, lst_sdf, :] = all_pred_sdf[vid][-1*num_samples:].cpu().data.numpy()

			""" Performing Marching Cubes on Predicted """
			for view_id in range(num_views):
				sdf = predicted_sdf[view_id].reshape((reso, reso, reso))
				try:
					verts, faces, normals, values = measure.marching_cubes(sdf, 0.0)
					with open(os.path.join(
						output_dir, 'results', '%s_t%d_pred_view_%d.obj'%(
							'-'.join(all_shirtname), topk, view_id)), 'w') as fp:

						for v in verts:
							fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
						for f in faces+1: # faces are 1-based, not 0-based in obj
							fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
				except:
					print ('Failed to perform marching cubes on %s View %d'%(shirtname, view_id))

			""" Perform Marching Vubes on GT SDF for comparison """
			for view_id in range(num_views):
				gt_sdf = all_gt_sdf[view_id]
				tmp_shirtname = all_shirtname[view_id]
				print ('processing %s GT Mesh'%tmp_shirtname)
				gt_sdf = gt_sdf.reshape((reso, reso, reso))
				verts, faces, normals, values = measure.marching_cubes(gt_sdf, 0.0)
				with open(os.path.join(
					output_dir, 'results', '%s_gt.obj'%(tmp_shirtname)), 'w') as fp:

					for v in verts:
						fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
					for f in faces+1: # faces are 1-based, not 0-based in obj
						fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

# -*- coding: utf-8 -*-

import trimesh
import glob
import tqdm
import os
from igl import signed_distance
import numpy as np
import argparse
from skimage import measure
import matplotlib.pyplot as plt


def get_normals(obj_path):
	shirtname = os.path.split(obj_path)[-1]
	reso = 256
	grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T

	mesh = trimesh.load(obj_path)

	# resize mesh
	scene = trimesh.Scene(mesh)
	scene = scene.scaled(1.5/scene.scale)
	mesh = scene.geometry[shirtname]
	mesh.rezero()
	mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)

	sdf = signed_distance(grid, mesh.vertices, mesh.faces)[0][:, np.newaxis]
	sdf = sdf.reshape((reso, reso, reso))
	verts, faces, normals, values = measure.marching_cubes(sdf, 0.0)

	view_ids = [180, 300, 60]

	for view_id in view_ids:
		ray_direction = np.array(
		                [[np.sin(np.radians(view_id)), 0, np.cos(np.radians(view_id))]]).T
		dot_product = np.dot(normals, ray_direction)[:, 0]
		idx = np.where(dot_product >= 0.0)

		subset_verts = verts[idx]
		subset_norms = dot_product[idx]
		[x_med, y_med, z_med] = np.median(verts, axis=0)

		canvas = np.zeros((reso, reso))
		for ((x, y, z), norm) in zip(subset_verts, subset_norms):
			x1 = (x-x_med)*np.cos(np.radians(view_id)) - (z-z_med)*np.sin(np.radians(view_id))
			x1 = x1 + x_med
			canvas[reso - int(y), reso - int(x1)] = norm

		np.save(os.path.join(opt.output_dir, '%s_%d.npy'%(shirtname, view_id)), canvas)
		plt.imshow(canvas)
		plt.savefig(os.path.join(opt.output_dir, '%s_normal_%d.png'%(shirtname[:-4], view_id)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get normal maps')
	parser.add_argument('--input_dir', type=str, default='output/*.obj',
		help='Enter input dir to meshes')
	parser.add_argument('--output_dir', type=str, default='output/',
		help='Enter output dir')
	opt = parser.parse_args()

	print ('Options:\n', opt)

	obj_shirt_list = glob.glob(opt.input_dir)

	for objpath in tqdm.tqdm(obj_shirt_list):
		get_normals(objpath)

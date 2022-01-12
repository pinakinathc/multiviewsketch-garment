# -*- coding: utf-8 -*-

import os
import numpy as np
import tqdm
import torch
import trimesh
from chamferdist import ChamferDistance
from options import opts

def mesh_resize(mesh):
	mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
	scene = trimesh.Scene(mesh)
	scene = scene.scaled(1.5/scene.scale)
	mesh = scene.geometry[list(scene.geometry.keys())[0]]
	return mesh

distance_fn = ChamferDistance()

def process_similar(list_shirts):
	list_shirts = np.array(list_shirts)
	obj_closest = {}
	all_objs = []
	print ('loading meshes ...')
	for shirtname in tqdm.tqdm(list_shirts):
		shirtpath = os.path.join(
			opts.data_dir, "GEO", "OBJ", shirtname, "%s.obj"%shirtname)
		obj = trimesh.load(shirtpath)
		obj = mesh_resize(obj)
		obj_tensor = torch.tensor(obj.vertices, dtype=torch.float32).unsqueeze(0)
		all_objs.append(obj_tensor.cuda())

	# find closest shirt
	print ('calculating closest ...')
	for q_idx in tqdm.tqdm(range(len(list_shirts))):
		shirtname, query_verts = list_shirts[q_idx], all_objs[q_idx]
		all_distances = []
		for s_idx, shirt_verts in enumerate(all_objs):
			all_distances.append(distance_fn(query_verts, shirt_verts, bidirectional=True).item())

		sorted_idx = np.argsort(all_distances)
		obj_closest[shirtname] = list_shirts[sorted_idx]
		np.savetxt(os.path.join(opts.data_dir, 'closest_mesh', '%s.txt'%shirtname), list_shirts[sorted_idx], fmt='%s')
	return obj_closest


if __name__ == '__main__':
	os.makedirs(os.path.join(opts.data_dir, 'closest_mesh'), exist_ok=True)
	list_shirts = os.listdir(os.path.join(opts.data_dir, "GEO", "OBJ"))
	val_shirts = np.loadtxt(os.path.join(opts.data_dir, 'val.txt'), dtype=str)[:21]
	train_shirts = sorted(list(set(list_shirts) - set(val_shirts)))

	process_similar(train_shirts)
	process_similar(val_shirts)

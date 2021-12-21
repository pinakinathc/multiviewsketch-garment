# -*- coding: utf-8 -*-

import os
import numpy as np
import tqdm
import torch
import trimesh
from chamferdist import ChamferDistance
from options import opt
from utils.render_npr import render as render_npr

distance_fn = ChamferDistance()

def process_similar(val_shirts, train_shirts, output_dir):
	train_shirts = np.array(train_shirts)
	val_shirts = np.array(val_shirts)
	val_objs = []
	train_objs = []
	
	print ('loading meshes ...')
	for shirtname in tqdm.tqdm(val_shirts):
		shirtpath = os.path.join(
			opt.root_dir, "GEO", "OBJ", shirtname, "shirt_mesh_r_tmp_watertight.obj")
		obj = trimesh.load(shirtpath)
		obj_tensor = torch.tensor(obj.vertices, dtype=torch.float32).unsqueeze(0)
		val_objs.append(obj_tensor.cuda())

	for shirtname in tqdm.tqdm(train_shirts):
		shirtpath = os.path.join(
			opt.root_dir, "GEO", "OBJ", shirtname, "shirt_mesh_r_tmp_watertight.obj")
		obj = trimesh.load(shirtpath)
		obj_tensor = torch.tensor(obj.vertices, dtype=torch.float32).unsqueeze(0)
		train_objs.append(obj_tensor.cuda())

	# find closest shirt
	print ('calculating closest ...')
	for q_idx in tqdm.tqdm(range(len(val_shirts))):
		shirtname, query_verts = val_shirts[q_idx], val_objs[q_idx]
		all_distances = []

		os.makedirs(os.path.join(output_dir, shirtname), exist_ok=True)
		render_npr(
			os.path.join(opt.root_dir, "GEO", "OBJ", shirtname, "shirt_mesh_r_tmp_watertight.obj"),
			os.path.join(output_dir, shirtname), filename="query.obj")

		for s_idx, shirt_verts in enumerate(train_objs):
			all_distances.append(distance_fn(query_verts, shirt_verts, bidirectional=True).item())

		sorted_idx = np.argsort(all_distances)
		print (train_shirts[sorted_idx])
		np.savetxt('data/train-test-diff/%s.txt'%shirtname, train_shirts[sorted_idx], fmt='%s')

		for idx, shirt_idx in enumerate(sorted_idx[::10][:4]):
			render_npr(
				os.path.join(opt.root_dir, "GEO", "OBJ", train_shirts[shirt_idx], "shirt_mesh_r_tmp_watertight.obj"),
				os.path.join(output_dir, shirtname), filename="%d_%s.obj"%(idx, train_shirts[shirt_idx]))


if __name__ == '__main__':
	list_shirts = os.listdir(os.path.join(opt.root_dir, "GEO", "OBJ"))
	val_shirts = np.loadtxt('data/val.txt', dtype=str)
	train_shirts = sorted(list(set(list_shirts) - set(val_shirts)))

	output_dir = 'experiments/train-test-diff'
	os.makedirs(output_dir, exist_ok=True)

	process_similar(val_shirts, train_shirts, output_dir)

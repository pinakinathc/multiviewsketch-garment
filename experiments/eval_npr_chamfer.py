# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import trimesh
from utils import calculate_npr_dist, calculate_mesh_view_dist

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluate distance between meshes and their NPRs')
	parser.add_argument('--input_dir', type=str, default='experiments/results', help='enter path to pred and gt meshes')
	opt = parser.parse_args()

	output_dir = 'experiments'

	# os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
	os.path.exists(opt.input_dir)
	
	""" Getting Data for Evaluation """
	list_shirts = np.loadtxt(os.path.join('data', 'val.txt'), dtype=str)
	list_shirts = ['T7UNJRFVTMZV', 'L9UVRMFHQWVK']

	for shirtname in list_shirts:
		related_shirts = np.loadtxt(os.path.join(
			'data', 'closest_mesh', '%s.txt'%shirtname), dtype=str)

		# for topk in [2, 10, 20]:
		for topk in [2]:
			all_shirtname = [shirtname, related_shirts[topk]]
			# all_shirtname = [shirtname, shirtname, shirtname]

			# for vid, view_id in enumerate([180, 300, 60]):
			for vid, view_id in enumerate([180, 0]):
				for idx in range(1):
					filename1 = '%s_t%d_pred_view_%d_NPR%d.png'%(
						'-'.join(all_shirtname), topk, idx, view_id)

					mesh1 = trimesh.load(os.path.join(opt.input_dir,
						'%s_t%d_pred_view_%d.obj'%('-'.join(all_shirtname), topk, idx)))
					
					# filename2 = '%s_t%d_pred_view_%d_NPR%d.png'%(
					# 	'-'.join(all_shirtname), topk, vid, view_id)

					# distance = calculate_npr_dist(
					# 		os.path.join(output_dir, 'results', filename1),
					# 		os.path.join(output_dir, 'results', filename2))
					# print ('Distance between %s and %s is: %f'%(filename1, filename2, distance))

					for gt_shirtname in all_shirtname:
						filename2 = '%s_gt_NPR%d.png'%(gt_shirtname, view_id)

						# npr_distance = calculate_npr_dist(
						# 		os.path.join(opt.input_dir, filename1),
						# 		os.path.join(opt.input_dir, filename2))
						npr_distance = 0.0

						mesh2 = trimesh.load(os.path.join(opt.input_dir, '%s_gt.obj'%(gt_shirtname)))
						mesh_dist = calculate_mesh_view_dist(mesh1, mesh2, vid)

						print ('Distance between %s and %s is: %f(NPR) %f(Mesh)'%(
							filename1, filename2, npr_distance, mesh_dist))

					# for vid in range(0, idx):
					# 	filename2 = '%s_t%d_pred_view_%d_NPR%d.png'%(
					# 		'-'.join(all_shirtname), topk, vid, view_id)
					# 	distance = calculate_npr_dist(
					# 				os.path.join(output_dir, 'results', filename1),
					# 				os.path.join(output_dir, 'results', filename2))
					# 	print ('Distance between %s and %s is: %f'%(filename1, filename2, distance))

				print ('\n\n')